#pragma once
#include <numeric>
#include "env.h"
#include "net.h"

struct PPOCfg {
  int iters = 2;          // PPO iterations per genome per generation (0 => pure neuroevolution)
  int rollout = 256, epochs = 4, minibatch = 64;
  bool greedy_eval = false;   // fitness evaluation episodes take argmax over legal actions instead of sampling
  float lr = 2e-3f, gamma = 0.99f, lam = 0.95f, clip = 0.2f, vf = 0.5f, ent = 0.01f, max_grad = 0.5f;
};

// One Trainer per worker thread: owns env instance, buffers, phenotype. No allocation after warm-up.
class Trainer {
public:
  std::unique_ptr<Env> env; PPOCfg cfg; long steps = 0, train_steps = 0, eval_steps = 0;   // steps = train + eval
  Trainer(std::unique_ptr<Env> e, PPOCfg c) : env(std::move(e)), cfg(c), rng(1) {
    pa = env->per_action(); ns = env->n_state(); n_in = env->n_in(); n_logit = env->n_logit(); nact = env->n_actions();
    cap = cfg.minibatch * (pa ? nact : 1);
    leg.resize(nact); z.resize(nact); p.resize(nact); lp.resize(nact); dz.resize(nact); tmp.resize(n_in);
    off.resize(cfg.minibatch);
  }

  // Evaluate an already-trained policy without further PPO updates.
  float evaluate_policy(const Genome& G, int eval_eps, uint64_t seed) {
    rng = Rng(seed); steps = train_steps = eval_steps = 0;
    P.build(G, n_logit, cap);
    assert(P.n_in == n_in);
    return eval(eval_eps);
  }

  // Held-out evaluation with per-episode detail. greedy=true takes argmax over legal actions, else samples.
  // Exact episode results are used when available; otherwise outcome counts fall back to return sign.
  struct EvalDetail {
    double mean = 0;
    int pos = 0, neg = 0, zero = 0, wins = 0, losses = 0, draws = 0;
    int episodes = 0, full_len = 0;
    double mean_len = 0;
  };
  EvalDetail evaluate_detail(const Genome& G, int eps, uint64_t seed, bool greedy, int full_len_threshold = 500) {
    rng = Rng(seed); steps = train_steps = eval_steps = 0;
    P.build(G, n_logit, cap); assert(P.n_in == n_in);
    EvalDetail d; d.episodes = eps; double tot = 0, len_sum = 0;
    for (int e = 0; e < eps; e++) {
      env->reset(rng); bool done = false; int guard = 0, len = 0; double ret = 0;
      while (!done && guard++ < 100000) {
        int nl = env->legal(leg.data()); X.clear();
        int nr = gather(nl); load(0, nr, 0); fill_bias(nr); P.forward(nr);
        read(0, nl, leg.data(), z.data()); softmax(nl);
        int j = 0;
        if (greedy) { for (int k = 1; k < nl; k++) if (z[k] > z[j]) j = k; } else j = sample(nl);
        ret += env->step(leg[j], done, rng); len++; eval_steps++; steps++;
      }
      tot += ret; len_sum += len;
      const int outcome = env->episode_result();
      if (outcome == 1) d.wins++;
      else if (outcome == -1) d.losses++;
      else if (outcome == 0) d.draws++;
      else if (ret > 0) d.wins++;
      else if (ret < 0) d.losses++;
      else d.draws++;
      if (ret > 0) d.pos++; else if (ret < 0) d.neg++; else d.zero++;
      if (len >= full_len_threshold) d.full_len++;
    }
    d.mean = tot / std::max(1, eps); d.mean_len = len_sum / std::max(1, eps);
    return d;
  }

  // Train a copy of G's weights with PPO; returns fitness (mean eval return). write_back => Lamarckian.
  float evaluate(Genome& G, bool write_back, int eval_eps, uint64_t seed, uint64_t eval_seed) {
    rng = Rng(seed); steps = train_steps = eval_steps = 0;
    P.build(G, n_logit, cap);
    assert(P.n_in == n_in);
    am.assign(P.w.size(), 0.f); av.assign(P.w.size(), 0.f); at = 0;   // fresh Adam per genome
    env->reset(rng);
    for (int it = 0; it < cfg.iters; it++) { rollout(); update(); }
    // Evaluate with a separate, common random stream so genomes in the same
    // generation face matched stochastic conditions. Restore the training RNG
    // afterwards so evaluation cannot perturb PPO training randomness.
    Rng training_rng = rng;
    rng = Rng(eval_seed);
    float f = eval(eval_eps);
    rng = training_rng;
    if (write_back && cfg.iters > 0) P.write_back(G);
    return f;
  }

private:
  Phenotype P; Rng rng; bool pa; int ns, n_in, n_logit, nact, cap; long at = 0;
  std::vector<float> X, t_logp, t_val, t_rew, t_adv, t_ret, z, p, lp, dz, tmp, am, av;
  std::vector<int> t_row0, t_nrows, t_leg0, t_nleg, t_ch, L, leg, idx, off;
  std::vector<uint8_t> t_done;

  int gather(int nl) {
    env->state_feat(tmp.data());
    if (!pa) { X.insert(X.end(), tmp.begin(), tmp.begin() + n_in); return 1; }
    for (int j = 0; j < nl; j++) {
      size_t o = X.size(); X.resize(o + n_in);
      std::memcpy(&X[o], tmp.data(), sizeof(float) * ns);
      env->action_feat(leg[j], &X[o + ns]);
    }
    return nl;
  }
  void load(int r0, int nr, int o) {
    const float* src = &X[(size_t)r0 * n_in];
    for (int r = 0; r < nr; r++) for (int i = 0; i < n_in; i++) P.A(i)[o + r] = src[(size_t)r * n_in + i];
  }
  void fill_bias(int R) { float* b = P.A(P.bias_slot); for (int i = 0; i < R; i++) b[i] = 1.f; }
  float read(int r0, int nl, const int* lg, float* zz) {
    if (pa) {
      const float* o0 = P.A(P.out_slot[0]) + r0; const float* o1 = P.A(P.out_slot[1]) + r0; float v = 0;
      for (int j = 0; j < nl; j++) { zz[j] = o0[j]; v += o1[j]; }
      return v / nl;
    }
    for (int j = 0; j < nl; j++) zz[j] = P.A(P.out_slot[lg[j]])[r0];
    return P.A(P.out_slot[n_logit])[r0];
  }
  void softmax(int nl) {
    float mx = z[0]; for (int j = 1; j < nl; j++) mx = std::max(mx, z[j]);
    float s = 0; for (int j = 0; j < nl; j++) { p[j] = std::exp(z[j] - mx); s += p[j]; }
    float ls = std::log(s);
    for (int j = 0; j < nl; j++) { lp[j] = z[j] - mx - ls; p[j] /= s; }
  }
  int sample(int nl) { float u = rng.uniform(), c = 0; for (int j = 0; j < nl; j++) { c += p[j]; if (u < c) return j; } return nl - 1; }

  void rollout() {
    X.clear(); L.clear(); t_logp.clear(); t_val.clear(); t_rew.clear(); t_done.clear();
    t_row0.clear(); t_nrows.clear(); t_leg0.clear(); t_nleg.clear(); t_ch.clear();
    for (int s = 0; s < cfg.rollout; s++) {
      int nl = env->legal(leg.data());
      int r0 = (int)(X.size() / n_in), nr = gather(nl);
      load(r0, nr, 0); fill_bias(nr); P.forward(nr);
      float v = read(0, nl, leg.data(), z.data()); softmax(nl);
      int j = sample(nl);
      t_row0.push_back(r0); t_nrows.push_back(nr); t_leg0.push_back((int)L.size()); t_nleg.push_back(nl); t_ch.push_back(j);
      L.insert(L.end(), leg.begin(), leg.begin() + nl);
      t_logp.push_back(lp[j]); t_val.push_back(v);
      bool done = false; float r = env->step(leg[j], done, rng); steps++; train_steps++;
      t_rew.push_back(r); t_done.push_back(done);
      if (done) env->reset(rng);
    }
    // bootstrap value of the state after the last step
    float boot = 0;
    if (!t_done.back()) {
      int nl = env->legal(leg.data()); X.resize(X.size());  // keep rows; append temp rows then drop
      size_t keep = X.size(); int r0 = (int)(keep / n_in), nr = gather(nl);
      load(r0, nr, 0); fill_bias(nr); P.forward(nr); boot = read(0, nl, leg.data(), z.data());
      X.resize(keep);
    }
    int T = (int)t_val.size(); t_adv.assign(T, 0.f); t_ret.assign(T, 0.f);
    float gae = 0;
    for (int t = T - 1; t >= 0; t--) {
      float nv = (t == T - 1) ? boot : t_val[t + 1], nt = t_done[t] ? 0.f : 1.f;
      float delta = t_rew[t] + cfg.gamma * nv * nt - t_val[t];
      gae = delta + cfg.gamma * cfg.lam * nt * gae;
      t_adv[t] = gae; t_ret[t] = gae + t_val[t];
    }
    double m = 0, s2 = 0; for (float a : t_adv) m += a; m /= T;
    for (float a : t_adv) s2 += (a - m) * (a - m);
    float sd = (float)std::sqrt(s2 / T) + 1e-6f;
    for (float& a : t_adv) a = (a - (float)m) / sd;
  }

  void update() {
    int T = (int)t_val.size(), mb = cfg.minibatch;
    idx.resize(T); std::iota(idx.begin(), idx.end(), 0);
    for (int ep = 0; ep < cfg.epochs; ep++) {
      for (int i = T - 1; i > 0; i--) std::swap(idx[i], idx[rng.below(i + 1)]);
      for (int s = 0; s < T; s += mb) {
        int M = std::min(mb, T - s), R = 0;
        for (int m = 0; m < M; m++) { int t = idx[s + m]; off[m] = R; load(t_row0[t], t_nrows[t], R); R += t_nrows[t]; }
        fill_bias(R); P.forward(R); P.begin_backward(R);
        std::fill(P.g.begin(), P.g.end(), 0.f);
        const float inv = 1.f / M;
        for (int m = 0; m < M; m++) {
          int t = idx[s + m], nl = t_nleg[t], c = t_ch[t]; const int* lg = &L[t_leg0[t]];
          float v = read(off[m], nl, lg, z.data()); softmax(nl);
          float ratio = std::exp(std::min(20.f, lp[c] - t_logp[t])), A = t_adv[t];
          bool clipped = (A >= 0 && ratio > 1 + cfg.clip) || (A < 0 && ratio < 1 - cfg.clip);
          float g = clipped ? 0.f : -A * ratio, H = 0;
          for (int j = 0; j < nl; j++) H -= p[j] * lp[j];
          for (int j = 0; j < nl; j++) dz[j] = (g * ((j == c) - p[j]) + cfg.ent * p[j] * (lp[j] + H)) * inv;
          float dv = cfg.vf * (v - t_ret[t]) * inv;
          if (pa) {
            for (int j = 0; j < nl; j++) { P.D(P.out_slot[0])[off[m] + j] = dz[j]; P.D(P.out_slot[1])[off[m] + j] = dv / nl; }
          } else {
            for (int j = 0; j < nl; j++) P.D(P.out_slot[lg[j]])[off[m]] = dz[j];
            P.D(P.out_slot[n_logit])[off[m]] = dv;
          }
        }
        P.backward(R);
        double n2 = 0; for (float x : P.g) n2 += (double)x * x;
        float sc = (float)std::sqrt(n2); sc = sc > cfg.max_grad ? cfg.max_grad / sc : 1.f;
        at++; float b1 = 0.9f, b2 = 0.999f, c1 = 1.f - std::pow(b1, (float)at), c2 = 1.f - std::pow(b2, (float)at);
        for (size_t e = 0; e < P.w.size(); e++) {
          float gj = P.g[e] * sc;
          am[e] = b1 * am[e] + (1 - b1) * gj; av[e] = b2 * av[e] + (1 - b2) * gj * gj;
          P.w[e] -= cfg.lr * (am[e] / c1) / (std::sqrt(av[e] / c2) + 1e-5f);
        }
      }
    }
  }

  float eval(int eps) {
    double tot = 0;
    for (int e = 0; e < eps; e++) {
      env->reset(rng); bool done = false; int guard = 0;
      while (!done && guard++ < 100000) {
        int nl = env->legal(leg.data()); X.clear();
        int nr = gather(nl); load(0, nr, 0); fill_bias(nr); P.forward(nr);
        read(0, nl, leg.data(), z.data()); softmax(nl);
        int aj = 0; if (cfg.greedy_eval) { for (int k = 1; k < nl; k++) if (z[k] > z[aj]) aj = k; } else aj = sample(nl);
        tot += env->step(leg[aj], done, rng); steps++; eval_steps++;
      }
    }
    return (float)(tot / std::max(1, eps));
  }
};
