#pragma once
#include <numeric>
#include "env.h"
#include "net.h"

struct PPOCfg {
  int iters = 2;          // PPO iterations per genome per generation (0 => pure neuroevolution)
  int rollout = 256, epochs = 4, minibatch = 64;
  bool train_fitness = false;
  bool persist_adam = false;
  bool greedy_eval = false;   // fitness evaluation episodes take argmax over legal actions instead of sampling
  float imitation = 0.f;
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
    ad = env->act_dim(); cont = ad > 0; act.resize(std::max(ad, 1));
    if (cont && (pa || ad != nact || n_logit != ad))
      throw std::invalid_argument("continuous environments require a fixed head with n_actions and n_logit equal to act_dim");
  }

  // Evaluate an already-trained policy without further PPO updates.
  float evaluate_policy(const Genome& G, int eval_eps, uint64_t seed) {
    rng = Rng(seed); steps = train_steps = eval_steps = 0;
    P.build(G, n_logit, cap); load_ls(G, false);
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
    P.build(G, n_logit, cap); load_ls(G, false); assert(P.n_in == n_in);
    EvalDetail d; d.episodes = eps; double tot = 0, len_sum = 0;
    for (int e = 0; e < eps; e++) {
      env->reset(rng); bool done = false; int guard = 0, len = 0; double ret = 0;
      while (!done && guard++ < 100000) {
        int nl = env->legal(leg.data()); X.clear();
        int nr = gather(nl); load(0, nr, 0); fill_bias(nr); P.forward(nr);
        read(0, nl, leg.data(), z.data());
        if (cont) {
          draw_action(z.data(), greedy);
          ret += env->step_cont(act.data(), done, rng);
        } else {
          softmax(nl);
          int j = 0;
          if (greedy) { for (int k = 1; k < nl; k++) if (z[k] > z[j]) j = k; } else j = sample(nl);
          ret += env->step(leg[j], done, rng);
        } len++; eval_steps++; steps++;
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
    P.build(G, n_logit, cap); load_ls(G, cfg.persist_adam);
    assert(P.n_in == n_in);
    am.assign(P.w.size(), 0.f); av.assign(P.w.size(), 0.f); at = 0;   // fresh Adam per genome
    if (cfg.persist_adam) {
      for (size_t e = 0; e < P.w.size(); e++) {
        am[e] = G.conns[P.econn[e]].am;
        av[e] = G.conns[P.econn[e]].av;
      }
      at = G.adam_t;
    }
    env->reset(rng);
    ep_sum = 0; ep_n = 0; cur_ret = 0;
    for (int it = 0; it < cfg.iters; it++) { rollout(); update(); }
    const float train_fit = ep_n > 0 ? (float)(ep_sum / ep_n) : (float)cur_ret;
    // Evaluate with a separate, common random stream so genomes in the same
    // generation face matched stochastic conditions. Restore the training RNG
    // afterwards so evaluation cannot perturb PPO training randomness.
    Rng training_rng = rng;
    rng = Rng(eval_seed);
    float f = (cfg.train_fitness && cfg.iters > 0) ? train_fit : eval(eval_eps);
    rng = training_rng;
    if (write_back && cfg.iters > 0) {
      P.write_back(G);
      if (cont) {
        G.logstd = ls;
        if (cfg.persist_adam) { G.ls_m = ls_m; G.ls_v = ls_v; }
      }
      if (cfg.persist_adam) {
        for (size_t e = 0; e < P.w.size(); e++) {
          G.conns[P.econn[e]].am = am[e];
          G.conns[P.econn[e]].av = av[e];
        }
        G.adam_t = at;
      }
    }
    return f;
  }

private:
  Phenotype P; Rng rng; bool pa; int ns, n_in, n_logit, nact, cap; bool cont = false; int ad = 0;
  std::vector<float> ls, ls_m, ls_v, ls_g, act, t_act;
  long at = 0; double ep_sum = 0, cur_ret = 0; long ep_n = 0;
  std::vector<float> X, t_logp, t_val, t_rew, t_adv, t_ret, z, p, lp, dz, tmp, am, av;
  std::vector<int> t_row0, t_nrows, t_leg0, t_nleg, t_ch, t_expert, L, leg, idx, off;
  std::vector<uint8_t> t_done;

  void load_ls(const Genome& G, bool moments) {
    if (!cont) return;
    ls = G.logstd.size() == (size_t)ad ? G.logstd : std::vector<float>(ad, -0.7f);
    ls_m.assign(ad, 0.f); ls_v.assign(ad, 0.f); ls_g.assign(ad, 0.f);
    if (moments && G.ls_m.size() == (size_t)ad) { ls_m = G.ls_m; ls_v = G.ls_v; }
  }
  float gauss_logp(const float* action, const float* mean) const {
    float result = 0.f;
    for (int i = 0; i < ad; i++) {
      float z = (action[i] - mean[i]) / std::exp(ls[i]);
      result += -0.5f * z * z - ls[i] - 0.9189385f;
    }
    return result;
  }
  void draw_action(const float* mean, bool greedy) {
    for (int i = 0; i < ad; i++) act[i] = greedy ? mean[i] : mean[i] + std::exp(ls[i]) * rng.normal();
  }
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
    t_expert.clear(); t_act.clear();
    for (int s = 0; s < cfg.rollout; s++) {
      int nl = env->legal(leg.data());
      const int expert = !cont && cfg.imitation > 0.f ? env->expert_action() : -1;
      int r0 = (int)(X.size() / n_in), nr = gather(nl);
      load(r0, nr, 0); fill_bias(nr); P.forward(nr);
      float v = read(0, nl, leg.data(), z.data()); if (!cont) softmax(nl);
      int j = 0; float logp;
      if (cont) {
        draw_action(z.data(), false);
        logp = gauss_logp(act.data(), z.data());
        t_act.insert(t_act.end(), act.begin(), act.begin() + ad);
      } else {
        j = sample(nl);
        if (expert >= 0) {
          for (int k = 0; k < nl; k++) if (leg[k] == expert) { j = k; break; }
        }
        logp = lp[j];
      }
      t_row0.push_back(r0); t_nrows.push_back(nr); t_leg0.push_back((int)L.size()); t_nleg.push_back(nl); t_ch.push_back(j);
      t_expert.push_back(expert);
      L.insert(L.end(), leg.begin(), leg.begin() + nl);
      t_logp.push_back(logp); t_val.push_back(v);
      bool done = false;
      float r = cont ? env->step_cont(act.data(), done, rng) : env->step(leg[j], done, rng);
      steps++; train_steps++; cur_ret += r;
      t_rew.push_back(r); t_done.push_back(done);
      if (done) { ep_sum += cur_ret; ep_n++; cur_ret = 0; env->reset(rng); }
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
        if (cont) std::fill(ls_g.begin(), ls_g.end(), 0.f);
        const float inv = 1.f / M;
        for (int m = 0; m < M; m++) {
          int t = idx[s + m], nl = t_nleg[t], c = t_ch[t]; const int* lg = &L[t_leg0[t]];
          float v = read(off[m], nl, lg, z.data());
          if (cont) {
            const float* action = &t_act[(size_t)t * ad];
            float ratio = std::exp(std::min(20.f, gauss_logp(action, z.data()) - t_logp[t]));
            float advantage = t_adv[t];
            bool clipped = (advantage >= 0 && ratio > 1 + cfg.clip) ||
                           (advantage < 0 && ratio < 1 - cfg.clip);
            float g = clipped ? 0.f : -advantage * ratio;
            for (int i = 0; i < ad; i++) {
              float delta = action[i] - z[i], variance = std::exp(2.f * ls[i]);
              P.D(P.out_slot[lg[i]])[off[m]] = g * delta / variance * inv;
              ls_g[i] += (g * (delta * delta / variance - 1.f) - cfg.ent) * inv;
            }
            P.D(P.out_slot[n_logit])[off[m]] = cfg.vf * (v - t_ret[t]) * inv;
            continue;
          }
          softmax(nl);
          float ratio = std::exp(std::min(20.f, lp[c] - t_logp[t])), A = t_adv[t];
          bool clipped = (A >= 0 && ratio > 1 + cfg.clip) || (A < 0 && ratio < 1 - cfg.clip);
          float g = clipped ? 0.f : -A * ratio, H = 0;
          for (int j = 0; j < nl; j++) H -= p[j] * lp[j];
          const int expert = t_expert[t];
          for (int j = 0; j < nl; j++) {
            const float imitation = expert >= 0 && lg[j] == expert
                ? 1.f : 0.f;
            const float policy_gradient = expert >= 0 ? 0.f : g;
            const float entropy_gradient = expert >= 0 ? 0.f : cfg.ent;
            dz[j] = (policy_gradient * ((j == c) - p[j])
                     + entropy_gradient * p[j] * (lp[j] + H)
                     + cfg.imitation * (p[j] - imitation)) * inv;
          }
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
        if (cont) for (float x : ls_g) n2 += (double)x * x;
        float sc = (float)std::sqrt(n2); sc = sc > cfg.max_grad ? cfg.max_grad / sc : 1.f;
        at++; float b1 = 0.9f, b2 = 0.999f, c1 = 1.f - std::pow(b1, (float)at), c2 = 1.f - std::pow(b2, (float)at);
        for (size_t e = 0; e < P.w.size(); e++) {
          float gj = P.g[e] * sc;
          am[e] = b1 * am[e] + (1 - b1) * gj; av[e] = b2 * av[e] + (1 - b2) * gj * gj;
          P.w[e] -= cfg.lr * (am[e] / c1) / (std::sqrt(av[e] / c2) + 1e-5f);
        }
        if (cont) for (int i = 0; i < ad; i++) {
          float gj = ls_g[i] * sc;
          ls_m[i] = b1 * ls_m[i] + (1 - b1) * gj;
          ls_v[i] = b2 * ls_v[i] + (1 - b2) * gj * gj;
          ls[i] = std::clamp(ls[i] - cfg.lr * (ls_m[i] / c1) /
              (std::sqrt(ls_v[i] / c2) + 1e-5f), -3.f, 0.5f);
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
        read(0, nl, leg.data(), z.data());
        if (cont) {
          draw_action(z.data(), cfg.greedy_eval);
          tot += env->step_cont(act.data(), done, rng);
        } else {
          softmax(nl);
          int aj = 0;
          if (cfg.greedy_eval) { for (int k = 1; k < nl; k++) if (z[k] > z[aj]) aj = k; } else aj = sample(nl);
          tot += env->step(leg[aj], done, rng);
        }
        steps++; eval_steps++;
      }
    }
    return (float)(tot / std::max(1, eps));
  }
};
