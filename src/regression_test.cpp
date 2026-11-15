// Regression tests for the hidden-layer / Lamarckian / NEAT path. Prints quantitative results.
#include "neat.h"
#include "envs.h"
#include "concept_env.h"
#include <cstdio>
#include <cstdlib>
#include <set>
static int fails = 0;
#define CHECK(c, ...) do { bool ok_ = (c); std::printf("[%s] ", ok_ ? "PASS" : "FAIL"); std::printf(__VA_ARGS__); std::printf("\n"); if (!ok_) fails++; } while (0)

static int count_type(const Genome& g, int t) { int n = 0; for (auto& x : g.nodes) n += x.type == t; return n; }

int main() {
  ConceptEnv env(std::make_unique<Nim>(), {"action_norm"});
  NeatCfg cfg = preset("lam"); cfg.mlp_hidden = {16}; cfg.eval_episodes = 10; cfg.threads = 1; cfg.seed = 5; cfg.pop = 12;
  NetSpec spec; spec.n_in = env.n_in(); spec.n_logit = env.n_logit();
  Innovations I; Rng rng(123);

  // 1. hidden layer exists and is used by the phenotype
  Genome g = make_mlp(spec, cfg.mlp_hidden, I, rng);
  CHECK(count_type(g, N_HID) == 16, "initial genome has %d hidden nodes (expect 16)", count_type(g, N_HID));
  Phenotype P; P.build(g, spec.n_logit, 4);
  CHECK(P.n_comp == 16 + spec.n_out(), "phenotype compute nodes=%d (16 hidden + %d outputs)", P.n_comp, spec.n_out());

  // 2. forward uses the hidden layer: compare against a hand-computed reference
  {
    for (int i = 0; i < P.n_in; i++) P.A(i)[0] = 0.3f * (i + 1);
    P.A(P.bias_slot)[0] = 1.f; P.forward(1);
    std::vector<double> h(16, 0.0);
    int hi = 0;
    for (auto& nd : g.nodes) if (nd.type == N_HID) {
      double s = 0; for (auto& c : g.conns) if (c.to == nd.id) { s += c.w * (c.from == BIAS_ID ? 1.0 : 0.3 * ((c.from) + 1)); }
      h[hi++] = std::tanh(s);
    }
    double o = 0; int oi = OUT_BASE;
    for (auto& c : g.conns) if (c.to == oi) { double src = 0; int k = 0; for (auto& nd : g.nodes) if (nd.type == N_HID) { if (nd.id == c.from) src = h[k]; k++; } if (c.from == BIAS_ID) src = 1.0; o += c.w * src; }
    float got = P.A(P.out_slot[0])[0];
    CHECK(std::fabs(got - o) < 1e-4, "forward output %.6f vs reference %.6f (|diff|=%.2e)", got, o, std::fabs(got - o));
  }

  // 3. numerical gradient check through the hidden layer (loss = output0 of one sample)
  {
    P.begin_backward(1); std::fill(P.g.begin(), P.g.end(), 0.f);
    P.D(P.out_slot[0])[0] = 1.f; P.backward(1);
    double worst = 0; int checked = 0, hidden_checked = 0;
    for (size_t e = 0; e < P.w.size(); e += 3) {
      float w0 = P.w[e]; const float h = 1e-2f;
      P.w[e] = w0 + h; P.forward(1); float fp = P.A(P.out_slot[0])[0];
      P.w[e] = w0 - h; P.forward(1); float fm = P.A(P.out_slot[0])[0];
      P.w[e] = w0; P.forward(1);
      double num = (fp - fm) / (2 * h), ana = P.g[e];
      double err = std::fabs(num - ana) / std::max(1e-3, std::fabs(num) + std::fabs(ana));
      worst = std::max(worst, err); checked++;
      int conn_to = g.conns[P.econn[e]].to; if (conn_to >= HID_BASE) hidden_checked++;
    }
    CHECK(worst < 2e-3, "gradient check: %d weights (%d into hidden units), worst rel err %.2e", checked, hidden_checked, worst);
    CHECK(hidden_checked > 0, "gradient check covered hidden-layer weights");
  }

  // 4. Lamarckian: PPO changes weights, written back; Darwinian leaves them untouched
  {
    Genome lam = g, dar = g; std::vector<float> before; for (auto& c : g.conns) before.push_back(c.w);
    Trainer T(env.make(), cfg.ppo);
    T.evaluate(lam, true, 10, 456, 789);
    T.evaluate(dar, false, 10, 456, 789);
    int ch = 0; double mag = 0; for (size_t i = 0; i < lam.conns.size(); i++) { float d = std::fabs(lam.conns[i].w - before[i]); ch += d > 1e-9f; mag += d; }
    int chd = 0; for (size_t i = 0; i < dar.conns.size(); i++) chd += std::fabs(dar.conns[i].w - before[i]) > 1e-9f;
    int hid_ch = 0, hid_tot = 0; for (size_t i = 0; i < lam.conns.size(); i++) if (lam.conns[i].to >= HID_BASE) { hid_tot++; hid_ch += std::fabs(lam.conns[i].w - before[i]) > 1e-9f; }
    CHECK(ch == (int)lam.conns.size(), "Lamarckian: %d/%zu weights changed (mean |dw|=%.4f; %d/%d into hidden units)", ch, lam.conns.size(), mag / lam.conns.size(), hid_ch, hid_tot);
    CHECK(chd == 0, "Darwinian: %d weights changed (expect 0)", chd);

    // 5. offspring inheritance: clone keeps trained weights exactly; crossover/mutation behaviour
    Genome clone = lam; bool same = true; for (size_t i = 0; i < clone.conns.size(); i++) same &= clone.conns[i].w == lam.conns[i].w;
    CHECK(same, "clone of trained genome preserves all trained weights");
    Genome child = crossover(lam, lam, rng); double dmax = 0; for (size_t i = 0; i < child.conns.size(); i++) dmax = std::max(dmax, (double)std::fabs(child.conns[i].w - lam.conns[i].w));
    CHECK(child.conns.size() == lam.conns.size() && dmax == 0, "crossover(trained, trained) preserves trained weights (max diff %.1e)", dmax);
    MutCfg m = cfg.mut; Genome mut = lam; Innovations I2 = I; mutate(mut, m, I2, rng);
    double md = 0; int touched = 0; for (size_t i = 0; i < lam.conns.size(); i++) { double d = std::fabs(mut.conns[i].w - lam.conns[i].w); md += d; touched += d > 0; }
    CHECK(md / lam.conns.size() < 0.1, "'lam' mutation perturbs %d/%zu weights, mean |dw|=%.4f (small vs learned change)", touched, lam.conns.size(), md / lam.conns.size());
  }

  // 6. initial population shares history (crossover can align genes)
  {
    Population p(env, cfg);
    int minshared = 1 << 30; for (int i = 1; i < (int)p.pop.size(); i++) { std::set<int> a, b; for (auto& c : p.pop[0].conns) a.insert(c.innov); for (auto& c : p.pop[i].conns) b.insert(c.innov); int s = 0; for (int v : a) s += b.count(v); minshared = std::min(minshared, s); }
    CHECK(minshared == (int)p.pop[0].conns.size(), "all initial genomes share all %zu innovations (min shared=%d)", p.pop[0].conns.size(), minshared);
    for (auto& g2 : p.pop) { Phenotype Q; Q.build(g2, spec.n_logit, 4); if (Q.n_comp != 16 + spec.n_out()) { CHECK(false, "genome lacks hidden layer"); } }
    CHECK(true, "every initial genome compiles with a hidden layer");
  }

  // 7. structural evolution still works over generations
  {
    NeatCfg c2 = cfg; c2.mut.p_node = 0.5f; c2.mut.p_conn = 0.5f; c2.pop = 16;
    Population p(env, c2);
    for (int i = 0; i < 6; i++) p.step();
    float nodes = 0, conns = 0; int maxhid = 0;
    for (auto& g2 : p.pop) { nodes += g2.nodes.size(); conns += g2.conns.size(); maxhid = std::max(maxhid, count_type(g2, N_HID)); }
    CHECK(maxhid > 16, "add-node mutation grew hidden nodes (max hidden=%d > 16)", maxhid);
    bool ok = true; for (auto& g2 : p.pop) { Phenotype Q; Q.build(g2, spec.n_logit, 4); for (float w : Q.w) ok &= std::isfinite(w); }
    CHECK(ok, "all evolved genomes compile with finite weights after 6 generations (mean nodes %.1f conns %.1f)", nodes / p.pop.size(), conns / p.pop.size());
  }

  std::printf("%s (%d failures)\n", fails ? "REGRESSION TESTS FAILED" : "ALL REGRESSION TESTS PASSED", fails);
  return fails ? 1 : 0;
}
