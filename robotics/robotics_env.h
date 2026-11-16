#pragma once
// Robotics proxy tasks for the NEAT+PPO learner (continuous actions, Gaussian head).
//
// PointMass: a 2-D point mass with momentum must reach a random target and stop on it. The action is a 2-D
// force in [-1,1]^2. Success = within `tol` of the target (and slow) at the end of the episode.
//   - posonly=1 hides velocity from the observation (only target-relative position), so a memoryless policy
//     cannot damp its motion; StackEnv restores it from previous observations / the previous action.
// Perturbations used by the reliability harness (all deterministic given the training Rng):
//   noise  : Gaussian observation noise (std, in observation units)
//   delay  : the force actually applied is the one commanded `delay` steps ago
//   mass   : dynamics mass multiplier (policy was trained on mass=1)
//   fnoise : Gaussian noise on the applied force
#include "env.h"
#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <deque>
#include <sstream>
#include <stdexcept>

struct PMCfg {
  float rscale = 1.f, noise = 0.f, fnoise = 0.f, mass = 1.f, tol = 0.1f, max_speed = 0.3f;
  int delay = 0, horizon = 100; bool posonly = false, tfeat = true;   // tfeat: append t/horizon so the value function is well-posed under the time limit
};

inline PMCfg parse_pm_args(const std::string& s, PMCfg c = PMCfg{}) {
  std::stringstream ss(s); std::string kv;
  while (std::getline(ss, kv, ',')) {
    if (kv.empty()) continue;
    size_t eq = kv.find('='); if (eq == std::string::npos) throw std::runtime_error("bad env arg: " + kv);
    std::string k = kv.substr(0, eq); float v = (float)std::atof(kv.c_str() + eq + 1);
    if (k == "rscale") c.rscale = v; else if (k == "noise") c.noise = v; else if (k == "fnoise") c.fnoise = v; else if (k == "mass") c.mass = v;
    else if (k == "tol") c.tol = v; else if (k == "delay") c.delay = (int)v; else if (k == "horizon") c.horizon = (int)v;
    else if (k == "posonly") c.posonly = v != 0.f; else if (k == "tfeat") c.tfeat = v != 0.f; else throw std::runtime_error("unknown env arg: " + k);
  }
  return c;
}

struct PointMass : Env {
  PMCfg cfg; float x = 0, y = 0, vx = 0, vy = 0, tx = 0, ty = 0; int t = 0; bool ok = false;
  std::deque<std::pair<float, float>> q; Rng orng{7};
  static constexpr float DT = 0.05f, FORCE = 4.f, DRAG = 0.2f, BOUND = 2.f;
  explicit PointMass(PMCfg c = PMCfg{}) : cfg(c) {}
  bool per_action() const override { return false; }
  int n_state() const override { return (cfg.posonly ? 2 : 4) + (cfg.tfeat ? 1 : 0); }
  int n_actions() const override { return 2; }
  int act_dim() const override { return 2; }
  void reset(Rng& r) override {
    x = r.range(-1.f, 1.f); y = r.range(-1.f, 1.f); tx = r.range(-1.f, 1.f); ty = r.range(-1.f, 1.f);
    vx = vy = 0; t = 0; ok = false; q.assign(cfg.delay, {0.f, 0.f}); orng = Rng(r.next());
  }
  int legal(int* o) override { o[0] = 0; o[1] = 1; return 2; }
  void state_feat(float* o) override {
    o[0] = (tx - x) / 2.f + cfg.noise * orng.normal(); o[1] = (ty - y) / 2.f + cfg.noise * orng.normal();
    int k = 2;
    if (!cfg.posonly) { o[2] = vx + cfg.noise * orng.normal(); o[3] = vy + cfg.noise * orng.normal(); k = 4; }
    if (cfg.tfeat) o[k] = (float)t / cfg.horizon;
  }
  float step_cont(const float* a, bool& done, Rng& r) override {
    float ax = std::clamp(a[0], -1.f, 1.f), ay = std::clamp(a[1], -1.f, 1.f);
    float effort = ax * ax + ay * ay, fx = ax, fy = ay;
    if (cfg.delay > 0) { q.push_back({ax, ay}); fx = q.front().first; fy = q.front().second; q.pop_front(); }
    if (cfg.fnoise > 0.f) { fx += cfg.fnoise * r.normal(); fy += cfg.fnoise * r.normal(); }
    vx += DT * (FORCE * fx / cfg.mass - DRAG * vx); vy += DT * (FORCE * fy / cfg.mass - DRAG * vy);
    x += DT * vx; y += DT * vy;
    if (std::fabs(x) > BOUND) { x = std::clamp(x, -BOUND, BOUND); vx = 0; }
    if (std::fabs(y) > BOUND) { y = std::clamp(y, -BOUND, BOUND); vy = 0; }
    t++; done = t >= cfg.horizon;
    float dist = std::hypot(tx - x, ty - y), speed = std::hypot(vx, vy);
    ok = dist < cfg.tol && speed < cfg.max_speed;
    return cfg.rscale * (-0.1f * dist - 0.01f * effort);
  }
  float step(int, bool& d, Rng&) override { d = true; return 0.f; }   // discrete entry point unused (continuous env)
  int episode_result() const override { return ok ? 1 : -1; }
  std::unique_ptr<Env> make() const override { return std::make_unique<PointMass>(cfg); }
};

// Memory for a stateless policy. Layout: [obs_t, g*(obs_t-obs_{t-1}), g*(obs_{t-1}-obs_{t-2}), ..., a_{t-1}] (K-1 scaled deltas).
// Deltas (not raw previous frames) make velocity an explicit, well-scaled input; g (gain) lifts the small per-step change.
// Zero-filled history at episode start (the first deltas are therefore not meaningful for one step).
struct StackEnv : Env {
  std::unique_ptr<Env> in; int K, ns, ad; float gain; std::vector<float> hist, prev_a, cur; bool have = false; int nfilled = 0;
  StackEnv(std::unique_ptr<Env> e, int k, float g = 10.f) : in(std::move(e)), K(std::max(1, k)), ns(in->n_state()), ad(in->act_dim()), gain(g) {
    hist.assign((size_t)(K - 1) * ns, 0.f); prev_a.assign(ad, 0.f); cur.assign(ns, 0.f);
  }
  bool per_action() const override { return in->per_action(); }
  int n_state() const override { return ns * K + (K > 1 ? ad : 0); }
  int n_actions() const override { return in->n_actions(); }
  int act_dim() const override { return ad; }
  void reset(Rng& r) override { in->reset(r); std::fill(hist.begin(), hist.end(), 0.f); std::fill(prev_a.begin(), prev_a.end(), 0.f); have = false; nfilled = 0; }
  int legal(int* o) override { return in->legal(o); }
  void state_feat(float* o) override {
    in->state_feat(cur.data()); std::copy(cur.begin(), cur.end(), o);
    // hist[0] = obs_{t-1}, hist[1] = obs_{t-2}, ... ; delta_k = gain*(obs_{t-k} - obs_{t-k-1}); zero until history is filled
    for (int k = 0; k < K - 1; k++) for (int i = 0; i < ns; i++) {
      float newer = k == 0 ? cur[i] : hist[(size_t)(k - 1) * ns + i], older = hist[(size_t)k * ns + i];
      o[ns * (k + 1) + i] = (have && (k == 0 || nfilled > k)) ? gain * (newer - older) : 0.f;
    }
    if (K > 1) std::copy(prev_a.begin(), prev_a.end(), o + ns * K);
  }
  float step_cont(const float* a, bool& d, Rng& r) override {
    // shift history using the observation the policy just saw (cur was filled by the last state_feat call)
    if (K > 1) {
      for (int k = K - 2; k > 0; k--) std::copy(hist.begin() + (size_t)(k - 1) * ns, hist.begin() + (size_t)k * ns, hist.begin() + (size_t)k * ns);
      std::copy(cur.begin(), cur.end(), hist.begin()); have = true; nfilled++;
      for (int i = 0; i < ad; i++) prev_a[i] = std::clamp(a[i], -1.f, 1.f);
    }
    return in->step_cont(a, d, r);
  }
  float step(int a, bool& d, Rng& r) override { return in->step(a, d, r); }
  int episode_result() const override { return in->episode_result(); }
  std::unique_ptr<Env> make() const override { return std::make_unique<StackEnv>(in->make(), K, gain); }
};
