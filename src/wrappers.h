#pragma once
// Representation wrappers used by the experiment runner. They change only how a task is presented to the
// learner (action encoding), never the task rules. They are created fresh for every run (no weights, no state).
#include "env.h"
#include <cmath>

// FIXED-head view: the net sees state features only and has one logit per action (illegal actions masked).
struct FixedHeadEnv : Env {
  std::unique_ptr<Env> base;
  explicit FixedHeadEnv(std::unique_ptr<Env> b) : base(std::move(b)) {}
  bool per_action() const override { return false; }
  int n_state() const override { return base->n_state(); }
  int n_actions() const override { return base->n_actions(); }
  void reset(Rng& r) override { base->reset(r); }
  int legal(int* o) override { return base->legal(o); }
  void state_feat(float* o) override { base->state_feat(o); }
  float step(int a, bool& d, Rng& r) override { return base->step(a, d, r); }
  std::unique_ptr<Env> make() const override { return std::make_unique<FixedHeadEnv>(base->make()); }
};

// PER_ACTION view with a one-hot action-identity encoding: net input = [state || onehot(action)], one logit per legal action.
struct OneHotEnv : Env {
  std::unique_ptr<Env> base;
  explicit OneHotEnv(std::unique_ptr<Env> b) : base(std::move(b)) {}
  bool per_action() const override { return true; }
  int n_state() const override { return base->n_state(); }
  int n_action_feat() const override { return base->n_actions(); }
  int n_actions() const override { return base->n_actions(); }
  void reset(Rng& r) override { base->reset(r); }
  int legal(int* o) override { return base->legal(o); }
  void state_feat(float* o) override { base->state_feat(o); }
  void action_feat(int a, float* o) override { for (int i = 0; i < base->n_actions(); i++) o[i] = i == a ? 1.f : 0.f; }
  float step(int a, bool& d, Rng& r) override { return base->step(a, d, r); }
  std::unique_ptr<Env> make() const override { return std::make_unique<OneHotEnv>(base->make()); }
};

// Diagnostic only (Nim): replaces the scalar state s = remaining/21 with a one-hot of remaining (0..21). This is a generic
// encoding of a small integer, not a strategy, but it changes the observation, so results using it are labelled separately.
struct OneHotScalarStateEnv : Env {
  std::unique_ptr<Env> base; int K;
  OneHotScalarStateEnv(std::unique_ptr<Env> b, int k) : base(std::move(b)), K(k) {}
  bool per_action() const override { return true; }
  int n_state() const override { return K + 1; }
  int n_action_feat() const override { return base->n_actions(); }
  int n_actions() const override { return base->n_actions(); }
  void reset(Rng& r) override { base->reset(r); }
  int legal(int* o) override { return base->legal(o); }
  void state_feat(float* o) override { float s; base->state_feat(&s); int v = (int)std::lround(s * (K - 1)); for (int i = 0; i <= K; i++) o[i] = (i == v) ? 1.f : 0.f; }
  void action_feat(int a, float* o) override { for (int i = 0; i < base->n_actions(); i++) o[i] = i == a ? 1.f : 0.f; }
  float step(int a, bool& d, Rng& r) override { return base->step(a, d, r); }
  std::unique_ptr<Env> make() const override { return std::make_unique<OneHotScalarStateEnv>(base->make(), K); }
};

// Raw state-action control: per-action features [action_norm, state_i * action_norm for K evenly spaced state coordinates].
// No GP, no discovered structure; same wrapper family as ConceptEnv (per-action, action_norm always present).
struct RawSxAEnv : Env {
  std::unique_ptr<Env> base; int K; bool act_oh; int state_k; std::vector<int> idx; mutable std::vector<float> st;
  RawSxAEnv(std::unique_ptr<Env> b, int k, bool act_onehot = false, int state_onehot_k = 0) : base(std::move(b)), K(k), act_oh(act_onehot), state_k(state_onehot_k) {
    int n = base->n_state(), m = std::min(K, n); st.assign(n, 0.f);
    for (int j = 0; j < m; j++) idx.push_back(m == 1 ? 0 : (int)std::lround((double)j * (n - 1) / (m - 1)));
  }
  bool per_action() const override { return true; }
  int n_state() const override { return state_k > 0 ? state_k + 1 : base->n_state(); }
  int n_action_feat() const override { return 1 + (int)idx.size() + (act_oh ? base->n_actions() : 0); }
  int n_actions() const override { return base->n_actions(); }
  void reset(Rng& r) override { base->reset(r); }
  int legal(int* o) override { return base->legal(o); }
  void state_feat(float* o) override {
    if (state_k > 0) { float v; base->state_feat(&v); int k = (int)std::lround(v * (state_k - 1)); for (int i = 0; i <= state_k; i++) o[i] = i == k ? 1.f : 0.f; }
    else base->state_feat(o);
  }
  void action_feat(int a, float* o) override {
    float an = base->n_actions() > 1 ? (2.f * a / (base->n_actions() - 1) - 1.f) : 0.f;
    base->state_feat(st.data()); o[0] = an;
    for (size_t j = 0; j < idx.size(); j++) o[1 + j] = st[idx[j]] * an;
    if (act_oh) for (int i = 0; i < base->n_actions(); i++) o[1 + idx.size() + i] = i == a ? 1.f : 0.f;
  }
  float step(int a, bool& d, Rng& r) override { return base->step(a, d, r); }
  std::unique_ptr<Env> make() const override { return std::make_unique<RawSxAEnv>(base->make(), K, act_oh, state_k); }
};

// Positive-control wrapper (NOT a discovered or transferred concept): per-action features
// [action_norm, one-hot action, state-independent afterstate tactics: wins-now, blocks-opponent-win]. Hand-supplied game knowledge.
struct TacticEnv : Env {
  std::unique_ptr<Env> base;
  explicit TacticEnv(std::unique_ptr<Env> b) : base(std::move(b)) {}
  bool per_action() const override { return true; }
  int n_state() const override { return base->n_state(); }
  int n_action_feat() const override { return 1 + base->n_actions() + base->n_action_tactics(); }
  int n_actions() const override { return base->n_actions(); }
  void reset(Rng& r) override { base->reset(r); }
  int legal(int* o) override { return base->legal(o); }
  void state_feat(float* o) override { base->state_feat(o); }
  void action_feat(int a, float* o) override {
    int n = base->n_actions(); o[0] = n > 1 ? (2.f * a / (n - 1) - 1.f) : 0.f;
    for (int i = 0; i < n; i++) o[1 + i] = i == a ? 1.f : 0.f;
    base->action_tactics(a, o + 1 + n);
  }
  float step(int a, bool& d, Rng& r) override { return base->step(a, d, r); }
  std::unique_ptr<Env> make() const override { return std::make_unique<TacticEnv>(base->make()); }
};
