
#pragma once
#include "concept_env.h"

inline bool fixed_head_usable_concept(const std::string& expr) {
  static const char* banned[] = {
    "action_norm", "action_wins_now", "action_blocks_opponent", "transition_"
  };
  for (const char* primitive : banned)
    if (expr.find(primitive) != std::string::npos) return false;
  return true;
}

class FixedHeadEnv : public Env {
  ConceptEnv inner;
  int n_concepts;
  std::vector<std::string> kept;
  std::vector<float> row;

  static std::vector<std::string> filter(const std::vector<std::string>& expressions) {
    std::vector<std::string> usable;
    for (const auto& expression : expressions)
      if (fixed_head_usable_concept(expression)) usable.push_back(expression);
    return usable;
  }

  FixedHeadEnv(std::unique_ptr<Env> base, std::vector<std::string> usable, int)
      : inner(std::move(base), usable), n_concepts((int)usable.size()),
        kept(std::move(usable)), row(n_concepts + 1) {}

public:
  FixedHeadEnv(std::unique_ptr<Env> base, const std::vector<std::string>& expressions)
      : FixedHeadEnv(std::move(base), filter(expressions), 0) {}

  bool per_action() const override { return false; }
  int n_state() const override { return inner.n_state() + n_concepts; }
  int n_actions() const override { return inner.n_actions(); }
  int act_dim() const override { return inner.act_dim(); }
  int expert_action() const override { return inner.expert_action(); }
  void reset(Rng& rng) override { inner.reset(rng); }
  int legal(int* out) override { return inner.legal(out); }
  void state_feat(float* out) override {
    inner.state_feat(out);
    if (n_concepts > 0) {
      inner.action_feat(0, row.data());
      for (int i = 0; i < n_concepts; i++) out[inner.n_state() + i] = row[i];
    }
  }
  float step(int action, bool& done, Rng& rng) override { return inner.step(action, done, rng); }
  float step_cont(const float* action, bool& done, Rng& rng) override {
    return inner.step_cont(action, done, rng);
  }
  int episode_result() const override { return inner.episode_result(); }
  std::unique_ptr<Env> make() const override {
    return std::unique_ptr<Env>(new FixedHeadEnv(inner.make_base(), kept, 0));
  }
};
