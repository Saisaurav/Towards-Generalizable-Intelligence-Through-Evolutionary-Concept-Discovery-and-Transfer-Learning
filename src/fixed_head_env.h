#pragma once
// Optional "fixed head" scorer for external (LLM-generated) environments.
//
// Default ConceptEnv scoring is per-action: the net sees [state || action features] and returns one logit per
// legal action, so the only thing distinguishing actions is a scalar action_norm. For small discrete action sets
// that make the action's *meaning* depend on the state (Snake directions, Pong paddle moves) that is a hard
// function for a minimal NEAT net. FixedHeadEnv instead feeds the net the state (plus state-level transferred
// concepts) once and reads one logit per action from separate outputs, the standard policy head.
//
// Concepts that need an action or a previous state are dropped (action_norm, action_wins_now,
// action_blocks_opponent, transition_*): the fixed head has no per-action rows, and the Python policy mirror is
// stateless between calls. Everything else about learning (NEAT + PPO, concepts, legal-action masking) is unchanged.
#include "concept_env.h"
#include <memory>
#include <string>
#include <vector>

inline bool fixed_head_usable_concept(const std::string& expr) {
  static const char* banned[] = {"action_norm", "action_wins_now", "action_blocks_opponent", "transition_"};
  for (const char* b : banned) if (expr.find(b) != std::string::npos) return false;
  return true;
}

class FixedHeadEnv : public Env {
  ConceptEnv inner;      // owns the base env; supplies state features, concept evaluation, and transition tracking
  int n_conc;            // concept values appended to the state (inner appends one extra action_norm concept, ignored)
  std::vector<std::string> kept;
  std::vector<float> row;
  static std::vector<std::string> filter(const std::vector<std::string>& ex) {
    std::vector<std::string> out;
    for (const auto& e : ex) if (fixed_head_usable_concept(e)) out.push_back(e);
    return out;
  }
  FixedHeadEnv(std::unique_ptr<Env> base, std::vector<std::string> usable, int)
    : inner(std::move(base), usable), n_conc((int)usable.size()), kept(std::move(usable)), row(n_conc + 1) {}
public:
  FixedHeadEnv(std::unique_ptr<Env> base, const std::vector<std::string>& ex) : FixedHeadEnv(std::move(base), filter(ex), 0) {}
  bool per_action() const override { return false; }
  int n_state() const override { return inner.n_state() + n_conc; }
  int n_actions() const override { return inner.n_actions(); }
  void reset(Rng& r) override { inner.reset(r); }
  int legal(int* out) override { return inner.legal(out); }
  void state_feat(float* out) override {
    inner.state_feat(out);
    if (n_conc > 0) {
      inner.action_feat(0, row.data());   // state-level concepts do not depend on the action
      for (int i = 0; i < n_conc; i++) out[inner.n_state() + i] = row[i];
    }
  }
  float step(int a, bool& d, Rng& r) override { return inner.step(a, d, r); }
  int episode_result() const override { return inner.episode_result(); }
  std::unique_ptr<Env> make() const override { return std::unique_ptr<Env>(new FixedHeadEnv(inner.make_base(), kept, 0)); }
};
