#pragma once
#include <memory>
#include <string>
#include <stdexcept>
#include <vector>
#include "rng.h"
// Single-agent env interface used by the brain. The LLM-generated IGame gets wrapped into this.
//  FIXED mode      : per_action()==false. Net input = state feats, outputs = n_actions logits (+1 value).
//  PER_ACTION mode : per_action()==true.  Net input = [state ‖ action feats], 1 logit per legal action (+1 value).
// Input column order == NetSpec.in_ids order. To add a concept: grow the env's feature vector by one
// column AND call Population::add_input(id).
struct Env {
  virtual ~Env() = default;
  virtual bool per_action() const = 0;
  virtual int n_state() const = 0;
  virtual int n_action_feat() const { return 0; }
  virtual int n_actions() const = 0;          // FIXED: #logits. PER_ACTION: max #legal actions
  virtual void reset(Rng&) = 0;
  virtual int legal(int* out) = 0;            // legal action ids; must return >=1 while not done
  virtual void state_feat(float* out) = 0;
  virtual void action_feat(int, float*) {}
  // Optional action-conditioned tactics (afterstate lookahead): out[0]=this move wins immediately, out[1]=this move blocks the opponent's immediate win.
  virtual int n_action_tactics() const { return 0; }
  virtual void action_tactics(int, float*) {}
  virtual bool additional_primitive(const std::string&, const std::vector<float>&, float&) const { return false; }
  virtual float step(int a,bool& done,Rng&) = 0;
  // Continuous control: act_dim()>0 => Gaussian policy head; n_actions() must equal act_dim(), legal() returns ids 0..act_dim()-1,
  // and the trainer calls step_cont() with act_dim() raw actions (the env clips them).
  virtual int act_dim() const { return 0; }
  virtual float step_cont(const float*, bool&, Rng&) {
    throw std::logic_error("continuous actions are not supported by this environment");
  }
  // Optional teacher action for imitation (only used when PPOCfg.imitation > 0); -1 = none.
  virtual int expert_action() const { return -1; }
  // 1 = win, -1 = loss, 0 = draw, 2 = unavailable.
  virtual int episode_result() const { return 2; }
  virtual std::unique_ptr<Env> make() const = 0;
  int n_in() const { return n_state() + n_action_feat(); }
  int n_logit() const { return per_action() ? 1 : n_actions(); }
};
