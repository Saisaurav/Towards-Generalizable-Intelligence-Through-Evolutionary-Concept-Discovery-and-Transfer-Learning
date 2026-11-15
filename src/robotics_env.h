#pragma once
#include "env.h"
#include <algorithm>
#include <cmath>
#include <deque>
#include <sstream>
#include <stdexcept>

struct PMCfg {
  float rscale = 1.f, noise = 0.f, fnoise = 0.f, mass = 1.f, tol = 0.1f, max_speed = 0.3f;
  int delay = 0, horizon = 100;
  bool posonly = false, tfeat = true;
};

inline void validate_pmcfg(const PMCfg& cfg) {
  if (!std::isfinite(cfg.rscale) || cfg.rscale <= 0.f ||
      !std::isfinite(cfg.noise) || cfg.noise < 0.f ||
      !std::isfinite(cfg.fnoise) || cfg.fnoise < 0.f ||
      !std::isfinite(cfg.mass) || cfg.mass <= 0.f ||
      !std::isfinite(cfg.tol) || cfg.tol <= 0.f ||
      !std::isfinite(cfg.max_speed) || cfg.max_speed < 0.f ||
      cfg.delay < 0 || cfg.delay > 1000 || cfg.horizon < 1 || cfg.horizon > 100000)
    throw std::runtime_error("invalid PointMass configuration");
}

inline PMCfg parse_pm_args(const std::string& args, PMCfg cfg = PMCfg{}) {
  std::stringstream input(args);
  std::string pair;
  while (std::getline(input, pair, ',')) {
    if (pair.empty()) continue;
    size_t equals = pair.find('=');
    if (equals == std::string::npos) throw std::runtime_error("bad env arg: " + pair);
    const std::string key = pair.substr(0, equals);
    const std::string text = pair.substr(equals + 1);
    size_t parsed = 0;
    float value;
    try {
      value = std::stof(text, &parsed);
    } catch (const std::exception&) {
      throw std::runtime_error("invalid PointMass value for " + key);
    }
    if (parsed != text.size() || !std::isfinite(value))
      throw std::runtime_error("invalid PointMass value for " + key);
    if (key == "rscale") cfg.rscale = value;
    else if (key == "noise") cfg.noise = value;
    else if (key == "fnoise") cfg.fnoise = value;
    else if (key == "mass") cfg.mass = value;
    else if (key == "tol") cfg.tol = value;
    else if (key == "max_speed") cfg.max_speed = value;
    else if (key == "delay" || key == "horizon") {
      if (std::floor(value) != value) throw std::runtime_error("PointMass " + key + " must be an integer");
      if (key == "delay") cfg.delay = (int)value;
      else cfg.horizon = (int)value;
    } else if (key == "posonly") cfg.posonly = value != 0.f;
    else if (key == "tfeat") cfg.tfeat = value != 0.f;
    else throw std::runtime_error("unknown env arg: " + key);
  }
  validate_pmcfg(cfg);
  return cfg;
}

struct PointMass : Env {
  PMCfg cfg;
  float x = 0, y = 0, vx = 0, vy = 0, tx = 0, ty = 0;
  int t = 0;
  bool ok = false;
  std::deque<std::pair<float, float>> queue;
  Rng observation_rng{7};
  static constexpr float DT = 0.05f, FORCE = 4.f, DRAG = 0.2f, BOUND = 2.f;

  explicit PointMass(PMCfg config = PMCfg{}) : cfg(config) { validate_pmcfg(cfg); }
  bool per_action() const override { return false; }
  int n_state() const override { return (cfg.posonly ? 2 : 4) + (cfg.tfeat ? 1 : 0); }
  int n_actions() const override { return 2; }
  int act_dim() const override { return 2; }
  void reset(Rng& rng) override {
    x = rng.range(-1.f, 1.f); y = rng.range(-1.f, 1.f);
    tx = rng.range(-1.f, 1.f); ty = rng.range(-1.f, 1.f);
    vx = vy = 0.f; t = 0; ok = false;
    queue.assign(cfg.delay, {0.f, 0.f});
    observation_rng = Rng(rng.next());
  }
  int legal(int* out) override { out[0] = 0; out[1] = 1; return 2; }
  void state_feat(float* out) override {
    out[0] = (tx - x) / 2.f + cfg.noise * observation_rng.normal();
    out[1] = (ty - y) / 2.f + cfg.noise * observation_rng.normal();
    int next = 2;
    if (!cfg.posonly) {
      out[2] = vx + cfg.noise * observation_rng.normal();
      out[3] = vy + cfg.noise * observation_rng.normal();
      next = 4;
    }
    if (cfg.tfeat) out[next] = (float)t / cfg.horizon;
  }
  float step_cont(const float* action, bool& done, Rng& rng) override {
    float ax = std::clamp(action[0], -1.f, 1.f);
    float ay = std::clamp(action[1], -1.f, 1.f);
    const float effort = ax * ax + ay * ay;
    float fx = ax, fy = ay;
    if (cfg.delay > 0) {
      queue.push_back({ax, ay});
      fx = queue.front().first; fy = queue.front().second;
      queue.pop_front();
    }
    if (cfg.fnoise > 0.f) { fx += cfg.fnoise * rng.normal(); fy += cfg.fnoise * rng.normal(); }
    vx += DT * (FORCE * fx / cfg.mass - DRAG * vx);
    vy += DT * (FORCE * fy / cfg.mass - DRAG * vy);
    x += DT * vx; y += DT * vy;
    if (std::fabs(x) > BOUND) { x = std::clamp(x, -BOUND, BOUND); vx = 0.f; }
    if (std::fabs(y) > BOUND) { y = std::clamp(y, -BOUND, BOUND); vy = 0.f; }
    t++;
    done = t >= cfg.horizon;
    const float distance = std::hypot(tx - x, ty - y);
    const float speed = std::hypot(vx, vy);
    ok = distance < cfg.tol && speed < cfg.max_speed;
    return cfg.rscale * (-0.1f * distance - 0.01f * effort);
  }
  float step(int, bool& done, Rng&) override { done = true; return 0.f; }
  int episode_result() const override { return ok ? 1 : -1; }
  std::unique_ptr<Env> make() const override { return std::make_unique<PointMass>(cfg); }
};

struct StackEnv : Env {
  std::unique_ptr<Env> inner;
  int depth, state_size, action_size;
  float gain;
  std::vector<float> history, previous_action, current;
  bool has_history = false;
  int filled = 0;

  StackEnv(std::unique_ptr<Env> env, int k, float delta_gain = 10.f)
      : inner(std::move(env)), depth(std::max(1, k)), state_size(inner->n_state()),
        action_size(inner->act_dim()), gain(delta_gain),
        history((size_t)(depth - 1) * state_size, 0.f),
        previous_action(action_size, 0.f), current(state_size, 0.f) {}
  bool per_action() const override { return inner->per_action(); }
  int n_state() const override { return state_size * depth + (depth > 1 ? action_size : 0); }
  int n_actions() const override { return inner->n_actions(); }
  int act_dim() const override { return action_size; }
  void reset(Rng& rng) override {
    inner->reset(rng);
    std::fill(history.begin(), history.end(), 0.f);
    std::fill(previous_action.begin(), previous_action.end(), 0.f);
    has_history = false; filled = 0;
  }
  int legal(int* out) override { return inner->legal(out); }
  void state_feat(float* out) override {
    inner->state_feat(current.data());
    std::copy(current.begin(), current.end(), out);
    for (int k = 0; k < depth - 1; k++) {
      for (int i = 0; i < state_size; i++) {
        float newer = k == 0 ? current[i] : history[(size_t)(k - 1) * state_size + i];
        float older = history[(size_t)k * state_size + i];
        out[state_size * (k + 1) + i] =
            (has_history && (k == 0 || filled > k)) ? gain * (newer - older) : 0.f;
      }
    }
    if (depth > 1) std::copy(previous_action.begin(), previous_action.end(), out + state_size * depth);
  }
  float step_cont(const float* action, bool& done, Rng& rng) override {
    if (depth > 1) {
      for (int k = depth - 2; k > 0; k--)
        std::copy(history.begin() + (size_t)(k - 1) * state_size,
                  history.begin() + (size_t)k * state_size,
                  history.begin() + (size_t)k * state_size);
      std::copy(current.begin(), current.end(), history.begin());
      has_history = true; filled++;
      for (int i = 0; i < action_size; i++)
        previous_action[i] = std::clamp(action[i], -1.f, 1.f);
    }
    return inner->step_cont(action, done, rng);
  }
  float step(int action, bool& done, Rng& rng) override { return inner->step(action, done, rng); }
  int episode_result() const override { return inner->episode_result(); }
  std::unique_ptr<Env> make() const override {
    return std::make_unique<StackEnv>(inner->make(), depth, gain);
  }
};
