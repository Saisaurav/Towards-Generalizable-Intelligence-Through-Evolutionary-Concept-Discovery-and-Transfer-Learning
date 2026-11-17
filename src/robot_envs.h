#pragma once
// Fast surrogate environments for the two Gazebo demos (train here at ~1e5-1e6 steps/s, deploy in Gazebo).
//   TurtleBotNav : TurtleBot3-Burger-like diff-drive robot, 4x4 m arena with random circular obstacles, 24-ray lidar.
//                  Action  = [v, w] in [-1,1]^2  -> /cmd_vel (v*0.22 m/s, w*2.0 rad/s), 10 Hz.
//   Arm2Link     : planar 2-link arm reaching a target with the end effector.
//                  Action  = joint velocity commands in [-1,1]^2 (x2.0 rad/s), 20 Hz.
// Both model first-order actuator lag, command delay, sensor noise, and per-episode domain randomization of all three.
// Observation layouts are documented in ENV_SPECS.md and in the exported policy JSON ("obs_names").
#include "env.h"
#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <deque>
#include <stdexcept>
#include <map>
#include <sstream>
#include <string>
#include <vector>

// "key=value,key=value" -> map; unknown keys are ignored by each env (shared grammar between envs).
inline std::map<std::string, float> parse_kv(const std::string& s) {
  std::map<std::string, float> m; std::stringstream ss(s); std::string kv;
  while (std::getline(ss, kv, ',')) { size_t e = kv.find('='); if (e != std::string::npos) m[kv.substr(0, e)] = (float)std::atof(kv.c_str() + e + 1); }
  return m;
}
inline float kvget(const std::map<std::string, float>& m, const char* k, float d) { auto it = m.find(k); return it == m.end() ? d : it->second; }

struct RobotCfg {
  float noise = 0.f, lag = 1.f;   int delay = 0;          // nominal perturbations (eval grid / fixed deployment mismatch)
  float rnoise = 0.f, rlag = 1.f; int rdelay = 0;         // per-episode randomization: noise~U(0,rnoise), lag log-uniform [1/rlag, rlag], delay~U{0..rdelay}
  int horizon = 0; bool tfeat = true; int feat = 0; float fk = 0.25f;   // feat (arm only): 0 = base, 1 = + fine EE error, 2 = + fine error + J^T e
  static RobotCfg parse(const std::string& s, int default_horizon) {
    auto m = parse_kv(s); RobotCfg c;
    static const char* known[] = {"noise","lag","delay","rnoise","rlag","rdelay","horizon","tfeat","feat","fk"};
    for (auto& kvp : m) { bool ok = false; for (auto k : known) if (kvp.first == k) ok = true; if (!ok) throw std::runtime_error("unknown env arg: " + kvp.first); }
    c.noise = kvget(m, "noise", 0); c.lag = kvget(m, "lag", 1); c.delay = (int)kvget(m, "delay", 0);
    c.rnoise = kvget(m, "rnoise", 0); c.rlag = kvget(m, "rlag", 1); c.rdelay = (int)kvget(m, "rdelay", 0);
    c.horizon = (int)kvget(m, "horizon", (float)default_horizon); c.tfeat = kvget(m, "tfeat", 1) != 0; c.feat = (int)kvget(m, "feat", 0); c.fk = kvget(m, "fk", 0.25f);
    return c;
  }
};

// Shared actuator-channel logic: delay queue + per-episode randomization.
struct Channel {
  RobotCfg cfg, eff; std::deque<std::pair<float, float>> q;
  void begin(Rng& r) {
    eff = cfg;
    if (cfg.rnoise > 0) eff.noise = r.range(0.f, cfg.rnoise);
    if (cfg.rdelay > 0) eff.delay = r.below(cfg.rdelay + 1);
    if (cfg.rlag > 1.f) eff.lag = cfg.lag * std::exp(r.range(-std::log(cfg.rlag), std::log(cfg.rlag)));
    q.assign(eff.delay, {0.f, 0.f});
  }
  std::pair<float, float> apply(float a, float b) {
    if (eff.delay <= 0) return {a, b};
    q.push_back({a, b}); auto o = q.front(); q.pop_front(); return o;
  }
};

// ---------------------------------------------------------------- TurtleBot with obstacles
struct TurtleBotNav : Env {
  static constexpr int NR = 24, MAXOB = 6;
  static constexpr float DT = 0.1f, VMAX = 0.22f, WMAX = 2.0f, RAD = 0.105f, ARENA = 2.0f, RANGE = 3.5f, TAU = 0.2f, GOAL_TOL = 0.15f;
  Channel ch; Rng orng{3};
  float x = 0, y = 0, th = 0, v = 0, w = 0, gx = 0, gy = 0, dprev = 0, pa0 = 0, pa1 = 0;
  int t = 0, nob = 0; bool reached = false, crashed = false;
  float ox[MAXOB], oy[MAXOB], orad[MAXOB];
  explicit TurtleBotNav(RobotCfg c = RobotCfg::parse("", 300)) { ch.cfg = ch.eff = c; }
  bool per_action() const override { return false; }
  int n_state() const override { return NR + 3 + 2 + 2 + (ch.cfg.tfeat ? 1 : 0); }
  int n_actions() const override { return 2; }
  int act_dim() const override { return 2; }
  int legal(int* o) override { o[0] = 0; o[1] = 1; return 2; }
  float step(int, bool& d, Rng&) override { d = true; return 0.f; }
  static float wrap(float a) { while (a > 3.14159265f) a -= 6.2831853f; while (a < -3.14159265f) a += 6.2831853f; return a; }
  void reset(Rng& r) override {
    ch.begin(r); orng = Rng(r.next()); t = 0; reached = crashed = false; v = w = pa0 = pa1 = 0;
    x = r.range(-1.7f, 1.7f); y = r.range(-1.7f, 1.7f); th = r.range(-3.14159f, 3.14159f);
    do { gx = r.range(-1.7f, 1.7f); gy = r.range(-1.7f, 1.7f); } while (std::hypot(gx - x, gy - y) < 1.0f);
    nob = 3 + r.below(MAXOB - 2);
    for (int i = 0; i < nob; i++) {
      for (int tries = 0; tries < 50; tries++) {
        orad[i] = r.range(0.1f, 0.25f); ox[i] = r.range(-1.7f, 1.7f); oy[i] = r.range(-1.7f, 1.7f);
        if (std::hypot(ox[i] - x, oy[i] - y) > orad[i] + RAD + 0.25f && std::hypot(ox[i] - gx, oy[i] - gy) > orad[i] + GOAL_TOL + 0.2f) break;
        if (tries == 49) { ox[i] = 9.f; oy[i] = 9.f; }   // could not place: park it outside the arena
      }
    }
    dprev = std::hypot(gx - x, gy - y);
  }
  float cast(float ang) const {
    float dx = std::cos(ang), dy = std::sin(ang), best = RANGE;
    for (int i = 0; i < nob; i++) {
      float fx = x - ox[i], fy = y - oy[i], b = fx * dx + fy * dy, c = fx * fx + fy * fy - orad[i] * orad[i], disc = b * b - c;
      if (disc >= 0) { float tt = -b - std::sqrt(disc); if (tt >= 0 && tt < best) best = tt; }
    }
    if (dx > 1e-6f) best = std::min(best, (ARENA - x) / dx); else if (dx < -1e-6f) best = std::min(best, (-ARENA - x) / dx);
    if (dy > 1e-6f) best = std::min(best, (ARENA - y) / dy); else if (dy < -1e-6f) best = std::min(best, (-ARENA - y) / dy);
    return std::max(0.f, best);
  }
  void state_feat(float* o) override {
    const float nz = ch.eff.noise; int k = 0;
    for (int i = 0; i < NR; i++) o[k++] = std::clamp(cast(th + 6.2831853f * i / NR) + nz * orng.normal(), 0.f, RANGE) / RANGE;
    float dx = gx - x + nz * orng.normal(), dy = gy - y + nz * orng.normal(), d = std::hypot(dx, dy), b = wrap(std::atan2(dy, dx) - th);
    o[k++] = std::min(d, 4.f) / 4.f; o[k++] = std::sin(b); o[k++] = std::cos(b);
    o[k++] = v / VMAX; o[k++] = w / WMAX; o[k++] = pa0; o[k++] = pa1;
    if (ch.cfg.tfeat) o[k++] = (float)t / ch.cfg.horizon;
  }
  float step_cont(const float* a, bool& done, Rng&) override {
    float a0 = std::clamp(a[0], -1.f, 1.f), a1 = std::clamp(a[1], -1.f, 1.f); pa0 = a0; pa1 = a1;
    auto c = ch.apply(a0, a1); float al = 1.f - std::exp(-DT / (TAU * ch.eff.lag));
    v += al * (c.first * VMAX - v); w += al * (c.second * WMAX - w);
    x += v * std::cos(th) * DT; y += v * std::sin(th) * DT; th = wrap(th + w * DT); t++;
    float d = std::hypot(gx - x, gy - y), r = (dprev - d) - 0.002f; dprev = d;
    if (d < GOAL_TOL) { reached = true; done = true; return r + 2.f; }
    bool hit = std::fabs(x) > ARENA - RAD || std::fabs(y) > ARENA - RAD;
    for (int i = 0; i < nob && !hit; i++) hit = std::hypot(x - ox[i], y - oy[i]) < orad[i] + RAD;
    if (hit) { crashed = true; done = true; return r - 1.f; }
    done = t >= ch.cfg.horizon; return r;
  }
  int episode_result() const override { return reached ? 1 : -1; }
  std::unique_ptr<Env> make() const override { return std::make_unique<TurtleBotNav>(ch.cfg); }
};

// ---------------------------------------------------------------- planar 2-link arm
struct Arm2Link : Env {
  static constexpr float DT = 0.05f, L1 = 0.30f, L2 = 0.25f, VMAX = 2.0f, TAU = 0.1f, Q1 = 3.0f, Q2 = 2.6f, TOL = 0.03f;
  Channel ch; Rng orng{5};
  float qh1 = 0, qh2 = 0, qf1 = 0, qf2 = 0; bool qinit = false;   // feat>=3: complementary-filter joint estimate (predict with commanded velocity, correct with noisy reading)
  float q1 = 0, q2 = 0, d1 = 0, d2 = 0, tx = 0, ty = 0, pa0 = 0, pa1 = 0, last_speed = 0; int t = 0; bool ok = false;
  explicit Arm2Link(RobotCfg c = RobotCfg::parse("", 100)) { ch.cfg = ch.eff = c; }
  bool per_action() const override { return false; }
  int n_state() const override { return 4 + 2 + 2 + 2 + (ch.cfg.feat >= 1 ? 2 : 0) + (ch.cfg.feat >= 2 ? 2 : 0) + (ch.cfg.tfeat ? 1 : 0); }
  int n_actions() const override { return 2; }
  int act_dim() const override { return 2; }
  int legal(int* o) override { o[0] = 0; o[1] = 1; return 2; }
  float step(int, bool& d, Rng&) override { d = true; return 0.f; }
  static void fk(float a, float b, float& x, float& y) { x = L1 * std::cos(a) + L2 * std::cos(a + b); y = L1 * std::sin(a) + L2 * std::sin(a + b); }
  void reset(Rng& r) override {
    ch.begin(r); orng = Rng(r.next()); qinit = false; t = 0; ok = false; d1 = d2 = pa0 = pa1 = last_speed = 0;
    q1 = r.range(-1.5f, 1.5f); q2 = r.range(-1.5f, 1.5f);
    fk(r.range(-2.5f, 2.5f), r.range(-2.4f, 2.4f), tx, ty);   // reachable by construction
  }
  void state_feat(float* o) override {
    const float nz = ch.eff.noise; float a = q1 + nz * orng.normal(), b = q2 + nz * orng.normal(), ex, ey;
    if (ch.cfg.feat >= 3) {
      float p1 = qinit ? qh1 + pa0 * VMAX * DT : a, p2 = qinit ? qh2 + pa1 * VMAX * DT : b;
      a = qf1 = p1 + ch.cfg.fk * (a - p1); b = qf2 = p2 + ch.cfg.fk * (b - p2);
    }
    fk(a, b, ex, ey);
    int k = 0; o[k++] = std::sin(a); o[k++] = std::cos(a); o[k++] = std::sin(b); o[k++] = std::cos(b);
    o[k++] = d1 / VMAX + 5.f * nz * orng.normal(); o[k++] = d2 / VMAX + 5.f * nz * orng.normal();
    float errx = tx - ex, erry = ty - ey;
    o[k++] = errx / 0.55f; o[k++] = erry / 0.55f; o[k++] = pa0; o[k++] = pa1;
    if (ch.cfg.feat >= 1) { o[k++] = std::clamp(errx / 0.08f, -1.f, 1.f); o[k++] = std::clamp(erry / 0.08f, -1.f, 1.f); }
    if (ch.cfg.feat >= 2) {   // J^T e from observed joint angles: joint-space direction that reduces the error
      float s1 = std::sin(a), c1 = std::cos(a), s12 = std::sin(a + b), c12 = std::cos(a + b);
      float j11 = -L1 * s1 - L2 * s12, j12 = -L2 * s12, j21 = L1 * c1 + L2 * c12, j22 = L2 * c12;
      o[k++] = std::clamp((j11 * errx + j21 * erry) / 0.05f, -1.f, 1.f); o[k++] = std::clamp((j12 * errx + j22 * erry) / 0.05f, -1.f, 1.f);
    }
    if (ch.cfg.tfeat) o[k++] = (float)t / ch.cfg.horizon;
  }
  float step_cont(const float* a, bool& done, Rng&) override {
    if (ch.cfg.feat >= 3) { qh1 = qf1; qh2 = qf2; qinit = true; }
    float a0 = std::clamp(a[0], -1.f, 1.f), a1 = std::clamp(a[1], -1.f, 1.f); pa0 = a0; pa1 = a1;
    auto c = ch.apply(a0, a1); float al = 1.f - std::exp(-DT / (TAU * ch.eff.lag));
    d1 += al * (c.first * VMAX - d1); d2 += al * (c.second * VMAX - d2);
    q1 += d1 * DT; q2 += d2 * DT;
    if (q1 > Q1 || q1 < -Q1) { q1 = std::clamp(q1, -Q1, Q1); d1 = 0; }
    if (q2 > Q2 || q2 < -Q2) { q2 = std::clamp(q2, -Q2, Q2); d2 = 0; }
    float ex, ey; fk(q1, q2, ex, ey); float dist = std::hypot(tx - ex, ty - ey); last_speed = std::hypot(d1, d2) / VMAX;
    t++; done = t >= ch.cfg.horizon; ok = dist < TOL && last_speed < 0.1f;
    return -0.1f * dist / 0.3f - 0.01f * (a0 * a0 + a1 * a1);
  }
  int episode_result() const override { return ok ? 1 : -1; }
  std::unique_ptr<Env> make() const override { return std::make_unique<Arm2Link>(ch.cfg); }
};
