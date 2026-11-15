#pragma once
// Environment backed by an LLM-generated Python game running in a sandbox.
//
// The C++ learner never spawns or talks to Python directly. It speaks a tiny
// line protocol on its own stdout/stdin (portable to Windows and POSIX), and the
// Python trainer relays each request to one sandbox process per Env instance.
//
//   brain -> relay   "@NEW"            relay -> brain  "OK <id> <n_state> <n_actions>"
//                    "@R <id> <seed>"                  "S <reward> <done> <n_legal> <legal...> <n_state> <state...> <result> <expert_action>"
//                    "@A <id> <action>"                "S ..." | "ERR <message>"
//                    "@DEL <id>"                       (no reply)
//
// Any stdout line not starting with '@' is ordinary training progress output.
// The Python side owns multi-agent logic: opponents, seat selection and result
// bookkeeping all happen there, so from the learner's point of view this is the
// same single-agent Env interface as every built-in game. Observations are a
// fixed-length float vector; actions are discrete ids with a per-step legal list
// (PER_ACTION mode), so nothing here assumes a grid or four directions.
#include "env.h"
#include <algorithm>
#include <cstdio>
#include <iostream>
#include <memory>
#include <mutex>
#include <sstream>
#include <stdexcept>

struct ExtChannel {
  std::mutex m;
  std::string request(const std::string& line) {
    std::lock_guard<std::mutex> lock(m);
    std::fputs(line.c_str(), stdout);
    std::fputc('\n', stdout);
    std::fflush(stdout);
    std::string reply;
    if (!std::getline(std::cin, reply)) throw std::runtime_error("trainer relay closed the connection");
    while (!reply.empty() && (reply.back() == '\r' || reply.back() == '\n')) reply.pop_back();
    if (reply.rfind("ERR", 0) == 0) throw std::runtime_error("environment error: " + (reply.size() > 4 ? reply.substr(4) : std::string()));
    return reply;
  }
  void notify(const std::string& line) {
    std::lock_guard<std::mutex> lock(m);
    std::fputs(line.c_str(), stdout);
    std::fputc('\n', stdout);
    std::fflush(stdout);
  }
};

class ExternalEnv : public Env {
  std::shared_ptr<ExtChannel> ch;
  int id = -1, ns = 0, na = 0;
  int episode_step_limit = 0, episode_steps = 0;
  int expert = -1;
  std::vector<float> st;
  std::vector<int> lg;
  int result = 2;

  void connect() {
    std::istringstream in(ch->request("@NEW"));
    std::string tag;
    if (!(in >> tag >> id >> ns >> na) || tag != "OK" || ns < 1 || na < 1)
      throw std::runtime_error("bad @NEW reply from trainer relay");
    st.assign(ns, 0.f);
  }
  float parse(const std::string& reply, bool& done) {
    std::istringstream in(reply);
    std::string tag; float reward; int d, nl, n;
    if (!(in >> tag >> reward >> d >> nl) || tag != "S" || nl < 0 || nl > na)
      throw std::runtime_error("bad state reply from trainer relay: " + reply.substr(0, 80));
    lg.assign(nl, 0);
    for (int i = 0; i < nl; i++) if (!(in >> lg[i])) throw std::runtime_error("truncated legal-action list");
    if (!(in >> n) || n != ns) throw std::runtime_error("observation length changed during training");
    for (int i = 0; i < ns; i++) if (!(in >> st[i])) throw std::runtime_error("truncated observation");
    result = 2;
    int parsed_result;
    if (in >> parsed_result) {
      if (parsed_result < -1 || parsed_result > 2)
        throw std::runtime_error("invalid episode result from trainer relay");
      result = parsed_result;
    }
    expert = -1;
    int parsed_expert;
    if (in >> parsed_expert) {
      if (parsed_expert < -1 || parsed_expert >= na)
        throw std::runtime_error("invalid expert action from trainer relay");
      if (parsed_expert >= 0 &&
          std::find(lg.begin(), lg.end(), parsed_expert) == lg.end())
        throw std::runtime_error("expert action is not currently legal");
      expert = parsed_expert;
    }
    done = d != 0;
    return reward;
  }
public:
  explicit ExternalEnv(std::shared_ptr<ExtChannel> channel, int max_episode_steps = 0)
      : ch(std::move(channel)), episode_step_limit(max_episode_steps) {
    if (episode_step_limit < 0) throw std::invalid_argument("episode step limit cannot be negative");
    connect();
  }
  void set_episode_step_limit(int max_episode_steps) {
    if (max_episode_steps < 0) throw std::invalid_argument("episode step limit cannot be negative");
    episode_step_limit = max_episode_steps;
  }
  ~ExternalEnv() override { if (id >= 0) { try { ch->notify("@DEL " + std::to_string(id)); } catch (...) {} } }
  bool per_action() const override { return true; }
  int n_state() const override { return ns; }
  int n_actions() const override { return na; }
  int legal(int* out) override { for (size_t i = 0; i < lg.size(); i++) out[i] = lg[i]; return (int)lg.size(); }
  void state_feat(float* out) override { for (int i = 0; i < ns; i++) out[i] = st[i]; }
  int episode_result() const override { return result; }
  int expert_action() const override { return expert; }
  void reset(Rng& r) override {
    episode_steps = 0;
    bool done = false;
    parse(ch->request("@R " + std::to_string(id) + " " + std::to_string(1 + r.next() % 2000000000ULL)), done);
  }
  float step(int a, bool& done, Rng&) override {
    float reward = parse(ch->request("@A " + std::to_string(id) + " " + std::to_string(a)), done);
    episode_steps++;
    if (!done && episode_step_limit > 0 && episode_steps >= episode_step_limit) {
      done = true;
      result = 0;
    }
    return reward;
  }
  std::unique_ptr<Env> make() const override {
    return std::make_unique<ExternalEnv>(ch, episode_step_limit);
  }
};
