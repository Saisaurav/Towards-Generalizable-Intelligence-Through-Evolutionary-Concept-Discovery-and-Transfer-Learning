// Integration-boundary parity harness (added by the merge; no research code is modified).
// Runs V8's real ConceptEnv over a mock single-agent Env -- the same wrapping the brain applies to an
// ExternalEnv (generated game) -- and prints each concept's value for every legal action.
//   usage: ext_primitive_parity_test <expressions-file> < cases
//   case : "<n_actions> <n_legal> <legal...> <n_state> <state...>"   output: one line per legal action
#include "concept_env.h"
#include <fstream>
#include <iomanip>
#include <iostream>
#include <memory>

struct MockEnv : Env {
  std::vector<float> st; std::vector<int> lg; int na = 1;
  bool per_action() const override { return true; }
  int n_state() const override { return (int)st.size(); }
  int n_actions() const override { return na; }
  void reset(Rng&) override {}
  int legal(int* out) override { for (size_t i = 0; i < lg.size(); i++) out[i] = lg[i]; return (int)lg.size(); }
  void state_feat(float* out) override { for (size_t i = 0; i < st.size(); i++) out[i] = st[i]; }
  float step(int, bool& d, Rng&) override { d = true; return 0.f; }
  std::unique_ptr<Env> make() const override { return std::make_unique<MockEnv>(*this); }
};

int main(int argc, char** argv) {
  if (argc < 2) { std::cerr << "usage: " << argv[0] << " <expressions-file>\n"; return 2; }
  std::vector<std::string> ex; std::ifstream f(argv[1]); std::string line;
  while (std::getline(f, line)) { while (!line.empty() && (line.back() == '\r' || line.back() == ' ')) line.pop_back(); if (!line.empty()) ex.push_back(line); }
  std::cout << std::setprecision(9);
  int na, nl, ns;
  while (std::cin >> na >> nl) {
    auto m = std::make_unique<MockEnv>(); m->na = na; m->lg.resize(nl);
    for (int& a : m->lg) std::cin >> a;
    std::cin >> ns; m->st.resize(ns); for (float& v : m->st) std::cin >> v;
    std::vector<int> legal = m->lg;
    ConceptEnv ce(std::move(m), ex);
    std::vector<float> out(ce.n_action_feat());
    for (int a : legal) {
      ce.action_feat(a, out.data());
      for (size_t i = 0; i < ex.size(); i++) std::cout << out[i] << (i + 1 < ex.size() ? " " : "\n");
    }
  }
  return 0;
}
