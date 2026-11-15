#pragma once
#include <vector>
#include <unordered_map>
#include <algorithm>
#include <cmath>
#include "rng.h"

enum NodeType : uint8_t { N_IN = 0, N_BIAS = 1, N_OUT = 2, N_HID = 3 };
constexpr int BIAS_ID = 1 << 24, OUT_BASE = BIAS_ID + 1, HID_BASE = 1 << 25;
// input node id == stable id (concept id / feature id). outputs: OUT_BASE+i, last output = value head.

struct NetSpec {
  int n_in = 0, n_logit = 0;
  std::vector<int> in_ids;                       // empty => 0..n_in-1
  int n_out() const { return n_logit + 1; }
  int in_id(int i) const { return in_ids.empty() ? i : in_ids[i]; }
};

struct Node { int id; uint8_t type; };
struct Conn { int innov, from, to; float w; bool en; float am = 0.f, av = 0.f; };   // am/av: persisted Adam moments (used only with PPOCfg.persist_adam)
struct Genome {
  std::vector<Node> nodes;
  std::vector<Conn> conns;                       // kept sorted by innov
  float fitness = 0, prev_fit = 0; int species = -1; long adam_t = 0;
  std::vector<float> logstd, ls_m, ls_v;           // continuous-action head: state-independent log-std (+ persisted Adam moments)
  int type_of(int id) const { for (auto& n : nodes) if (n.id == id) return n.type; return -1; }
  bool has_conn(int f, int t) const { for (auto& c : conns) if (c.from == f && c.to == t) return true; return false; }
};

struct Innovations {                             // main-thread only
  std::unordered_map<uint64_t, int> conn; int next_innov = 0;
  std::unordered_map<int, int> split; int next_hid = HID_BASE;
  int conn_id(int f, int t) {
    uint64_t k = ((uint64_t)(uint32_t)f << 32) | (uint32_t)t;
    auto it = conn.find(k); if (it != conn.end()) return it->second;
    return conn[k] = next_innov++;
  }
  std::unordered_map<int, int> init_hid;
  int initial_hidden(int layer, int unit) {
    int key = layer * 100000 + unit;
    auto it = init_hid.find(key); if (it != init_hid.end()) return it->second;
    return init_hid[key] = next_hid++;
  }
  int hidden_for(int innov) {
    auto it = split.find(innov); if (it != split.end()) return it->second;
    return split[innov] = next_hid++;
  }
};

inline void sort_conns(Genome& g) {
  std::sort(g.conns.begin(), g.conns.end(), [](const Conn& a, const Conn& b) { return a.innov < b.innov; });
}

inline void add_io_nodes(Genome& g, const NetSpec& s) {
  for (int i = 0; i < s.n_in; i++) g.nodes.push_back({s.in_id(i), N_IN});
  g.nodes.push_back({BIAS_ID, N_BIAS});
  for (int o = 0; o < s.n_out(); o++) g.nodes.push_back({OUT_BASE + o, N_OUT});
}

// Sparse minimal genome: each (input|bias)->output link exists with prob p.
inline Genome make_minimal(const NetSpec& s, Innovations& I, Rng& r, float p) {
  Genome g; add_io_nodes(g, s);
  for (int o = 0; o < s.n_out(); o++) {
    for (int i = 0; i <= s.n_in; i++) {
      int from = i < s.n_in ? s.in_id(i) : BIAS_ID, to = OUT_BASE + o;
      if (r.chance(p)) g.conns.push_back({I.conn_id(from, to), from, to, r.normal(), true});
    }
  }
  sort_conns(g); return g;
}

// Fully-connected MLP genome (tanh hidden, linear out). Use with MutCfg.topology=false for the fixed-MLP baseline.
inline Genome make_mlp(const NetSpec& s, const std::vector<int>& hid, Innovations& I, Rng& r) {
  Genome g; add_io_nodes(g, s);
  std::vector<int> prev;
  for (int i = 0; i < s.n_in; i++) prev.push_back(s.in_id(i));
  prev.push_back(BIAS_ID);
  auto link = [&](const std::vector<int>& src, int to) {
    float sc = 1.f / std::sqrt((float)src.size());
    for (int f : src) g.conns.push_back({I.conn_id(f, to), f, to, r.normal() * sc, true});
  };
  int layer_no = 0;
  for (int h : hid) {
    std::vector<int> layer;
    // Hidden ids are keyed by (layer, unit) so every initial genome in a population shares the same
    // historical markings; without this, crossover cannot align genes and speciation is degenerate.
    for (int k = 0; k < h; k++) { int id = I.initial_hidden(layer_no, k); g.nodes.push_back({id, N_HID}); layer.push_back(id); link(prev, id); }
    layer_no++;
    prev = layer; prev.push_back(BIAS_ID);
  }
  for (int o = 0; o < s.n_out(); o++) link(prev, OUT_BASE + o);
  sort_conns(g); return g;
}

inline bool reaches(const Genome& g, int src, int dst) {
  std::unordered_map<int, std::vector<int>> adj;
  for (auto& c : g.conns) if (c.en) adj[c.from].push_back(c.to);
  std::vector<int> st{src}; std::unordered_map<int, bool> seen;
  while (!st.empty()) {
    int u = st.back(); st.pop_back();
    if (u == dst) return true;
    if (seen[u]) continue; seen[u] = true;
    auto it = adj.find(u); if (it != adj.end()) for (int v : it->second) st.push_back(v);
  }
  return false;
}

struct MutCfg {
  float w_rate = 0.8f, w_power = 0.5f, w_replace = 0.1f;
  float p_conn = 0.05f, p_node = 0.03f, p_toggle = 0.01f;
  bool topology = true;
};

inline bool mut_add_conn(Genome& g, Innovations& I, Rng& r) {
  std::vector<int> src, dst;
  for (auto& n : g.nodes) {
    if (n.type != N_OUT) src.push_back(n.id);
    if (n.type == N_HID || n.type == N_OUT) dst.push_back(n.id);
  }
  for (int t = 0; t < 30; t++) {
    int f = src[r.below((int)src.size())], to = dst[r.below((int)dst.size())];
    if (f == to || g.has_conn(f, to) || reaches(g, to, f)) continue;
    g.conns.push_back({I.conn_id(f, to), f, to, r.normal(), true});
    return true;
  }
  return false;
}

inline bool mut_add_node(Genome& g, Innovations& I, Rng& r) {
  std::vector<int> cand;
  for (int i = 0; i < (int)g.conns.size(); i++) if (g.conns[i].en) cand.push_back(i);
  if (cand.empty()) return false;
  int ci = cand[r.below((int)cand.size())]; Conn c = g.conns[ci];
  g.conns[ci].en = false;
  int h = I.hidden_for(c.innov); if (g.type_of(h) >= 0) h = I.next_hid++;
  g.nodes.push_back({h, N_HID});
  g.conns.push_back({I.conn_id(c.from, h), c.from, h, 1.0f, true});
  g.conns.push_back({I.conn_id(h, c.to), h, c.to, c.w, true});
  return true;
}

inline void mutate(Genome& g, const MutCfg& m, Innovations& I, Rng& r) {
  for (auto& c : g.conns) if (r.chance(m.w_rate)) {
    if (r.chance(m.w_replace)) c.w = r.normal(); else c.w += r.normal() * m.w_power;
    c.w = std::clamp(c.w, -30.f, 30.f);
  }
  if (m.topology) {
    if (r.chance(m.p_node)) mut_add_node(g, I, r);
    if (r.chance(m.p_conn)) mut_add_conn(g, I, r);
    if (r.chance(m.p_toggle) && !g.conns.empty()) {
      Conn& c = g.conns[r.below((int)g.conns.size())];
      if (c.en) c.en = false; else if (!reaches(g, c.to, c.from)) c.en = true;
    }
  }
  sort_conns(g);
}

// a must be the fitter parent (disjoint/excess come from a). Child nodes = a's nodes.
inline Genome crossover(const Genome& a, const Genome& b, Rng& r) {
  Genome c; c.nodes = a.nodes; c.fitness = a.fitness; c.adam_t = a.adam_t; c.logstd = a.logstd; c.ls_m = a.ls_m; c.ls_v = a.ls_v;
  size_t j = 0;
  for (auto& ca : a.conns) {
    while (j < b.conns.size() && b.conns[j].innov < ca.innov) j++;
    Conn x = ca;
    if (j < b.conns.size() && b.conns[j].innov == ca.innov) {
      const Conn& cb = b.conns[j];
      if (r.chance(0.5f)) x.w = cb.w;
      if (!ca.en || !cb.en) x.en = !r.chance(0.75f);
    }
    c.conns.push_back(x);
  }
  return c;
}

inline float distance(const Genome& a, const Genome& b, float cd, float cw) {
  size_t i = 0, j = 0; int un = 0, nm = 0; double ws = 0;
  while (i < a.conns.size() && j < b.conns.size()) {
    int x = a.conns[i].innov, y = b.conns[j].innov;
    if (x == y) { ws += std::fabs(a.conns[i].w - b.conns[j].w); nm++; i++; j++; }
    else if (x < y) { un++; i++; } else { un++; j++; }
  }
  un += (int)(a.conns.size() - i + b.conns.size() - j);
  float N = (float)std::max(a.conns.size(), b.conns.size()); if (N < 20) N = 1;
  return cd * un / N + cw * (nm ? (float)(ws / nm) : 0.f);
}
