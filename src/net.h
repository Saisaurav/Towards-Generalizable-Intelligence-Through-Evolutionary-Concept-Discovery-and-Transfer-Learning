#pragma once
#include <cstring>
#include <cassert>
#include <queue>
#include "genome.h"

// Compiled phenotype. Slots: [0..n_in) inputs, n_in = bias, then compute nodes in topological order.
// Edges grouped by target. Activations are [slot][batch] so inner loops are contiguous over the batch.
struct Phenotype {
  int n_in = 0, n_out = 0, n_slots = 0, n_comp = 0, bias_slot = 0, cap = 0;
  std::vector<int> estart, esrc, econn, out_slot;
  std::vector<uint8_t> act;                       // 1 = tanh (hidden), 0 = linear (output)
  std::vector<float> w, g, a, d;
  float* A(int s) { return a.data() + (size_t)s * cap; }
  float* D(int s) { return d.data() + (size_t)s * cap; }

  void build(const Genome& G, int n_logit, int cap_) {
    n_out = n_logit + 1; cap = cap_;
    std::unordered_map<int, int> slot;
    n_in = 0;
    for (auto& n : G.nodes) if (n.type == N_IN) slot[n.id] = n_in++;
    bias_slot = n_in; slot[BIAS_ID] = bias_slot;
    std::vector<int> cid; std::unordered_map<int, int> cidx;
    for (auto& n : G.nodes) if (n.type == N_HID || n.type == N_OUT) { cidx[n.id] = (int)cid.size(); cid.push_back(n.id); }
    int nc = (int)cid.size();
    std::vector<std::vector<int>> in_e(nc);
    for (int ci = 0; ci < (int)G.conns.size(); ci++) {
      auto& c = G.conns[ci]; if (!c.en) continue;
      auto it = cidx.find(c.to); if (it != cidx.end()) in_e[it->second].push_back(ci);
    }
    std::vector<char> live(nc, 0); std::vector<int> st;
    for (int k = 0; k < nc; k++) if (cid[k] >= OUT_BASE && cid[k] < HID_BASE) { live[k] = 1; st.push_back(k); }
    while (!st.empty()) {
      int k = st.back(); st.pop_back();
      for (int ci : in_e[k]) { auto jt = cidx.find(G.conns[ci].from); if (jt != cidx.end() && !live[jt->second]) { live[jt->second] = 1; st.push_back(jt->second); } }
    }
    std::vector<int> indeg(nc, 0); std::vector<std::vector<int>> adj(nc);
    for (int k = 0; k < nc; k++) if (live[k]) for (int ci : in_e[k]) {
      auto jt = cidx.find(G.conns[ci].from); if (jt != cidx.end()) { indeg[k]++; adj[jt->second].push_back(k); }
    }
    std::queue<int> q; std::vector<int> order;
    for (int k = 0; k < nc; k++) if (live[k] && !indeg[k]) q.push(k);
    while (!q.empty()) { int k = q.front(); q.pop(); order.push_back(k); for (int m : adj[k]) if (--indeg[m] == 0) q.push(m); }
    n_comp = (int)order.size(); n_slots = n_in + 1 + n_comp;
    for (int p = 0; p < n_comp; p++) slot[cid[order[p]]] = n_in + 1 + p;
    estart.assign(1, 0); esrc.clear(); econn.clear(); w.clear(); act.clear();
    for (int p = 0; p < n_comp; p++) {
      int k = order[p]; act.push_back(cid[k] >= HID_BASE);
      for (int ci : in_e[k]) {
        auto it = slot.find(G.conns[ci].from); if (it == slot.end()) continue;
        esrc.push_back(it->second); econn.push_back(ci); w.push_back(G.conns[ci].w);
      }
      estart.push_back((int)esrc.size());
    }
    out_slot.assign(n_out, -1);
    for (int o = 0; o < n_out; o++) { auto it = slot.find(OUT_BASE + o); assert(it != slot.end()); out_slot[o] = it->second; }
    g.assign(w.size(), 0.f);
    if (a.size() < (size_t)n_slots * cap) { a.assign((size_t)n_slots * cap, 0.f); d.assign((size_t)n_slots * cap, 0.f); }
  }

  void forward(int B) {
    for (int k = 0; k < n_comp; k++) {
      float* __restrict o = A(n_in + 1 + k);
      for (int b = 0; b < B; b++) o[b] = 0.f;
      for (int e = estart[k]; e < estart[k + 1]; e++) {
        const float wv = w[e]; const float* __restrict s = A(esrc[e]);
        for (int b = 0; b < B; b++) o[b] += wv * s[b];
      }
      if (act[k]) for (int b = 0; b < B; b++) o[b] = std::tanh(o[b]);
    }
  }
  void begin_backward(int B) { for (int s = n_in + 1; s < n_slots; s++) std::memset(D(s), 0, sizeof(float) * B); }
  void backward(int B) {
    for (int k = n_comp - 1; k >= 0; k--) {
      int sl = n_in + 1 + k; float* __restrict ds = D(sl); const float* o = A(sl);
      if (act[k]) for (int b = 0; b < B; b++) ds[b] *= 1.f - o[b] * o[b];
      for (int e = estart[k]; e < estart[k + 1]; e++) {
        int sr = esrc[e]; const float* __restrict as = A(sr); float gs = 0.f;
        for (int b = 0; b < B; b++) gs += ds[b] * as[b];
        g[e] += gs;
        if (sr > n_in) { float* __restrict dsr = D(sr); const float wv = w[e]; for (int b = 0; b < B; b++) dsr[b] += wv * ds[b]; }
      }
    }
  }
  void write_back(Genome& G) const { for (size_t e = 0; e < w.size(); e++) G.conns[econn[e]].w = w[e]; }
};
