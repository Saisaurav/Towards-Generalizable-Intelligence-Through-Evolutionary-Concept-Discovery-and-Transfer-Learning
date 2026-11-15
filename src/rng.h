#pragma once
#include <cstdint>
#include <cmath>
// xoshiro256++ (one instance per thread / per genome task)
struct Rng {
  uint64_t s[4];
  static uint64_t sm(uint64_t& x) {
    uint64_t z = (x += 0x9E3779B97F4A7C15ULL);
    z = (z ^ (z >> 30)) * 0xBF58476D1CE4E5B9ULL;
    z = (z ^ (z >> 27)) * 0x94D049BB133111EBULL;
    return z ^ (z >> 31);
  }
  explicit Rng(uint64_t seed = 1) { for (auto& v : s) v = sm(seed); }
  static inline uint64_t rotl(uint64_t x, int k) { return (x << k) | (x >> (64 - k)); }
  uint64_t next() {
    uint64_t r = rotl(s[0] + s[3], 23) + s[0], t = s[1] << 17;
    s[2] ^= s[0]; s[3] ^= s[1]; s[1] ^= s[2]; s[0] ^= s[3]; s[2] ^= t; s[3] = rotl(s[3], 45);
    return r;
  }
  float uniform() { return (next() >> 40) * (1.0f / 16777216.0f); }
  float range(float a, float b) { return a + (b - a) * uniform(); }
  int below(int n) { return (int)(next() % (uint64_t)n); }
  bool chance(float p) { return uniform() < p; }
  float normal() {
    float u1 = uniform(); if (u1 < 1e-7f) u1 = 1e-7f;
    return std::sqrt(-2.f * std::log(u1)) * std::cos(6.2831853f * uniform());
  }
};
