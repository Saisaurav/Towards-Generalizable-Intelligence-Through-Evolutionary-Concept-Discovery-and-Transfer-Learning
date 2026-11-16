#pragma once
#include <chrono>
#include <string>
#include <thread>
#include <atomic>
#include <cstdio>
#include <stdexcept>
#include "ppo.h"

struct NeatCfg {
  int pop = 48; uint64_t seed = 1; int threads = 0;
  bool write_back = true;            // true = Lamarckian, false = Darwinian (Baldwinian)
  int eval_episodes = 5;
  bool budget_train_only = false;    // true => run() budget counts PPO rollout steps only (eval steps logged separately)
  float init_prob = 0.3f;            // sparse initial wiring (minimal genome)
  std::vector<int> mlp_hidden;       // non-empty => fully-connected MLP init (baseline)
  MutCfg mut; PPOCfg ppo;
  float cx_prob = 0.6f, survive = 0.3f, thr = 3.f, c_d = 1.f, c_w = 0.4f;
  int target_species = 6; bool adapt_thr = true;
  float cont_out_gain = 0.05f, carry = 0.f;
};

struct Stats { int gen; long steps; float best, mean; int species; float nodes, conns; long train_steps = 0, eval_steps = 0; std::vector<float> fitness; float diversity = 0, champ_nodes = 0, champ_conns = 0; };

// Loop control for Population::run_until_win_rate (no effect on learning): a hard generation cap, a
// wall-clock cap (stops gracefully so the champion is still exported), a minimum number of
// generations that overrides the step budget, and an optional first generation checkpoint for
// target-based stopping.
struct StopLimits {
  int max_gens = 0;
  int min_gens = 0;
  int target_stop_generation = 0;
  double max_seconds = 0.0;
};

class Population {
public:
  NeatCfg cfg; NetSpec spec; Innovations inns; std::vector<Genome> pop; Genome champion;
  Env* proto; Rng rng; int gen = 0; std::atomic<long> steps{0}, train_steps{0}, eval_steps{0};

  // Constructing a Population always starts a fresh learner for this game:
  // new genomes/topologies and fresh PPO optimiser state. Cross-game transfer
  // is intentionally limited to future compatible concept features supplied
  // by the Env; no genome, policy weights, or optimiser state is imported.
  Population(Env& e, NeatCfg c) : cfg(c), proto(&e), rng(c.seed) {
    spec.n_in = e.n_in(); spec.n_logit = e.n_logit();
    for (int i = 0; i < cfg.pop; i++)
      pop.push_back(cfg.mlp_hidden.empty() ? make_minimal(spec, inns, rng, cfg.init_prob) : make_mlp(spec, cfg.mlp_hidden, inns, rng));
    if (e.act_dim() > 0) {
      for (auto& g : pop)
        for (auto& connection : g.conns)
          if (connection.to >= OUT_BASE && connection.to < OUT_BASE + spec.n_logit)
            connection.w *= cfg.cont_out_gain;
    }
  }

  // New input (e.g. new concept). Behavior unchanged until mutation wires it in. Env must expose the new column.
  void add_input(int stable_id) {
    if (spec.in_ids.empty()) for (int i = 0; i < spec.n_in; i++) spec.in_ids.push_back(i);
    spec.in_ids.push_back(stable_id); spec.n_in++;
    for (auto& g : pop) g.nodes.push_back({stable_id, N_IN});
    champion.nodes.push_back({stable_id, N_IN});
    workers.clear();
  }

  Stats step() {
    evaluate_all();
    Stats s{gen, steps.load(), -1e30f, 0, 0, 0, 0, train_steps.load(), eval_steps.load()};
    int bi = 0;
    for (int i = 0; i < (int)pop.size(); i++) {
      s.mean += pop[i].fitness; s.nodes += pop[i].nodes.size(); s.conns += pop[i].conns.size();
      if (pop[i].fitness > pop[bi].fitness) bi = i;
    }
    s.best = pop[bi].fitness; champion = pop[bi];
    s.mean /= pop.size(); s.nodes /= pop.size(); s.conns /= pop.size();
    for (auto& g : pop) s.fitness.push_back(g.fitness);
    s.champ_nodes = (float)champion.nodes.size(); s.champ_conns = (float)champion.conns.size();
    { double ds = 0; int dn = 0; int N = (int)pop.size();   // mean pairwise genome distance (population diversity)
      for (int i = 0; i < N; i++) for (int j = i + 1; j < N; j++) { ds += distance(pop[i], pop[j], cfg.c_d, cfg.c_w); dn++; }
      s.diversity = dn ? (float)(ds / dn) : 0.f; }
    speciate(); s.species = (int)species.size();
    reproduce(); gen++;
    return s;
  }

  float evaluate_champion(int episodes, uint64_t seed) {
    Trainer evaluator(proto->make(), cfg.ppo);
    return evaluator.evaluate_policy(champion, episodes, seed);
  }

  Trainer::EvalDetail evaluate_champion_detail(int episodes, uint64_t seed, bool greedy = false) {
    Trainer evaluator(proto->make(), cfg.ppo);
    return evaluator.evaluate_detail(champion, episodes, seed, greedy);
  }

  void run(long budget, const char* csv = nullptr, const StopLimits& limits = StopLimits{}) {
    FILE* f = csv ? std::fopen(csv, "w") : nullptr;
    if (f) std::fprintf(f, "gen,steps,best,mean,species,nodes,conns\n");
    const auto started = std::chrono::steady_clock::now();
    auto may_continue = [&]() {
      if (limits.max_gens > 0 && gen >= limits.max_gens) return false;
      if (limits.max_seconds > 0.0 &&
          std::chrono::duration<double>(std::chrono::steady_clock::now() - started).count() >= limits.max_seconds)
        return false;
      if (gen < limits.min_gens) return true;
      return budget > 0 && (cfg.budget_train_only ? train_steps.load() : steps.load()) < budget;
    };
    while (may_continue()) {
      Stats s = step();
      std::printf("gen %3d steps %9ld best %8.2f mean %8.2f sp %2d nodes %5.1f conns %5.1f\n", s.gen, s.steps, s.best, s.mean, s.species, s.nodes, s.conns);
      if (f) { std::fprintf(f, "%d,%ld,%.4f,%.4f,%d,%.2f,%.2f\n", s.gen, s.steps, s.best, s.mean, s.species, s.nodes, s.conns); std::fflush(f); }
    }
    if (f) std::fclose(f);
  }

  bool run_until_win_rate(double target, int eval_episodes, int eval_every_generations,
                          long max_steps, const char* csv = nullptr,
                          const StopLimits& limits = StopLimits{}) {
    if (!(target > 0.0 && target <= 1.0) || eval_episodes < 1 || eval_every_generations < 1 ||
        limits.max_gens < 0 || limits.min_gens < 0 || limits.target_stop_generation < 0 ||
        limits.max_seconds < 0.0)
      throw std::invalid_argument("invalid win-rate stopping configuration");
    FILE* f = csv ? std::fopen(csv, "w") : nullptr;
    if (f) std::fprintf(f, "gen,steps,best,mean,species,nodes,conns\n");
    bool reached = false;
    int last_eval_gen = -1;
    auto check_quality = [&]() {
      Trainer::EvalDetail detail = evaluate_champion_detail(
          eval_episodes, cfg.seed ^ (0xD1B54A32D192ED03ULL + (uint64_t)gen * 7919ULL),
          cfg.ppo.greedy_eval);
      const double win_rate = (double)detail.wins / std::max(1, detail.episodes);
      std::printf("QUALITY gen %d steps %ld win_rate %.4f wins %d losses %d draws %d mean_return %.4f\n",
                  gen - 1, steps.load(), win_rate, detail.wins, detail.losses, detail.draws, detail.mean);
      last_eval_gen = gen;
      return win_rate >= target;
    };
    const auto started = std::chrono::steady_clock::now();
    auto may_continue = [&]() {
      if (limits.max_gens > 0 && gen >= limits.max_gens) return false;
      if (limits.max_seconds > 0.0 &&
          std::chrono::duration<double>(std::chrono::steady_clock::now() - started).count() >= limits.max_seconds)
        return false;
      if (gen < limits.min_gens) return true;
      return max_steps <= 0 || (cfg.budget_train_only ? train_steps.load() : steps.load()) < max_steps;
    };
    auto may_stop_for_quality = [&](int generation) {
      return limits.target_stop_generation == 0 ||
          generation == limits.target_stop_generation ||
          (limits.max_gens > 0 && generation >= limits.max_gens);
    };
    while (may_continue()) {
      Stats s = step();
      std::printf("gen %3d steps %9ld best %8.2f mean %8.2f sp %2d nodes %5.1f conns %5.1f train %ld eval %ld\n",
                  s.gen, s.steps, s.best, s.mean, s.species, s.nodes, s.conns, s.train_steps, s.eval_steps);
      if (f) {
        std::fprintf(f, "%d,%ld,%.4f,%.4f,%d,%.2f,%.2f\n",
                     s.gen, s.steps, s.best, s.mean, s.species, s.nodes, s.conns);
        std::fflush(f);
      }
      const bool checkpoint = gen == limits.target_stop_generation ||
          (limits.max_gens > 0 && gen >= limits.max_gens);
      if (gen % eval_every_generations == 0 || checkpoint) {
        if (check_quality() && may_stop_for_quality(gen)) {
          reached = true;
          break;
        }
      }
    }
    if (!reached && gen > 0 && last_eval_gen != gen)
      reached = check_quality() && may_stop_for_quality(gen);
    std::printf("TARGET_REACHED %d\n", reached ? 1 : 0);
    std::printf("GENERATION_LIMIT_REACHED %d\n",
                (!reached && limits.max_gens > 0 && gen >= limits.max_gens) ? 1 : 0);
    if (f) std::fclose(f);
    return reached;
  }

private:
  struct Species { Genome rep; std::vector<int> mem; };
  std::vector<Species> species;
  std::vector<std::unique_ptr<Trainer>> workers;

  void evaluate_all() {
    int nt = cfg.threads > 0 ? cfg.threads : (int)std::max(1u, std::thread::hardware_concurrency());
    nt = std::min(nt, (int)pop.size());
    if ((int)workers.size() != nt) { workers.clear(); for (int t = 0; t < nt; t++) workers.push_back(std::make_unique<Trainer>(proto->make(), cfg.ppo)); }
    std::atomic<int> next{0};
    auto fn = [&](int t) {
      Trainer& T = *workers[t];
      for (;;) {
        int i = next++; if (i >= (int)pop.size()) break;
        float fresh_fitness = T.evaluate(pop[i], cfg.write_back, cfg.eval_episodes,
            cfg.seed * 1000003ULL + (uint64_t)gen * 7919ULL + i,
            cfg.seed * 0x9E3779B97F4A7C15ULL + (uint64_t)gen * 7919ULL);
        pop[i].fitness = (gen > 0 && cfg.carry > 0.f)
            ? cfg.carry * pop[i].prev_fit + (1.f - cfg.carry) * fresh_fitness
            : fresh_fitness;
        steps += T.steps; train_steps += T.train_steps; eval_steps += T.eval_steps;
      }
    };
    std::vector<std::thread> th; for (int t = 1; t < nt; t++) th.emplace_back(fn, t);
    fn(0); for (auto& x : th) x.join();
  }

  void speciate() {
    for (auto& s : species) s.mem.clear();
    for (int i = 0; i < (int)pop.size(); i++) {
      int f = -1;
      for (int s = 0; s < (int)species.size(); s++) if (distance(pop[i], species[s].rep, cfg.c_d, cfg.c_w) < cfg.thr) { f = s; break; }
      if (f < 0) { species.push_back({pop[i], {}}); f = (int)species.size() - 1; }
      species[f].mem.push_back(i); pop[i].species = f;
    }
    species.erase(std::remove_if(species.begin(), species.end(), [](const Species& s) { return s.mem.empty(); }), species.end());
    if (cfg.adapt_thr && pop.size() >= 8) {
      int n = (int)species.size();
      if (n > cfg.target_species + 1) cfg.thr += 0.3f; else if (n < cfg.target_species - 1) cfg.thr = std::max(0.5f, cfg.thr - 0.3f);
    }
  }

  void reproduce() {
    int S = (int)species.size(), N = (int)pop.size();
    float mn = 1e30f; for (auto& g : pop) mn = std::min(mn, g.fitness);
    std::vector<double> avg(S, 0); double tot = 0;
    for (int s = 0; s < S; s++) {   // fitness sharing: species share = mean shifted fitness
      for (int i : species[s].mem) avg[s] += pop[i].fitness - mn + 1e-3f;
      avg[s] /= species[s].mem.size(); tot += avg[s];
    }
    std::vector<int> n_off(S); int used = 0, best_s = 0;
    for (int s = 0; s < S; s++) { n_off[s] = (int)(N * avg[s] / tot); used += n_off[s]; if (avg[s] > avg[best_s]) best_s = s; }
    n_off[best_s] += N - used;
    std::vector<Genome> next; next.reserve(N);
    for (int s = 0; s < S; s++) {
      auto mem = species[s].mem;
      std::sort(mem.begin(), mem.end(), [&](int a, int b) { return pop[a].fitness > pop[b].fitness; });
      int n = n_off[s]; if (n <= 0) continue;
      next.push_back(pop[mem[0]]); n--;                       // champion carried over unchanged
      int surv = std::max(1, (int)std::ceil(cfg.survive * mem.size()));
      for (int k = 0; k < n; k++) {
        Genome c;
        if (surv >= 2 && rng.chance(cfg.cx_prob)) {
          int a = mem[rng.below(surv)], b = mem[rng.below(surv)];
          if (pop[b].fitness > pop[a].fitness) std::swap(a, b);
          c = crossover(pop[a], pop[b], rng);
        } else c = pop[mem[rng.below(surv)]];
        mutate(c, cfg.mut, inns, rng);
        next.push_back(std::move(c));
      }
      species[s].rep = pop[mem[rng.below((int)mem.size())]];
    }
    while ((int)next.size() < N) { Genome c = champion; mutate(c, cfg.mut, inns, rng); next.push_back(std::move(c)); }
    next.resize(N);
    for (auto& g : next) { g.prev_fit = g.fitness; g.fitness = 0; g.species = -1; }
    pop = std::move(next);
  }
};

// Presets = the designs under test. All share the same env interface / variable in-out spec.
inline NeatCfg preset(const std::string& n) {
  NeatCfg c;
  if (n == "lam") {                       // NEAT+PPO, Lamarckian: trained weights written back; gentle weight mutation
    c.write_back = true; c.mut.w_rate = 0.15f; c.mut.w_power = 0.1f; c.mut.w_replace = 0.f;
  } else if (n == "dar") {                // NEAT+PPO, Darwinian/Baldwinian: fitness after training, genome keeps original weights
    c.write_back = false;
  } else if (n == "mlp") {                // fixed MLP + PPO (pop 1, no mutation) = plain PPO baseline
    c.pop = 1; c.write_back = true; c.mlp_hidden = {16}; c.mut.w_rate = 0.f; c.mut.topology = false;
    c.ppo.iters = 4; c.eval_episodes = 3; c.threads = 1;
  } else if (n == "neat") {               // pure neuroevolution (no PPO)
    c.pop = 100; c.write_back = false; c.ppo.iters = 0;
  }
  return c;
}
