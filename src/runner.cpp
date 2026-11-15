// Experiment runner: trains one NEAT+PPO(+Lamarckian) learner from scratch on one task and writes a
// machine-readable JSON record (config, learning curve, held-out evaluation, random baseline, failure status).
//   usage: ./runner key=value ...      (see parse() for keys)
#include "neat.h"
#include "envs.h"
#include "concept_env.h"
#include "wrappers.h"
#include <chrono>
#include <ctime>
#include <fstream>
#include <iomanip>
#include <map>
#include <sstream>

#ifndef REPO_VERSION
#define REPO_VERSION "unknown"
#endif

using KV = std::map<std::string, std::string>;
static std::string get(const KV& k, const std::string& n, const std::string& d) { auto it = k.find(n); return it == k.end() ? d : it->second; }
static double getd(const KV& k, const std::string& n, double d) { auto it = k.find(n); return it == k.end() ? d : std::atof(it->second.c_str()); }

static std::vector<std::string> read_lines(const std::string& path) {
  std::vector<std::string> out; std::ifstream f(path); std::string s;
  if (!f) throw std::runtime_error("cannot open concepts: " + path);
  while (std::getline(f, s)) if (!s.empty() && s[0] != '#') out.push_back(s);
  return out;
}

static std::string g_grid_spec;   // path of a grid-spec file (game=grid)
static std::unique_ptr<Env> make_base(const std::string& game, int opp, int ttt_feat) {
  if (game == "ttt") { auto e = std::make_unique<TicTacToe>(); e->nf = ttt_feat ? 4 : 0; e->opp = opp; return e; }
  if (game == "connect_four") { auto e = std::make_unique<ConnectFour>(); e->opp = opp; return e; }
  if (game == "nim") { auto e = std::make_unique<Nim>(); e->opp = opp; return e; }
  if (game == "cartpole") return std::make_unique<CartPole>();
  if (game == "grid") return std::make_unique<GridWorld>(GridWorld::from_file(g_grid_spec.c_str()));
  if (game == "lineworld") return std::make_unique<LineWorld>();
  throw std::runtime_error("unknown game: " + game);
}

static int g_act_onehot = 0, g_state_onehot = 0;   // ConceptEnv options (runner keys act_onehot=, state_onehot=)
static std::unique_ptr<Env> wrap(std::unique_ptr<Env> base, const std::string& repr, const std::vector<std::string>& concepts) {
  if (repr == "native") return base;
  if (repr == "fixed") return std::make_unique<FixedHeadEnv>(std::move(base));
  if (repr == "onehot") return std::make_unique<OneHotEnv>(std::move(base));
  if (repr == "onehot_state") return std::make_unique<OneHotScalarStateEnv>(std::move(base), 22);
  if (repr == "sxa") return std::make_unique<RawSxAEnv>(std::move(base), 4, g_act_onehot != 0, g_state_onehot);
  if (repr == "tactics") return std::make_unique<TacticEnv>(std::move(base));
  if (repr == "aid") return std::make_unique<ConceptEnv>(std::move(base), std::vector<std::string>{"action_norm"}, g_act_onehot != 0, g_state_onehot);
  if (repr == "concepts") return std::make_unique<ConceptEnv>(std::move(base), concepts, g_act_onehot != 0, g_state_onehot);
  throw std::runtime_error("unknown repr: " + repr);
}

struct Outcome { double mean = 0, mean_len = 0; int pos = 0, neg = 0, zero = 0, episodes = 0, full_len = 0; };
static Outcome random_policy(const std::string& game, int opp, int ttt_feat, int eps, uint64_t seed) {
  auto env = make_base(game, opp, ttt_feat); Rng rng(seed); Outcome o; o.episodes = eps; int lg[128]; double tot = 0, len = 0;
  for (int e = 0; e < eps; e++) {
    env->reset(rng); bool done = false; double ret = 0; int l = 0;
    while (!done) { int n = env->legal(lg); ret += env->step(lg[rng.below(n)], done, rng); l++; }
    tot += ret; len += l; if (ret > 0) o.pos++; else if (ret < 0) o.neg++; else o.zero++; if (l >= 500) o.full_len++;
  }
  o.mean = tot / eps; o.mean_len = len / eps; return o;
}
static std::string js(const Outcome& o) {
  std::ostringstream s; s << std::setprecision(8);
  s << "{\"mean_return\":" << o.mean << ",\"win\":" << o.pos << ",\"loss\":" << o.neg << ",\"draw_or_zero\":" << o.zero
    << ",\"episodes\":" << o.episodes << ",\"mean_length\":" << o.mean_len << ",\"full_length_episodes\":" << o.full_len << "}";
  return s.str();
}
static std::string js(const Trainer::EvalDetail& d) {
  Outcome o; o.mean = d.mean; o.mean_len = d.mean_len; o.pos = d.pos; o.neg = d.neg; o.zero = d.zero; o.episodes = d.episodes; o.full_len = d.full_len; return js(o);
}
static std::string esc(const std::string& s) { std::string o; for (char c : s) { if (c == '"' || c == '\\') o += '\\'; if (c == '\n') o += "\\n"; else o += c; } return o; }
static uint64_t fnv(const std::string& s) { uint64_t h = 1469598103934665603ULL; for (unsigned char c : s) { h ^= c; h *= 1099511628211ULL; } return h; }

int main(int argc, char** argv) {
  KV kv; std::string cfg_str;
  for (int i = 1; i < argc; i++) { std::string a = argv[i]; auto p = a.find('='); if (p == std::string::npos) { std::fprintf(stderr, "bad arg %s\n", a.c_str()); return 2; } kv[a.substr(0, p)] = a.substr(p + 1); cfg_str += a + ";"; }
  std::string game = get(kv, "game", "nim"), repr = get(kv, "repr", "aid"), tag = get(kv, "tag", "run"), out = get(kv, "out", "");
  uint64_t seed = (uint64_t)getd(kv, "seed", 11);
  long budget = (long)getd(kv, "budget", 150000);
  int ttt_feat = (int)getd(kv, "ttt_feat", 0);
  std::string concept_file = get(kv, "concepts", "");
  g_grid_spec = get(kv, "grid", ""); g_act_onehot = (int)getd(kv, "act_onehot", 0); g_state_onehot = (int)getd(kv, "state_onehot", 0);
  auto t0 = std::chrono::steady_clock::now();
  std::time_t now = std::time(nullptr); char ts[40]; std::strftime(ts, sizeof ts, "%Y-%m-%dT%H:%M:%SZ", std::gmtime(&now));
  char idbuf[32]; std::snprintf(idbuf, sizeof idbuf, "%08llx", (unsigned long long)(fnv(cfg_str) & 0xffffffffULL));
  std::string exp_id = game + "_" + tag + "_s" + std::to_string(seed) + "_" + idbuf;
  std::ostringstream J; J << std::setprecision(8);
  std::string status = "ok";
  std::ostringstream curve, final_block, arch;
  Outcome rnd0, rnd1; bool have_final = false; NeatCfg c;
  try {
    std::vector<std::string> concepts; if (!concept_file.empty()) concepts = read_lines(concept_file);
    c = preset(get(kv, "preset", "lam")); 
    std::string hid = get(kv, "hidden", "16"); c.mlp_hidden.clear();
    { std::stringstream hs(hid); std::string t; while (std::getline(hs, t, ',')) if (!t.empty() && t != "0") c.mlp_hidden.push_back(std::atoi(t.c_str())); }
    c.eval_episodes = (int)getd(kv, "eval_eps", 10); c.threads = (int)getd(kv, "threads", 1); c.seed = seed; c.pop = (int)getd(kv, "pop", 24);
    c.ppo.iters = (int)getd(kv, "ppo_iters", c.ppo.iters); c.ppo.rollout = (int)getd(kv, "rollout", c.ppo.rollout);
    c.ppo.epochs = (int)getd(kv, "epochs", c.ppo.epochs); c.ppo.minibatch = (int)getd(kv, "minibatch", c.ppo.minibatch);
    c.ppo.lr = (float)getd(kv, "lr", c.ppo.lr); c.ppo.ent = (float)getd(kv, "ent", c.ppo.ent); c.ppo.max_grad = (float)getd(kv, "max_grad", c.ppo.max_grad);
    c.ppo.greedy_eval = getd(kv, "greedy_eval", 0) != 0;
    c.ppo.gamma = (float)getd(kv, "gamma", c.ppo.gamma); c.ppo.lam = (float)getd(kv, "gae_lambda", c.ppo.lam);
    c.write_back = getd(kv, "write_back", c.write_back ? 1 : 0) != 0;
    c.mut.w_rate = (float)getd(kv, "w_rate", c.mut.w_rate); c.mut.w_power = (float)getd(kv, "w_power", c.mut.w_power);
    c.mut.p_node = (float)getd(kv, "p_node", c.mut.p_node); c.mut.p_conn = (float)getd(kv, "p_conn", c.mut.p_conn);
    c.survive = (float)getd(kv, "survive", c.survive);
    c.budget_train_only = getd(kv, "budget_train_only", 1) != 0;
    int ckpt_every = (int)getd(kv, "ckpt_every", 5), ckpt_eps = (int)getd(kv, "ckpt_eps", 100);

    auto train_env = wrap(make_base(game, 0, ttt_feat), repr, concepts);
    Population p(*train_env, c);
    bool first = true;
    while ((c.budget_train_only ? p.train_steps.load() : p.steps.load()) < budget) {
      Stats s = p.step();
      if (!first) curve << ","; first = false;
      curve << "{\"gen\":" << s.gen << ",\"train_steps\":" << s.train_steps << ",\"eval_steps\":" << s.eval_steps << ",\"best\":" << s.best
            << ",\"mean\":" << s.mean << ",\"species\":" << s.species << ",\"avg_nodes\":" << s.nodes << ",\"avg_conns\":" << s.conns
            << ",\"champ_nodes\":" << s.champ_nodes << ",\"champ_conns\":" << s.champ_conns << ",\"diversity\":" << s.diversity << ",\"fitness\":[";
      for (size_t i = 0; i < s.fitness.size(); i++) curve << (i ? "," : "") << s.fitness[i];
      curve << "]";
      if (ckpt_every > 0 && (s.gen % ckpt_every == 0)) {
        Trainer T(train_env->make(), c.ppo);
        auto d = T.evaluate_detail(p.champion, ckpt_eps, (seed * 31 + (uint64_t)s.gen) ^ 0xA5A5A5A5ULL, true);
        curve << ",\"heldout_greedy\":" << d.mean << ",\"heldout_len\":" << d.mean_len;
      }
      curve << "}";
    }
    // ---- final held-out evaluation (1000 episodes), seeds disjoint from training streams
    final_block << "{";
    bool fb = true;
    for (int opp = 0; opp <= 1; opp++) {
      if (game == "cartpole" && opp == 1) break;
      auto eenv = wrap(make_base(game, opp, ttt_feat), repr, concepts);
      for (int greedy = 0; greedy <= 1; greedy++) {
        Trainer T(eenv->make(), c.ppo);
        auto d = T.evaluate_detail(p.champion, 1000, seed ^ (0xD1B54A32D192ED03ULL + (uint64_t)opp * 977 + (uint64_t)greedy * 31), greedy == 1);
        if (!fb) final_block << ","; fb = false;
        final_block << "\"" << (opp ? "vs_strategic" : "vs_random") << "_" << (greedy ? "greedy" : "stochastic") << "\":" << js(d);
      }
    }
    final_block << "}"; have_final = true;
    rnd0 = random_policy(game, 0, ttt_feat, 1000, seed ^ 0x1234567ULL); rnd1 = random_policy(game, 1, ttt_feat, 1000, seed ^ 0x7654321ULL);
    int hid_n = 0, in_n = 0, en = 0; for (auto& n : p.champion.nodes) { hid_n += n.type == N_HID; in_n += n.type == N_IN; } for (auto& cn : p.champion.conns) en += cn.en;
    arch << "{\"inputs\":" << in_n << ",\"hidden\":" << hid_n << ",\"outputs\":" << (p.spec.n_out()) << ",\"nodes\":" << p.champion.nodes.size()
         << ",\"connections\":" << p.champion.conns.size() << ",\"enabled_connections\":" << en << "}";
    J << "\"train_steps\":" << p.train_steps.load() << ",\"eval_steps\":" << p.eval_steps.load() << ",\"generations\":" << p.gen << ",";
  } catch (const std::exception& e) { status = std::string("failed: ") + e.what(); }
  double secs = std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
  std::ostringstream R; R << std::setprecision(8);
  R << "{\"experiment_id\":\"" << exp_id << "\",\"repo_version\":\"" << REPO_VERSION << "\",\"started_utc\":\"" << ts << "\",\"status\":\"" << esc(status)
    << "\",\"runtime_sec\":" << secs << ",\"config\":{";
  { bool f = true; for (auto& k : kv) { R << (f ? "" : ",") << "\"" << k.first << "\":\"" << esc(k.second) << "\""; f = false; } }
  R << "},\"resolved\":{\"hidden\":\"" << get(kv, "hidden", "16") << "\",\"pop\":" << c.pop << ",\"eval_episodes\":" << c.eval_episodes << ",\"write_back\":" << c.write_back
    << ",\"ppo\":{\"iters\":" << c.ppo.iters << ",\"rollout\":" << c.ppo.rollout << ",\"epochs\":" << c.ppo.epochs << ",\"minibatch\":" << c.ppo.minibatch << ",\"lr\":" << c.ppo.lr
    << ",\"ent\":" << c.ppo.ent << ",\"max_grad\":" << c.ppo.max_grad << ",\"greedy_eval\":" << c.ppo.greedy_eval << ",\"gamma\":" << c.ppo.gamma << "},\"budget_counts\":\"" << (c.budget_train_only ? "ppo_rollout_steps_only" : "train_plus_eval_steps")
    << "\",\"concept_file\":\"" << esc(concept_file) << "\"},";
  R << J.str();
  R << "\"curve\":[" << curve.str() << "],";
  R << "\"architecture\":" << (arch.str().empty() ? "null" : arch.str()) << ",";
  R << "\"heldout\":" << (have_final ? final_block.str() : "null") << ",";
  R << "\"random_baseline\":" << (have_final ? "{\"vs_random\":" + js(rnd0) + ",\"vs_strategic\":" + js(rnd1) + "}" : "null") << "}\n";
  if (!out.empty()) { std::ofstream f(out); f << R.str(); } else std::fputs(R.str().c_str(), stdout);
  return status == "ok" ? 0 : 1;
}
