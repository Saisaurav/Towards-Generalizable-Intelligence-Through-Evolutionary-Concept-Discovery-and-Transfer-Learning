#include "neat.h"
#include "envs.h"
#include "concept_env.h"
#include "external_env.h"
#include "fixed_head_env.h"
#include "robotics_env.h"
#include <fstream>
#include <cstdlib>
#include <iomanip>
#include <string>

static std::vector<std::string> load_concepts(const char* path) {
  std::vector<std::string> out; std::ifstream f(path); std::string s;
  if(!f) throw std::runtime_error(std::string("cannot open concepts: ")+path);
  while(std::getline(f,s)) if(!s.empty() && s[0]!='#') out.push_back(s);
  return out;
}

static void save_policy(const char* path, const Genome& genome, const Env& env) {
  std::ofstream out(path);
  if (!out) throw std::runtime_error(std::string("cannot write policy: ") + path);
  out << std::setprecision(9);
  out << "{\"n_logit\":" << env.n_logit() << ",\"head\":\"" << (env.per_action() ? "per_action" : "fixed") << "\",\"input_ids\":[";
  bool first = true;
  for (const auto& node : genome.nodes) if (node.type == N_IN) {
    if (!first) out << ",";
    out << node.id;
    first = false;
  }
  out << "],\"nodes\":[";
  first = true;
  for (const auto& node : genome.nodes) {
    if (!first) out << ",";
    out << "{\"id\":" << node.id << ",\"type\":" << (int)node.type << "}";
    first = false;
  }
  out << "],\"connections\":[";
  first = true;
  for (const auto& connection : genome.conns) {
    if (!std::isfinite(connection.w))
      throw std::runtime_error("cannot export a policy with non-finite weights");
    if (!first) out << ",";
    out << "{\"from\":" << connection.from << ",\"to\":" << connection.to
        << ",\"weight\":" << connection.w << ",\"enabled\":"
        << (connection.en ? "true" : "false") << "}";
    first = false;
  }
  out << "]}\n";
  if (!out) throw std::runtime_error("failed while writing policy");
}

int main(int argc,char**argv){
  if(argc==2 && std::string(argv[1])=="--primitive-contract"){
    std::printf("%s\n",kConceptPrimitiveContract);
    return 0;
  }
  try {
    if(argc<5){std::fprintf(stderr,"usage: %s <game> <concept-file> <budget> <seed> [pop] [model-output]\n       %s ext <concept-file> <budget> <seed> [pop] [model-output] [--target-win-rate F --eval-episodes N --eval-interval N]\n       %s grid <grid-spec-file> <concept-file> <budget> <seed> [pop] [model-output]\n",argv[0],argv[0],argv[0]);return 2;}
    std::string game=argv[1];
    int arg=2;
    std::unique_ptr<Env> base;
    if(game=="grid") {
      if(argc<6){std::fprintf(stderr,"grid requires a spec, concepts, budget, and seed\n");return 2;}
      base=std::make_unique<GridWorld>(GridWorld::from_file(argv[arg++]));
    } else if(game=="ext") {
      base=std::make_unique<ExternalEnv>(std::make_shared<ExtChannel>());
    } else if(game=="ttt"){auto e=std::make_unique<TicTacToe>();e->nf=0;base=std::move(e);}
    else if(game=="connect_four") base=std::make_unique<ConnectFour>();
    else if(game=="cartpole") base=std::make_unique<CartPole>();
    else if(game=="nim") base=std::make_unique<Nim>();
    else if(game=="lineworld") base=std::make_unique<LineWorld>();
    else if(game=="pointmass") base=std::make_unique<PointMass>();
    else {std::fprintf(stderr,"unknown game\n");return 2;}

    auto ex=load_concepts(argv[arg++]);
    long budget=std::atol(argv[arg++]);
    uint64_t seed=std::atoll(argv[arg++]);
    NeatCfg c=preset("lam"); c.eval_episodes=100; c.threads=(game=="ext")?1:2; c.seed=seed;
    c.pop=argc>arg?std::atoi(argv[arg++]):24;
    const char* model_path=argc>arg && argv[arg][0]!='-'?argv[arg++]:nullptr;
    double target_win_rate = -1.0;
    int eval_episodes = 100, eval_interval = 5;
    std::string preset_name, hidden_str; int fitness_eps = -1, iters_o = -1, rollout_o = -1, train_fit_o = -1, persist_o = -1; float carry_o = -1.f, lr_o = -1.f, ent_o = -1.f;   // --preset mlp => pop 1 fixed MLP + PPO (no evolution)
    std::string env_args; int stack_k = 1, robust_eps = 0;
    std::string head = "per_action";   // ext only: "fixed" => FixedHeadEnv (see fixed_head_env.h)
    StopLimits limits;   // loop control only: generation/time stops for ext training
    while (arg < argc) {
      std::string option = argv[arg++];
      if (arg >= argc) throw std::runtime_error("missing value for " + option);
      if (option == "--target-win-rate") target_win_rate = std::atof(argv[arg++]);
      else if (option == "--eval-episodes") eval_episodes = std::atoi(argv[arg++]);
      else if (option == "--eval-interval") eval_interval = std::atoi(argv[arg++]);
      else if (option == "--head") head = argv[arg++];
      else if (option == "--preset") preset_name = argv[arg++];
      else if (option == "--hidden") hidden_str = argv[arg++];
      else if (option == "--fitness-episodes") fitness_eps = std::atoi(argv[arg++]);
      else if (option == "--env-args") env_args = argv[arg++];
      else if (option == "--stack") stack_k = std::atoi(argv[arg++]);
      else if (option == "--robust") robust_eps = std::atoi(argv[arg++]);
      else if (option == "--iters") iters_o = std::atoi(argv[arg++]);
      else if (option == "--rollout") rollout_o = std::atoi(argv[arg++]);
      else if (option == "--train-fitness") train_fit_o = std::atoi(argv[arg++]);
      else if (option == "--persist-adam") persist_o = std::atoi(argv[arg++]);
      else if (option == "--carry") carry_o = (float)std::atof(argv[arg++]);
      else if (option == "--ent") ent_o = (float)std::atof(argv[arg++]);
      else if (option == "--lr") lr_o = (float)std::atof(argv[arg++]);
      else if (option == "--max-generations") limits.max_gens = std::atoi(argv[arg++]);
      else if (option == "--min-generations") limits.min_gens = std::atoi(argv[arg++]);
      else if (option == "--max-seconds") limits.max_seconds = std::atof(argv[arg++]);
      else throw std::runtime_error("unknown option: " + option);
    }
    if (!preset_name.empty()) { uint64_t sd = c.seed; int th = c.threads; c = preset(preset_name); c.seed = sd; c.threads = th; }
    if (fitness_eps >= 0) c.eval_episodes = fitness_eps;
    if (iters_o >= 0) c.ppo.iters = iters_o; if (rollout_o > 0) c.ppo.rollout = rollout_o; if (lr_o > 0) c.ppo.lr = lr_o; if (ent_o >= 0) c.ppo.ent = ent_o;
    if (train_fit_o >= 0) c.ppo.train_fitness = train_fit_o; if (persist_o >= 0) c.ppo.persist_adam = persist_o; if (carry_o >= 0) c.carry = carry_o;
    if (!hidden_str.empty()) { c.mlp_hidden.clear(); size_t i = 0; while (i < hidden_str.size()) { size_t j = hidden_str.find(',', i); if (j == std::string::npos) j = hidden_str.size(); if (j > i) c.mlp_hidden.push_back(std::atoi(hidden_str.substr(i, j - i).c_str())); i = j + 1; } }
    if (head != "per_action" && head != "fixed") throw std::runtime_error("--head must be per_action or fixed");
    if (head == "fixed" && game != "ext") throw std::runtime_error("--head fixed is only supported for ext environments");
    std::unique_ptr<Env> wrapped;
    PMCfg pmc;
    auto build_pm = [&](const PMCfg& pc) { std::unique_ptr<Env> b = std::make_unique<PointMass>(pc); return stack_k > 1 ? std::unique_ptr<Env>(new StackEnv(std::move(b), stack_k)) : std::move(b); };
    if (game == "pointmass") { pmc = parse_pm_args(env_args); wrapped = build_pm(pmc); }
    else if (head == "fixed") wrapped = std::make_unique<FixedHeadEnv>(std::move(base), ex);
    else wrapped = std::make_unique<ConceptEnv>(std::move(base), ex);
    Env& e = *wrapped; Population p(e,c);
    std::string csv=game+"_concept_seed"+std::to_string(c.seed)+".csv";
    if (target_win_rate >= 0.0) {
      if (game != "ext") throw std::runtime_error("win-rate stopping is only supported for ext environments");
      p.run_until_win_rate(target_win_rate, eval_episodes, eval_interval, budget, csv.c_str(), limits);
    } else {
      p.run(budget,csv.c_str());
    }
    const int heldout_episodes = target_win_rate >= 0.0 ? eval_episodes : 1000;
    std::printf("HELDOUT return %.6f episodes %d\n",
                p.evaluate_champion(heldout_episodes,c.seed^0xD1B54A32D192ED03ULL), heldout_episodes);
    if (game == "pointmass") {   // greedy, held-out success of the champion on the training env and a perturbation grid
      auto grid = [&](const PMCfg& pc, const char* tag) {
        Trainer ev(build_pm(pc), c.ppo); auto d = ev.evaluate_detail(p.champion, robust_eps, c.seed ^ 0xABCDEF12345ULL, true);
        std::printf("ROBUST %s noise %.3f delay %d mass %.2f fnoise %.2f success %.4f mean_return %.4f\n", tag, pc.noise, pc.delay, pc.mass, pc.fnoise, (double)d.wins / std::max(1, d.episodes), d.mean);
      };
      if (robust_eps > 0) {
        grid(pmc, "train");
        for (float n : {0.05f, 0.1f, 0.2f}) { PMCfg q = pmc; q.noise = n; grid(q, "noise"); }
        for (int dl : {1, 2, 4}) { PMCfg q = pmc; q.delay = dl; grid(q, "delay"); }
        for (float ms : {0.5f, 1.5f, 2.f}) { PMCfg q = pmc; q.mass = ms; grid(q, "mass"); }
        { PMCfg q = pmc; q.fnoise = 0.3f; grid(q, "fnoise"); }
      }
    }
    if(model_path) save_policy(model_path,p.champion,e);
  } catch (const std::exception& error) {
    std::fprintf(stderr,"brain error: %s\n",error.what());
    return 1;
  }
  return 0;
}
