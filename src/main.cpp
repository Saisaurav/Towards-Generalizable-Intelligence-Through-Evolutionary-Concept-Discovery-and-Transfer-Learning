#include "neat.h"
#include "envs.h"
#include "concept_env.h"
#include "external_env.h"
#include "fixed_head_env.h"
#include "robotics_env.h"
#include "robot_envs.h"
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

static std::string turtlebot_obs_names(int n) {
  std::string s = "[";
  for (int i = 0; i < TurtleBotNav::NR; ++i) s += std::string(i ? "," : "") + "\"lidar" + std::to_string(i) + "\"";
  for (auto k : {"goal_dist", "goal_sin", "goal_cos", "v", "w", "prev_a0", "prev_a1"}) s += std::string(",\"") + k + "\"";
  if (n > TurtleBotNav::NR + 7) s += ",\"t_frac\"";
  return s + "]";
}
static std::string arm_obs_names(const std::string& args) {
  auto kv = parse_kv(args); int feat = (int)kvget(kv, "feat", 0); bool tf = kvget(kv, "tfeat", 1) != 0;
  std::string s = "["; bool first = true;
  for (auto k : {"sin_q1", "cos_q1", "sin_q2", "cos_q2", "dq1", "dq2", "err_x", "err_y", "prev_a0", "prev_a1"}) {
    s += std::string(first ? "" : ",") + "\"" + k + "\""; first = false;
  }
  if (feat >= 1) s += ",\"err_x_fine\",\"err_y_fine\"";
  if (feat >= 2) s += ",\"jt_e1\",\"jt_e2\"";
  if (tf) s += ",\"t_frac\"";
  return s + "]";
}

static void save_policy(const char* path, const Genome& genome, const Env& env, bool action_onehot,
                        const std::string& env_name = "", const std::string& env_args = "") {
  std::ofstream out(path);
  if (!out) throw std::runtime_error(std::string("cannot write policy: ") + path);
  out << std::setprecision(9);
  const char* head = env.act_dim() > 0 ? "continuous" : (env.per_action() ? "per_action" : "fixed");
  out << "{\"n_logit\":" << env.n_logit() << ",\"head\":\"" << head
      << "\",\"action_onehot\":" << (action_onehot ? "true" : "false") << ",\"input_ids\":[";
  bool first = true;
  for (const auto& node : genome.nodes) if (node.type == N_IN) {
    if (!first) out << ",";
    out << node.id;
    first = false;
  }
  out << "]";
  if (env.act_dim() > 0) {
    out << ",\"act_dim\":" << env.act_dim() << ",\"action_dim\":" << env.act_dim() << ",\"logstd\":[";
    for (int i = 0; i < env.act_dim(); ++i) {
      if (!std::isfinite(genome.logstd.size() == (size_t)env.act_dim() ? genome.logstd[i] : -0.7f)) throw std::runtime_error("non-finite logstd");
      out << (i ? "," : "") << (genome.logstd.size() == (size_t)env.act_dim() ? genome.logstd[i] : -0.7f);
    }
    out << "],\"deterministic_action\":\"mean_output_clipped_to_unit\",\"env\":\"" << env_name
        << "\",\"env_args\":\"" << env_args << "\"";
    if (env_name == "turtlebot") out << ",\"obs_names\":" << turtlebot_obs_names(env.n_state())
        << ",\"action_scale\":{\"linear_x_mps\":0.22,\"angular_z_radps\":2.0},\"rate_hz\":10";
    if (env_name == "arm2link") out << ",\"obs_names\":" << arm_obs_names(env_args)
        << ",\"action_scale\":{\"joint_vel_radps\":2.0},\"rate_hz\":20";
  }
  out << ",\"nodes\":["; first = true;
  for (const auto& node : genome.nodes) {
    if (!first) out << ",";
    out << "{\"id\":" << node.id << ",\"type\":" << (int)node.type << "}"; first = false;
  }
  out << "],\"connections\":["; first = true;
  for (const auto& connection : genome.conns) {
    if (!std::isfinite(connection.w)) throw std::runtime_error("cannot export policy with non-finite weights");
    if (!first) out << ",";
    out << "{\"from\":" << connection.from << ",\"to\":" << connection.to
        << ",\"weight\":" << connection.w << ",\"enabled\":" << (connection.en ? "true" : "false") << "}"; first = false;
  }
  out << "]}\n";
  if (!out) throw std::runtime_error("failed while writing policy");
}

int main(int argc,char**argv){
  // Keep generation telemetry streaming when the browser launches the trainer through a pipe.
  std::setvbuf(stdout, nullptr, _IOLBF, 0);
  if(argc==2 && std::string(argv[1])=="--primitive-contract"){
    std::printf("%s\n",kConceptPrimitiveContract);
    return 0;
  }
  try {
    if(argc<5){std::fprintf(stderr,"usage: %s <game> <concept-file> <budget> <seed> [pop] [model-output] [options]\n       %s ext <concept-file> <budget> <seed> [pop] [model-output] [--head fixed|per_action --max-episode-steps N --target-win-rate F --target-stop-generation N --max-generations N --max-seconds S --genome-dir DIR]\n       %s pointmass <concept-file> <budget> <seed> [pop] [model-output] [--env-args key=value,... --stack N | --filt MASK --filt-k K --robust N]\n       %s <turtlebot|arm2link> <concept-file> <budget> <seed> [pop] [model-output] [--preset lam --env-args key=value,... --genome-dir DIR]\n       %s grid <grid-spec-file> <concept-file> <budget> <seed> [pop] [model-output]\n",argv[0],argv[0],argv[0],argv[0],argv[0]);return 2;}
    std::string game=argv[1];
    int arg=2;
    std::unique_ptr<Env> base;
    if(game=="grid") {
      if(argc<6){std::fprintf(stderr,"grid requires a spec, concepts, budget, and seed\n");return 2;}
      base=std::make_unique<GridWorld>(GridWorld::from_file(argv[arg++]));
    } else if(game=="ext") {
      base=std::make_unique<ExternalEnv>(std::make_shared<ExtChannel>());
    } else if(game=="pointmass" || game=="turtlebot" || game=="arm2link") {
      // Configured after command-line options are parsed.
    } else if(game=="ttt"){auto e=std::make_unique<TicTacToe>();e->nf=0;base=std::move(e);}
    else if(game=="connect_four") base=std::make_unique<ConnectFour>();
    else if(game=="cartpole") base=std::make_unique<CartPole>();
    else if(game=="nim") base=std::make_unique<Nim>();
    else if(game=="lineworld") base=std::make_unique<LineWorld>();
    else {std::fprintf(stderr,"unknown game\n");return 2;}

    auto ex=load_concepts(argv[arg++]);
    long budget=std::atol(argv[arg++]);
    uint64_t seed=std::atoll(argv[arg++]);
    NeatCfg c=preset("lam"); c.eval_episodes=100; c.threads=(game=="ext")?1:2; c.seed=seed;
    if (game == "ext") {
      c.eval_episodes = 10;
      c.mlp_hidden = {64};
      c.ppo.iters = 4;
      c.ppo.greedy_eval = true;
      c.ppo.imitation = 1.f;
    }
    const bool pop_arg_set = argc > arg && argv[arg][0] != '-';
    const int requested_pop = pop_arg_set ? std::atoi(argv[arg++]) : 24;
    c.pop = requested_pop;
    const char* model_path=argc>arg && argv[arg][0]!='-'?argv[arg++]:nullptr;
    double target_win_rate = -1.0;
    int eval_episodes = 100, eval_interval = 5, fitness_eps = -1, stack_k = 1, robust_eps = 0;
    int max_episode_steps = 0;
    int iters = -1, rollout = -1, train_fitness = -1, persist_adam = -1, filt_mask = 0;
    float carry = -1.f, ent = -1.f, lr = -1.f, imitation = -1.f, filt_k = 0.2f;
    std::string head = "per_action", preset_name, hidden, env_args;
    StopLimits limits;
    std::string genome_dir;
    while (arg < argc) {
      std::string option = argv[arg++];
      if (arg >= argc) throw std::runtime_error("missing value for " + option);
      if (option == "--target-win-rate") target_win_rate = std::atof(argv[arg++]);
      else if (option == "--eval-episodes") eval_episodes = std::atoi(argv[arg++]);
      else if (option == "--eval-interval") eval_interval = std::atoi(argv[arg++]);
      else if (option == "--head") head = argv[arg++];
      else if (option == "--preset") preset_name = argv[arg++];
      else if (option == "--hidden") hidden = argv[arg++];
      else if (option == "--fitness-episodes") fitness_eps = std::atoi(argv[arg++]);
      else if (option == "--env-args") env_args = argv[arg++];
      else if (option == "--filt") filt_mask = std::atoi(argv[arg++]);
      else if (option == "--filt-k") filt_k = (float)std::atof(argv[arg++]);
      else if (option == "--stack") stack_k = std::atoi(argv[arg++]);
      else if (option == "--robust") robust_eps = std::atoi(argv[arg++]);
      else if (option == "--iters") iters = std::atoi(argv[arg++]);
      else if (option == "--rollout") rollout = std::atoi(argv[arg++]);
      else if (option == "--train-fitness") train_fitness = std::atoi(argv[arg++]);
      else if (option == "--persist-adam") persist_adam = std::atoi(argv[arg++]);
      else if (option == "--carry") carry = (float)std::atof(argv[arg++]);
      else if (option == "--ent") ent = (float)std::atof(argv[arg++]);
      else if (option == "--lr") lr = (float)std::atof(argv[arg++]);
      else if (option == "--imitation") imitation = (float)std::atof(argv[arg++]);
      else if (option == "--max-generations") limits.max_gens = std::atoi(argv[arg++]);
      else if (option == "--min-generations") limits.min_gens = std::atoi(argv[arg++]);
      else if (option == "--target-stop-generation") limits.target_stop_generation = std::atoi(argv[arg++]);
      else if (option == "--max-seconds") limits.max_seconds = std::atof(argv[arg++]);
      else if (option == "--max-episode-steps") max_episode_steps = std::atoi(argv[arg++]);
      else if (option == "--genome-dir") genome_dir = argv[arg++];
      else throw std::runtime_error("unknown option: " + option);
    }
    if (!preset_name.empty()) {
      uint64_t configured_seed = c.seed;
      int configured_threads = c.threads;
      c = preset(preset_name);
      c.seed = configured_seed;
      c.threads = configured_threads;
      // An explicit positional population size must survive preset selection.
      // Without one, keep the preset's own population (e.g. the MLP preset uses pop=1).
      if (pop_arg_set) c.pop = requested_pop;
    }
    if (fitness_eps >= 0) c.eval_episodes = fitness_eps;
    if (iters >= 0) c.ppo.iters = iters;
    if (rollout > 0) c.ppo.rollout = rollout;
    if (train_fitness >= 0) c.ppo.train_fitness = train_fitness != 0;
    if (persist_adam >= 0) c.ppo.persist_adam = persist_adam != 0;
    if (carry >= 0.f) c.carry = carry;
    if (ent >= 0.f) c.ppo.ent = ent;
    if (lr > 0.f) c.ppo.lr = lr;
    if (imitation >= 0.f) c.ppo.imitation = imitation;
    if (!hidden.empty()) {
      c.mlp_hidden.clear();
      size_t start = 0;
      while (start < hidden.size()) {
        size_t end = hidden.find(',', start);
        if (end == std::string::npos) end = hidden.size();
        int width = std::atoi(hidden.substr(start, end - start).c_str());
        if (width < 1 || width > 4096) throw std::runtime_error("--hidden widths must be in 1..4096");
        c.mlp_hidden.push_back(width);
        start = end + 1;
      }
    }
    if (head != "per_action" && head != "fixed")
      throw std::runtime_error("--head must be per_action or fixed");
    if (head == "fixed" && game != "ext")
      throw std::runtime_error("--head fixed is only supported for ext environments");
    if (stack_k < 1 || stack_k > 32) throw std::runtime_error("--stack must be in 1..32");
    if (filt_mask < 0 || filt_mask > 3 || !std::isfinite(filt_k) || filt_k <= 0.f)
      throw std::runtime_error("--filt must be in 0..3 and --filt-k must be positive");
    if ((filt_mask > 0 || stack_k > 1) && game != "pointmass")
      throw std::runtime_error("--filt and --stack are only supported for pointmass");
    if (robust_eps < 0 || max_episode_steps < 0 || limits.max_gens < 0 || limits.min_gens < 0 ||
        limits.target_stop_generation < 0 ||
        !std::isfinite(limits.max_seconds) || limits.max_seconds < 0.0)
      throw std::runtime_error("invalid robustness or stopping limit");
    if (eval_episodes < 1 || eval_interval < 1)
      throw std::runtime_error("evaluation episode count and interval must be positive");
    if (c.pop < 1 || c.pop > 4096 || c.ppo.iters < 0 ||
        c.ppo.rollout < 1 || c.ppo.epochs < 1 || c.ppo.minibatch < 1 ||
        !std::isfinite(c.carry) || c.carry < 0.f || c.carry > 1.f ||
        !std::isfinite(c.ppo.ent) || c.ppo.ent < 0.f ||
        !std::isfinite(c.ppo.imitation) || c.ppo.imitation < 0.f ||
        !std::isfinite(c.ppo.lr) || c.ppo.lr <= 0.f)
      throw std::runtime_error("invalid learner configuration");
    if (game == "pointmass" && budget <= 0)
      throw std::runtime_error("pointmass training budget must be positive");
    if (max_episode_steps > 0) {
      if (game != "ext") throw std::runtime_error("--max-episode-steps is only supported for ext environments");
      static_cast<ExternalEnv*>(base.get())->set_episode_step_limit(max_episode_steps);
    }
    const bool action_onehot = game == "ext" && head == "per_action";
    PMCfg pm_cfg;
    if (game == "pointmass") pm_cfg = parse_pm_args(env_args);
    RobotCfg robot_cfg;
    const bool robot = game == "turtlebot" || game == "arm2link";
    if (robot) robot_cfg = RobotCfg::parse(env_args, game == "turtlebot" ? 300 : 100);
    auto build_robot = [&](const RobotCfg& cfg) -> std::unique_ptr<Env> {
      if (game == "turtlebot") return std::make_unique<TurtleBotNav>(cfg);
      return std::make_unique<Arm2Link>(cfg);
    };
    auto build_pm = [&](const PMCfg& cfg) -> std::unique_ptr<Env> {
      std::unique_ptr<Env> inner = std::make_unique<PointMass>(cfg);
      if (filt_mask > 0) return std::make_unique<FilterEnv>(std::move(inner), filt_mask, 10.f, 0.99f, filt_k);
      if (stack_k > 1) return std::make_unique<StackEnv>(std::move(inner), stack_k);
      return inner;
    };
    std::unique_ptr<Env> wrapped;
    if (robot) {
      wrapped = build_robot(robot_cfg);
    } else if (game == "pointmass") {
      wrapped = std::make_unique<FixedHeadEnv>(build_pm(pm_cfg), ex);
      if (train_fitness < 0) c.ppo.train_fitness = true;
      if (persist_adam < 0) c.ppo.persist_adam = true;
      c.eval_episodes = 1;
    } else if (game == "ext" && head == "fixed") {
      wrapped = std::make_unique<FixedHeadEnv>(std::move(base), ex);
      // Real-time (fixed-head) defaults: the configuration measured on the Snake fixture in baseline/results
      // (pop 8, 16 hidden units, 16 PPO iterations, fitness from PPO rollouts, persisted Adam moments, carry 0.5,
      // no entropy bonus). Applied only for options the caller did not set; --preset opts out entirely.
      // Imitation of the env's scripted_action stays OFF here: the learner must learn from the reward.
      if (preset_name.empty()) {
        if (hidden.empty()) c.mlp_hidden = {16};
        if (iters < 0) c.ppo.iters = 16;
        if (train_fitness < 0) c.ppo.train_fitness = true;
        if (carry < 0.f) c.carry = 0.5f;
        if (ent < 0.f) c.ppo.ent = 0.f;
        if (imitation < 0.f) c.ppo.imitation = 0.f;
        c.ppo.greedy_eval = false;
      }
      if (persist_adam < 0) c.ppo.persist_adam = true;
    } else {
      wrapped = std::make_unique<ConceptEnv>(std::move(base), ex, action_onehot);
    }
    Env& e = *wrapped;
    Population p(e,c);
    if (!genome_dir.empty()) p.enable_genome_recording(genome_dir);
    std::string csv=game+"_concept_seed"+std::to_string(c.seed)+".csv";
    if (target_win_rate >= 0.0) {
      if (game != "ext") throw std::runtime_error("win-rate stopping is only supported for ext environments");
      p.run_until_win_rate(target_win_rate, eval_episodes, eval_interval, budget, csv.c_str(), limits);
    } else {
      p.run(budget,csv.c_str(),limits);
    }
    const int heldout_episodes = robot ? 300 : (target_win_rate >= 0.0 || game == "ext" ? eval_episodes : 1000);
    if (robot) {
      Trainer heldout(build_robot(robot_cfg), c.ppo);
      auto d = heldout.evaluate_detail(p.champion, heldout_episodes, c.seed ^ 0xD1B54A32D192ED03ULL, true);
      std::printf("HELDOUT success %.4f mean_return %.4f episodes %d\n", (double)d.wins / std::max(1, d.episodes), d.mean, heldout_episodes);
    } else {
      std::printf("HELDOUT return %.6f episodes %d\n",
                  p.evaluate_champion(heldout_episodes,c.seed^0xD1B54A32D192ED03ULL), heldout_episodes);
    }
    if (game == "pointmass" && robust_eps > 0) {
      auto evaluate_robustness = [&](const PMCfg& config, const char* label) {
        std::unique_ptr<Env> env = std::make_unique<FixedHeadEnv>(build_pm(config), ex);
        Trainer evaluator(std::move(env), c.ppo);
        Trainer::EvalDetail detail = evaluator.evaluate_detail(
            p.champion, robust_eps, c.seed ^ 0xABCDEF12345ULL, true);
        std::printf("ROBUST %s noise %.3f delay %d mass %.2f fnoise %.2f success %.4f mean_return %.4f\n",
                    label, config.noise, config.delay, config.mass, config.fnoise,
                    (double)detail.wins / std::max(1, detail.episodes), detail.mean);
      };
      evaluate_robustness(pm_cfg, "train");
      for (float noise : {0.05f, 0.1f, 0.2f}) {
        PMCfg config = pm_cfg; config.noise = noise; evaluate_robustness(config, "noise");
      }
      for (int delay : {1, 2, 4}) {
        PMCfg config = pm_cfg; config.delay = delay; evaluate_robustness(config, "delay");
      }
      for (float mass : {0.5f, 1.5f, 2.f}) {
        PMCfg config = pm_cfg; config.mass = mass; evaluate_robustness(config, "mass");
      }
      PMCfg noisy_force = pm_cfg; noisy_force.fnoise = 0.3f;
      evaluate_robustness(noisy_force, "fnoise");
    }
    if (robot && robust_eps > 0) {
      auto rgrid = [&](RobotCfg cfg, const char* tag) {
        cfg.rnoise = 0.f; cfg.rdelay = 0; cfg.rlag = 1.f;
        Trainer ev(build_robot(cfg), c.ppo);
        auto d = ev.evaluate_detail(p.champion, robust_eps, c.seed ^ 0xABCDEF12345ULL, true);
        std::printf("ROBUST %s noise %.3f delay %d lag %.2f success %.4f mean_return %.4f\n", tag,
                    cfg.noise, cfg.delay, cfg.lag, (double)d.wins / std::max(1, d.episodes), d.mean);
      };
      rgrid(robot_cfg, "train");
      for (float n : {0.02f, 0.05f, 0.1f}) { RobotCfg q = robot_cfg; q.noise = n; rgrid(q, "noise"); }
      for (int dl : {1, 2, 3}) { RobotCfg q = robot_cfg; q.delay = dl; rgrid(q, "delay"); }
      for (float lg : {0.5f, 2.f}) { RobotCfg q = robot_cfg; q.lag = lg; rgrid(q, "lag"); }
    }
    if(model_path) save_policy(model_path,p.champion,e,action_onehot,robot ? game : "",env_args);
  } catch (const std::exception& error) {
    std::fprintf(stderr,"brain error: %s\n",error.what());
    return 1;
  }
  return 0;
}
