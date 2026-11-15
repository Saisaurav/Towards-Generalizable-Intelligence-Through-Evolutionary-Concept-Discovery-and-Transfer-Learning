#include "neat.h"
#include "envs.h"
#include "concept_env.h"
#include "external_env.h"
#include <fstream>
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
  out << "{\"n_logit\":" << env.n_logit() << ",\"input_ids\":[";
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
    if(argc<5){std::fprintf(stderr,"usage: %s <game> <concept-file> <budget> <seed> [pop] [model-output]\n       %s ext <concept-file> <budget> <seed> [pop] [model-output]   (generated env via trainer relay on stdin/stdout)\n       %s grid <grid-spec-file> <concept-file> <budget> <seed> [pop] [model-output]\n",argv[0],argv[0],argv[0]);return 2;}
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
    else {std::fprintf(stderr,"unknown game\n");return 2;}

    auto ex=load_concepts(argv[arg++]);
    long budget=std::atol(argv[arg++]);
    uint64_t seed=std::atoll(argv[arg++]);
    NeatCfg c=preset("lam"); c.eval_episodes=100; c.threads=(game=="ext")?1:2; c.seed=seed;
    c.pop=argc>arg?std::atoi(argv[arg++]):24;
    const char* model_path=argc>arg && argv[arg][0]!='-'?argv[arg++]:nullptr;
    double target_win_rate = -1.0;
    int eval_episodes = 100, eval_interval = 5;
    while (arg < argc) {
      std::string option = argv[arg++];
      if (arg >= argc) throw std::runtime_error("missing value for " + option);
      if (option == "--target-win-rate") target_win_rate = std::atof(argv[arg++]);
      else if (option == "--eval-episodes") eval_episodes = std::atoi(argv[arg++]);
      else if (option == "--eval-interval") eval_interval = std::atoi(argv[arg++]);
      else throw std::runtime_error("unknown option: " + option);
    }
    ConceptEnv e(std::move(base),ex); Population p(e,c);
    std::string csv=game+"_concept_seed"+std::to_string(c.seed)+".csv";
    if (target_win_rate >= 0.0) {
      if (game != "ext") throw std::runtime_error("win-rate stopping is only supported for ext environments");
      p.run_until_win_rate(target_win_rate, eval_episodes, eval_interval, budget, csv.c_str());
    } else {
      p.run(budget,csv.c_str());
    }
    std::printf("HELDOUT return %.6f episodes 1000\n",p.evaluate_champion(1000,c.seed^0xD1B54A32D192ED03ULL));
    if(model_path) save_policy(model_path,p.champion,e);
  } catch (const std::exception& error) {
    std::fprintf(stderr,"brain error: %s\n",error.what());
    return 1;
  }
  return 0;
}
