#include "neat.h"
#include "fixed_head_env.h"
#include "robotics_env.h"
#include <cassert>
#include <cmath>
#include <memory>
#include <filesystem>
#include <fstream>
#include <sstream>

class FixedHeadTestEnv : public Env {
public:
  int value = 0;
  bool per_action() const override { return true; }
  int n_state() const override { return 1; }
  int n_action_feat() const override { return 0; }
  int n_actions() const override { return 3; }
  void reset(Rng&) override { value = 0; }
  int legal(int* out) override { out[0] = 0; out[1] = 1; out[2] = 2; return 3; }
  void state_feat(float* out) override { out[0] = (float)value; }
  float step(int action, bool& done, Rng&) override {
    value++;
    done = true;
    return action == 1 ? 1.f : -1.f;
  }
  std::unique_ptr<Env> make() const override { return std::make_unique<FixedHeadTestEnv>(); }
};

class ExpertLabelTestEnv : public Env {
public:
  bool per_action() const override { return false; }
  int n_state() const override { return 1; }
  int n_actions() const override { return 2; }
  int expert_action() const override { return 1; }
  void reset(Rng&) override {}
  int legal(int* out) override { out[0] = 0; out[1] = 1; return 2; }
  void state_feat(float* out) override { out[0] = 0.f; }
  float step(int, bool& done, Rng&) override { done = true; return 0.f; }
  std::unique_ptr<Env> make() const override { return std::make_unique<ExpertLabelTestEnv>(); }
};

int main() {
  {
    std::vector<std::string> concepts = {
        "state_mean", "action_norm", "transition_delta_mean"};
    FixedHeadEnv env(std::make_unique<FixedHeadTestEnv>(), concepts);
    assert(!env.per_action());
    assert(env.n_actions() == 3);
    assert(env.n_state() == 2);
    Rng rng(4);
    env.reset(rng);
    float features[2] = {};
    env.state_feat(features);
    assert(features[0] == 0.f && features[1] == 0.f);
    bool done = false;
    assert(env.step(1, done, rng) == 1.f && done);
  }

  {
    ExpertLabelTestEnv env;
    PPOCfg ppo;
    ppo.iters = 8;
    ppo.rollout = 32;
    ppo.epochs = 4;
    ppo.minibatch = 32;
    ppo.imitation = 1.f;
    Trainer trainer(env.make(), ppo);
    NetSpec spec;
    spec.n_in = env.n_in();
    spec.n_logit = env.n_logit();
    Innovations innovations;
    Rng rng(91);
    Genome genome = make_mlp(spec, {8}, innovations, rng);
    trainer.evaluate(genome, true, 1, 17, 19);
    Phenotype phenotype;
    phenotype.build(genome, spec.n_logit, 1);
    phenotype.A(0)[0] = 0.f;
    phenotype.A(phenotype.bias_slot)[0] = 1.f;
    phenotype.forward(1);
    assert(phenotype.A(phenotype.out_slot[1])[0] >
           phenotype.A(phenotype.out_slot[0])[0]);
  }

  {
    PMCfg config;
    config.horizon = 8;
    config.posonly = true;
    auto pointmass = std::make_unique<FixedHeadEnv>(
        std::make_unique<PointMass>(config), std::vector<std::string>{"state_mean"});
    StackEnv env(std::move(pointmass), 3);
    assert(env.act_dim() == 2);
    assert(env.n_state() == 3 * (2 + 1 + 1) + 2);

    NeatCfg learner = preset("mlp");
    learner.pop = 1;
    learner.threads = 1;
    learner.eval_episodes = 1;
    learner.ppo.rollout = 16;
    learner.ppo.epochs = 1;
    learner.ppo.minibatch = 8;
    learner.ppo.train_fitness = true;
    learner.ppo.persist_adam = true;
    const auto record_dir = std::filesystem::temp_directory_path() / "frameforge_genome_recorder_test";
    std::filesystem::remove_all(record_dir);
    {
      Population population(env, learner);
      population.enable_genome_recording(record_dir);
      population.step();
      assert(population.pop[0].logstd.size() == 2);
      assert(population.pop[0].ls_m.size() == 2);
      assert(population.pop[0].adam_t > 0);
      population.step();
      assert(population.pop[0].adam_t > 0);
      for (float value : population.pop[0].logstd) assert(std::isfinite(value));
    }  // recorder joins and flushes its pending snapshots here
    const auto snapshot_path = record_dir / "generation_0.json";
    assert(std::filesystem::is_regular_file(snapshot_path));
    std::ifstream snapshot_file(snapshot_path);
    std::ostringstream snapshot_text; snapshot_text << snapshot_file.rdbuf();
    assert(snapshot_text.str().find("\"population\"") != std::string::npos);
    assert(snapshot_text.str().find("\"connections\"") != std::string::npos);
    std::filesystem::remove_all(record_dir);
  }
  return 0;
}
