# Robotics proxy: continuous control, velocity from memory, multi-seed reliability

## What was added (all native C++, ~200k env steps/s, no Python relay)
- Continuous actions: `Env::act_dim()/step_cont()`, Gaussian policy head with a learned state-independent log-std (PPO update, Adam, persisted on the genome, Lamarckian write-back, crossover copies it).
- `src/robotics_env.h`: `PointMass` (2-D point mass with momentum, reach a random target and stop; force in [-1,1]^2), `StackEnv` (previous-frame deltas + previous action as inputs).
- `brain pointmass none.txt STEPS SEED POP ... --env-args "posonly=1,noise=..,delay=..,mass=.." --stack K --robust N`; `--robust` prints greedy held-out success/return on the training env and a perturbation grid.
- `robotics/run_reliability.py run|summary`: configs x seeds, median and worst case per perturbation.

## Two real bugs found on the way (both would have blocked any robotics result)
1. **Time limit without a clock.** Fixed-horizon episodes were treated as terminal with zero bootstrap while the observation had no time input, so the value function was ill-posed. PPO sat near -20 for millions of steps. Adding t/horizon to the observation fixed it (to about -1.9 within 300k steps; PD controller = -1.3). Anything with a hard time limit needs this, or proper truncation handling in the trainer (not implemented).
2. **Saturated initial action means.** With a normal output layer the means start large, the env clips them, and the policy gradient through the clip is zero. Seeds that started saturated never recovered. Mean-output weights are now scaled by 0.05 at init for continuous envs (`cont_out_gain`).

## Results (5 seeds each, 1.5M env steps, greedy eval on 100 held-out episodes per perturbation)
"Solved" = train-env mean return > -3.0 (PD controller -1.3, do-nothing -10). The sparse success flag (within 0.1 and slow at the end) is brittle: some seeds have near-identical returns but flip on the tolerance, so read both.

| config | seeds solved | train-env success median [worst] | return median [worst] |
|---|---|---|---|
| ppo_full (velocity observed, pop 1) | 4/5 | 0.00 [0.00] | -2.17 [-6.17] |
| neat_full (velocity observed, pop 8) | 5/5 | 1.00 [0.00] | -1.88 [-2.05] |
| ppo_pos (velocity hidden, no memory) | 0/5 | 0.00 [0.00] | -7.06 [-8.11] |
| ppo_pos_s3 (hidden, 3-frame deltas + prev action) | 4/5 | 1.00 [0.00] | -1.96 [-6.01] |
| neat_pos_s3 | 5/5 | 1.00 [1.00] | -1.58 [-1.61] |

At 4M steps (full observation only): NEAT solves 5/5 with train success 1.00 [1.00]; PPO solves 3/5 with success 1.00 [0.00].

Perturbation tables (obs noise 0.05/0.1/0.2, delay 1/2/4, mass x0.5/1.5/2, force noise 0.3) are produced by `python3 robotics/run_reliability.py summary --steps 1500000`. Headlines:
- Mass changes and delay <= 2 barely hurt the surviving policies.
- **Observation noise is the weak point:** success falls to about 0.7-0.8 at 0.1 and 0.15 at 0.2 for NEAT-full; the delta-stack policies fall much faster (neat_pos_s3 return -3.9 at noise 0.05) because differencing noisy positions amplifies noise.
- **Delay 4 breaks everything** (success about 0-0.2; return -4 to -6).
- No policy was trained under perturbations. These numbers measure robustness of nominal training, not of a hardened controller.

## Caveats
- n = 5 seeds. NEAT 5/5 vs PPO 4/5 (3/5 at 4M) is suggestive, not significant. Configs are not hyperparameter-matched (NEAT: pop 8 x 16 iters, carry 0.5; PPO: pop 1 x 8 iters); total env steps are equal.
- One task, one dynamics model, 2-D. Not a robot.
- The 3-frame stack uses gain 10 on deltas, chosen by reasoning, not tuned.
- Untested: truncation-aware bootstrapping, reward/observation normalization, domain randomization during training, higher-dimensional tasks, evolving log-std/architecture.
