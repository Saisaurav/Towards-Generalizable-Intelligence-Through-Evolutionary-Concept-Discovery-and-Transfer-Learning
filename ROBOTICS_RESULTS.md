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

---
# Addendum: domain randomization (4M env steps, 5 seeds)
Training env samples per episode: obs noise ~U(0,0.15), delay ~U{0..3}, mass log-uniform in [1/2,2], force noise ~U(0,0.3).
Evaluation points are fixed (never randomized), so noise 0.2 and delay 4 are *outside* the training range. Configs `*_dr` in the harness; `python3 robotics/run_reliability.py summary --steps 4000000` prints all tables.

Worst seed on any perturbation point (mean return, higher is better; PD = -1.3, do-nothing = -10):

| config | nominal (median) | worst seed, worst perturbation |
|---|---|---|
| neat_full | -1.71 | -8.23 (delay 4) |
| neat_full_dr | -2.32 | -3.30 (noise 0.2) |
| ppo_full | -1.66 | -8.88 (delay 4) |
| ppo_full_dr | -2.83 | -4.78 (delay 4) |

- Randomization removes the catastrophic cases for full-observation policies: delay-4 return goes from -5.5/-6.3 (NEAT/PPO median) to -2.7/-3.4, and NEAT success at delay 4 from 0.00 to 0.93. Noise 0.2 return improves from -3.8/-4.6 to -3.2/-3.6.
- The price is nominal precision: about 0.6 to 1.2 return worse on the unperturbed env. The sparse success flag (within 0.1 and slow) collapses for the hedged policies (e.g. ppo_full_dr train success 0.00 despite return -2.8), so use returns for these.
- NEAT solves 5/5 (return > -3) with randomization, PPO 3/5; the same split as without. n = 5, so only suggestive.
- **Randomization hurts the velocity-hidden, frame-delta policies badly** (neat_pos_s3_dr 1/5 solved, ppo_pos_s3_dr 0/5; without randomization 5/5 and 3/5). Differencing positions with gain 10 turns noise up to 0.15 into delta noise of about 1.5, larger than the signal. Velocity-from-history needs filtering (longer stack, smoothed deltas, or a recurrent state), not more training. Not tried.
- Not done: separate randomization ranges per channel, curriculum, matched hyperparameters, more than 5 seeds.

---
# Addendum 2: velocity from history under noise (velocity hidden, with domain randomization, 4M steps, 5 seeds)
`FilterEnv` (src/robotics_env.h, `--filt MASK`, `--filt-k K`) replaces the raw frame deltas. bit0 = two EMAs (alpha 0.5 / 0.1) of the observation deltas; bit1 = leaky integral of the *commanded actions* (u_t = 0.99 u_{t-1} + k a_t, an efference copy that never touches the noisy observation). `k` = 0.2 is the unscaled version, `*k` configs use `--filt-k 0.02` so the input stays in about [-2,2].

Mean return on the unperturbed env (median [worst seed]), then perturbation points; solved = return > -3:

| config | train | obs noise 0.1 | obs noise 0.2 | delay 4 | mass x2 | solved |
|---|---|---|---|---|---|---|
| neat_full_dr (velocity *observed*, reference) | -2.32 [-2.55] | -2.58 | -3.20 | -2.72 | -2.79 | 5/5 |
| neat raw 3-frame deltas | -3.47 [-4.19] | -5.82 | -8.14 | -7.27 | -4.21 | 1/5 |
| neat EMA deltas | -3.01 [-3.31] | -3.66 | -4.88 | -4.79 | -3.80 | 2/5 |
| neat action integral (k=0.2) | -2.52 [-3.01] | -2.78 | -3.52 | -2.92 | -3.65 | 4/5 |
| **neat action integral (k=0.02)** | **-2.15 [-2.41]** | -2.54 | -3.35 | -3.14 | -2.85 | **5/5** |
| neat both (k=0.02) | -2.43 [-2.53] | -3.19 | -4.51 | -3.68 | -3.00 | 5/5 |
| ppo action integral (k=0.02) | -3.43 [-24.22] | -3.60 | -4.03 | -3.84 | -4.07 | 2/5 |
| ppo both (k=0.02) | -25.61 [-26.60] | -25.34 | -24.65 | -24.97 | -23.76 | 0/5 |
| ppo raw 3-frame deltas | -7.09 [-8.52] | -7.44 | -10.13 | -9.20 | -7.22 | 0/5 |

- **With the action integral, NEAT with velocity hidden matches NEAT with velocity observed** under randomization, including at noise 0.2 and delay 4, which are outside the training ranges. Filtering the observation deltas helps only partly; the integral of past commands is what carries the signal.
- **Scale matters.** The unscaled integral reaches +-20 (it tracks velocity, top speed 4/0.2 = 20) and PPO collapsed to a saturated policy on all 10 seeds (return about -23 to -28; lowering lr to 5e-4 did not help). Scaling to +-2 fixed NEAT's residual failures and let PPO learn on some seeds. The input-scale explanation is my inference from this pattern, not isolated by a dedicated ablation.
- **PPO is unstable with these inputs even after scaling** (ppo action integral: 2/5 seeds learn, the other 3 sit at about -24; with both filters 0/5). I did not diagnose why. NEAT's selection discards the collapsing genomes; PPO has no such guard.
- Caveats: n = 5. The leak 0.99 was chosen from the plant's known damping (drag 0.2 x dt 0.05), so this uses a model assumption; it still held at mass x2 and delay 4, but it is open-loop and would drift where the true velocity is changed by contact (here only the arena walls). A learned recurrent state would not need that assumption and is untested.
