# Extended v9 Evaluation (14 November 2026)

## Scope

This is an extended evaluation of the supplied v9 archive, without modifying its learner or concept generator during the runs. The C++ learner was compiled with the repository Makefile and `make test-envs` passed. The focused GP/novelty tests passed (24 tests). The optional LLM platform test module could not be imported because the `openai` Python package is not installed; this does not affect the core game/GP experiments.

## Standalone game-playing baseline

Fresh learners received no transferred concept expressions (empty concept file; runtime action feature behavior unchanged). Each run used 5,000 nominal environment steps, population 8, and 1,000 held-out evaluation episodes. Three seeds (11, 22, 33) were run per game.

| Game | Seed 11 | Seed 22 | Seed 33 |
|---|---:|---:|---:|
| Tic-Tac-Toe | -0.028 | -0.007 | 0.045 |
| CartPole | 22.731 | 22.247 | 22.365 |
| Connect Four | -0.039 | 0.116 | 0.079 |
| Nim | 0.002 | 0.046 | 0.022 |
| LineWorld | 0.735 | -0.148 | -0.128 |

These values use different reward scales and cannot be compared directly between games. Three seeds are only an initial competence/stability check. The results show reliable CartPole performance at this budget, mixed LineWorld outcomes, and weak/near-zero returns in the game environments. They do not establish strong strategic play.

## Concept discovery and transfer screen

Five cross-game discovery runs were attempted using v9, population 24, 8 generations, shortlist 3, and 5 paired downstream seeds at 5,000 steps/population 8:

| Source → target | Teacher | Result |
|---|---|---|
| Tic-Tac-Toe → CartPole | Oracle | No downstream-admitted candidate |
| CartPole → Tic-Tac-Toe | Rollout | Two candidates passed the initial 5-seed admission heuristic |
| Connect Four → Nim | Rollout | Two candidates passed the initial 5-seed admission heuristic |
| Nim → LineWorld | Oracle | No downstream-admitted candidate |
| LineWorld → Connect Four | Oracle | Run did not complete before the experiment time limit; no result claimed |

The two pairs with initial passes were then re-evaluated more extensively: 20 paired learner seeds, same target baseline per seed, 5,000-step budget, population 8.

### CartPole → Tic-Tac-Toe

Baseline target return mean across 20 seeds: 0.03295 (SD 0.04273).

| Candidate | Mean paired change | Median | Improved seeds | Approx. 95% CI for mean paired change |
|---|---:|---:|---:|---:|
| `div(add(legal_norm,state_sign_change_fraction),div(action_norm,state_median))` | +0.00615 | 0.000 | 8/20 (40%) | [-0.0225, +0.0348] |
| `sub(div(add(legal_norm,state_sign_change_fraction),div(action_norm,state_median)),square(state_positive_fraction))` | -0.00105 | -0.004 | 7/20 (35%) | [-0.0269, +0.0248] |

Neither candidate replicated the initial five-seed admission. Both fail the 60% win gate and their paired-change intervals include zero.

### Connect Four → Nim

Baseline target return mean across 20 seeds: 0.04890 (SD 0.07630).

| Candidate | Mean paired change | Median | Improved seeds | Approx. 95% CI for mean paired change |
|---|---:|---:|---:|---:|
| `mul(add(log1p(state_positive_fraction),state_abs_mean),add(div(state_positive_fraction,action_norm),state_sign_change_fraction))` | -0.00430 | +0.006 | 10/20 (50%) | [-0.0561, +0.0475] |
| `mul(add(log1p(log1p(state_positive_fraction)),min(state_max_abs,state_mean)),action_norm)` | +0.00490 | -0.002 | 9/20 (45%) | [-0.0293, +0.0391] |

Neither candidate replicated the initial five-seed admission. Both fail the 60% win gate and their paired-change intervals include zero.

Intervals above use a t critical value approximation (2.093, df=19); they are descriptive, not a substitute for a preregistered inferential analysis or multiple-comparison correction.

## Interpretation

1. The v9 pipeline runs end-to-end for the completed pairs: discovery, compatibility filtering, fresh target training, and paired-seed evaluation.
2. The initial five-seed admission gate was too optimistic for these candidates. Extending to 20 seeds overturned all four apparent passes. This is an important negative result and validates the need for larger paired evaluations.
3. The experiment does not demonstrate reliable cross-game concept transfer in these pairs. It also does not disprove transfer in general.
4. Standalone performance is not yet uniformly convincing. In particular, Tic-Tac-Toe, Connect Four, and Nim returns are near zero under these short budgets and random-opponent/task definitions; LineWorld is unstable across seeds. CartPole is the clearest functioning baseline.
5. The 5,000-step budget is nominal: the learner completes whole generations and can overshoot it substantially. The learner's printed generation step counts show this. Therefore compare runs with identical configuration and record actual steps in future experiments.
6. One GP discovery seed per source-target pair and small GP populations/generation counts mean this is a screening experiment, not a definitive ranking of the v9 architecture.

## Recommended next experiment

Before adding more primitives, fix the experimental protocol and improve standalone game competence:
- Evaluate each environment against stronger, explicit baselines (random, heuristic, and exact oracle where available), not just absolute reward.
- Use at least 20 independent learner seeds for any candidate that passes screening; ideally 30+ and multiple GP discovery seeds.
- Keep training budgets matched by actual environment steps, not nominal generation stopping thresholds.
- For Tic-Tac-Toe, Nim, and LineWorld, compare the learned policy against exact optimal play / shortest-path policies. For Connect Four, use a fixed, documented opponent ladder rather than random-only opponents.
- Use a held-out target state distribution and separate discovery seeds from learner evaluation seeds.
- Add a random-feature control and a target-trained concept-discovery upper-bound control.
- Report normalized regret / win rate / success rate and learning curves, not just one terminal return.

## Reproduction

The original v9 source tree is unchanged. Baseline outputs and completed transfer run directories are included alongside this report. Core commands are documented in the root README; the exact completed command settings are listed above.
