# Stagnation-triggered GP concept regeneration: experiment report

## Executive summary

The restart mechanism was tested against a no-restart control in two directed cross-game settings:

- Nim → LineWorld
- Connect Four → Nim

For each direction, three GP random seeds were run. Both conditions used the same initial population, source/target datasets, random seed, population size (20), and evolution budget (14 generations). The restart arm was deliberately forced into a known exhausted state so the restart path could be stress-tested.

**Result:** the restart arm discovered more unique behavioural fingerprints in both transfer directions. The small downstream pilots were directionally positive relative to the no-restart control, but the sample is far too small to claim a statistically reliable transfer advantage.

A separate ordinary run using the default 8-generation stagnation threshold did not stagnate in 30 generations and therefore did not restart. This is expected: the restart mechanism is triggered by lack of behavioural novelty, not by low fitness alone.

## Method

### Forced-stagnation stress test

- Initial population: 20 copies of `action_norm`.
- The novelty monitor was pre-seeded with this population's behavioural fingerprint, representing an exhausted archive.
- Restart arm: stagnation window = 1 generation, so the known lack of new behaviour triggers regeneration immediately.
- Control arm: stagnation window = 999 generations, effectively disabling restarts for the 14-generation run.
- Restart configuration: preserve Pareto elites, inject fresh random expression trees (65% of the non-elite population), inject mutated elite descendants, and use stronger mutation for 3 generations.
- Three independent GP seeds per source-target direction: 731, 732, 733.
- Target learner: always initialized from scratch. Only the expression feature was supplied.

Candidate selection was performed from the final population plus Pareto archive in both arms. Candidates had to be behaviourally novel relative to the initial exhausted bank, use only primitives available in the target game, have nontrivial target output variance, and pass source-side validation selection. Source test correlation was measured separately.

### Downstream transfer pilot

- Three paired target learner seeds: 11, 22, 33.
- 3,000 environment steps per learner.
- NEAT+PPO population: 8.
- Each candidate was compared with a fresh action-identity baseline on the same target seed.
- Reported transfer improvement is the mean paired return difference, first averaged over the three target seeds for each GP seed, then summarized across the three GP seeds.

These are pilot settings, not a final statistical evaluation.

## Results

### Discovery

| Source → Target | Condition | Mean unique structures | Mean unique behaviours | Mean target-compatible novel candidates |
|---|---|---:|---:|---:|
| Nim → LineWorld | Restart | 88.7 | 47.0 | 15.7 |
| Nim → LineWorld | No restart | 48.7 | 32.0 | 13.3 |
| Connect Four → Nim | Restart | 75.7 | 50.0 | 13.7 |
| Connect Four → Nim | No restart | 62.0 | 43.0 | 12.3 |

Restarting increased mean behavioural diversity by:

- **Nim → LineWorld:** 46.9% (47 vs 32).
- **Connect Four → Nim:** 16.3% (50 vs 43).

Across the six forced-stagnation GP runs, 10 restarts were triggered. Eight were confirmed successful, none were confirmed unsuccessful, and two were still pending their observation window at the end of the fixed generation budget.

### Downstream transfer

| Source → Target | Restart mean improvement | No-restart mean improvement |
|---|---:|---:|
| Nim → LineWorld | +0.2913 | +0.1932 |
| Connect Four → Nim | +0.0244 | -0.0102 |

Across the three GP seeds, restart minus control in per-GP-seed mean target improvement was:

- Nim → LineWorld: +0.1646, +0.0076, +0.1221.
- Connect Four → Nim: +0.0307, +0.0253, +0.0480.

The restart condition was directionally better in these pilots. However, each candidate was evaluated on only three target seeds and 3,000 steps. The returns are noisy, and this is not enough evidence for statistical significance or a general transfer claim.

### Generalization warning

The restart-selected candidates had lower average held-out source-test correlation than the no-restart candidates in both directions:

| Source → Target | Restart candidate mean source-test | No-restart candidate mean source-test |
|---|---:|---:|
| Nim → LineWorld | 0.0905 | 0.1456 |
| Connect Four → Nim | 0.0864 | 0.1863 |

This is important: regeneration finds more behaviours, but the extra behaviours are not necessarily better source concepts. Behavioural novelty and predictive usefulness remain separate objectives.

## Default-threshold natural run

A normal Nim → LineWorld discovery run used:

- Population: 30
- Generations: 30
- Stagnation window: 8
- Default restart settings

Results:

- 199 unique structures
- 98 unique behavioural fingerprints
- 0 restarts triggered
- 4 candidates shortlisted (downstream evaluation disabled)

The ordinary run continued finding new behaviours, so it correctly did not restart. This shows that the restart system is not firing indiscriminately under this run; it does not prove that natural stagnation will occur in every task.

## Engineering checks

- Python unit tests: **10/10 passed**.
- Focused novelty monitor tests: passed.
- C++ brain: compiled successfully.
- C++ environment smoke test: passed after correcting an assertion to expect `tanh(action_norm)` rather than raw ±1.
- C++ concept primitive/parity test: compiled and passed.

The C++ build still emits existing `-Wmisleading-indentation` warnings in `genome.h` and `concept_env.h`; these did not fail compilation or tests.

## Interpretation

The experiments support three separate conclusions:

1. **The restart mechanism works mechanically.** It preserves elites, injects new random/mutated expressions, and resumes evolution.
2. **It increases behavioural exploration under forced stagnation.** Both tested transfer directions showed more unique behavioural fingerprints than the no-restart control.
3. **More novelty does not automatically mean better concepts.** Downstream pilots were encouraging, but source-test correlations were weaker for restart-selected candidates and the sample is too small to establish superiority.

## Recommended next improvements

1. Keep behavioural stagnation as one restart trigger.
2. Add an optional *fitness-stagnation* trigger: restart when validation fitness or Pareto-front hypervolume fails to improve for a configurable number of generations, even if structurally/behaviourally new but useless expressions continue appearing. Use a tolerance and smoothed metric to avoid reacting to noise.
3. Track each concept's origin (`initial`, `random_restart`, `elite_mutation`, `ordinary_mutation`, `crossover`) so downstream utility can be attributed to restart-generated material.
4. Run at least 10 independent GP seeds and 10–20 paired downstream seeds, with 10× or larger learner budgets.
5. Compare restart candidates, no-restart candidates, random features, and target-trained GP concepts; add per-concept ablation.

**Bottom line:** the generator now demonstrably escapes a deliberately exhausted population and discovers more distinct behaviours. Whether those behaviours are reliably more useful remains an open experimental question.
