# Cross-game GP concept transfer experiment

## Executive result

A first two-direction transfer experiment was run using the two distinct environments in the codebase at that time: Tic-Tac-Toe (TTT) and CartPole. Every target-game learner was initialised from scratch: new NEAT genomes/topologies, new PPO weights, and fresh optimiser state. Only GP expression features were transferred.

**Promising result:** TTT-evolved concepts improved CartPole held-out return relative to an action-ID-only baseline across 10 paired seeds. CartPole-evolved concepts also improved TTT mean return across 10 seeds, although that reverse-direction result is less robust after accounting for multiple comparisons.

This is preliminary evidence for transferring reusable mathematical feature programs. It is not yet proof that the GP has discovered high-level, semantically meaningful game strategies.

## Games used in the original experiment and shared primitive contract

The current repository contains two distinct environments:
- Tic-Tac-Toe versus a random opponent.
- CartPole.

`ttt0`, `ttt2`, and `ttrand` are Tic-Tac-Toe feature controls, not separate games.

Both games were wrapped in a common per-action interface. The GP programs consume six shared normalized primitives:

- `state_mean`
- `state_abs_mean`
- `state_var`
- `state_max_abs`
- `action_norm`
- `legal_norm`

A concept is an expression tree over those primitives. The C++ runtime evaluates the same expression syntax and applies protected division and final tanh normalization. A Python/C++ parity test passed.

Important limitation: these primitives have identical mathematical definitions but their underlying state dimensions have different meanings in TTT and CartPole. The experiment therefore tests transfer of computational feature programs over generic state/action statistics, not transfer of identical strategic semantics.

## GP discovery procedure

- Population: 60 expression trees.
- Evolution: 16 generations, tournament-like elite selection, crossover and mutation.
- Operators: add, subtract, multiply, protected divide, min and max.
- Fitness: absolute correlation between concept output and rollout-estimated relative action advantage, minus a small expression-complexity penalty.
- Source data was split from a held-out source set; target held-out labels were not used to evolve or select source concepts.
- Four candidate expressions were retained per source direction.
- A target compatibility filter removed concepts with near-constant output (`std < 0.01`) or behavior nearly duplicating another retained feature (`|corr| >= 0.995`). This filter used target feature values only, not target reward labels.

The GP datasets are deliberately small and noisy in this prototype: TTT source training used 100 sampled states and 5 rollouts per legal action; CartPole source training used 45 sampled states and 2 rollouts per action. This is a major limitation and should be expanded.

## Novelty/stagnation monitor

The monitor tracks two different kinds of novelty:

1. **Structural novelty:** canonicalized expression trees not seen previously. Commutative operands are sorted for canonicalization.
2. **Behavioral novelty:** expression outputs on a fixed probe set. Output vectors are normalized; a candidate is treated as behaviorally duplicate if its absolute correlation with an existing fingerprint is at least 0.995. Constant outputs are not counted as new behavior.

If a generation produces no behaviorally new concept for 8 consecutive generations, the GP run is flagged as stagnant and stops. The monitor also records per-generation structural novelty, behavioral novelty and behavioral diversity.

Observed discovery runs:

| Source game | Generations | Unique structures | Unique behaviors | Stagnation stop |
|---|---:|---:|---:|---|
| TTT | 16 | 461 | 230 | No |
| CartPole | 16 | 718 | 402 | No |

A unit test passed: repeated populations with the same behavior trigger stagnation after 8 consecutive generations, and a new behavior resets the counter.

## Target learner experiment

Every run used:
- Fresh NEAT population and PPO optimiser state.
- Population size: 12.
- Training budget: 150,000 environment steps (generation completion can cause a small overshoot).
- 100 evaluation episodes per genome per generation.
- Held-out evaluation: 1,000 episodes per final champion.
- Ten paired seeds: 11, 22, 33, 44, 55, 66, 77, 88, 99, 111.

Conditions:
- **Baseline:** action identity feature plus four zero-valued padding inputs.
- **Random features:** action identity plus four fixed random algebraic features.
- **Transferred concepts:** action identity plus GP expressions evolved on the other game.
- For TTT→CartPole, both the full four-expression bank and the two nonconstant, behaviorally distinct concepts were tested.

No genome, topology, policy weight, PPO optimiser state, or training trajectory was transferred.

## Results

### TTT → CartPole

CartPole return is mean episode survival steps (maximum 500).

| Condition | Mean held-out return | SD |
|---|---:|---:|
| Baseline | 22.428 | 0.370 |
| Random features | 22.253 | 0.441 |
| Four transferred expressions | 28.194 | 3.948 |
| Two compatible transferred expressions | **30.554** | 6.865 |

Paired comparison against baseline:
- Random features: mean difference -0.176, paired t-test p = 0.210.
- Four transferred expressions: +5.766, p = 0.00109.
- Two compatible transferred expressions: +8.126, p = 0.00518.

The compatibility-filtered transfer performed best on average, but variance was high. CartPole is not solved: the average remains far below the 500-step maximum.

### CartPole → TTT

TTT return is win (+1), draw (0), loss (-1), averaged over 1,000 held-out episodes against the random opponent.

| Condition | Mean held-out return | SD |
|---|---:|---:|
| Baseline | 0.0276 | 0.0392 |
| Random features | 0.0309 | 0.0542 |
| Three compatible transferred expressions | **0.0741** | 0.0553 |

Paired comparison against baseline:
- Random features: mean difference +0.0033, p = 0.808.
- Transferred concepts: +0.0465, p = 0.0295.

This is a positive directional result, but it is less conclusive after accounting for multiple comparisons. Absolute TTT performance is still weak.

## Interpretation

1. Random algebraic features did not improve either target learner over the action-ID baseline.
2. GP-evolved features transferred from the other environment improved mean target performance in both directions in this prototype.
3. The strongest result is TTT→CartPole. The reverse direction is promising but needs more independent replication.
4. The GP source held-out correlations were modest. The result may reflect useful action-conditioned nonlinear feature transformations rather than discovery of robust game-strategy concepts.
5. The output-variance compatibility filter removed degenerate features without looking at target rewards. This improved the TTT→CartPole mean, but the variance also increased.
6. These are small-scale experiments with simplified environments, random opponents, small populations and noisy GP screening. They do not establish general transfer across arbitrary games.

## Decision

**Keep the concept-transfer path and novelty monitor as an experimental branch. Do not yet claim the research hypothesis is proven.**

Next tests:
1. Increase GP source datasets and rollout estimates.
2. Add more seeds and repeat with independent GP evolution seeds, not just independent NEAT/PPO seeds.
3. Test whether the same transferred expressions remain useful under different observation normalizations and opponents.
4. Compare source-trained GP concepts against target-trained GP concepts as an upper-bound reference.
5. Add stronger semantic primitive definitions before claiming that the same concept means the same thing across games.
6. Track concept contribution by ablation: remove one concept at a time and measure the change in target learning curves and final performance.

## Next generator milestone

The GP pipeline now has an implementation path for the three proposed priorities. This is a protocol change, not an experimental result; the historical tables above were produced by the previous single-objective generator and remain unchanged.

### Primitive contract v2

The original six generic primitives remain available. The added primitives are:

- `state_magnitude`: root mean square of normalized state features, available in both games.
- `state_dispersion`: standard deviation of normalized state features, available in both games.
- `state_boundary_margin`: one minus the larger absolute normalized CartPole position/angle, available only in CartPole.

Expression generation uses only the source game's available primitives. Target compatibility additionally requires every dependency to be implemented by the target. Other suggested primitives are deferred rather than approximated with reward labels or unavailable transition/history information. The CartPole C++ runtime exposes the same boundary-margin definition; unavailable primitive use raises an explicit runtime error.

### Multi-objective search and staged admission

The scalar fitness and elite-only ordering have been replaced with NSGA-II ranking and crowding over four maximization objectives: source-training absolute correlation, source-validation absolute correlation, negative tree size, and behavioral novelty on validation probes. A separate source test set is reported but is not part of evolution or shortlist ranking. The shortlist is screened for nonconstant behavior, target primitive compatibility, target-output variance, and duplicate target behavior; that compatibility screen does not use target reward labels.

Supplying `--downstream-budget` runs each shortlisted concept in a fresh target NEAT+PPO learner and compares it against a fresh action-identity baseline at the same fixed budget, population size, and paired seeds. Both conditions use the same five action-feature slots. Admission requires positive mean paired held-out return improvement by default; `--downstream-min-improvement` can set a stricter threshold. Runs without a downstream budget explicitly leave candidates pending and produce an empty validated concept bank.

This gate is a practical screening criterion, not a significance test. Candidate count, seed count, and budget should be reported with future results, and admitted concepts still need independent transfer replications and ablation. The new pipeline has only received small smoke/test validation so far; it has not yet produced a new transfer result.

Stagnation handling now restarts rather than immediately terminating evolution. A bounded Pareto archive is carried into each restart, alongside random expression injection and temporarily stronger mutation. A restart is counted as successful only if a globally behaviorally novel expression appears within its observation window; evolution stops after the configured number of consecutive unsuccessful restarts. Restart configuration and outcomes are recorded in `novelty_history.json`.

## Additional experiment environments

Connect Four, Nim, and LineWorld have been added to the environment interface and GP dataset pipeline:

- Connect Four is 6x7, with legal column drops and a random opponent that may open.
- Nim uses a single 10–21 object heap; each player takes 1–3, and the random opponent replies after each learner move.
- LineWorld is an 11-position deterministic navigation task with left/right actions, a 30-step limit, and a goal reward.

The GP can now select any of the five games as a source and any different game as a transfer target. These additions have dataset/unit smoke coverage, but **no NEAT+PPO learning or transfer result has been run for them yet**. They also have not been compiled in the current Windows environment because no C++ compiler is available there or in the configured WSL distribution. See `README.md` for commands and current game rules.

## Stagnation-triggered regeneration follow-up

See `RESTART_EXPERIMENT_REPORT.md` for the forced-stagnation vs no-restart experiments across Nim → LineWorld and Connect Four → Nim, downstream paired-seed pilots, the default-threshold natural run, and engineering test results. Raw JSON and runnable experiment scripts are in `restart_experiment/` and `restart_experiment_c4_nim/`.
