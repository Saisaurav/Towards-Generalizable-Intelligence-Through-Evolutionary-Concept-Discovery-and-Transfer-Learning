# Concept Generator v8 — Game-calibrated counterfactual rollout data

## Summary

v8 improves the quality of action-advantage labels used by GP without changing the primitive vocabulary or C++ inference contract. The previous dataset builders evaluated each legal action using independent random streams. v8 adds a shared helper that can evaluate actions with either common random numbers (CRN) or independent rollout seeds.

CRN is not assumed to be universally beneficial. It reduces variance when outcomes under different actions are positively correlated, but can increase variance when they are negatively correlated. A small fixed-state pilot was therefore run before selecting a per-environment policy.

- TicTacToe: CRN enabled.
- Connect Four: CRN enabled.
- LineWorld: CRN enabled.
- CartPole: independent seeds retained.
- Nim: independent seeds retained.

The dataset protocol is recorded as `game_calibrated_rollout_seeds_v3`. The primitive contract remains `generic_state_action_v7`; no C++ runtime changes were required.

## Variance pilot

For each environment, repeated experiments estimated the difference between two fixed legal actions. The paired condition reused the same rollout seed list for both actions; the independent condition generated separate seed lists. The reported ratio is sample variance under CRN divided by sample variance under independent seeds. A value below 1 indicates a reduction in this pilot.

| Environment | Fixed state/action comparison | Repetitions | Rollouts/action | Variance ratio | Selected protocol |
|---|---|---:|---:|---:|---|
| TicTacToe | Empty board, action 0 vs 1 | 1,200 | 8 | 0.376 | CRN |
| Connect Four | Empty board, column 0 vs 3 | 1,200 | 8 | 0.765 | CRN |
| CartPole | `[0.02, 0.01, 0.03, 0.0]`, action 0 vs 1 | 500 | 4 | 1.364 | Independent |
| Nim | 15 objects remaining, take 1 vs 2 | 1,200 | 8 | 1.469 | Independent |
| LineWorld | Position 4, step 5, left vs right | 1,200 | 8 | 0.381 | CRN |

**Limitation:** this is a calibration pilot on one representative state and action pair per environment, not a universal proof that the selected protocol reduces variance for every state. A stronger follow-up should estimate covariance across a stratified sample of states and optionally choose the seed protocol per state or state cluster.

## Implementation

`paired_action_values(actions, rollout_fn, rollouts, rng, common_random_numbers=True)` returns action means and per-rollout samples. In CRN mode, every action receives the same seed list. In independent mode, each action receives a separately generated seed list.

The dataset builders use CRN for TicTacToe, Connect Four, and LineWorld, and independent seeds for CartPole and Nim. The downstream target-learner admission protocol is unchanged; it already uses paired learner seeds for candidate-vs-baseline comparisons.

## Validation

- Python compilation: passed.
- `python -m unittest -v test_gp_selection.py test_novelty_monitor.py`: **18 tests passed**.
- New tests verify shared seed sequences, independent seed sequences, zero-rollout validation, and deterministic dataset generation.
- Small dataset generation and schema checks across all five environments: passed; all available primitive rows and labels were finite, and unavailable primitive rows remained NaN.
- C++ build: passed.
- C++ environment smoke test: passed.
- C++ primitive contract: `generic_state_action_v7`, as expected because no runtime primitive changed.
- End-to-end GP discovery smoke test (TicTacToe → Connect Four; population 30, 5 generations, 30 source states/split, 3 rollouts/action): completed. It produced 78 unique expression structures, 60 unique behaviours, and a shortlist of 3 candidates. Downstream admission was not enabled in this smoke test, so no transfer improvement is claimed.

The C++ compiler emitted existing `-Wmisleading-indentation` warnings in `genome.h` and `concept_env.h`; compilation and smoke tests nevertheless succeeded. These warnings were not introduced by v8.

## Research interpretation and next step

This version improves the experimental control of action-value labels. It does not make the rollout opponents stronger and does not solve teacher-policy bias. The next high-value step is to add stronger, explicitly documented teacher policies or exact evaluators where feasible, then compare label quality and downstream transfer under matched compute budgets. The CRN policy should also be recalibrated across many states rather than one representative state per game.

No claim is made that v8 discovers more transferable concepts than v7. Its demonstrated contribution is a measured, environment-dependent change to counterfactual rollout sampling.
