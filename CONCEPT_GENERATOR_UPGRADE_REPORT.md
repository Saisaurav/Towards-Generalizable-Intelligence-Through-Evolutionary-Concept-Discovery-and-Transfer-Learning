# Concept Generator v4 Upgrade Report

## What changed

This is an experimental upgrade of the existing GP concept-discovery pipeline. It preserves the existing expression-tree representation, NSGA-II selection, novelty/restart system, source/target environment datasets, concept-bank format, and downstream NEAT+PPO admission test.

### 1. Primitive contract v4

Four distribution primitives were added to the generic state/action feature vector and implemented in both Python and C++:

- `state_median`: median normalized state value.
- `state_iqr`: 75th percentile minus 25th percentile, using linear interpolation.
- `state_skewness`: standardized third central moment; returns zero for near-constant state vectors.
- `state_entropy`: normalized Shannon entropy of absolute state-value mass; returns zero for all-zero or singleton vectors.

All four are available in TTT, CartPole, Connect Four, Nim, and LineWorld. The C++ runtime and Python policy evaluator were updated. The contract identifier is now `generic_state_action_v4`; the Python concept loader continues to recognize v1-v3 banks.

These features are permutation-invariant summaries. They do not encode board geometry, coordinate identity, temporal change, or action-conditioned transition effects. They are deliberately generic statistical features, not a solution to structured-state representation.

### 2. Fitness v1: signed and generalization-aware

`population_objectives` now maximizes five objectives:

1. `min(train_signed_corr, validation_signed_corr)` — robust predictive direction across both splits.
2. `validation_signed_corr` — validation predictive quality.
3. `-abs(train_signed_corr - validation_signed_corr)` — train/validation consistency.
4. `-tree_size` — simplicity.
5. Behavioral novelty — distance from other population behaviours.

Previously, absolute correlation treated a strongly inverted feature as equally predictive. The GP can still discover `neg(...)` explicitly when reversing a feature is useful.

The source test split remains outside evolutionary selection. Downstream target-learner evaluation remains the admission gate; source fitness and target feature compatibility alone do not admit a concept.

### 3. Dataset-size controls

Added:

- `--nstates N`: override state samples per dataset split.
- `--rollouts N`: override rollouts per candidate action.

A value of zero (the default) retains environment-specific defaults. The controls apply independently to source training, validation, test, and target feature datasets.

## Validation performed

### Automated checks

- `python -m unittest -v test_gp_selection.py test_novelty_monitor.py`: **13 tests passed**.
- `python -m py_compile gp_transfer.py llm_game_platform/policy.py test_gp_selection.py test_llm_game_platform.py`: passed.
- C++ `make clean all test-envs`: compiled successfully and environment smoke test passed.
- C++ concept expression parity test: compiled and passed with the v4 contract and the four new primitive names.
- Python policy parser/evaluator smoke test for all four new primitives: passed.

The full `test_llm_game_platform.py` suite could not be executed because the sandbox does not have the `openai` dependency installed and network access prevented installing it. Its Python source compiled successfully; the v4 loader/evaluator path was separately exercised without importing the LLM client.

### End-to-end discovery and downstream smoke test

Command configuration:

- Source: Nim
- Target: LineWorld
- Seed: 20261117
- Population: 24
- Generations: 6
- Source/target samples: 25 states per split
- Rollouts per action: 2
- Shortlist: 3
- Downstream budget: 5,000 environment steps per condition
- Downstream learner population: 8
- Paired seeds: 11, 22, 33, 44, 55, 66, 77, 88, 99, 111

Result:

- Primitive contract: `generic_state_action_v4`
- Shortlisted: 3
- Downstream-admitted: 1
- Admitted expression: `sub(state_max_abs,action_norm)`
- Mean paired return difference: +0.1861

The paired differences were heterogeneous: seeds 11, 55, and 111 were neutral; seed 22 produced a much larger gain than the others. This is a pipeline smoke test, not evidence that v4 improves transfer. The sample count, environment budget, and single GP evolution seed are too small for a research conclusion.

## Important limitations still open

1. **State representation remains the main bottleneck.** The current feature vector still compresses each observation into global statistics. It loses coordinate identity and board topology.
2. **No temporal primitives yet.** The data pipeline currently evaluates sampled states/actions; it does not consistently carry previous observations or transition histories into GP.
3. **Action-value labels remain rollout-policy-dependent.** Random or simple rollout policies can produce noisy, policy-specific advantages.
4. **Semantic compatibility remains weak.** Matching primitive names across environments does not prove that the same expression has the same meaning.
5. **No broad A/B claim.** The historical v3 and v4 smoke runs are not a controlled multi-seed comparison. A proper comparison needs identical datasets, GP seeds, downstream budgets, and paired learner seeds across both versions.

## Recommended next experiment

Run v3 and v4 under a matched protocol on at least three source-target pairs, with multiple independent GP seeds and the full downstream seed set. Report shortlist quality, admission rate, paired return differences with confidence intervals, and concept ablations. Only after that should we decide whether these additional distribution primitives are worth keeping.
