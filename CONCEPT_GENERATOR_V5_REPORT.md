# Concept Generator v5 — implementation and sandbox validation

## Scope

This upgrade builds on the v4 branch. It does not replace GP, NSGA-II, the novelty/restart monitor, the C++ NEAT+PPO learner, or the fresh-target-learner transfer boundary.

## Changes

### 1. Primitive contract v5

The shared contract is now `generic_state_action_v5`. Four order-sensitive observation-vector descriptors were added in Python discovery, Python policy inference, and the C++ runtime:

- `state_position_moment`: signed first moment over normalized feature indices.
- `state_position_spread`: absolute-mass-weighted spread over normalized feature indices.
- `state_adjacent_difference`: mean absolute difference between adjacent state entries.
- `state_sign_change_fraction`: fraction of adjacent pairs with opposite nonzero signs.

All four are available in the five current environments. They are order-sensitive and should not be interpreted as having identical domain meaning across environments. They do not encode board topology, object relations, or temporal dynamics. For one-dimensional Nim observations, the descriptors are degenerate or provide little information.

The Python concept loader accepts v1-v5 contracts, so existing v4 banks remain readable. The C++ learner must be rebuilt because its primitive contract is now v5.

### 2. Better candidate screening

A candidate must now:

- Depend on at least one state primitive (not just `action_norm` / `legal_norm`).
- Meet `--min-source-robust-corr` (default `0.02`), where robust correlation is the worse signed correlation across source train and validation splits.
- Pass existing depth, variance, target-compatibility, and target-output-diversity checks.

This prevents action-only expressions from being passed off as discovered state concepts and removes candidates with essentially no positive source predictive signal.

### 3. More conservative downstream admission

The old admission test only required a positive mean paired improvement. It could admit a concept based on one unusually favorable seed.

The new default gate requires all three:

- Mean paired held-out return difference > `--downstream-min-improvement` (default 0).
- Median paired held-out return difference > 0.
- At least 60% of paired seeds improve (`--downstream-min-win-fraction`, default 0.6).

The report stores the median, win fraction, and threshold. This is a practical screening rule, not a confidence interval or formal hypothesis test.

## Validation

### Automated tests

- `python -m unittest -v test_gp_selection.py test_novelty_monitor.py`: **14 tests passed**.
- Fifteen non-provider tests from `test_llm_game_platform.py` passed using a minimal local stub for the unavailable `openai` package; this includes the five `GridPolicyTests` covering v5 bank parsing, v4 backward compatibility, v3 compatibility, and grid-policy selection. The live provider-error test was excluded because the real SDK is unavailable.
- `python -m py_compile gp_transfer.py llm_game_platform/policy.py test_gp_selection.py test_llm_game_platform.py`: passed.
- `cd src && make clean all test-envs`: C++ build and environment smoke test passed.
- C++ expression parity test compiled and passed with all 23 primitives; an additional numerical check confirmed C++/Python root-evaluation parity for all four new v5 primitives.
- `src/brain --primitive-contract` reports `generic_state_action_v5`.

The complete `test_llm_game_platform.py` suite could not be imported normally because the sandbox does not have the `openai` Python package installed. No dependency installation was attempted. The relevant policy test class was run with a minimal import stub; this does not validate live LLM API behavior.

### Transfer pilots with final v5 screening

**Nim → LineWorld**, seed 20261115, population 40, 10 generations, 100 sampled states, 6 rollouts, 5 paired learner seeds, 5,000 downstream steps per learner:

- 143 unique GP structures and 55 unique behaviors.
- Four candidates survived source/target screening.
- **Zero concepts admitted.**
- All four candidates had positive source robust correlation, but each improved only 1/5 paired learner seeds (win fraction 0.2). Their positive mean improvements were driven by a single favorable seed and were rejected by the new consistency gate.
- At least one candidate used the new `state_position_moment` descriptor, but it did not demonstrate repeatable transfer.

**TicTacToe → Connect Four**, seed 20261116, population 40, 10 generations, 100 sampled states, 5 rollouts, 5 paired learner seeds, 3,000 downstream steps per learner:

- 152 unique GP structures and 98 unique behaviors.
- One candidate survived source/target screening: `square(log1p(abs(div(state_position_moment,action_norm))))`.
- Its mean downstream improvement was approximately +0.014, but only 2/5 seeds improved (win fraction 0.4), so it was rejected.
- **Zero concepts admitted.**

These pilots do not demonstrate that v5 improves transfer. They demonstrate that stricter admission rejects candidates with inconsistent downstream outcomes, including candidates that would have passed the previous mean-only rule.

## Important limitation: temporal concepts are not implemented yet

The current discovery dataset is a matrix of independently sampled state/action rows with estimated action advantages. It does not preserve a reliable `(previous_state, action, next_state, reward, done)` transition record for every sample. The C++ concept runtime also evaluates the current state/action feature vector and does not currently maintain a previous-observation buffer for concept evaluation.

Adding `state_delta` or `state_acceleration` as if they were ordinary current-state primitives would therefore be incorrect. A real temporal upgrade requires coordinated changes to:

1. Dataset builders: emit trajectory-aligned transitions and episode boundaries.
2. Train/validation splitting: split by episode/trajectory to prevent temporal leakage.
3. Primitive evaluator: define explicit history semantics and reset behavior.
4. C++ runtime: maintain previous observations and clear history on environment reset.
5. Python policy runtime: implement exactly the same history contract.
6. Parity tests: compare Python and C++ temporal features on identical transition sequences.

This was intentionally not faked in v5. The next major research milestone should be a transition-aware dataset/runtime contract, followed by action-conditioned delta, trend, stability, and short-horizon change features. Structured board topology should be handled separately with typed grid/graph primitives rather than generic feature-index arithmetic.

## Conclusion

v5 improves candidate discipline and introduces a small experimental family of ordered-vector features. It is not yet the substantial representation upgrade needed for genuinely reusable concepts. The pilots give no evidence that the new descriptors themselves improve cross-game transfer. The next step should be a properly transition-aware and typed structured-state representation, with ablations against v5 and the original baseline.
