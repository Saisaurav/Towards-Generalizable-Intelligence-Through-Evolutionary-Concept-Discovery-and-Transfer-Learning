# Concept Generator v7 — typed board structure and validation

## Summary

v7 advances the primitive contract to `generic_state_action_v7`. It retains v6's first-order temporal features and adds six board-structure primitives specifically for TicTacToe and Connect Four. These features are computed from the known board layouts and actual win geometry, rather than from flattened-vector adjacency alone.

## New board primitives

| Primitive | Definition | Availability |
|---|---|---|
| `board_own_open_lines` | Fraction of winning lines containing at least one own mark and no opponent marks | TicTacToe, Connect Four |
| `board_opponent_open_lines` | Fraction of winning lines containing at least one opponent mark and no own marks | TicTacToe, Connect Four |
| `board_own_immediate_wins` | Distinct empty cells completing an own winning line, divided by board cells | TicTacToe, Connect Four |
| `board_opponent_immediate_wins` | Distinct empty cells completing an opponent winning line, divided by board cells | TicTacToe, Connect Four |
| `board_center_control` | Own-minus-opponent center occupancy, normalized for Connect Four | TicTacToe, Connect Four |
| `board_occupancy` | Fraction of occupied cells | TicTacToe, Connect Four |

TicTacToe uses length-three winning lines; Connect Four uses length-four winning lines. Immediate-win features count distinct playable cells, so a cell that completes multiple lines is counted once.

Python discovery and C++ inference use matching definitions. The C++ environments expose the typed features through `additional_primitive`; the shared concept runtime places them in primitive slots 27–32. The generic grid-game Python policy accepts v7 bank files but filters both temporal and board-specific primitives from grid policies that do not have those semantics.

## Validation

- Python compilation for `gp_transfer.py`, `llm_game_platform/policy.py`, and both test modules: passed.
- `python -m unittest -v test_gp_selection.py test_novelty_monitor.py`: **14 tests passed**.
- Seven `GridPolicyTests` passed using a minimal import stub for the unavailable optional `openai` SDK. This includes v6 and v7 concept-bank compatibility checks.
- C++ `make clean all test-envs`: build and environment/runtime smoke test passed.
- C++ concept expression parity test compiled and passed with the 33-primitive v7 contract.
- C++ board-feature smoke checks passed for a TicTacToe two-in-a-row position and a Connect Four three-in-a-row position.
- Dataset schema checks across all five environments passed: available primitive rows are finite and unavailable primitive rows are NaN.
- `src/brain --primitive-contract` reports `generic_state_action_v7`.

The complete LLM-platform suite was not run because the real optional `openai` SDK is unavailable in the sandbox; the relevant grid-policy tests were run with a minimal import stub.

## Transfer discovery pilot: TicTacToe → Connect Four

Configuration: population 35, 6 generations, 45 states per source split, 2 rollouts per action, shortlist 5.

- 95 unique expression structures and 67 unique behaviours.
- Three candidates passed source and target compatibility screening.
- The top candidate explicitly used `board_opponent_immediate_wins` and `board_opponent_open_lines`:

```text
max(
  div(min(board_opponent_immediate_wins,state_adjacent_difference),add(state_var,action_norm)),
  mul(board_opponent_open_lines,sub(state_range,state_max))
)
```

The candidate had source robust signed correlation approximately +0.100. This shows the new structural primitives are reachable by GP and can appear in shortlisted expressions.

### Downstream test of the structural candidate

Configuration: fresh Connect Four learners, 1,200 training steps, population 8, paired seeds 11/22/33/44/55.

| Seed | Baseline | Candidate | Difference |
|---:|---:|---:|---:|
| 11 | -0.050 | -0.050 | 0.000 |
| 22 | 0.138 | 0.138 | 0.000 |
| 33 | 0.056 | 0.035 | -0.021 |
| 44 | 0.102 | 0.120 | +0.018 |
| 55 | 0.024 | 0.024 | 0.000 |

- Mean paired improvement: -0.0006.
- Median paired improvement: 0.
- Win fraction: 0.20.
- Admission result: **rejected**.

This is not evidence of improved transfer. It does show that the structural concepts can be generated, evaluated in the target runtime, and rejected when the downstream evidence is weak.

## Limitations and next research steps

1. The board primitives are handcrafted structural descriptors, not yet autonomously discovered relational operators.
2. The current source advantage labels still rely on relatively weak random rollout policies. Strategic features may be hard to discover when the teacher signal is noisy.
3. The small downstream budget and five paired seeds are screening evidence, not a definitive statistical conclusion.
4. Temporal features still use only the previous transition; no multi-step trajectory encoder is implemented.
5. The system still lacks hierarchical composition of validated concepts.
6. The next useful upgrade is better teacher/data generation and action-conditioned counterfactual evaluation, followed by larger paired-seed transfer tests. More primitives alone are unlikely to solve the weak-signal problem.

## Conclusion

v7 adds a real typed representation for immediate threats, open winning lines, center control, and occupancy in two board games. The GP discovered candidates using these features, but the tested structural candidate failed downstream admission. The next bottleneck is now clearer: improving the quality of the learning signal and evaluating concepts over stronger target learners, not simply expanding the primitive vocabulary.
