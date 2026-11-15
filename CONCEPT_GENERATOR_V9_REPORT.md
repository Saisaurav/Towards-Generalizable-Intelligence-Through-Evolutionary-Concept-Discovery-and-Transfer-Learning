# Concept Generator v9 — Exact oracle teachers for solvable environments

## Summary

v9 adds an optional `--teacher-policy oracle` dataset mode while preserving the v8 rollout mode as the default. The primitive vocabulary and C++ runtime remain unchanged (`generic_state_action_v7`). The goal is to separate the quality of the supervision signal from the GP search architecture.

Exact action-value evaluators are currently implemented for:

- TicTacToe: minimax under perfect play.
- Nim: exact optimal-play recursion for the one-heap, take-1-to-3, last-take-wins rules.
- LineWorld: exact deterministic return when the first action is followed by the shortest-path continuation.

CartPole and Connect Four are explicitly unsupported in oracle mode. The code raises an error instead of quietly using a weaker substitute.

## Interface

```bash
python gp_transfer.py --source ttt --target connect_four --teacher-policy oracle
python gp_transfer.py --source ttt --target connect_four --teacher-policy rollout
```

The default remains `rollout`. The selected policy is recorded in the summary and each candidate's source-data metadata. The target feature dataset uses oracle labels when that target has an implemented oracle; otherwise it uses the rollout builder. This distinction affects labels only; target features are still derived from the same observation/action representation.

## Tests

- Python compilation: passed.
- `python -m unittest -v test_gp_selection.py test_novelty_monitor.py`: **23 tests passed**.
- Oracle-specific tests verify TicTacToe immediate wins, Nim winning/losing states, LineWorld goal-directed action ranking, deterministic/reproducible oracle datasets, and rejection of unsupported games.
- Rollout-mode dataset schema checks across all five environments: passed.
- Oracle dataset generation checks for all three supported environments: passed.
- C++ build and environment smoke test: passed; the primitive contract remains `generic_state_action_v7`.

## Controlled experiment: TicTacToe → Connect Four

Discovery settings for both conditions: seed 20261115, population 40, 8 generations, 60 states per source split, 4 rollouts/action where applicable. The downstream candidate was tested in fresh Connect Four learners with population 8 and 1,200 environment steps per learner. Baseline and candidate used identical learner seeds.

The top candidate from each discovery condition was evaluated on ten paired seeds: 11, 22, 33, 44, 55, 66, 77, 88, 99, 111.

| Metric | Exact oracle labels | Random-rollout labels |
|---|---:|---:|
| Unique expression structures | 132 | 148 |
| Unique behaviours | 101 | 116 |
| Shortlisted candidates | 1 | 1 |
| Mean paired downstream improvement | +0.0511 | +0.0151 |
| Median paired improvement | +0.0130 | 0.0000 |
| Seeds improved | 5/10 | 3/10 |
| Admission gate (60% win fraction) | Rejected | Rejected |

Oracle candidate expression:

```text
div(
  sqrt(mul(
    max(state_abs_mean,state_position_moment),
    max(state_skewness,board_center_control)
  )),
  sub(state_iqr,action_norm)
)
```

Rollout candidate expression:

```text
min(
  action_norm,
  div(
    min(state_position_moment,state_max_abs),
    tanh(mul(state_magnitude,board_own_open_lines))
  )
)
```

The oracle-derived candidate had a larger mean improvement and improved more seeds in this pilot. However, it improved only 5/10 seeds, below the configured 60% admission threshold. **Neither candidate was admitted in the ten-seed comparison.** This is suggestive evidence that cleaner labels may help, not proof that oracle supervision improves transfer generally. The comparison uses one GP seed and one shortlisted candidate per condition; broader repeated runs are required.

The first five-seed screen had provisionally passed the oracle candidate (mean +0.0346, median +0.026, 3/5 wins). Expanding to ten seeds changed the decision to rejection. This demonstrates why the larger paired-seed check matters.

## Limitations and next steps

1. Exact teachers exist for only three of five environments.
2. LineWorld's oracle encodes shortest-path continuation; it is not a general planner for arbitrary environments.
3. The experiment is one discovery seed and one candidate per condition, so candidate-selection variance remains high.
4. CartPole needs a documented expert controller or policy-derived action-value estimator; Connect Four needs a stronger tactical or search-based teacher.
5. The next experiment should repeat oracle-vs-rollout discovery across several GP seeds and evaluate every shortlisted candidate under the same larger downstream seed set.

## Conclusion

v9 adds a clean experimental switch between noisy rollout labels and exact action-value labels in three solvable games. The first TTT→Connect Four pilot favoured the oracle-derived candidate on mean improvement and seed win count, but neither method passed the ten-seed admission gate. The correct conclusion is that oracle supervision is promising enough to investigate further, not that it has solved cross-game transfer.
