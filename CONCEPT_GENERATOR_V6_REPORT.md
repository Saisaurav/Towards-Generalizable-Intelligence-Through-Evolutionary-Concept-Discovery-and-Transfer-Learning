# Concept Generator v6 — transition-aware upgrade and sandbox report

## Summary

v6 advances the shared contract from `generic_state_action_v5` to `generic_state_action_v6`. It adds first-order transition features and carries their history through the C++ concept runtime. This is a limited temporal upgrade, not a full trajectory model or a structured graph/grid concept system.

## Implementation

### New primitives

| Primitive | Definition | Available environments |
|---|---|---|
| `transition_delta_mean` | Mean of current normalized observation minus previous normalized observation | CartPole, LineWorld |
| `transition_delta_abs_mean` | Mean absolute transition delta | CartPole, LineWorld |
| `transition_delta_max_abs` | Maximum absolute transition delta | CartPole, LineWorld |
| `transition_state_alignment` | Cosine similarity between current observation and its transition delta | CartPole, LineWorld |

The initial observation after reset has zero transition features. `ConceptEnv::step` captures the observation before and after the base environment step and stores the normalized delta. `ConceptEnv::reset` clears history. Action-feature evaluation reads the same stored transition for each legal candidate action, so evaluating multiple candidate actions does not mutate history.

Python dataset generation uses the last observed transition for CartPole samples and a one-step predecessor for LineWorld samples. Temporal primitives are marked unavailable in TicTacToe, Connect Four, and Nim. Existing v1-v5 banks remain readable by the Python concept loader. The Python grid policy accepts v6 banks but filters transition-only concepts because its grid-game inference interface does not maintain transition history.

## Validation

- `python -m py_compile gp_transfer.py llm_game_platform/policy.py test_gp_selection.py test_llm_game_platform.py`: passed.
- `python -m unittest -v test_gp_selection.py test_novelty_monitor.py`: **14 tests passed**.
- C++ `make clean all test-envs`: build and environment/runtime smoke test passed.
- C++ expression parity test compiled and ran with the 27-primitive contract.
- C++ runtime smoke test verifies transition is zero immediately after reset, becomes nonzero after a step, and is cleared by reset.
- Dataset schema check for all five games: all available primitive rows were finite and all unavailable primitive rows were NaN.
- `src/brain --primitive-contract`: `generic_state_action_v6`.

The full `test_llm_game_platform.py` suite was not run in this sandbox because the optional `openai` SDK is unavailable. Its source compiles, and a v6 bank-compatibility test was added.

## Transfer experiments

### LineWorld → CartPole: downstream pilot

Configuration: population 50, 10 generations, 80 states per source split, 4 rollouts per action, shortlist 4, 1,200 downstream steps, population 8, paired seeds 11/22/33/44/55.

- 195 unique expression structures and 137 unique behaviors.
- Four candidates shortlisted.
- One candidate passed the existing downstream gate.
- Admitted expression: `div(state_mean,action_norm)`.
- Paired differences: 0.000, +5.934, +0.561, +13.718, 0.000.
- Mean improvement: +4.043; median: +0.561; win fraction: 0.60.

Important interpretation: the admitted expression does **not** use any of the new temporal primitives. This is evidence that the v6 pipeline and admission process run end-to-end, not evidence that temporal concepts improved transfer.

### CartPole → LineWorld: discovery pilot

Configuration: population 40, 8 generations, 50 states per split, 2 rollouts per action.

- 133 unique structures and 112 unique behaviors.
- No candidates passed source/target screening.
- No concepts admitted.

### Direct temporal-feature probe

On a separate 500-state, 8-rollout dataset, individual transition features had near-zero correlation with action advantage, as expected for state-only features shared by the action alternatives. Interactions with `action_norm` were more informative in LineWorld: `transition_delta_abs_mean * action_norm` had correlation approximately +0.614.

However, a direct downstream CartPole test of `mul(transition_delta_abs_mean,action_norm)` over the same five paired seeds produced differences 0.000, +0.047, 0.000, 0.000, and -0.637 (mean -0.118; median 0; win fraction 0.20). It failed the admission criteria. This is a useful negative result: the temporal feature was measurable and executable, but this particular temporal expression did not transfer reliably.

## Limitations and next work

1. The temporal representation contains only the immediately preceding transition; it does not include multi-step windows, acceleration, event sequences, or learned dynamics.
2. LineWorld's transition is highly constrained and its delta magnitudes are nearly fixed, so it is a weak testbed for rich temporal abstraction.
3. CartPole's data sampler is still built from independently generated sampled states, with only the most recent random-walk transition retained. A stronger next version should preserve complete trajectories and split by episode.
4. Board games still need typed topology-aware primitives (neighbors, lines, threats, connectivity) rather than treating flattened board cells as a generic numeric vector.
5. Hierarchical concept composition is not implemented in v6.
6. The temporal primitive's downstream transfer has not been demonstrated. Do not describe v6 as a proven transfer improvement.

## Conclusion

v6 establishes a functioning Python/C++ first-order transition contract and preserves reset semantics. The end-to-end transfer pilot admitted one non-temporal concept; the direct temporal candidate failed. The next research step should improve trajectory data quality and introduce typed structural representations, with explicit temporal and structural ablations rather than simply increasing primitive count.
