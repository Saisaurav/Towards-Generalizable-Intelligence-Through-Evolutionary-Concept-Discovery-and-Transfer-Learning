# Integration report

Sources: `changed_files.zip` (new frontend, environment generation, runtime, learner bridge) and `cross_game_transfer_v8_final.zip` (research brain, concept generator, experiments, results). Counts come from a SHA-256 comparison against the original V8 tree.

## Files
* **Retained byte-identical from V8:** 1,838 of 1,844 files, including all research assets (results*, restart/extended experiments, experiments/, demo/, reports, CSVs), the C++ learner headers, `gp_transfer.py`, and `policy.py`, `trainer.py`, `spec.py`, `runtime.py`, `generator.py`, `__init__.py`. Nothing from V8 was removed.
* **Replaced by the new version:** `llm_game_platform/server.py` (then patched), `web/index.html` (then patched), `llm.py` (pure superset of V8's), `src/main.cpp` (V8 + four additive `ext` edits).
* **Merged:** `src/Makefile` (V8's + `external_env.h` dependency), `README.md` (V8 base + new section), `.gitignore`.
* **Added verbatim from `changed_files.zip`:** the 21 new files (sandbox, generator, validator, service, store, trainer relay, `external_env.h`, fixtures, `test_generated_envs.py`, `GENERATED_ENVIRONMENTS.md`, ...).
* **Added by the integration:** `test_ext_primitive_parity.py`, `src/ext_primitive_parity_test.cpp`, `integration_tests/`, `docs/`. `neat_policy.py` is from the new archive plus the adapter edit below.

## Compatibility issues
| # | Issue | Resolution |
|---|---|---|
| 1 | New `Makefile` dropped V8's `runner` and `regression_test` targets | Kept V8's, added one dependency |
| 2 | New `README.md` is older than V8's | V8 base, new sections added |
| 3 | New server dropped V8's `/research` + `/data/*.json` | Routes restored, one nav link added (judgment call) |
| 4 | `NeatPolicy` computed 7 primitives; V8's screened bank uses 8 more -> `KeyError` (3 test failures at baseline) | `concept_features` reuses V8's `_state_primitives`; only `legal_norm` is overridden |
| 5 | Both archives contain a different `llm_game_platform/` package | Combined; new code imports V8 `policy`/`trainer`, so those stay |
| 6 | `ExternalEnv` cannot supply board/tactic/boundary primitives (NaN -> C++ throw) | Bank screening via V8's `screen_concepts` already excludes them; verified |
| 7 | Float32 (C++) vs float64 (Python) sign flip where a divisor cancels to rounding noise | Not changed (V8 concept maths); the parity test bounds and reports such points |

No research algorithm, training parameter or concept representation was modified.

## Tests actually executed
* Baseline on the unfixed merge: `test_generated_envs` + `test_llm_game_platform`, 66 run, 63 pass, 3 fail (all issue 4).
* After the fix: `test_generated_envs`, `test_llm_game_platform`, `test_concept_bank`, `test_ext_primitive_parity`: 78 run, all pass. Research tests (`test_novelty_monitor`, `test_gp_selection`, `test_action_primitives`): 27 pass.
* C++: `regression_test` 0 failures; `env_smoke_test` exit 0 (silent on success).
* Parity: 22 primitives, 17,270 comparisons, max diff 5.6e-7; 12 bank concepts, 9,420 comparisons, max diff 3e-4, 4 ill-conditioned points excused.
* Learner regression: original V8 `brain` vs merged `brain`, same seeds on ttt/nim/lineworld: stdout and exported models byte-identical.
* Browser end-to-end (jsdom, real server, real sandbox, real C++ training, **stub LLM**): 24/24 checks (`integration_tests/`).

## Not verified / limitations
* No live LLM was used; generation quality from a real model is untested.
* End-to-end ran only the Tic-Tac-Toe fixture, with a 40k-step budget; held-out return 0.097 is a plumbing check, not evidence of learning or transfer. V8's bank note says none of its concepts were admitted as demonstrated transfer wins.
* jsdom is not a real browser (layout/CSS untested). The legacy grid tab was covered only by `test_llm_game_platform`, not in a browser.
* V8's long experiments were not re-run; no prior result is claimed as reproduced.
* `make test-envs` fails as it did in V8 (`lamarckian_hidden_test.cpp` absent from the V8 archive).
* Linux/g++ only; binaries are not shipped, build with `cd src && make brain runner`.
