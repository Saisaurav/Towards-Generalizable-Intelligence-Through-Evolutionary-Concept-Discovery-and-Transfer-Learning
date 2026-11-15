# Architecture of the merged system

Three independently developed parts live in one tree. Ownership follows the integration brief.

| Layer | Files | Origin |
|---|---|---|
| Frontend | `llm_game_platform/web/index.html` (+ `/research` viewer `web/research.html`, `web/data/*.json`) | new frontend; research viewer retained from V8 |
| HTTP API | `llm_game_platform/server.py` | new, plus V8's `/research` routes restored |
| Environment generation | `env_generator.py`, `env_prompts.py`, `static_check.py`, `env_validator.py`, `env_store.py`, `env_interface.py`, `generate_env.py` | new |
| Runtime loop / sandbox | `sandbox.py`, `sandbox_worker.py`, `env_service.py`, `render_protocol.py`, `play_cli.py` | new |
| Learner bridge | `env_trainer.py` (relay), `src/external_env.h`, `src/main.cpp` (`ext` mode) | new |
| Research brain | `src/*.h`, `src/runner.cpp`, `llm_game_platform/policy.py`, `trainer.py` | V8, unmodified |
| Concept generator / transfer | `gp_transfer.py`, `experiments/`, `results*/` | V8, unmodified |
| Adapter | `llm_game_platform/neat_policy.py` | new file, edited at the integration boundary |
| Legacy grid path | `spec.py`, `runtime.py`, `generator.py`, `__init__.py` | V8, unmodified (still used by the "Grid maze" tab) |

## Data flow

```
browser (index.html)
   | POST /api/envs {prompt}
   v
server.py -> EnvService -> EnvGenerator --LLM--> env.py + meta.json (generated_envs/<id>/)
                              | static_check -> sandbox load -> 24 random episodes -> determinism -> brain/human paths
                              v
                      sandbox_worker.py  (one subprocess per env instance)
                              ^ line protocol "R seed" / "A action"
browser editor --POST /api/envs/<id>/train {source}--> revalidate + persist exact edited source
EnvTrainer relay <----stdout/stdin----> src/brain ext   (ExternalEnv -> ConceptEnv -> NEAT+PPO, V8 code)
   |                                                      concepts: results/concept_bank.json via V8 screen_concepts
   v
NeatPolicy (Python inference of the exported policy) -> /agent endpoint -> browser
```

## The integration boundary (compatibility contract)

* `ExternalEnv` implements V8's `Env`; the V8-only optional hooks (`n_action_tactics`, `additional_primitive`) keep their defaults, so board/tactic/boundary primitives evaluate to NaN. `ConceptEnv` throws on NaN primitives, so only concepts passing V8's `screen_concepts` (`_GRID_PRIMITIVES`: state statistics, `action_norm`, `legal_norm`) are sent to generated games.
* The exported policy runs in Python. `neat_policy.concept_features` computes concept inputs with V8's `_state_primitives`, overriding only `legal_norm = #legal / n_actions` (the C++ definition for generic envs). `test_ext_primitive_parity.py` checks this against the real C++ `ConceptEnv`.
* Each training run starts a new `brain ext` process and fresh population; only concept expressions carry over.

## Lifecycles

* **Environment:** prompt -> generating (repair loop, up to `max_attempts`) -> ready | failed. Validated source is stored with a SHA-256 and re-validated on reload; the ready source is editable in the browser.
* **Training:** waiting -> validating edited source -> queued -> running (per-generation points parsed from the learner's log) -> ready | failed | skipped (non-trainable env, e.g. continuous actions). Generation alone never starts training; the `train` request validates and persists its submitted source before queuing the learner. The runtime starts from the same accepted source when the user requests training. Brain play is enabled only when `brain_ready`.
* **Episode:** `/reset` (seat, opponent) -> repeated `/action` or `/agent` -> terminal; the runtime truncates at `max_steps`.

## Errors

Generation errors go back to the LLM as repair prompts; after the limit the status is `failed` with `generation_error`. Sandbox failures surface as `SandboxError` (HTTP 4xx/5xx with `{"error": ...}`); training failures set `training_status = "failed"` and `training_error`. A missing brain binary raises `BrainTrainingError` ("build it from src").

## Limits

Python sandboxing is defence in depth, not a security boundary (see `GENERATED_ENVIRONMENTS.md` section 6). Discrete actions only (<=128) are trainable.
