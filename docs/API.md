# HTTP API (`python -m llm_game_platform.server`, default http://127.0.0.1:8000)

All bodies are JSON; errors are `{"error": "..."}`.

## Generated environments (primary path)
| Method & path | Body | Response |
|---|---|---|
| `POST /api/envs` | `{"prompt": str (<=4000), "training_opponent"?: str}` | 202 + generation status snapshot; generation does not start training |
| `GET /api/envs` | - | `{"environments": [{id, title, created, ...}]}` saved envs |
| `POST /api/envs/load` | `{"id": hex32}` | snapshot of the reloaded, re-validated env |
| `GET /api/envs/<id>/status` | - | snapshot (below) |
| `GET /api/envs/<id>/source` | - | generated Python source (text) for editing |
| `POST /api/envs/<id>/train` | `{"source": str (<=50000 chars), "training_opponent"?: str}` | 202 + queued snapshot; edited source is sandbox-validated and persisted before training starts; validation failures return 422 and leave the prior source unchanged |
| `POST /api/envs/<id>/reset` | `{"seat"?: int, "opponent"?: str}` | snapshot with fresh `view` |
| `POST /api/envs/<id>/action` | `{"action": int}` (illegal -> 400 "illegal") | snapshot |
| `POST /api/envs/<id>/agent` | `{}` brain plays the human seat | snapshot (4xx if no brain) |

Snapshot fields used by the frontend: `id, generation_status, generation_notes, generation_error, attempts, spec, validation, training_status (waiting|validating|queued|running|ready|failed|skipped), training_error, training_note, training_concepts_added [expression], training_history [{gen, steps, best, mean, species, nodes, conns, win_rate?, wins?, losses?, draws?}], training_heldout, training_target_reached, training_quality_mode (result|return), training_success_return, training_opponent, concepts [expression], brain_ready, human_seat, opponent, view` (render payload returned by the sandbox for the current state), plus `ok, stats, warnings`. In `return` quality mode `win_rate` is the success rate: the fraction of episodes whose return beats the calibrated random-play bar (`training_success_return`).

## Legacy grid contract (V8, unchanged)
`POST /api/games {prompt}`, `GET /api/games/<id>/status`, `POST /api/games/<id>/reset|actions|agent`.

## Research viewer (V8, read-only)
`GET /research`, `GET /data/{concepts,experiments,demo_games}.json` (cached results; regenerate with `experiments/build_frontend_data.py`).

## Configuration
`LLM_API_KEY, LLM_MODEL, LLM_BASE_URL, LLM_REQUEST_TIMEOUT_SECONDS` (OpenAI-compatible endpoint); generated-env training uses `GAME_ENV_TARGET_WIN_RATE` (fraction, default 0.90; checked at G30 and G60 for per-action-head refresh runs), `GAME_ENV_EVAL_EPISODES` (default 100), `GAME_ENV_EVAL_INTERVAL_GENERATIONS` (default 5), `GAME_ENV_TRAINING_POPULATION` (default 3 for turn-based games, 8 for real-time games; falls back to `GAME_TRAINING_POPULATION` if set), `GAME_ENV_HEAD` (`auto|per_action|fixed`; auto = fixed for real-time), `GAME_ENV_QUALITY_MODE` (`auto|result|return`), `GAME_ENV_MAX_GENERATIONS`, `GAME_ENV_MIN_GENERATIONS`, `GAME_ENV_IMITATION` (weight of the auxiliary imitation of the env's `scripted_action`; default on for the per-action head, off for the real-time fixed head), `GAME_ENV_CONCEPT_REFRESH_GENERATION` (default 60; per-action head only; 0 disables the GP refresh/restart; a failed G30 check continues the same learner through G60 before refresh), and `GAME_ENV_MAX_STEPS` (default 0, no learner step cap; real-time games also use `SPEC.max_steps = 0` and terminate by game rules or the runtime frame cap; `GAME_ENV_TRAINING_BUDGET` is a legacy fallback). Per-action-head PPO uses the measured action-identity setup (one-hot action inputs, 16 hidden units, 4 PPO iterations, greedy fitness/held-out evaluation); the real-time fixed head defaults to the configuration measured on the Snake fixture (16 hidden units, 16 PPO iterations, fitness from rollouts, persisted Adam moments, carry 0.5, no entropy bonus). `GAME_TRAINING_TIMEOUT` is the per-phase limit (default 900 seconds). Other learner settings: `GAME_TRAINING_BUDGET/POPULATION/TIMEOUT`, `GAME_CONCEPT_BANK`, `GAME_BRAIN_BIN`.

## Learner line protocol (internal, brain <-> relay)
`@NEW`, `@R <id> <seed>`, `@A <id> <action>`, `@DEL <id>`; replies `OK <id> <n_state> <n_actions>` / `S <reward> <done> <n_legal> <legal..> <n_state> <state..> <result>` / `ERR <msg>`. Result is `1` win, `-1` loss, `0` draw, or `2` unavailable. Non-`@` stdout lines are training progress and quality checkpoints.
