# HTTP API (`python -m llm_game_platform.server`, default http://127.0.0.1:8000)

All bodies are JSON; errors are `{"error": "..."}`.

## Generated environments (primary path)
| Method & path | Body | Response |
|---|---|---|
| `POST /api/envs` | `{"prompt": str (<=4000), "training_opponent": str}` (valid opponents: `TRAINING_OPPONENTS` in `env_service.py`) | 202 + status snapshot |
| `GET /api/envs` | - | `{"environments": [{id, title, created, ...}]}` saved envs |
| `POST /api/envs/load` | `{"id": hex32}` | snapshot of the reloaded, re-validated env |
| `GET /api/envs/<id>/status` | - | snapshot (below) |
| `GET /api/envs/<id>/source` | - | generated Python source (text) |
| `POST /api/envs/<id>/reset` | `{"seat"?: int, "opponent"?: str}` | snapshot with fresh `view` |
| `POST /api/envs/<id>/action` | `{"action": int}` (illegal -> 400 "illegal") | snapshot |
| `POST /api/envs/<id>/agent` | `{}` brain plays the human seat | snapshot (4xx if no brain) |

Snapshot fields used by the frontend: `id, generation_status, generation_notes, generation_error, attempts, spec, validation, training_status (waiting|running|ready|failed|skipped), training_error, training_history [{gen, steps, best, mean, species, nodes, conns, win_rate?, wins?, losses?, draws?}], training_heldout, training_target_reached, training_opponent, concepts [expression], brain_ready, human_seat, opponent, view` (render payload returned by the sandbox for the current state), plus `ok, stats, warnings`.

## Legacy grid contract (V8, unchanged)
`POST /api/games {prompt}`, `GET /api/games/<id>/status`, `POST /api/games/<id>/reset|actions|agent`.

## Research viewer (V8, read-only)
`GET /research`, `GET /data/{concepts,experiments,demo_games}.json` (cached results; regenerate with `experiments/build_frontend_data.py`).

## Configuration
`LLM_API_KEY, LLM_MODEL, LLM_BASE_URL, LLM_REQUEST_TIMEOUT_SECONDS` (OpenAI-compatible endpoint); generated-env training uses `GAME_ENV_TARGET_WIN_RATE` (fraction, default 0.90), `GAME_ENV_EVAL_EPISODES` (default 100), `GAME_ENV_EVAL_INTERVAL_GENERATIONS` (default 5), and `GAME_ENV_MAX_STEPS` (default 0, no learner step cap; real-time games also use `SPEC.max_steps = 0` and terminate by game rules; `GAME_ENV_TRAINING_BUDGET` is a legacy fallback); `GAME_TRAINING_TIMEOUT` still limits wall time. Other learner settings: `GAME_TRAINING_BUDGET/POPULATION/TIMEOUT`, `GAME_CONCEPT_BANK`, `GAME_BRAIN_BIN`.

## Learner line protocol (internal, brain <-> relay)
`@NEW`, `@R <id> <seed>`, `@A <id> <action>`, `@DEL <id>`; replies `OK <id> <n_state> <n_actions>` / `S <reward> <done> <n_legal> <legal..> <n_state> <state..> <result>` / `ERR <msg>`. Result is `1` win, `-1` loss, `0` draw, or `2` unavailable. Non-`@` stdout lines are training progress and quality checkpoints.
