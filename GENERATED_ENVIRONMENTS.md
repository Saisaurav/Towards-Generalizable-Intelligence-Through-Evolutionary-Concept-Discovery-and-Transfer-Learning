# General-purpose generated environments

The LLM now writes **actual Python game logic** (not a JSON config of a grid template). The code is
validated in an isolated process and then trained on by the *unchanged* C++ NEAT+PPO learner through a
generic interface. The old grid pipeline (`/api/games`, `GameSpec`, `GridPolicy`) is untouched and still
available under the "Grid maze (legacy contract)" tab.

## 1. What the old architecture assumed

| Layer | Grid-specific assumption |
|---|---|
| `spec.py` / `generator.py` | LLM only emits a JSON `GameSpec` (width/height/walls/goal/hazards/…); the "game" is a fixed template |
| `runtime.py` | Fixed templates; only the generated *settings* vary |
| `policy.py` `GridPolicy` | Hard-coded grid features and exactly 4 actions (`legal_norm = n/4`, `action_norm = 2a/3-1`) |
| `server.py` / `index.html` | Responses carry grid cells; the page draws a maze and listens for arrow keys |
| `main.cpp` | `grid` mode builds `GridWorld` from a spec file; one learning agent, opponent hard-coded inside the C++ env |
| C++ `Env` / `ConceptEnv` / PPO / NEAT | **Already generic**: any fixed `n_state`, discrete `n_actions`, per-step legal list. Concepts are expressions over state statistics + `action_norm` + `legal_norm`, so they were never grid-specific. |

So the real blockers were Python-side (templated generation, grid policy, grid renderer) and the missing
bridge from an *arbitrary* environment (multi-agent, opponents) into the single-agent C++ `Env` interface.
No change to NEAT, PPO, genomes or the concept machinery was needed.

## 2. New architecture

```
natural language ──► EnvGenerator (LLM, repair loop) ──► env.py + meta.json (generated_envs/<id>/)
                          │  static_check (AST) → sandbox load → 24 random episodes → determinism
                          │  → learner path → human-play path;   errors go back to the LLM
                          ▼
                 sandbox_worker.py  (one subprocess per env instance; limits; restricted builtins)
                  ▲ JSON ops (validate, play, evaluate)        ▲ text "R seed"/"A action"
                  │                                            │
   server.py / EnvService (human play, renderer data)      EnvTrainer relay ◄── stdout/stdin ──► C++ `brain ext`
                                                                                 (ExternalEnv : Env, NEAT+PPO, concepts)
```

* **Environment logic** – generated `Environment` class (`env_interface.py` documents the contract).
* **Environment runtime** – `sandbox_worker.py` `Runner`: contract checks, turn-based truncation at `max_steps`,
  seat selection, opponents (random / rule-based / frozen policy), result bookkeeping.
* **Renderer** – `render()` returns a *scene* (`render_protocol.py`); `index.html` has a generic SVG
  renderer. Scenes never reach the learner; observations never reach the renderer.
* **Learner bridge** – `external_env.h` (`ExternalEnv`) implements the existing C++ `Env` interface by
  speaking a line protocol on the brain's own stdout/stdin; `env_trainer.py` relays each request to a
  sandbox worker. (Portable: C++ never spawns processes, so the same code works on Windows.)

"Every new game trains its own learner from scratch; only previously discovered concepts transfer" is
preserved: each `train_env` call starts a new `brain ext` process / new population. The only thing passed
in is the compatible concept list from `results/concept_bank.json` (`load_compatible_concepts`, which still
excludes the grid-only `state_boundary_margin` primitive). A test checks that identical seeds restart
identically and different seeds differ.

## 3. The environment interface

```python
SPEC = {"title", "description", "agents": [0, 1], "observation_size": N,
        "action": {"type": "discrete", "n": K}  # or continuous {dim, low, high}
        "turn_based": bool, "max_steps": int,  # turn-based: 1..10000; real-time: 0 (unlimited)
        # optional: real_time, learner_agent, symmetric_agents, noop_action, tick_ms, hold_last_action,
        #           controls, action_names, cooperative
       }
class Environment:
    reset(seed) · observe(agent) · legal_actions(agent) · active_agents() · step({agent: action}) -> {agent: reward}
    is_terminal() · result(agent) -> "win"|"loss"|"draw" · render() -> scene
    # optional: is_truncated() · scripted_action(agent) · clone()
```

* Turn-based: one active agent per step. Real-time/simultaneous: all agents are active and act in one `step`.
* Variable legal actions: `legal_actions` (action mask as an id list); the runtime refuses anything else.
* Rewards are per-agent dicts, so competitive, cooperative and self-play views all work.
* Termination vs truncation: `is_terminal()` is the game's own end; the runtime truncates turn-based
  episodes at `max_steps` (and honours optional `is_truncated()`); truncated episodes count as draws.
  Real-time games set `max_steps: 0` and end only through game rules, not a frame count.
* Deterministic seeding: `reset(seed)`; `random` inside generated code only exposes `random.Random`, and
  the validator replays an episode on the same and on a fresh instance and compares trajectory hashes.

## 4. How LLM generation works

`env_prompts.py` gives the model the interface contract, the sandbox rules, design guidance (perspective-
relative observations, normalisation, symmetric seats, scripted opponent hook, render protocol) and one
*unrelated* tiny example game (Coin Runner) – no template for any game the user may request. The user text
is passed as data. `env_generator.py` extracts the ```python block, validates it, and on failure sends the
precise errors (with generated-code line numbers) back for up to `max_attempts` (default 4) rounds.
On success it writes `generated_envs/<id>/env.py` + `meta.json` (prompt, model, attempts, spec, validation
report, SHA-256). `EnvStore.load` re-checks the hash and the server re-validates on reload.

## 5. Validation pipeline (what each stage checks)

1. AST static check: imports whitelist, forbidden names (`eval/exec/open/getattr/…`), dunder access, `SPEC` + required methods.
2. Sandboxed load: executes in a separate process, validates `SPEC` schema (`EnvSpec`).
3. 24 random-play episodes: observation length / finiteness, legal actions are ints in range and non-empty while running, rewards finite, `active_agents` consistent, `legal_actions == []` and `result()` valid at terminal, scenes valid (start, middle, end), illegal-action rejection (warning only; runtime blocks anyway).
4. Termination reachable; non-zero reward seen; every agent becomes active.
5. Determinism (same instance + fresh instance).
6. Learner path through the real brain-mode protocol (incl. seat randomisation) and human-play path.
7. Failure → `ValidationReport.feedback()` → repair prompt.

## 6. Isolation – what it is and is not

Separate process (`python -I`), scrubbed environment (no API keys), temp cwd, `RLIMIT_AS/CPU/FSIZE/NOFILE`
(POSIX), restricted builtins, import whitelist (no os/socket/subprocess/io), per-call `SIGALRM` timeout
plus a parent watchdog that kills the worker. Tested: infinite loop, 80 GB allocation, forbidden imports,
dunder escapes. **This is defence in depth, not a security boundary against a determined attacker** – a
Python-level sandbox can't be. For untrusted multi-tenant use run the worker in a container/seccomp/gVisor.
On Windows the rlimits and SIGALRM are unavailable; only the watchdog, scrubbed env and restricted
builtins apply (untested there).

## 7. Learner ↔ environment

For the learner the generated game is a single-agent `Env` (`PER_ACTION` mode: network scores
`[observation ‖ action concept features]` per legal action, so the number of legal actions may vary per
step). The worker picks the learner's seat (random each episode if `symmetric_agents`), plays all other
agents with the configured opponent until the learner must act, and returns the learner's accumulated reward,
the done flag, the learner's legal ids and the learner's observation.

Opponents: `random`; `rule_based` (the env's own `scripted_action`, else a generic win-now / avoid-immediate-loss
look-ahead on copies); `policy` (a frozen exported brain – used for staged self-play: each round a *fresh*
learner trains against the previous round's champion); and the human in the UI.

## 7b. Measured results (hand-written fixtures through the real pipeline, one CPU core, single seeds)

*Environment execution* (all pass; `test_generated_envs.py`): TTT, Snake and Pong validate first try in the
sandbox, obey their rules (unit-tested: 9 initial legal moves, occupied cells rejected, win/draw/no-moves-after-end;
snake movement/collision/food/growth; pong bounce/miss/score/termination), run through the C++ learner, and are
playable in the browser (jsdom smoke test: generate → board renders → click → training completes → brain moves).

*Agent training* (mean return from the learner's seat; greedy exported policy vs the same opponent type):

| Env | Budget | Opponent | Trained | Random baseline | Verdict |
|---|---|---|---|---|---|
| Tic-Tac-Toe | 600k steps | random | win 49% / loss 41%, return +0.08 | win 43% / loss 46%, return -0.03 | marginal |
| Tic-Tac-Toe | 2M steps | random | return -0.04 (sampled C++ held-out +0.11) | -0.03 | **not learned** |
| Tic-Tac-Toe | 200k steps | rule-based | return -0.82 | -0.73 | **not learned** |
| Snake | 2M steps | - | dies in 5 steps (return -1.01) | 14 steps (-0.96) | **not learned** |
| Pong | 250k steps (1 generation) | random | return -0.13 | +0.78 (60 episodes, noisy) | **not learned / inconclusive** |

Python policy inference matches the C++ evaluation (|diff| ≤ 0.02 over 3 seeds at 150k budget; unit test uses tolerance 0.15).
For reference the original built-in Tic-Tac-Toe reached held-out returns of -0.03 / 0.005 at 150k steps, so these numbers are
in line with the existing learner, not a regression. **Do not read this as "the learner plays Tic-Tac-Toe".**

Likely cause (not tested by an ablation): in per-action mode the network sees `[observation ‖ concept features]`, and the
only per-action information is the scalar `action_norm` (id rescaled to [-1,1]) inside concepts. Choosing "the empty cell
that blocks" or "the direction without a wall" needs the net to decode that scalar against the observation. That is the
existing design (and the `baseline_action_id` experiment condition); I did not change it because it would change what
transfers. A gated option (e.g. one-hot action features, or letting the env supply per-action features) is the obvious next experiment.

## 7c. Text mode / terminal player (no graphics)

The LLM also writes `render_text()` in the same `Environment` (the prompt asks for an ASCII picture of the state that
shows the numbers/names a player can type). It is validated like everything else (must return a string; a missing one is
only a warning and falls back to the scene's text items + HUD). Two ways to use it, both on the same sandboxed game:

```bash
python play_cli.py <env-id>                          # saved environment from generated_envs/
python play_cli.py generated_env_fixtures/ttt.py --opponent rule_based --seat 1
python play_cli.py <env-id> --opponent brain         # needs generated_envs/<id>/policies/brain.json
```
At the prompt type a move number or action name; `Enter` = default move (real-time games, via `noop_action`);
`b` = brain plays your move; `r` = restart; `q` = quit. In the browser tick **Text mode (no graphics)** to hide the
canvas and play the same text view with an input box.

Design choice: the "CLI app" is a generic runner over `render_text()`, not a second LLM-written program, so the generated
code never runs outside the sandbox and the validator covers both views. Real-time games (Snake, Pong) become
one-tick-per-input in the terminal, so they are playable but slow-paced.

## 8. Known limitations (read these)

* **Learner capability is unchanged.** The generalisation removes *interface* limits, not the limits of
  NEAT+PPO with the current concept bank and these budgets. See section 7b: none of the three acceptance games is
  learned to a useful level; Tic-Tac-Toe matches the original built-in game's weak result at the same budget. Arbitrary observations do not make arbitrary tasks learnable.
* Observations are fixed-length vectors (entity lists must be padded by the generated code). There is no
  variable-length/set encoder in the network.
* Discrete actions only for training (≤128 actions, a C++ buffer limit). Continuous action spaces run and validate
  but are reported as not trainable.
* Concepts using `state_boundary_margin` (grid geometry) are excluded; the other concepts are state statistics and
  action/legal normalisations, so what transfers is intentionally generic.
* No gradient bootstrapping on truncation (inherited from the existing PPO).
* Training uses 1 learner thread in `ext` mode (relay is serial): ≈10–15k env steps/s for simple games; slower for
  Pong-like long episodes and for the rule-based opponent (copies the env per candidate move).
* Opponent-in-training choices: random, rule-based, staged self-play. There is no simultaneous co-evolution of two learners.
* Real-time environments set `real_time: True` and `max_steps: 0`, use one `step()` per frame, and configure the frame interval with `tick_ms`. The learner supplies an action each frame; browser human/brain playback uses that interval. Older non-turn-based specs default to real-time behavior.
* The page has no *live* population view beyond per-generation fitness/species/node/connection counts parsed from the
  learner's log (the README already listed a full visualizer as not implemented).
* Python-level sandbox limits above; the generated code was never run against a **live** LLM in this repo's tests –
  tests use hand-written fixtures in `generated_env_fixtures/` that go through the identical extract→validate→store path.
  Run `python generate_env.py --acceptance ttt --train` with your API key to exercise a real model.

## 9. Running

```bash
cd src && make && cd ..              # builds brain (now with `ext` mode)
pip install -r requirements.txt
export LLM_API_KEY=...               # as before (see llm_game_platform/llm.py)
export GAME_ENV_TARGET_WIN_RATE=0.90     # stop when win rate reaches 90% (fraction, default 0.90)
export GAME_ENV_EVAL_EPISODES=100        # episodes per quality check (default 100)
export GAME_ENV_EVAL_INTERVAL_GENERATIONS=5  # generations between checks (default 5)
export GAME_ENV_MAX_STEPS=0              # optional cap; 0 means no step cap
# GAME_TRAINING_TIMEOUT still applies (default 900 seconds).
python -m llm_game_platform.server   # open the page; default tab = "Any game (generated Python)"
python generate_env.py --acceptance ttt --train      # CLI: generate + validate + train
python -m unittest test_generated_envs test_llm_game_platform   # ≈2 min on one core
cd src && make test-envs             # original C++ env smoke tests
```

## Real-time safety and learnability (Nov 2026)

- **Frame cap / stall truncation.** Every real-time episode is truncated at `SPEC["max_frames"]` (default 3000) and, if `SPEC["stall_frames"]` > 0, after that many frames without a new best cumulative return. Truncation is a draw, never a win, so a policy that learns to survive forever cannot hang training.
- **Quality metric.** Games with real win/loss outcomes keep win rate. Single-agent score games where no win is ever observed (Snake) are scored by *success rate*: an episode succeeds when its return is above a bar calibrated from random play (90th percentile, raised a quarter of the way to the scripted policy when that is clearly better). The UI labels it accordingly. Override with `GAME_ENV_QUALITY_MODE=result|return`.
- **Learnability gate (validator).** 60 calibration episodes compare random play with `scripted_action()` (or the generic look-ahead player for turn-based games). A reward under which competent play scores significantly *worse* than random is a validation error and goes back to the LLM for repair; "not clearly better", raw-grid observations, no `scripted_action` and always-capped episodes are warnings.
- **Training limits.** `GAME_ENV_MAX_GENERATIONS`, `GAME_ENV_MIN_GENERATIONS` (auto: 4 for real-time games with a step budget) and a soft wall-clock stop at 85% of the timeout. The soft stop is graceful: the champion is still exported instead of the run failing with a timeout. These are `brain ext` options (`--max-generations`, `--min-generations`, `--max-seconds`); rebuild `brain` / `brain.exe` to use them.
- **Fixed output head (`--head fixed`, `GAME_ENV_HEAD=auto|per_action|fixed`).** The default per-action scorer only tells actions apart by a scalar `action_norm`, which is a hard function for a minimal NEAT net when an action's meaning depends on the state (Snake headings, Pong paddle moves). The fixed head feeds the state (plus state-level concepts) once and reads one logit per action. `auto` = fixed for real-time games, per-action for turn-based ones. Concepts that use `action_norm`, `action_wins_now`, `action_blocks_opponent` or `transition_*` are dropped under the fixed head. The exported policy carries `"head"`, and `NeatPolicy` mirrors it. Measured on the Snake fixture (one seed each): per-action head about -1.0 return for 3M steps (no learning); fixed head reached +5.2 (greedy, Python) after about 280 s / 100 generations, scripted +25, random -1.0. Pong did not show learning in 18 generations (about 9M steps).
