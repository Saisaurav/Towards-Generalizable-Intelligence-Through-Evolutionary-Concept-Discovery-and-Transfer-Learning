# General-purpose generated environments

The LLM now writes **actual Python game logic** (not a JSON config of a grid template). The code is
validated in an isolated process and trained by the C++ NEAT+PPO learner through a generic interface.
The learner supports fixed-output discrete policies for real-time games, and its standalone CLI also
supports Gaussian continuous-action PPO for the PointMass research proxy. The existing online concept
refresh and staged-training path remain in place. The old grid pipeline (`/api/games`, `GameSpec`,
`GridPolicy`) is untouched and still available under the "Grid maze (legacy contract)" tab.

## 1. What the old architecture assumed

| Layer | Grid-specific assumption |
|---|---|
| `spec.py` / `generator.py` | LLM only emits a JSON `GameSpec` (width/height/walls/goal/hazards/…); the "game" is a fixed template |
| `runtime.py` | Fixed templates; only the generated *settings* vary |
| `policy.py` `GridPolicy` | Hard-coded grid features and exactly 4 actions (`legal_norm = n/4`, `action_norm = 2a/3-1`) |
| `server.py` / `index.html` | Responses carry grid cells; the page draws a maze and listens for arrow keys |
| `main.cpp` | `grid` mode builds `GridWorld` from a spec file; one learning agent, opponent hard-coded inside the C++ env |
| C++ `Env` / `ConceptEnv` / PPO / NEAT | Generic state and discrete-action interfaces, extended with fixed-output heads for real-time games and Gaussian continuous-action PPO for the standalone PointMass proxy. Existing concept expressions and online refresh remain available. |

The original blockers were Python-side (templated generation, grid policy, grid renderer) and the missing
bridge from an *arbitrary* environment (multi-agent, opponents) into the single-agent C++ `Env` interface.
The learner extensions add fixed discrete action heads and continuous-action training without replacing
the current concept generator or its staged/online discovery workflow.

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

For the learner the generated game is a single-agent `Env`. Turn-based environments use a `PER_ACTION`
head that scores `[observation ‖ action concept features]` separately for each legal action, so the number
of legal actions may vary per step. Discrete real-time games use a fixed output per action and choose among
the current legal ids each frame; action-dependent and transition concepts are excluded from that head.
When an environment supplies a valid `scripted_action(agent_id)`, its legal action is used to guide
training rollouts and as an auxiliary PPO imitation target. Evaluation and fitness selection still use
the learned policy against the game itself. The worker picks the learner's seat (random each episode if `symmetric_agents`), plays all other agents
with the configured opponent until the learner acts, and returns the learner's reward, done flag, legal
ids and observation.

Opponents: `random`; `rule_based` (the env's own `scripted_action`, else a generic win-now / avoid-immediate-loss
look-ahead on copies); `policy` (a frozen exported brain – used for staged self-play: each round a *fresh*
learner trains against the previous round's champion); and the human in the UI.

If the target win rate is still unmet after generation 20, generated-game training collects episodes from
that exact environment, runs a bounded GP search over policy-supported state/action summaries, and starts
one fresh learner with the old and newly screened concepts. This refresh is attempted once; its discovery
and restarted training share one additional timeout window. If no candidates pass the train/validation
correlation and output-diversity checks, the learner still restarts with the existing concepts and the
status explains that no new features were admitted.

Snake is an exception to win-rate stopping: filling the entire board is an impractical default target.
Training instead ends at its configured step budget; if no positive budget is supplied, it is capped at
60 generations. This makes the run bounded, not evidence that the learned policy is strong. The recorded
Snake performance below remains poor and must be remeasured after generating and training a corrected
real-time Snake environment.

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
in line with the existing learner, not a regression. These results predate the generated-game fixes below and are not measurements of the updated setup. **Do not read this as "the learner plays Tic-Tac-Toe".**

The low generated-game results exposed two training mismatches, now corrected: an opponent-only terminal reward previously
left the learner with zero reward instead of a loss penalty, and the exported brain's action inputs did not match the
stronger action-identity experiment. Generated games now append one-hot action identity after concept features in both
C++ training and Python inference. Their defaults use the measured TTT experiment's 16-unit hidden layer, four PPO
iterations, and greedy fitness/held-out evaluation; generated-game population defaults to 3 (override with
`GAME_ENV_TRAINING_POPULATION`). The historical table above is retained as a pre-fix baseline; the post-fix single-seed
check is in section 7c and is not enough to establish multi-seed performance.

## 7c. Post-fix Tic-Tac-Toe check (single seed)

Using the updated `src/brain.exe`, the hand-written TTT fixture, a 150k-step budget, population 3, seed 11, and no
concept refresh, the greedy exported policy was evaluated for 1,000 episodes against random with seed 3001. It won
82.5%, lost 13.0%, drew 4.5%, and returned +0.695. Random-vs-random on the same seeds won 43.0%, lost 45.0%, drew
12.0%, and returned -0.020. The 200-episode training checkpoint was 81.5% wins / 87.0% wins-or-draws, so the strict
90% win-only stopping target was not reached. This is a one-seed sanity check, not a multi-seed benchmark.

## 7d. Text mode / terminal player (no graphics)

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

* **Updated generated-game training has not yet been benchmarked at scale.** Section 7b's low scores predate the
  terminal-reward, action-identity, and PPO-configuration fixes above. Rerun the benchmark before drawing conclusions
  about the updated setup. The generalisation removes *interface* limits, not the limits of NEAT+PPO; arbitrary
  observations do not make arbitrary tasks learnable.
* Observations are fixed-length vectors (entity lists must be padded by the generated code). There is no
  variable-length/set encoder in the network.
* Generated environments currently train with discrete actions (≤128 actions, a C++ buffer limit).
  Continuous-action training is available only through the standalone PointMass robotics proxy.
* Concepts using `state_boundary_margin` (grid geometry) are excluded; the other concepts are state statistics and
  action/legal normalisations, so what transfers is intentionally generic.
* No gradient bootstrapping on truncation (inherited from the existing PPO).
* Training uses 1 learner thread in `ext` mode (relay is serial): ≈10–15k env steps/s for simple games; slower for
  Pong-like long episodes and for the rule-based opponent (copies the env per candidate move).
* Opponent-in-training choices: random, rule-based, staged self-play. There is no simultaneous co-evolution of two learners.
* Real-time environments set `real_time: True` and `max_steps: 0`, use one `step()` per frame, and configure the frame interval with `tick_ms`. The browser's **Play frames** control steps a human-played game at that interval; keyboard controls steer each frame, and `hold_last_action` determines whether a direction persists after the key is released. Brain autoplay uses the same frame interval. Older non-turn-based specs default to real-time behavior.
* To keep evaluation bounded when a real-time policy can avoid ending a game, generated-game training truncates each episode after 256 frames (counted as a draw); this does not limit browser play.
* The standalone `pointmass` CLI task is a robotics research proxy, not a simulator or physical robot
  integration. It uses Gaussian continuous-action PPO, optional observation-difference/action stacking,
  persisted optimizer moments, and held-out noise/delay/mass robustness checks. Example:

  ```bash
  cd src
  ./brain pointmass concepts.txt 200000 11 24 pointmass-policy.json --stack 4 --robust 100
  ```
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
export GAME_ENV_MAX_STEPS=0              # optional training-step cap; 0 means no step cap
export GAME_ENV_CONCEPT_REFRESH_GENERATION=20  # one GP refresh/restart if target is unmet
# GAME_TRAINING_TIMEOUT is 900 seconds per phase by default; refresh allows one extra phase.
python -m llm_game_platform.server   # open the page; default tab = "Any game (generated Python)"
python generate_env.py --acceptance ttt --train      # CLI: generate + validate + train
python -m unittest test_generated_envs test_llm_game_platform   # ≈2 min on one core
cd src && make test-envs             # original C++ env smoke tests
```
