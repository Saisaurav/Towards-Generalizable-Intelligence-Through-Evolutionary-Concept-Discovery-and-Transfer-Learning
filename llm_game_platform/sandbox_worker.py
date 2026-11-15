#!/usr/bin/env python3
"""Sandbox worker: loads ONE generated environment in its own process and serves it.

Run as a script by ``sandbox.SandboxProcess`` (never imported into the main app):

    python -I sandbox_worker.py <env.py> [--mem-mb N] [--cpu-seconds N] [--call-timeout S]

Protocol on stdin/stdout, one request per line:
  * lines starting with '{'  -> JSON request, one JSON response line.
  * lines starting with 'R' / 'A' -> compact text "brain mode" used by the C++
    learner (via the trainer relay):  "R <seed>" reset, "A <action>" step.
    Response: "S <reward> <done> <n_legal> <legal...> <n_state> <state...>" or "ERR <msg>".

Isolation (best effort, see GENERATED_ENVIRONMENTS.md): separate process with a
scrubbed environment, RLIMIT_AS/CPU/FSIZE/NOFILE where the OS supports them,
no filesystem/network/process modules importable by the generated code,
restricted builtins, per-call SIGALRM timeout, parent-side watchdog.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import math
import os
import random as _random
import signal
import sys
import time
import traceback
import types

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

import render_protocol  # noqa: E402  (stdlib only)
from env_interface import RESULT_VALUES, EnvSpec  # noqa: E402

ALLOWED_IMPORTS = {
    "math", "random", "copy", "collections", "itertools", "functools", "dataclasses",
    "enum", "typing", "heapq", "bisect", "statistics", "operator", "fractions", "abc",
}
_SAFE_NAMES = (
    "abs all any bin bool callable chr dict divmod enumerate filter float format frozenset hex "
    "int isinstance issubclass iter len list map max min next object ord pow property range repr "
    "reversed round set slice sorted staticmethod classmethod str sum super tuple zip type "
    "Exception ValueError RuntimeError TypeError IndexError KeyError ZeroDivisionError "
    "StopIteration AssertionError NotImplementedError ArithmeticError OverflowError "
    "AttributeError LookupError NotImplemented Ellipsis True False None"
).split()


class EnvContractError(Exception):
    """The environment violated the interface contract (message is shown to the LLM)."""


class _CallTimeout(BaseException):
    pass


def _alarm(_signum, _frame):
    raise _CallTimeout()


def _safe_import(name, globals=None, locals=None, fromlist=(), level=0):
    root = name.split(".")[0]
    if level or root not in ALLOWED_IMPORTS:
        raise ImportError(f"import of '{name}' is not allowed in generated environments")
    if root == "random":
        proxy = types.ModuleType("random")
        proxy.Random = _random.Random  # module-level functions are deliberately absent
        return proxy
    return __import__(name, globals, locals, fromlist, level)


def _safe_print(*args, **kwargs):  # stdout is the protocol channel
    print(*args, file=sys.stderr)


def _build_builtins():
    import builtins

    safe = {n: getattr(builtins, n) for n in _SAFE_NAMES if hasattr(builtins, n)}
    safe["__build_class__"] = builtins.__build_class__
    safe["__import__"] = _safe_import
    safe["print"] = _safe_print
    return safe


def apply_limits(mem_mb: int, cpu_seconds: int) -> None:
    try:
        import resource
    except ImportError:  # Windows: rely on the parent watchdog
        return
    for name, value in (
        ("RLIMIT_AS", mem_mb * 1024 * 1024),
        ("RLIMIT_CPU", cpu_seconds),
        ("RLIMIT_FSIZE", 0),
        ("RLIMIT_NOFILE", 32),
        ("RLIMIT_CORE", 0),
    ):
        limit = getattr(resource, name, None)
        if limit is None:
            continue
        try:
            resource.setrlimit(limit, (value, value))
        except (ValueError, OSError):
            pass


def _generated_frames(error: BaseException, source_lines: list[str]) -> str:
    frames = []
    for frame in traceback.extract_tb(error.__traceback__):
        if frame.filename == "<generated_env>":
            text = source_lines[frame.lineno - 1].strip() if 0 < frame.lineno <= len(source_lines) else ""
            frames.append(f"  line {frame.lineno} in {frame.name}: {text}")
    return "\n".join(frames)


def _fmt(values) -> str:
    return " ".join(f"{float(v):.9g}" for v in values)


class Runner:
    """Runtime around a generated Environment: validation, truncation, controllers."""

    def __init__(self, env_cls, spec: EnvSpec) -> None:
        self.env_cls, self.spec = env_cls, spec
        self.env = env_cls()
        self.steps = 0
        self.truncated = False
        self.rng = _random.Random(0)
        self.default_controller = ("random",)
        self.controllers: dict[int, tuple] = {}
        self.learner = spec.learner_agent
        self.human: int | None = None
        self.brain = None
        self.last_rewards: dict[int, float] = {}
        self.log: list[dict] = []

    # ---- contract-checked primitives ------------------------------------
    def reset(self, seed) -> None:
        self.env.reset(seed)
        self.steps, self.truncated, self.last_rewards = 0, False, {}
        self.rng.seed((seed or 0) * 7919 + 13)

    def done(self) -> bool:
        if self.env.is_terminal():
            return True
        if self.steps >= self.spec.max_steps or (
            hasattr(self.env, "is_truncated") and self.env.is_truncated()
        ):
            self.truncated = True
            return True
        return False

    def active(self) -> list[int]:
        if self.env.is_terminal():
            return []
        agents = self.env.active_agents()
        try:
            agents = [int(a) for a in agents]
        except (TypeError, ValueError):
            raise EnvContractError("active_agents() must return a list of integer agent ids")
        if not agents:
            if self.done():
                return []
            raise EnvContractError("active_agents() returned [] but is_terminal() is False")
        for a in agents:
            if a not in self.spec.agents:
                raise EnvContractError(f"active_agents() returned {a}, which is not in SPEC.agents")
        return agents

    def legal(self, agent: int) -> list[int]:
        if not self.spec.discrete:
            return []
        raw = self.env.legal_actions(agent)
        try:
            legal = [int(a) for a in raw]
        except (TypeError, ValueError):
            raise EnvContractError("legal_actions() must return a list of integers")
        if len(set(legal)) != len(legal):
            raise EnvContractError(f"legal_actions({agent}) contains duplicates: {legal}")
        for a in legal:
            if not 0 <= a < self.spec.n_actions:
                raise EnvContractError(
                    f"legal_actions({agent}) returned {a}, outside 0..{self.spec.n_actions - 1}"
                )
        return legal

    def observe(self, agent: int) -> list[float]:
        raw = self.env.observe(agent)
        try:
            obs = [float(v) for v in raw]
        except (TypeError, ValueError):
            raise EnvContractError("observe() must return a flat list of numbers")
        if len(obs) != self.spec.observation_size:
            raise EnvContractError(
                f"observe({agent}) returned {len(obs)} values but SPEC.observation_size is {self.spec.observation_size}"
            )
        if any(not math.isfinite(v) or abs(v) > 1e4 for v in obs):
            raise EnvContractError(f"observe({agent}) contains NaN/inf/huge values (normalise features to about [-1, 1])")
        return obs

    def step(self, actions: dict[int, object]) -> dict[int, float]:
        if self.done():
            raise EnvContractError("step() called after the episode ended; call reset()")
        active = self.active()
        if set(actions) != set(active):
            raise EnvContractError(f"actions must be provided for exactly the active agents {active}, got {sorted(actions)}")
        clean: dict[int, object] = {}
        for agent, action in actions.items():
            if self.spec.discrete:
                if isinstance(action, bool) or not isinstance(action, int):
                    raise EnvContractError(f"action for agent {agent} must be an int, got {action!r}")
                legal = self.legal(agent)
                if action not in legal:
                    raise EnvContractError(f"illegal action {action} for agent {agent}; legal actions: {legal}")
                clean[agent] = action
            else:
                vec = list(action) if isinstance(action, (list, tuple)) else [action]
                if len(vec) != self.spec.n_actions or any(
                    isinstance(v, bool) or not isinstance(v, (int, float)) or not math.isfinite(v) for v in vec
                ):
                    raise EnvContractError(f"continuous action for agent {agent} must be {self.spec.n_actions} finite numbers")
                clean[agent] = [min(self.spec.action_high, max(self.spec.action_low, float(v))) for v in vec]
        raw = self.env.step(clean)
        rewards: dict[int, float] = {a: 0.0 for a in self.spec.agents}
        if raw is not None:
            if not isinstance(raw, dict):
                raise EnvContractError("step() must return a {agent_id: reward} dict (or None)")
            for key, value in raw.items():
                if key not in rewards:
                    raise EnvContractError(f"step() returned a reward for unknown agent {key!r}")
                if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or abs(value) > 1000:
                    raise EnvContractError(f"reward for agent {key} must be a finite number with |r| <= 1000, got {value!r}")
                rewards[key] = float(value)
        self.steps += 1
        self.last_rewards = rewards
        return rewards

    def result(self, agent: int) -> str:
        if not self.env.is_terminal():
            return "draw"  # truncated episodes count as draws
        value = self.env.result(agent)
        if value not in RESULT_VALUES:
            raise EnvContractError(f"result({agent}) must be one of {RESULT_VALUES} once terminal, got {value!r}")
        return value

    # ---- controllers ------------------------------------------------------
    def _clone(self, env):
        return env.clone() if hasattr(env, "clone") else copy.deepcopy(env)

    def _random_action(self, agent):
        if self.spec.discrete:
            return self.rng.choice(self.legal(agent))
        return [self.rng.uniform(self.spec.action_low, self.spec.action_high) for _ in range(self.spec.n_actions)]

    def _simulate(self, env, agent, action):
        actions = {agent: action}
        for other in env.active_agents():
            if other != agent:
                actions[other] = self.rng.choice(list(env.legal_actions(other)))
        return env.step(actions) or {}

    def rule_based(self, agent: int):
        legal = self.legal(agent)
        if not self.spec.discrete or len(legal) <= 1:
            return legal[0] if legal else self._random_action(agent)
        if hasattr(self.env, "scripted_action"):
            action = self.env.scripted_action(agent)
            if action in legal:
                return action
        turn_based = len(self.active()) == 1
        best, best_score = [], -1e18
        for action in legal:
            sim = self._clone(self.env)
            rewards = self._simulate(sim, agent, action)
            score = float(rewards.get(agent, 0.0))
            if sim.is_terminal():
                outcome = sim.result(agent)
                score += {"win": 1000.0, "loss": -1000.0}.get(outcome, 0.0)
            elif turn_based:
                nxt = list(sim.active_agents())
                if len(nxt) == 1 and nxt[0] != agent:  # can the opponent win next move?
                    for reply in sim.legal_actions(nxt[0]):
                        sim2 = self._clone(sim)
                        sim2.step({nxt[0]: reply})
                        if sim2.is_terminal() and sim2.result(agent) == "loss":
                            score -= 500.0
                            break
            if score > best_score + 1e-9:
                best, best_score = [action], score
            elif abs(score - best_score) <= 1e-9:
                best.append(action)
        return self.rng.choice(best)

    def choose(self, agent: int):
        kind = self.controllers.get(agent, self.default_controller)
        if kind[0] == "rule_based":
            return self.rule_based(agent)
        if kind[0] == "policy" and self.spec.discrete:
            return kind[1].choose(self.observe(agent), self.legal(agent), self.rng, greedy=kind[2])
        return self._random_action(agent)

    def set_controller(self, config: dict | None):
        config = config or {"kind": "random"}
        kind = config.get("kind", "random")
        if kind == "policy":
            from neat_policy import NeatPolicy

            policy = NeatPolicy.from_file(config["policy_path"], self.spec.n_actions)
            return ("policy", policy, bool(config.get("greedy", False)))
        if kind not in ("random", "rule_based"):
            raise EnvContractError(f"unknown controller kind {kind!r}")
        return (kind,)

    # ---- learner (brain-mode) episode management ---------------------------
    def advance(self, focus: int) -> float:
        """Run controllers until `focus` must act (or the episode is done)."""
        if self.spec.real_time:
            if not self.done() and focus not in self.active():
                raise EnvContractError(
                    "real-time environments must include the learner in active_agents() every frame"
                )
            return 0.0
        total = 0.0
        while not self.done() and focus not in self.active():
            actions = {a: self.choose(a) for a in self.active()}
            rewards = self.step(actions)
            self.log.append({"agents": sorted(actions), "actions": actions})
            total += rewards.get(focus, 0.0)
        return total

    def learner_reset(self, seed: int) -> None:
        for attempt in range(20):
            self.reset(seed + attempt)
            self.learner = self.rng.choice(self.spec.agents) if self.spec.symmetric_agents else self.spec.learner_agent
            self.advance(self.learner)
            if not self.done():
                return
        raise EnvContractError("episode ended before the learner could act (checked 20 resets)")

    def learner_step(self, action: int) -> tuple[float, bool]:
        active = self.active()
        if self.spec.real_time and self.learner not in active:
            raise EnvContractError(
                "real-time environments must include the learner in active_agents() every frame"
            )
        actions = {a: self.choose(a) for a in active if a != self.learner}
        actions[self.learner] = action
        total = self.step(actions)[self.learner]
        if not self.spec.real_time:
            total += self.advance(self.learner)
        return total, self.done()

    def brain_line(self, reward: float, done: bool) -> str:
        obs = self.observe(self.learner)
        legal = [] if done else self.legal(self.learner)
        outcome = {"win": 1, "loss": -1, "draw": 0}[self.result(self.learner)] if done else 2
        return (
            f"S {reward:.9g} {int(done)} {len(legal)} {' '.join(map(str, legal))} "
            f"{len(obs)} {_fmt(obs)} {outcome}"
        )

    def evaluate(self, learner_cfg: dict | None, episodes: int, seed: int) -> dict:
        """Play full episodes with `learner_cfg` as the learner against the configured opponents."""
        ctrl = self.set_controller(learner_cfg)
        total, counts, lengths = 0.0, {"win": 0, "loss": 0, "draw": 0}, 0
        for episode in range(episodes):
            self.learner_reset(seed + episode * 1009)
            while not self.done():
                if ctrl[0] == "policy":
                    action = ctrl[1].choose(self.observe(self.learner), self.legal(self.learner), self.rng, greedy=ctrl[2])
                elif ctrl[0] == "rule_based":
                    action = self.rule_based(self.learner)
                else:
                    action = self._random_action(self.learner)
                reward, _done = self.learner_step(action)
                total += reward
                lengths += 1
            counts[self.result(self.learner)] += 1
        return {"mean_return": total / episodes, "mean_length": lengths / episodes,
                **{k: v / episodes for k, v in counts.items()}}

    # ---- human / UI play ---------------------------------------------------
    def view(self) -> dict:
        scene, render_error = None, None
        try:
            scene = render_protocol.validate_scene(self.env.render())
        except render_protocol.SceneError as error:
            render_error = str(error)
        done = self.done()
        active = [] if done else self.active()
        view = {
            "scene": scene, "render_error": render_error, "terminal": done,
            "truncated": self.truncated and not self.env.is_terminal(),
            "steps": self.steps, "active": active, "human_seat": self.human,
            "legal": self.legal(self.human) if self.human in active else [],
            "rewards": {str(k): v for k, v in self.last_rewards.items()},
            "results": {str(a): self.result(a) for a in self.spec.agents} if done else None,
            "moves": self.log[-8:],
            "text": self.text_view(),
        }
        self.log = []
        return view

    def text_view(self) -> str:
        """Plain-text picture of the state for the CLI / text mode."""
        if hasattr(self.env, "render_text"):
            text = self.env.render_text()
            if not isinstance(text, str):
                raise EnvContractError("render_text() must return a string")
            return text[:4000]
        # Fallback: summarise the scene's text items (no graphics needed).
        try:
            scene = render_protocol.validate_scene(self.env.render())
        except render_protocol.SceneError:
            return ""
        texts = sorted((i for i in scene["items"] if i["type"] == "text"), key=lambda i: (round(i["y"]), i["x"]))
        return "\n".join(filter(None, [" ".join(t["text"] for t in texts), scene.get("hud", "")]))[:4000]

    def human_reset(self, seed: int, seat: int | None) -> dict:
        self.reset(seed)
        self.human = seat if seat in self.spec.agents else self.spec.agents[0]
        self.advance(self.human)
        return self.view()

    def human_play(self, action, use_brain: bool = False) -> dict:
        if self.human is None:
            raise EnvContractError("reset the game first")
        if self.done():
            raise EnvContractError("the game is already finished; restart it")
        active = self.active()
        if self.human not in active:
            raise EnvContractError("it is not the human seat's turn")
        if use_brain:
            if self.brain is None:
                raise EnvContractError("the brain is not trained yet")
            action = self.brain.choose(self.observe(self.human), self.legal(self.human), self.rng, greedy=True)
        elif action is None:
            if self.spec.noop_action is None:
                raise EnvContractError("this game needs an explicit action")
            action = self.spec.noop_action
        actions = {a: self.choose(a) for a in active if a != self.human}
        actions[self.human] = action
        self.step(actions)
        self.advance(self.human)
        view = self.view()
        view["action"] = action
        return view

    def _check_clone(self, issues: list) -> None:
        """clone() must be an independent, equivalent copy (used by look-ahead opponents)."""
        copy_env = self.env.clone()
        for agent in self.spec.agents:
            if copy_env.observe(agent) != self.env.observe(agent):
                raise EnvContractError(f"clone() produced a different observe({agent}); it must copy all game state")
        before = [self.env.observe(a) for a in self.spec.agents]
        active = list(copy_env.active_agents())
        if active and self.spec.discrete and not copy_env.is_terminal():
            copy_env.step({a: copy_env.legal_actions(a)[0] for a in active})
            if [self.env.observe(a) for a in self.spec.agents] != before:
                raise EnvContractError("stepping the clone() changed the original environment; clone() must not share mutable state (lists, dicts, sets)")

    # ---- validator episode -------------------------------------------------
    def validate_episode(self, seed: int, action_seed: int, check_illegal: bool) -> dict:
        rng = _random.Random(action_seed)
        digest = hashlib.sha256()
        issues: list[dict] = []
        self.reset(seed)
        render_protocol.validate_scene(self.env.render())
        total = {a: 0.0 for a in self.spec.agents}
        nonzero = False
        acting = set()
        tested_illegal = False
        while not self.done():
            active = self.active()
            actions = {}
            for agent in active:
                digest.update(repr(self.observe(agent)).encode())
                if self.spec.discrete:
                    legal = self.legal(agent)
                    if not legal:
                        raise EnvContractError(f"agent {agent} has no legal actions while the game is not terminal")
                    actions[agent] = rng.choice(legal)
                    if check_illegal and not tested_illegal:
                        outside = [x for x in range(self.spec.n_actions) if x not in legal]
                        if outside:
                            tested_illegal = True
                            probe = self._clone(self.env)
                            try:
                                probe.step({agent: outside[0]})
                                issues.append({"level": "warning", "message": f"Environment.step() silently accepted illegal action {outside[0]}; it should raise ValueError (the runtime blocks it anyway)"})
                            except Exception:
                                pass
                else:
                    actions[agent] = [rng.uniform(self.spec.action_low, self.spec.action_high) for _ in range(self.spec.n_actions)]
                acting.add(agent)
            if hasattr(self.env, "clone") and not self.done() and self.steps in (0, 1):
                self._check_clone(issues)
            rewards = self.step(actions)
            digest.update(repr(sorted(rewards.items())).encode())
            for agent, value in rewards.items():
                total[agent] += value
                nonzero = nonzero or value != 0.0
            if self.steps == max(1, self.spec.max_steps // 2):
                render_protocol.validate_scene(self.env.render())
        terminal = bool(self.env.is_terminal())
        results = None
        if terminal:
            if self.env.active_agents():
                raise EnvContractError("active_agents() must return [] once is_terminal() is True")
            for agent in self.spec.agents:
                if self.spec.discrete and self.legal(agent):
                    raise EnvContractError(f"legal_actions({agent}) must be [] once the game is terminal")
            results = {str(a): self.result(a) for a in self.spec.agents}
            probe = self._clone(self.env)
            try:
                probe.step({self.spec.agents[0]: 0 if self.spec.discrete else [0.0] * self.spec.n_actions})
                issues.append({"level": "warning", "message": "Environment.step() did not raise after the game ended; the runtime blocks it anyway"})
            except Exception:
                pass
        for agent in self.spec.agents:  # terminal observation must still be well-formed
            digest.update(repr(self.observe(agent)).encode())
        render_protocol.validate_scene(self.env.render())
        if hasattr(self.env, "render_text"):
            self.text_view()
        return {
            "steps": self.steps, "terminal": terminal, "truncated": self.truncated and not terminal,
            "results": results, "total_rewards": {str(k): v for k, v in total.items()},
            "nonzero_reward": nonzero, "digest": digest.hexdigest(), "issues": issues,
            "agents_that_acted": sorted(acting),
        }


class Server:
    def __init__(self, source_path: str, call_timeout: float) -> None:
        self.source_path = source_path
        self.call_timeout = call_timeout
        self.source_lines: list[str] = []
        self.runner: Runner | None = None
        self.env_cls = None
        self.spec: EnvSpec | None = None

    def load(self) -> dict:
        with open(self.source_path, encoding="utf-8") as handle:
            source = handle.read()
        self.source_lines = source.splitlines()
        namespace = {"__builtins__": _build_builtins(), "__name__": "generated_env"}
        exec(compile(source, "<generated_env>", "exec"), namespace)
        raw_spec = namespace.get("SPEC")
        if raw_spec is None:
            raise EnvContractError("module-level SPEC dict is missing")
        try:
            raw_spec = json.loads(json.dumps(raw_spec))
        except (TypeError, ValueError):
            raise EnvContractError("SPEC must contain only JSON-serialisable literals")
        self.spec = EnvSpec.from_mapping(raw_spec)
        self.env_cls = namespace.get("Environment")
        if not isinstance(self.env_cls, type):
            raise EnvContractError("class Environment is missing")
        self.runner = Runner(self.env_cls, self.spec)
        spec_dict = self.spec.to_dict()
        spec_dict["has_render_text"] = hasattr(self.env_cls, "render_text")
        return {"ok": True, "event": "loaded", "spec": spec_dict}

    def error(self, error: BaseException) -> dict:
        if isinstance(error, _CallTimeout):
            return {"ok": False, "kind": "timeout", "error": f"environment call exceeded {self.call_timeout:g}s (infinite loop or far too slow?)", "frames": ""}
        if isinstance(error, EnvContractError):
            return {"ok": False, "kind": "contract", "error": str(error), "frames": _generated_frames(error, self.source_lines)}
        if isinstance(error, MemoryError):
            return {"ok": False, "kind": "memory", "error": "environment exceeded its memory limit", "frames": ""}
        return {
            "ok": False, "kind": "exception",
            "error": f"{type(error).__name__}: {error}",
            "frames": _generated_frames(error, self.source_lines),
        }

    def handle_json(self, request: dict) -> dict:
        op = request.get("op")
        r = self.runner
        if op == "configure":
            r.default_controller = r.set_controller(request.get("default"))
            r.controllers = {int(k): r.set_controller(v) for k, v in (request.get("controllers") or {}).items()}
            if request.get("brain"):
                from neat_policy import NeatPolicy

                r.brain = NeatPolicy.from_file(request["brain"]["policy_path"], self.spec.n_actions)
            return {"ok": True}
        if op == "reset":
            return {"ok": True, **r.human_reset(int(request.get("seed", 0)), request.get("seat"))}
        if op == "play":
            return {"ok": True, **r.human_play(request.get("action"), bool(request.get("brain")))}
        if op == "view":
            return {"ok": True, **r.view()}
        if op == "validate_episode":
            runner = Runner(self.env_cls, self.spec) if request.get("fresh") else r
            return {"ok": True, **runner.validate_episode(int(request["seed"]), int(request["action_seed"]), bool(request.get("check_illegal")))}
        if op == "evaluate":
            return {"ok": True, **r.evaluate(request.get("learner"), int(request.get("episodes", 100)), int(request.get("seed", 0)))}
        if op == "quit":
            raise SystemExit(0)
        raise EnvContractError(f"unknown op {op!r}")

    def handle_text(self, line: str) -> str:
        parts = line.split()
        r = self.runner
        if parts[0] == "R":
            r.learner_reset(int(parts[1]))
            return r.brain_line(0.0, False)
        if parts[0] == "A":
            reward, done = r.learner_step(int(parts[1]))
            return r.brain_line(reward, done)
        raise EnvContractError(f"bad brain-mode command {line!r}")

    def serve(self, out) -> None:
        use_alarm = hasattr(signal, "setitimer")
        if use_alarm:
            signal.signal(signal.SIGALRM, _alarm)
        for line in sys.stdin:
            line = line.strip()
            if not line:
                continue
            text_mode = line[0] != "{"
            try:
                if use_alarm:
                    signal.setitimer(signal.ITIMER_REAL, self.call_timeout)
                if text_mode:
                    response = self.handle_text(line)
                else:
                    response = json.dumps(self.handle_json(json.loads(line)))
            except SystemExit:
                return
            except BaseException as error:  # noqa: BLE001 - report everything to the parent
                payload = self.error(error)
                response = ("ERR " + payload["error"].replace("\n", " ") + " " + payload["frames"].replace("\n", " | ")) if text_mode else json.dumps(payload)
            finally:
                if use_alarm:
                    signal.setitimer(signal.ITIMER_REAL, 0)
            out.write(response + "\n")
            out.flush()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("source")
    parser.add_argument("--mem-mb", type=int, default=1024)
    parser.add_argument("--cpu-seconds", type=int, default=600)
    parser.add_argument("--call-timeout", type=float, default=10.0)
    args = parser.parse_args()
    out = sys.stdout
    sys.stdout = sys.stderr  # stray prints must never corrupt the protocol channel
    try:
        sys.stdin.reconfigure(encoding="utf-8")
    except AttributeError:
        pass
    server = Server(args.source, args.call_timeout)
    apply_limits(args.mem_mb, args.cpu_seconds)
    try:
        if hasattr(signal, "setitimer"):
            signal.signal(signal.SIGALRM, _alarm)
            signal.setitimer(signal.ITIMER_REAL, args.call_timeout)
        loaded = server.load()
        if hasattr(signal, "setitimer"):
            signal.setitimer(signal.ITIMER_REAL, 0)
    except BaseException as error:  # noqa: BLE001
        out.write(json.dumps(server.error(error)) + "\n")
        out.flush()
        return
    out.write(json.dumps(loaded) + "\n")
    out.flush()
    server.serve(out)


if __name__ == "__main__":
    main()
