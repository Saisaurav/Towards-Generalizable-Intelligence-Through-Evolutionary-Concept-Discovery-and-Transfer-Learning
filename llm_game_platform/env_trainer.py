"""Trains a fresh NEAT+PPO brain on a generated environment.

The C++ ``brain ext`` process asks for environment transitions over its
stdout/stdin; this relay forwards each request to one sandbox worker per Env
instance (the learner clones its env for worker threads and evaluators).
Every call to ``train`` starts a brand-new learner. The only knowledge reused
across games is the list of compatible GP concept expressions, exactly as for
the built-in games and the grid prototype.
"""

from __future__ import annotations

import json
import math
import os
import re
import secrets
import subprocess
import tempfile
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable

from . import learnability
from .neat_policy import NeatPolicy
from .policy import load_compatible_concepts
from .sandbox import SandboxError, SandboxProcess
from .trainer import BrainTrainer, BrainTrainingError

_PROGRESS = re.compile(
    r"gen\s+(\d+)\s+steps\s+(\d+)\s+best\s+(-?[\d.]+)\s+mean\s+(-?[\d.]+)\s+sp\s+(\d+)\s+nodes\s+([\d.]+)\s+conns\s+([\d.]+)"
)
_HELDOUT = re.compile(r"HELDOUT return\s+(-?[\d.]+)")
_QUALITY = re.compile(
    r"QUALITY gen\s+(\d+)\s+steps\s+(\d+)\s+win_rate\s+([\d.]+)\s+"
    r"wins\s+(\d+)\s+losses\s+(\d+)\s+draws\s+(\d+)\s+mean_return\s+(-?[\d.]+)"
)
_TARGET_REACHED = re.compile(r"TARGET_REACHED\s+([01])")


@dataclass
class EnvTrainResult:
    model: dict[str, Any]
    history: list[dict[str, float]]
    heldout_return: float | None
    concepts: list[str]
    seed: int
    opponent: str
    policy: NeatPolicy = field(repr=False, default=None)  # type: ignore[assignment]
    target_reached: bool = False
    # "result": win rate from real game outcomes. "return": success = episode return beats the
    # calibrated random-play bar (score-style games such as Snake, where "win" is unreachable).
    quality_mode: str = "result"
    success_return: float | None = None
    calibration: dict[str, Any] | None = None

    def save_policy(self, path: Path) -> Path:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(self.model), encoding="utf-8")
        return path


def opponent_label(opponent: dict[str, Any] | str) -> str:
    return opponent if isinstance(opponent, str) else str(opponent.get("kind", "random"))


class EnvTrainer(BrainTrainer):
    """Train generated games until a win-rate target or optional step cap is reached."""

    def __init__(
        self,
        brain_binary: Path | None = None,
        concept_bank: Path | None = None,
        budget: int | None = None,
        population: int | None = None,
        timeout: int | None = None,
        target_win_rate: float | None = None,
        evaluation_episodes: int | None = None,
        evaluation_interval_generations: int | None = None,
        max_generations: int | None = None,
        min_generations: int | None = None,
        max_seconds: float | None = None,
        quality: str | None = None,
        head: str | None = None,
    ) -> None:
        if budget is None:
            raw_budget = os.getenv(
                "GAME_ENV_MAX_STEPS", os.getenv("GAME_ENV_TRAINING_BUDGET", "0")
            )
            try:
                budget = int(raw_budget)
            except ValueError as error:
                raise ValueError("GAME_ENV_MAX_STEPS must be an integer") from error
        if budget < 0 or budget > 5_000_000:
            raise ValueError("environment training step cap must be between 0 and 5000000")
        super().__init__(
            brain_binary=brain_binary,
            concept_bank=concept_bank,
            budget=max(1, budget),
            population=population,
            timeout=timeout,
        )
        self.budget = budget
        if target_win_rate is None:
            try:
                target_win_rate = float(os.getenv("GAME_ENV_TARGET_WIN_RATE", "0.90"))
            except ValueError as error:
                raise ValueError("GAME_ENV_TARGET_WIN_RATE must be a number from 0 to 1") from error
        if isinstance(target_win_rate, bool) or not math.isfinite(target_win_rate) or not 0 < target_win_rate <= 1:
            raise ValueError("target_win_rate must be greater than 0 and at most 1")
        self.target_win_rate = target_win_rate
        self.evaluation_episodes = (
            evaluation_episodes if evaluation_episodes is not None
            else self._setting("GAME_ENV_EVAL_EPISODES", 100, 1, 10000)
        )
        self.evaluation_interval_generations = (
            evaluation_interval_generations
            if evaluation_interval_generations is not None
            else self._setting("GAME_ENV_EVAL_INTERVAL_GENERATIONS", 5, 1, 10000)
        )
        self.max_generations = (
            max_generations if max_generations is not None else self._setting("GAME_ENV_MAX_GENERATIONS", 0, 0, 100000)
        )
        # None => auto: real-time games with a step budget get a few generations minimum, because one
        # generation of long real-time episodes can already exceed a small step budget.
        self.min_generations = (
            min_generations if min_generations is not None
            else (self._setting("GAME_ENV_MIN_GENERATIONS", 0, 0, 100000) if os.getenv("GAME_ENV_MIN_GENERATIONS") else None)
        )
        self.max_seconds = max_seconds  # None => 85% of the hard timeout (graceful stop, model is kept)
        self.quality = quality or os.getenv("GAME_ENV_QUALITY_MODE", "auto")
        # "per_action": net scores each legal action from [state | action features] (original design).
        # "fixed": one output per action, state-only input (see src/fixed_head_env.h). auto => fixed for real-time games.
        self.head = head or os.getenv("GAME_ENV_HEAD", "auto")
        if self.head not in ("auto", "per_action", "fixed"):
            raise ValueError("head must be 'auto', 'per_action' or 'fixed'")
        if self.quality not in ("auto", "result", "return"):
            raise ValueError("quality must be 'auto', 'result' or 'return'")
        if self.evaluation_episodes < 1 or self.evaluation_episodes > 10000:
            raise ValueError("evaluation_episodes must be between 1 and 10000")
        if self.evaluation_interval_generations < 1 or self.evaluation_interval_generations > 10000:
            raise ValueError("evaluation_interval_generations must be between 1 and 10000")

    def train_env(
        self,
        source_path: Path,
        spec: dict[str, Any],
        opponent: dict[str, Any] | str = "random",
        seed: int | None = None,
        progress: Callable[[dict[str, float]], None] | None = None,
        concepts: list[str] | None = None,
    ) -> EnvTrainResult:
        if not spec.get("trainable"):
            raise BrainTrainingError(spec.get("untrainable_reason") or "environment is not trainable")
        opponent_cfg = {"kind": opponent} if isinstance(opponent, str) else dict(opponent)
        concepts = list(concepts) if concepts is not None else load_compatible_concepts(self.concept_bank)
        binary = self._binary()
        mode, bar, calibration = self._quality_plan(source_path, spec, opponent_cfg)
        min_gens = self.min_generations
        if min_gens is None:
            min_gens = 4 if (spec.get("real_time") and self.budget > 0) else 0
        head = self.head if self.head != "auto" else ("fixed" if spec.get("real_time") else "per_action")
        soft_seconds = self.max_seconds if self.max_seconds is not None else self.timeout * 0.85
        seed = seed if seed is not None else secrets.randbelow(2**31 - 1) + 1
        history: list[dict[str, float]] = []
        heldout: float | None = None
        target_reached = False
        errors: list[str] = []
        workers: dict[int, SandboxProcess] = {}
        with tempfile.TemporaryDirectory(prefix="env-training-") as directory:
            workdir = Path(directory)
            (workdir / "concepts.txt").write_text("\n".join(concepts) + "\n", encoding="ascii")
            model_path = workdir / "brain-policy.json"
            command = [str(binary), "ext", str(workdir / "concepts.txt"), str(self.budget), str(seed),
                       str(self.population), str(model_path), "--target-win-rate", str(self.target_win_rate),
                       "--eval-episodes", str(self.evaluation_episodes),
                       "--eval-interval", str(self.evaluation_interval_generations),
                       "--max-seconds", f"{soft_seconds:.1f}"]
            if head == "fixed":
                command += ["--head", "fixed"]
            if self.max_generations:
                command += ["--max-generations", str(self.max_generations)]
            if min_gens:
                command += ["--min-generations", str(min_gens)]
            brain = subprocess.Popen(command, cwd=workdir, stdin=subprocess.PIPE, stdout=subprocess.PIPE,
                                     stderr=subprocess.PIPE)
            stderr_lines: list[str] = []
            threading.Thread(target=lambda: stderr_lines.extend(
                l.decode("utf-8", "replace").rstrip() for l in brain.stderr), daemon=True).start()
            timed_out = threading.Event()

            def watchdog() -> None:
                deadline = time.monotonic() + self.timeout
                while brain.poll() is None:
                    if time.monotonic() > deadline:
                        timed_out.set()
                        brain.kill()
                        return
                    time.sleep(0.25)

            threading.Thread(target=watchdog, daemon=True).start()
            next_id = 0

            def reply(text: str) -> None:
                brain.stdin.write(text.encode("utf-8") + b"\n")  # type: ignore[union-attr]
                brain.stdin.flush()  # type: ignore[union-attr]

            try:
                while True:
                    raw = brain.stdout.readline()  # type: ignore[union-attr]
                    if not raw:
                        break
                    line = raw.decode("utf-8", "replace").rstrip()
                    if not line.startswith("@"):
                        match = _PROGRESS.search(line)
                        if match:
                            g, steps, best, mean, species, nodes, conns = match.groups()
                            point = {"gen": int(g), "steps": int(steps), "best": float(best), "mean": float(mean),
                                     "species": int(species), "nodes": float(nodes), "conns": float(conns)}
                            history.append(point)
                            if progress:
                                progress(point)
                        held = _HELDOUT.search(line)
                        if held:
                            heldout = float(held.group(1))
                        quality = _QUALITY.search(line)
                        if quality:
                            g, steps, win_rate, wins, losses, draws, mean_return = quality.groups()
                            point = next((p for p in reversed(history) if int(p["gen"]) == int(g)), None)
                            if point is not None:
                                point.update({
                                    "win_rate": float(win_rate), "wins": int(wins), "losses": int(losses),
                                    "draws": int(draws), "quality_return": float(mean_return),
                                    "quality_steps": int(steps),
                                })
                                if progress:
                                    progress(dict(point))
                        reached = _TARGET_REACHED.search(line)
                        if reached:
                            target_reached = reached.group(1) == "1"
                        continue
                    parts = line.split()
                    try:
                        if parts[0] == "@NEW":
                            worker = SandboxProcess(source_path, cpu_seconds=self.timeout + 60)
                            worker.request("configure", default=opponent_cfg, success_return=bar)
                            workers[next_id] = worker
                            reply(f"OK {next_id} {spec['observation_size']} {spec['action']['n']}")
                            next_id += 1
                        elif parts[0] == "@R":
                            reply(workers[int(parts[1])].raw_line(f"R {parts[2]}"))
                        elif parts[0] == "@A":
                            reply(workers[int(parts[1])].raw_line(f"A {parts[2]}"))
                        elif parts[0] == "@DEL":
                            worker = workers.pop(int(parts[1]), None)
                            if worker:
                                worker.close()
                        else:
                            reply("ERR unknown relay command")
                    except SandboxError as error:
                        errors.append(error.describe())
                        reply("ERR " + error.describe().replace("\n", " | "))
                    except (BrokenPipeError, OSError):
                        break
            finally:
                for worker in workers.values():
                    worker.close()
                try:
                    brain.stdin.close()  # type: ignore[union-attr]
                except OSError:
                    pass
                try:
                    brain.wait(timeout=30)
                except subprocess.TimeoutExpired:
                    brain.kill()
                    brain.wait()
                for stream in (brain.stdout, brain.stderr):
                    try:
                        stream.close()  # type: ignore[union-attr]
                    except OSError:
                        pass
            if timed_out.is_set():
                raise BrainTrainingError(f"brain training exceeded the {self.timeout}-second time limit")
            if brain.returncode != 0:
                detail = errors[-1] if errors else ("\n".join(stderr_lines[-5:]) or f"exit status {brain.returncode}")
                raise BrainTrainingError(f"brain training failed: {detail[-2000:]}")
            try:
                model = json.loads(model_path.read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError) as error:
                raise BrainTrainingError(f"brain did not produce a valid policy: {error}") from error
            model["concepts"] = concepts
            try:
                policy = NeatPolicy(model, spec["action"]["n"])
            except ValueError as error:
                raise BrainTrainingError(f"brain exported an unusable policy: {error}") from error
        return EnvTrainResult(
            model, history, heldout, concepts, seed, opponent_label(opponent_cfg), policy, target_reached,
            mode, bar, _calibration_summary(calibration),
        )

    def _quality_plan(self, source_path: Path, spec: dict[str, Any], opponent_cfg: dict[str, Any]):
        """Decide how progress is scored and, for return-based quality, the success bar."""
        single = len(spec["agents"]) == 1 or bool(spec.get("cooperative"))
        if self.quality == "result" or (self.quality == "auto" and not single):
            return "result", None, None
        episodes = self._setting("GAME_ENV_CALIBRATION_EPISODES", 100, 10, 2000)
        try:
            with SandboxProcess(source_path, cpu_seconds=self.timeout + 60) as sandbox:
                sandbox.request("configure", default=opponent_cfg)
                calibration = learnability.run_calibration(sandbox, episodes=episodes)
        except SandboxError as error:
            raise BrainTrainingError(f"baseline calibration failed: {error.describe()}") from error
        mode = learnability.quality_mode(spec, calibration, self.quality)
        if mode == "result":
            return "result", None, calibration
        return "return", learnability.success_bar(calibration), calibration

    def train_self_play(
        self,
        source_path: Path,
        spec: dict[str, Any],
        policy_dir: Path,
        rounds: int = 2,
        first_opponent: dict[str, Any] | str = "random",
        progress: Callable[[dict[str, float]], None] | None = None,
        concepts: list[str] | None = None,
    ) -> list[EnvTrainResult]:
        """Round 1 trains against ``first_opponent``; each later round starts a FRESH learner
        that trains against the frozen champion of the previous round (staged self-play)."""
        results: list[EnvTrainResult] = []
        opponent: dict[str, Any] | str = first_opponent
        for index in range(rounds):
            result = self.train_env(source_path, spec, opponent, progress=progress, concepts=concepts)
            path = result.save_policy(policy_dir / f"selfplay_round_{index + 1}.json")
            results.append(result)
            opponent = {"kind": "policy", "policy_path": str(path), "greedy": False}
        return results


def _calibration_summary(calibration: dict[str, Any] | None) -> dict[str, Any] | None:
    if calibration is None:
        return None
    scripted = calibration["scripted"]
    return {
        "random_mean_return": calibration["random"]["mean_return"],
        "random_mean_length": calibration["random"]["mean_length"],
        "scripted_mean_return": scripted["mean_return"] if scripted else None,
    }


def evaluate_policy(
    source_path: Path,
    learner: dict[str, Any] | str,
    opponent: dict[str, Any] | str = "random",
    episodes: int = 200,
    seed: int = 12345,
    call_timeout: float = 60.0,
) -> dict[str, float]:
    """Mean return and win/draw/loss rates of ``learner`` (a controller config such as
    {"kind": "policy", "policy_path": ..., "greedy": False}, or "random"/"rule_based")
    against ``opponent``, measured from the learner's seat."""
    as_cfg = lambda value: {"kind": value} if isinstance(value, str) else value  # noqa: E731
    with SandboxProcess(source_path, call_timeout=call_timeout) as sandbox:
        sandbox.request("configure", default=as_cfg(opponent))
        response = sandbox.request("evaluate", learner=as_cfg(learner), episodes=episodes, seed=seed)
    response.pop("ok", None)
    response.pop("returns", None)
    return response
