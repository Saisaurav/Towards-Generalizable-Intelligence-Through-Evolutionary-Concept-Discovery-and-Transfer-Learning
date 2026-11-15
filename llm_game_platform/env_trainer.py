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

from .neat_policy import NeatPolicy
from .online_concepts import discover_concepts
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
_GENERATION_LIMIT_REACHED = re.compile(r"GENERATION_LIMIT_REACHED\s+([01])")


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
    generation_limit_reached: bool = False
    concepts_added: list[str] = field(default_factory=list)
    concept_refresh_note: str | None = None

    def save_policy(self, path: Path) -> Path:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(self.model), encoding="utf-8")
        return path


def opponent_label(opponent: dict[str, Any] | str) -> str:
    return opponent if isinstance(opponent, str) else str(opponent.get("kind", "random"))


class EnvTrainer(BrainTrainer):
    """Train generated games to a target, with one optional generation-20 feature refresh."""

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
        concept_refresh_generation: int | None = None,
        concept_refresh_episodes: int = 16,
        concept_refresh_max_steps: int = 96,
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
        if population is None:
            population = self._setting(
                "GAME_ENV_TRAINING_POPULATION",
                self._setting("GAME_TRAINING_POPULATION", 3, 2, 128),
                2,
                128,
            )
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
        if self.evaluation_episodes < 1 or self.evaluation_episodes > 10000:
            raise ValueError("evaluation_episodes must be between 1 and 10000")
        if self.evaluation_interval_generations < 1 or self.evaluation_interval_generations > 10000:
            raise ValueError("evaluation_interval_generations must be between 1 and 10000")
        self.concept_refresh_generation = (
            concept_refresh_generation if concept_refresh_generation is not None
            else self._setting("GAME_ENV_CONCEPT_REFRESH_GENERATION", 20, 0, 10000)
        )
        self.concept_refresh_episodes = concept_refresh_episodes
        self.concept_refresh_max_steps = concept_refresh_max_steps
        if not 1 <= self.concept_refresh_episodes <= 64:
            raise ValueError("concept_refresh_episodes must be between 1 and 64")
        if not 1 <= self.concept_refresh_max_steps <= 10000:
            raise ValueError("concept_refresh_max_steps must be between 1 and 10000")

    def train_env(
        self,
        source_path: Path,
        spec: dict[str, Any],
        opponent: dict[str, Any] | str = "random",
        seed: int | None = None,
        progress: Callable[[dict[str, float]], None] | None = None,
        concepts: list[str] | None = None,
        status: Callable[[str], None] | None = None,
        _allow_concept_refresh: bool = True,
    ) -> EnvTrainResult:
        if not spec.get("trainable"):
            raise BrainTrainingError(spec.get("untrainable_reason") or "environment is not trainable")
        opponent_cfg = {"kind": opponent} if isinstance(opponent, str) else dict(opponent)
        concepts = list(concepts) if concepts is not None else load_compatible_concepts(self.concept_bank)
        binary = self._binary()
        seed = seed if seed is not None else secrets.randbelow(2**31 - 1) + 1
        history: list[dict[str, float]] = []
        heldout: float | None = None
        target_reached = False
        generation_limit_reached = False
        errors: list[str] = []
        workers: dict[int, SandboxProcess] = {}
        refresh_at = (
            self.concept_refresh_generation
            if _allow_concept_refresh else 0
        )
        with tempfile.TemporaryDirectory(prefix="env-training-") as directory:
            workdir = Path(directory)
            (workdir / "concepts.txt").write_text("\n".join(concepts) + "\n", encoding="ascii")
            model_path = workdir / "brain-policy.json"
            command = [str(binary), "ext", str(workdir / "concepts.txt"), str(self.budget), str(seed),
                       str(self.population), str(model_path), "--target-win-rate", str(self.target_win_rate),
                       "--eval-episodes", str(self.evaluation_episodes),
                       "--eval-interval", str(self.evaluation_interval_generations)]
            if refresh_at:
                command.extend(["--max-generations", str(refresh_at)])
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
                                win_count, loss_count, draw_count = int(wins), int(losses), int(draws)
                                point.update({
                                    "win_rate": float(win_rate),
                                    "win_or_draw_rate": (win_count + draw_count) / self.evaluation_episodes,
                                    "wins": win_count, "losses": loss_count, "draws": draw_count,
                                    "quality_return": float(mean_return),
                                    "quality_steps": int(steps),
                                })
                                if progress:
                                    progress(dict(point))
                        reached = _TARGET_REACHED.search(line)
                        if reached:
                            target_reached = reached.group(1) == "1"
                        generation_limit = _GENERATION_LIMIT_REACHED.search(line)
                        if generation_limit:
                            generation_limit_reached = generation_limit.group(1) == "1"
                        continue
                    parts = line.split()
                    try:
                        if parts[0] == "@NEW":
                            worker = SandboxProcess(source_path, cpu_seconds=self.timeout + 60)
                            worker.request("configure", default=opponent_cfg)
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
        result = EnvTrainResult(
            model, history, heldout, concepts, seed, opponent_label(opponent_cfg), policy,
            target_reached, generation_limit_reached
        )
        if not (refresh_at and generation_limit_reached and not target_reached):
            return result

        refresh_deadline = time.monotonic() + self.timeout
        if status:
            status(
                f"Generation {refresh_at} reached below the {self.target_win_rate:.0%} target; "
                "collecting game episodes for concept discovery."
            )
        try:
            with tempfile.TemporaryDirectory(prefix="env-concept-refresh-") as directory:
                policy_path = Path(directory) / "stage-one-policy.json"
                result.save_policy(policy_path)
                with SandboxProcess(
                    source_path, cpu_seconds=self.timeout + 60, call_timeout=120.0
                ) as sandbox:
                    sandbox.request("configure", default=opponent_cfg)
                    collected = sandbox.request(
                        "collect_concept_data",
                        learner={"kind": "policy", "policy_path": str(policy_path), "greedy": False},
                        episodes=self.concept_refresh_episodes,
                        seed=seed ^ 0x5EEDC0DE,
                        max_steps=self.concept_refresh_max_steps,
                    )["episodes"]
            discovered = discover_concepts(
                collected, spec["action"]["n"], concepts, seed ^ 0xC0FFEE
            )
        except (OSError, RuntimeError, ValueError, SandboxError) as error:
            raise BrainTrainingError(f"automatic concept discovery failed: {error}") from error

        combined_concepts = list(concepts)
        combined_concepts.extend(expression for expression in discovered if expression not in combined_concepts)
        note = (
            f"Generation {refresh_at}: discovered {len(discovered)} new concepts; "
            "restarted training with the combined concept set."
            if discovered else
            f"Generation {refresh_at}: no new concepts passed validation; "
            "restarted training with the current concept set."
        )
        if status:
            status(note)

        remaining = refresh_deadline - time.monotonic()
        if remaining < 1:
            raise BrainTrainingError(
                "automatic concept discovery used the available time; no time remained for the restarted learner"
            )
        second_trainer = EnvTrainer(
            brain_binary=binary,
            concept_bank=self.concept_bank,
            budget=self.budget,
            population=self.population,
            timeout=max(1, math.ceil(remaining)),
            target_win_rate=self.target_win_rate,
            evaluation_episodes=self.evaluation_episodes,
            evaluation_interval_generations=self.evaluation_interval_generations,
            concept_refresh_generation=0,
            concept_refresh_episodes=self.concept_refresh_episodes,
            concept_refresh_max_steps=self.concept_refresh_max_steps,
        )
        generation_offset = max((int(point["gen"]) for point in history), default=-1) + 1
        step_offset = max((int(point["steps"]) for point in history), default=0)

        def second_progress(point: dict[str, float]) -> None:
            if progress:
                adjusted = dict(point)
                adjusted["gen"] += generation_offset
                adjusted["steps"] += step_offset
                progress(adjusted)

        second_seed = seed + 1 if seed is not None else result.seed + 1
        restarted = second_trainer.train_env(
            source_path,
            spec,
            opponent_cfg,
            seed=second_seed,
            progress=second_progress,
            concepts=combined_concepts,
            status=status,
            _allow_concept_refresh=False,
        )
        restarted_history = []
        for point in restarted.history:
            adjusted = dict(point)
            adjusted["gen"] += generation_offset
            adjusted["steps"] += step_offset
            restarted_history.append(adjusted)
        return EnvTrainResult(
            restarted.model,
            history + restarted_history,
            restarted.heldout_return,
            combined_concepts,
            restarted.seed,
            restarted.opponent,
            restarted.policy,
            restarted.target_reached,
            False,
            discovered,
            note,
        )

    def train_self_play(
        self,
        source_path: Path,
        spec: dict[str, Any],
        policy_dir: Path,
        rounds: int = 2,
        first_opponent: dict[str, Any] | str = "random",
        progress: Callable[[dict[str, float]], None] | None = None,
        concepts: list[str] | None = None,
        status: Callable[[str], None] | None = None,
    ) -> list[EnvTrainResult]:
        """Round 1 trains against ``first_opponent``; each later round starts a FRESH learner
        that trains against the frozen champion of the previous round (staged self-play)."""
        results: list[EnvTrainResult] = []
        opponent: dict[str, Any] | str = first_opponent
        for index in range(rounds):
            result = self.train_env(
                source_path, spec, opponent, progress=progress, concepts=concepts, status=status
            )
            path = result.save_policy(policy_dir / f"selfplay_round_{index + 1}.json")
            results.append(result)
            opponent = {"kind": "policy", "policy_path": str(path), "greedy": False}
        return results


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
    return response
