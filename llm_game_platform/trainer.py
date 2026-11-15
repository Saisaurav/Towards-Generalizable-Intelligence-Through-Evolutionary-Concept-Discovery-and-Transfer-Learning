"""Runs the existing C++ brain on a validated grid contract and exports its policy."""

import json
import os
import secrets
import subprocess
import tempfile
from pathlib import Path

from .policy import GridPolicy, load_compatible_concepts
from .spec import GameSpec


_REPO_ROOT = Path(__file__).resolve().parents[1]


class BrainTrainingError(RuntimeError):
    pass


class BrainTrainer:
    def __init__(
        self,
        brain_binary: Path | None = None,
        concept_bank: Path | None = None,
        budget: int | None = None,
        population: int | None = None,
        timeout: int | None = None,
    ) -> None:
        configured_binary = os.getenv("GAME_BRAIN_BIN")
        self.brain_binary = Path(configured_binary) if configured_binary else brain_binary
        self.concept_bank = concept_bank or Path(
            os.getenv("GAME_CONCEPT_BANK", str(_REPO_ROOT / "results" / "concept_bank.json"))
        )
        self.budget = budget if budget is not None else self._setting(
            "GAME_TRAINING_BUDGET", 10000, 1, 5_000_000
        )
        self.population = population if population is not None else self._setting(
            "GAME_TRAINING_POPULATION", 24, 2, 128
        )
        self.timeout = timeout if timeout is not None else self._setting(
            "GAME_TRAINING_TIMEOUT", 900, 1, 3600
        )
        if self.budget < 1 or self.population < 2 or self.timeout < 1:
            raise ValueError("training budget, population, and timeout must be positive")

    @staticmethod
    def _setting(name: str, default: int, minimum: int, maximum: int) -> int:
        raw = os.getenv(name)
        try:
            value = default if raw is None else int(raw)
        except ValueError as error:
            raise ValueError(f"{name} must be an integer") from error
        if value < minimum or value > maximum:
            raise ValueError(f"{name} must be between {minimum} and {maximum}")
        return value

    def _binary(self) -> Path:
        if self.brain_binary is not None:
            candidates = [self.brain_binary]
        else:
            candidates = [_REPO_ROOT / "src" / "brain.exe", _REPO_ROOT / "src" / "brain"]
        for candidate in candidates:
            if candidate.is_file():
                return candidate.resolve()
        raise BrainTrainingError(
            "brain executable not found; build it from src before starting game training"
        )

    @staticmethod
    def _write_grid_spec(path: Path, spec: GameSpec) -> None:
        lines = [
            f"{spec.width} {spec.height} {spec.max_turns} "
            f"{spec.start[0]} {spec.start[1]} {spec.goal[0]} {spec.goal[1]}",
            str(len(spec.walls)),
            *(f"{x} {y}" for x, y in spec.walls),
            str(len(spec.hazards)),
            *(f"{x} {y}" for x, y in spec.hazards),
        ]
        path.write_text("\n".join(lines) + "\n", encoding="ascii")

    def train(self, spec: GameSpec) -> GridPolicy:
        concepts = load_compatible_concepts(self.concept_bank)
        binary = self._binary()
        with tempfile.TemporaryDirectory(prefix="llm-game-training-") as directory:
            workdir = Path(directory)
            spec_path = workdir / "grid-spec.txt"
            concept_path = workdir / "concepts.txt"
            model_path = workdir / "brain-policy.json"
            self._write_grid_spec(spec_path, spec)
            concept_path.write_text("\n".join(concepts) + "\n", encoding="ascii")
            seed = secrets.randbelow(2**31 - 1) + 1
            command = [
                str(binary),
                "grid",
                str(spec_path),
                str(concept_path),
                str(self.budget),
                str(seed),
                str(self.population),
                str(model_path),
            ]
            try:
                result = subprocess.run(
                    command,
                    cwd=workdir,
                    stdout=subprocess.DEVNULL,
                    stderr=subprocess.PIPE,
                    text=True,
                    timeout=self.timeout,
                    check=False,
                )
            except subprocess.TimeoutExpired as error:
                raise BrainTrainingError(
                    f"brain training exceeded the {self.timeout}-second time limit"
                ) from error
            except OSError as error:
                raise BrainTrainingError(f"could not start brain training: {error}") from error
            if result.returncode != 0:
                detail = result.stderr.strip()[-2000:] or f"exit status {result.returncode}"
                raise BrainTrainingError(f"brain training failed: {detail}")
            try:
                model = json.loads(model_path.read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError) as error:
                raise BrainTrainingError(f"brain did not produce a valid policy: {error}") from error
            if not isinstance(model, dict):
                raise BrainTrainingError("brain policy export must be a JSON object")
            model["concepts"] = concepts
            try:
                return GridPolicy(model)
            except ValueError as error:
                raise BrainTrainingError(f"brain exported an unusable policy: {error}") from error
