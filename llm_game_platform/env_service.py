"""Server-side lifecycle of generated environments: generate -> validate -> train -> play."""

from __future__ import annotations

import threading
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any
from uuid import uuid4

from .env_generator import EnvGenerationError, EnvGenerator
from .env_store import EnvStore, EnvStoreError
from .env_trainer import EnvTrainer
from .env_validator import ValidationReport, validate_environment
from .llm import LLMConfigurationError, LLMRequestError
from .sandbox import SandboxError, SandboxProcess
from .trainer import BrainTrainingError

OPPONENTS = ("random", "rule_based", "brain")
TRAINING_OPPONENTS = ("random", "rule_based", "self_play")


class EnvSourceValidationError(RuntimeError):
    def __init__(self, report: ValidationReport) -> None:
        self.report = report
        super().__init__("edited environment failed sandbox validation: " + "; ".join(report.errors))


class EnvNotReady(RuntimeError):
    pass


@dataclass
class EnvSession:
    id: str
    prompt: str = ""
    training_opponent: str = "random"
    generation_status: str = "queued"
    generation_notes: list[str] = field(default_factory=list)
    generation_error: str | None = None
    attempts: int = 0
    spec: dict[str, Any] | None = None
    validation: dict[str, Any] | None = None
    source_path: Path | None = None
    training_status: str = "waiting"
    training_error: str | None = None
    history: list[dict[str, float]] = field(default_factory=list)
    heldout: float | None = None
    target_reached: bool | None = None
    training_note: str | None = None
    concepts_added: list[str] = field(default_factory=list)
    quality_mode: str = "result"
    success_return: float | None = None
    concepts: list[str] = field(default_factory=list)
    policy_path: Path | None = None
    opponent: str = "random"
    human_seat: int | None = None
    view: dict[str, Any] | None = None
    sandbox: SandboxProcess | None = None
    lock: threading.RLock = field(default_factory=threading.RLock)


class EnvService:
    def __init__(self, generator: EnvGenerator, trainer: EnvTrainer, store: EnvStore) -> None:
        self.generator, self.trainer, self.store = generator, trainer, store
        self._sessions: dict[str, EnvSession] = {}
        self._lock = threading.Lock()

    # ---- lifecycle -----------------------------------------------------------
    def create_pending(self, prompt: str, training_opponent: str = "random") -> EnvSession:
        if not isinstance(training_opponent, str) or training_opponent not in TRAINING_OPPONENTS:
            raise ValueError(f"training_opponent must be one of {', '.join(TRAINING_OPPONENTS)}")
        session = EnvSession(id=uuid4().hex, prompt=prompt, training_opponent=training_opponent)
        with self._lock:
            self._sessions[session.id] = session
        return session

    def get(self, env_id: str) -> EnvSession:
        with self._lock:
            try:
                return self._sessions[env_id]
            except KeyError:
                raise KeyError("environment not found") from None

    def generate(self, session_id: str) -> None:
        """Generate and validate the editable environment source."""
        session = self.get(session_id)
        with session.lock:
            session.generation_status = "running"
        try:
            generated = self.generator.generate(session.prompt, progress=lambda m: self._note(session, m), env_id=session.id)
        except (EnvGenerationError, LLMConfigurationError, LLMRequestError, ValueError) as error:
            with session.lock:
                session.generation_status, session.generation_error = "failed", str(error)
                session.training_status = "skipped"
            return
        with session.lock:
            self._adopt(session, generated.env_id, generated.spec, generated.report.to_dict(), generated.attempts)
            session.training_status = "waiting"

    def _adopt(self, session: EnvSession, env_id: str, spec: dict, report: dict, attempts: int) -> None:
        session.spec, session.validation, session.attempts = spec, report, attempts
        session.source_path = self.store.source_path(env_id)
        session.generation_status = "ready"

    def prepare_training(self, env_id: str, source: object, training_opponent: object = "random") -> dict[str, Any]:
        if not isinstance(source, str) or not source.strip():
            raise ValueError("environment source must not be empty")
        if len(source) > 50_000:
            raise ValueError("environment source must be at most 50000 characters")
        if not isinstance(training_opponent, str) or training_opponent not in TRAINING_OPPONENTS:
            raise ValueError(f"training_opponent must be one of {', '.join(TRAINING_OPPONENTS)}")
        session = self.get(env_id)
        with session.lock:
            self._ready(session)
            if session.training_status in {"validating", "queued", "running"}:
                raise EnvNotReady("training is already in progress")
            session.training_status = "validating"

        try:
            report = validate_environment(source)
            if not report.ok:
                raise EnvSourceValidationError(report)
            spec = report.spec or {}
            if not spec.get("trainable"):
                raise ValueError(spec.get("untrainable_reason") or "the environment is not trainable")

            _, metadata = self.store.load(env_id)
            with session.lock:
                if session.sandbox is not None:
                    session.sandbox.close()
                    session.sandbox = None
            self.store.save(
                env_id,
                prompt=session.prompt,
                source=source,
                spec=spec,
                report=report.to_dict(),
                attempts=session.attempts,
                model=metadata.get("model"),
            )
            policy_path = self.store.path(env_id) / "policies" / "brain.json"
            if policy_path.is_file():
                policy_path.unlink()
            with session.lock:
                session.spec = spec
                session.validation = report.to_dict()
                session.source_path = self.store.source_path(env_id)
                session.training_opponent = training_opponent
                session.training_status = "queued"
                session.training_error = None
                session.history.clear()
                session.heldout = None
                session.target_reached = None
                session.training_note = None
                session.concepts_added.clear()
                session.concepts.clear()
                session.policy_path = None
                session.view = None
                return self.snapshot(env_id)
        except EnvSourceValidationError:
            with session.lock:
                session.training_status = "waiting"
            raise
        except (EnvStoreError, OSError, SandboxError):
            with session.lock:
                session.training_status = "waiting"
            raise
        except ValueError:
            with session.lock:
                session.training_status = "waiting"
            raise

    def load_saved(self, env_id: str) -> EnvSession:
        """Re-open a previously generated environment from disk (source hash checked)."""
        with self._lock:
            existing = self._sessions.get(env_id)
        if existing is not None and existing.generation_status == "ready":
            return existing
        source, meta = self.store.load(env_id)
        from .env_validator import validate_environment

        report = validate_environment(source)  # never trust files blindly on reload either
        if not report.ok:
            raise EnvStoreError("saved environment no longer validates: " + "; ".join(report.errors)[:400])
        session = EnvSession(id=env_id, prompt=meta.get("prompt", ""))
        self._adopt(session, env_id, report.spec or meta["spec"], report.to_dict(), meta.get("attempts", 1))
        policy = self.store.path(env_id) / "policies" / "brain.json"
        if policy.is_file():
            session.policy_path, session.training_status = policy, "ready"
        with self._lock:
            self._sessions[env_id] = session
        return session

    def train(self, session: EnvSession) -> None:
        with session.lock:
            session.training_status, session.history = "running", []
            session.target_reached = None
            session.training_note = None
            session.concepts_added = []
            if not session.spec or not session.spec.get("trainable"):
                session.training_status = "skipped"
                session.training_error = (session.spec or {}).get("untrainable_reason")
                return
        policy_dir = self.store.path(session.id) / "policies"

        def progress(point: dict[str, float]) -> None:
            with session.lock:
                existing = next(
                    (item for item in reversed(session.history) if item.get("gen") == point.get("gen")),
                    None,
                )
                if existing is None:
                    session.history.append(point)
                else:
                    existing.update(point)

        def training_note(message: str) -> None:
            with session.lock:
                session.training_note = message

        try:
            if session.training_opponent == "self_play":
                result = self.trainer.train_self_play(
                    session.source_path, session.spec, policy_dir, rounds=2,
                    progress=progress, status=training_note,
                )[-1]
            else:
                result = self.trainer.train_env(
                    session.source_path, session.spec, session.training_opponent,
                    progress=progress, status=training_note,
                )
            result.save_policy(policy_dir / "brain.json")
        except (BrainTrainingError, ValueError, SandboxError) as error:
            with session.lock:
                session.training_status, session.training_error = "failed", str(error)
            return
        with session.lock:
            session.concepts, session.heldout = result.concepts, result.heldout_return
            session.target_reached = result.target_reached
            session.concepts_added = result.concepts_added
            session.training_note = result.concept_refresh_note or session.training_note
            session.quality_mode, session.success_return = result.quality_mode, result.success_return
            session.policy_path, session.training_status = policy_dir / "brain.json", "ready"
            self._configure(session)

    def fail(self, env_id: str, message: str) -> None:
        session = self.get(env_id)
        with session.lock:
            session.generation_status, session.generation_error = "failed", message
            session.training_status = "skipped"

    def fail_training(self, env_id: str, message: str) -> None:
        session = self.get(env_id)
        with session.lock:
            session.training_status, session.training_error = "failed", message

    @staticmethod
    def _note(session: EnvSession, message: str) -> None:
        with session.lock:
            session.generation_notes.append(message)

    # ---- play ----------------------------------------------------------------
    def _configure(self, session: EnvSession) -> None:
        if session.sandbox is None:
            return
        default: dict[str, Any] = {"kind": session.opponent}
        if session.opponent == "brain" and session.policy_path:
            default = {"kind": "policy", "policy_path": str(session.policy_path), "greedy": True}
        elif session.opponent == "brain":
            default = {"kind": "random"}
        fields: dict[str, Any] = {"default": default}
        if session.policy_path:
            fields["brain"] = {"policy_path": str(session.policy_path)}
        session.sandbox.request("configure", **fields)

    def _ready(self, session: EnvSession) -> None:
        if session.generation_status != "ready" or session.source_path is None:
            raise EnvNotReady("the environment is still being generated")

    def reset(self, env_id: str, seat: int | None = None, opponent: str | None = None) -> dict[str, Any]:
        session = self.get(env_id)
        with session.lock:
            self._ready(session)
            if opponent is not None:
                if opponent not in OPPONENTS:
                    raise ValueError(f"opponent must be one of {', '.join(OPPONENTS)}")
                session.opponent = opponent
            if session.sandbox is None:
                session.sandbox = SandboxProcess(session.source_path, call_timeout=10.0)
            self._configure(session)
            agents = session.spec["agents"]
            if seat is not None and seat not in agents:
                raise ValueError(f"seat must be one of {agents}")
            session.human_seat = seat if seat is not None else (session.human_seat if session.human_seat in agents else agents[0])
            session.view = session.sandbox.request("reset", seed=uuid4().int % 2**31, seat=session.human_seat)
            return self.snapshot(env_id)

    def play(self, env_id: str, action: Any, use_brain: bool = False) -> dict[str, Any]:
        session = self.get(env_id)
        with session.lock:
            self._ready(session)
            if session.sandbox is None or session.view is None:
                raise ValueError("start the game first")
            if use_brain and session.training_status != "ready":
                raise EnvNotReady("brain is not ready: " + (session.training_error or f"training is {session.training_status}"))
            try:
                session.view = session.sandbox.request("play", action=action, brain=use_brain)
            except SandboxError as error:
                if error.kind == "contract":
                    raise ValueError(str(error)) from error
                raise
            return self.snapshot(env_id)

    def source(self, env_id: str) -> str:
        session = self.get(env_id)
        self._ready(session)
        return session.source_path.read_text(encoding="utf-8")  # type: ignore[union-attr]

    def snapshot(self, env_id: str) -> dict[str, Any]:
        session = self.get(env_id)
        with session.lock:
            return {
                "id": session.id,
                "generation_status": session.generation_status,
                "generation_notes": session.generation_notes[-12:],
                "generation_error": session.generation_error,
                "attempts": session.attempts,
                "spec": session.spec,
                "validation": session.validation and {
                    "ok": session.validation["ok"], "warnings": session.validation["warnings"],
                    "stats": session.validation["stats"],
                },
                "training_status": session.training_status,
                "training_error": session.training_error,
                "training_opponent": session.training_opponent,
                "training_history": session.history[-300:],
                "training_heldout": session.heldout,
                "training_target_reached": session.target_reached,
                "training_note": session.training_note,
                "training_concepts_added": session.concepts_added,
                "training_quality_mode": session.quality_mode,
                "training_success_return": session.success_return,
                "concepts": session.concepts,
                "opponent": session.opponent,
                "human_seat": session.human_seat,
                "brain_ready": session.training_status == "ready",
                "view": session.view,
            }

    def list_saved(self) -> list[dict[str, Any]]:
        return self.store.list()

    def close_all(self) -> None:
        with self._lock:
            sessions = list(self._sessions.values())
        for session in sessions:
            if session.sandbox is not None:
                session.sandbox.close()
