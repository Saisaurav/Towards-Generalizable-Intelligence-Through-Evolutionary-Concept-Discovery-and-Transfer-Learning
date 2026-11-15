"""LLM -> Python environment source, validated in a sandbox with automatic repair rounds."""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any, Callable, Protocol
from uuid import uuid4

from . import env_prompts
from .env_store import EnvStore
from .env_validator import ValidationReport, validate_environment

_FENCE = re.compile(r"```(?:python|py)?\s*\n(.*?)```", re.S)


class ChatClient(Protocol):
    def complete(self, system_prompt: str, user_prompt: str) -> str: ...


class EnvGenerationError(RuntimeError):
    def __init__(self, attempts: int, errors: list[str]) -> None:
        self.attempts, self.errors = attempts, errors
        super().__init__(f"could not produce a valid environment after {attempts} attempts: " + " | ".join(errors[-1:]))


@dataclass
class GeneratedEnv:
    env_id: str
    prompt: str
    source: str
    spec: dict[str, Any]
    report: ValidationReport
    attempts: int
    repairs: list[str] = field(default_factory=list)


def extract_source(text: str) -> str:
    blocks = _FENCE.findall(text)
    if blocks:
        return max(blocks, key=len).strip() + "\n"
    if "class Environment" in text:
        return text.strip() + "\n"
    raise ValueError("response did not contain a ```python code block")


class EnvGenerator:
    def __init__(self, client: ChatClient, store: EnvStore | None = None, max_attempts: int = 4) -> None:
        if not 1 <= max_attempts <= 6:
            raise ValueError("max_attempts must be between 1 and 6")
        self.client, self.store, self.max_attempts = client, store or EnvStore(), max_attempts

    @staticmethod
    def validate_prompt(text: object) -> str:
        if not isinstance(text, str) or not text.strip():
            raise ValueError("game prompt must not be empty")
        if len(text) > 4000:
            raise ValueError("game prompt must be at most 4000 characters")
        return text.strip()

    def _complete(self, system: str, user: str) -> str:
        complete = getattr(self.client, "complete_text", None) or self.client.complete
        return complete(system, user)

    def generate(self, description: str, progress: Callable[[str], None] | None = None,
                 env_id: str | None = None) -> GeneratedEnv:
        description = self.validate_prompt(description)
        note = progress or (lambda _m: None)
        errors: list[str] = []
        source = ""
        user = env_prompts.user_prompt(description)
        for attempt in range(1, self.max_attempts + 1):
            note(f"attempt {attempt}: asking the LLM for environment code")
            raw = self._complete(env_prompts.SYSTEM_PROMPT, user)
            try:
                source = extract_source(raw)
            except ValueError as error:
                errors.append(str(error))
                user = env_prompts.repair_prompt(description, raw[:8000], f"Problem: {error}")
                continue
            note(f"attempt {attempt}: validating in the sandbox")
            report = validate_environment(source)
            if report.ok:
                env_id = env_id or uuid4().hex
                self.store.save(env_id, prompt=description, source=source, spec=report.spec or {},
                                report=report.to_dict(), attempts=attempt,
                                model=getattr(self.client, "model", None))
                return GeneratedEnv(env_id, description, source, report.spec or {}, report, attempt, errors)
            errors.append("; ".join(report.errors)[:1500])
            note(f"attempt {attempt}: validation failed, requesting a repair")
            user = env_prompts.repair_prompt(description, source, report.feedback())
        raise EnvGenerationError(self.max_attempts, errors)
