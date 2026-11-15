"""Validation pipeline for LLM-generated environment source.

Stages: static AST checks -> sandboxed load (SPEC + interface) -> random-play
episodes (observations, legal actions, transitions, rewards, termination,
results, render scenes) -> determinism (same seed, same instance and fresh
instance) -> learner path (brain-mode text protocol) -> human-play path.
Any failure yields feedback text suitable for an automatic LLM repair round.
"""

from __future__ import annotations

import tempfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from .sandbox import SandboxError, SandboxProcess
from .static_check import static_check

EPISODES = 24
VALIDATOR_VERSION = 1


@dataclass
class ValidationReport:
    ok: bool
    errors: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    spec: dict[str, Any] | None = None
    stats: dict[str, Any] = field(default_factory=dict)

    def feedback(self) -> str:
        lines = ["The generated environment failed validation:"]
        lines += [f"- {e}" for e in self.errors]
        if self.warnings:
            lines += ["Warnings (fix if easy):"] + [f"- {w}" for w in self.warnings]
        return "\n".join(lines)

    def to_dict(self) -> dict[str, Any]:
        return {"ok": self.ok, "errors": self.errors, "warnings": self.warnings,
                "stats": self.stats, "validator_version": VALIDATOR_VERSION}


def validate_environment(source: str, call_timeout: float = 5.0) -> ValidationReport:
    problems = static_check(source)
    if problems:
        return ValidationReport(False, errors=[f"static check: {p}" for p in problems])
    with tempfile.TemporaryDirectory(prefix="env-validate-") as directory:
        path = Path(directory) / "env.py"
        path.write_text(source, encoding="utf-8")
        try:
            sandbox = SandboxProcess(path, mem_mb=768, cpu_seconds=120, call_timeout=call_timeout)
        except SandboxError as error:
            return ValidationReport(False, errors=[f"loading the environment failed: {error.describe()}"])
        try:
            return _run_checks(sandbox)
        except SandboxError as error:
            return ValidationReport(False, errors=[error.describe()], spec=sandbox.spec)
        finally:
            sandbox.close()


def _run_checks(sandbox: SandboxProcess) -> ValidationReport:
    spec = sandbox.spec
    errors: list[str] = []
    warnings: list[str] = []
    episodes = []
    for i in range(EPISODES):
        episodes.append(sandbox.request("validate_episode", seed=i, action_seed=1000 + i, check_illegal=(i == 0)))
    for episode in episodes:
        for issue in episode["issues"]:
            if issue["message"] not in warnings and issue["level"] == "warning":
                warnings.append(issue["message"])
    terminated = [e for e in episodes if e["terminal"]]
    if not terminated:
        errors.append(
            f"none of {EPISODES} random-play episodes reached a terminal state within max_steps={spec['max_steps']}; "
            "either max_steps is too small or the win/loss/draw condition is unreachable"
        )
    if not any(e["nonzero_reward"] for e in episodes):
        errors.append("every reward in every random-play episode was exactly 0; the learner needs a reward signal (e.g. +1 win, -1 loss)")
    acted = set().union(*(set(e["agents_that_acted"]) for e in episodes))
    for agent in spec["agents"]:
        if agent not in acted:
            errors.append(f"agent {agent} never became active in {EPISODES} episodes (active_agents() never returned it)")
    lengths = [e["steps"] for e in episodes]
    if len(set(lengths)) == 1 and len(set(e["digest"] for e in episodes)) == 1:
        warnings.append("all random-play episodes were identical; reset(seed) should vary the game with the seed where the game has randomness")

    # determinism: same seed again on the same instance, and on a fresh instance
    again = sandbox.request("validate_episode", seed=3, action_seed=1003, check_illegal=False)
    fresh = sandbox.request("validate_episode", seed=3, action_seed=1003, check_illegal=False, fresh=True)
    if again["digest"] != episodes[3]["digest"] or fresh["digest"] != episodes[3]["digest"]:
        errors.append(
            "not deterministic: replaying reset(seed=3) with identical actions gave a different trajectory "
            f"(same instance: {'match' if again['digest'] == episodes[3]['digest'] else 'DIFFERENT'}, "
            f"fresh instance: {'match' if fresh['digest'] == episodes[3]['digest'] else 'DIFFERENT'}). "
            "Use a random.Random(seed) instance created in reset(); do not keep state across episodes or use module-level randomness"
        )

    # learner path (what the C++ brain talks to)
    if spec["trainable"]:
        learner_done = 0
        for seed in (11, 12, 13):
            line = sandbox.raw_line(f"R {seed}")
            for _ in range(spec["max_steps"] + 5):
                if line.startswith("ERR"):
                    errors.append(f"learner interface: {line[4:]}")
                    break
                fields = line.split()
                done, n_legal = int(fields[2]), int(fields[3])
                if done:
                    learner_done += 1
                    break
                if n_legal < 1:
                    errors.append("learner interface: no legal action while the episode is running")
                    break
                line = sandbox.raw_line(f"A {fields[4]}")
            if errors:
                break
    else:
        warnings.append(spec["untrainable_reason"])

    if "has_render_text" in spec and not spec["has_render_text"]:
        warnings.append("no render_text(): the CLI/text mode falls back to a summary of the scene's text; add render_text() returning an ASCII picture of the state")

    # human-play path (discrete games only: the browser UI sends integer actions)
    if spec["action"]["type"] != "discrete":
        return ValidationReport(not errors, errors, warnings, spec, _stats(episodes, terminated, lengths))
    view = sandbox.request("reset", seed=5, seat=spec["agents"][-1])
    if view.get("render_error"):
        errors.append(f"render(): {view['render_error']}")
    for _ in range(spec["max_steps"] + 2):
        if view["terminal"]:
            break
        if not view["legal"] and spec["action"]["type"] == "discrete" and not view["terminal"]:
            errors.append("human seat is active but has no legal actions")
            break
        action = view["legal"][0] if view["legal"] else spec.get("noop_action")
        view = sandbox.request("play", action=action)

    return ValidationReport(not errors, errors, warnings, spec, _stats(episodes, terminated, lengths))


def _stats(episodes, terminated, lengths) -> dict[str, Any]:
    return {
        "episodes": EPISODES, "terminated": len(terminated), "truncated": EPISODES - len(terminated),
        "mean_steps": sum(lengths) / len(lengths), "max_steps_seen": max(lengths),
        "results": _result_counts(terminated),
    }


def _result_counts(episodes: list[dict[str, Any]]) -> dict[str, dict[str, int]]:
    counts: dict[str, dict[str, int]] = {}
    for episode in episodes:
        for agent, result in (episode["results"] or {}).items():
            counts.setdefault(agent, {}).setdefault(result, 0)
            counts[agent][result] += 1
    return counts
