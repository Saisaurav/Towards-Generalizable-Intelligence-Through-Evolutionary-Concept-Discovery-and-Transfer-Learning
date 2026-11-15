"""Game-agnostic learnability checks for generated environments.

Three jobs, all driven by cheap baseline rollouts of the *runtime* (no learner involved):

1. ``run_calibration``      - random vs scripted/rule-based returns from the learner's seat.
2. ``quality_mode`` / ``success_bar`` - pick how training progress is scored. Games with real
   win/loss outcomes keep win rate; score-style single-agent games (Snake) whose "win" is
   (almost) unreachable are scored by "return beats the random baseline" instead.
3. ``findings``             - the validator's learnability gate: does the reward actually
   separate competent play from random play, and is the observation sane for a small net?
"""

from __future__ import annotations

import math
from typing import Any

CALIBRATION_EPISODES = 60
_CHUNK = 10
RAW_GRID_WARN = 64


def _mean_se(values: list[float]) -> tuple[float, float]:
    n = len(values)
    if n == 0:
        return 0.0, 0.0
    mean = sum(values) / n
    if n == 1:
        return mean, 0.0
    var = sum((v - mean) ** 2 for v in values) / (n - 1)
    return mean, math.sqrt(var / n)


def percentile(values: list[float], q: float) -> float:
    ordered = sorted(values)
    if not ordered:
        return 0.0
    pos = (len(ordered) - 1) * q
    lo, hi = int(math.floor(pos)), int(math.ceil(pos))
    return ordered[lo] + (ordered[hi] - ordered[lo]) * (pos - lo)


def _merge(parts: list[dict[str, Any]]) -> dict[str, Any]:
    returns = [r for p in parts for r in p["returns"]]
    n = len(returns)
    weights = [len(p["returns"]) / n for p in parts]
    merged = {"returns": returns, "mean_return": sum(returns) / n}
    for key in ("mean_length", "win", "loss", "draw"):
        merged[key] = sum(w * p[key] for w, p in zip(weights, parts))
    return merged


def run_calibration(sandbox, episodes: int = CALIBRATION_EPISODES, seed: int = 777) -> dict[str, Any]:
    """Run ``calibrate`` in small chunks (each worker call has its own time limit)."""
    random_parts: list[dict[str, Any]] = []
    scripted_parts: list[dict[str, Any]] = []
    has_hook = False
    for index in range(max(1, episodes // _CHUNK)):
        part = sandbox.request("calibrate", episodes=_CHUNK, seed=seed + index * 100003)
        random_parts.append(part["random"])
        has_hook = bool(part.get("has_scripted_hook"))
        if part.get("scripted"):
            scripted_parts.append(part["scripted"])
    return {
        "random": _merge(random_parts),
        "scripted": _merge(scripted_parts) if scripted_parts else None,
        "has_scripted_hook": has_hook,
    }


def quality_mode(spec: dict[str, Any], calibration: dict[str, Any] | None, requested: str = "auto") -> str:
    """'result' => win rate from real game outcomes; 'return' => success = beating the random baseline."""
    if requested in ("result", "return"):
        return requested
    single = len(spec["agents"]) == 1 or bool(spec.get("cooperative"))
    if not single or calibration is None:
        return "result"
    wins = calibration["random"]["win"] + (calibration["scripted"]["win"] if calibration["scripted"] else 0.0)
    return "return" if wins == 0 else "result"


def success_bar(calibration: dict[str, Any]) -> float:
    """An episode counts as a success when its return is strictly above this bar.

    The bar is the 90th percentile of random-play returns (so random succeeds <= ~10% of the
    time), raised a quarter of the way towards the scripted policy when that policy is clearly better than random.
    """
    random_returns = calibration["random"]["returns"]
    bar = percentile(random_returns, 0.90)
    scripted = calibration.get("scripted")
    if scripted:
        random_mean = sum(random_returns) / len(random_returns)
        if scripted["mean_return"] > random_mean:
            bar = max(bar, random_mean + 0.25 * (scripted["mean_return"] - random_mean))
    return bar


def findings(spec: dict[str, Any], calibration: dict[str, Any]) -> tuple[list[str], list[str]]:
    """(errors, warnings) of the learnability gate."""
    errors: list[str] = []
    warnings: list[str] = []
    rnd = calibration["random"]
    scripted = calibration["scripted"]
    r_mean, r_se = _mean_se(rnd["returns"])
    real_time = spec["real_time"]
    if scripted is not None:
        s_mean, s_se = _mean_se(scripted["returns"])
        diff, se = s_mean - r_mean, math.sqrt(r_se ** 2 + s_se ** 2)
        who = "scripted_action()" if calibration["has_scripted_hook"] else "the generic look-ahead player"
        if diff < -2 * max(se, 1e-9):
            errors.append(
                f"learnability: {who} scores worse than random play (mean return {s_mean:.2f} vs {r_mean:.2f} "
                f"over {len(rnd['returns'])} episodes). Either the reward punishes competent play (e.g. a per-step "
                "penalty so large that dying early beats surviving or eating) or scripted_action() is wrong. "
                "Make the return higher for genuinely better play: reward progress/goals, keep step penalties "
                "much smaller than the reward for staying alive and succeeding."
            )
        elif diff < 2 * max(se, 1e-9):
            warnings.append(
                f"learnability: {who} is not clearly better than random play (mean return {s_mean:.2f} vs {r_mean:.2f}); "
                "the reward may not distinguish good play from random play, which makes the game hard to learn"
            )
    elif real_time:
        warnings.append(
            "learnability: no scripted_action(); add one (a simple competent policy) so the platform can verify "
            "that the reward separates good play from random play"
        )
    if len(set(round(v, 9) for v in rnd["returns"])) == 1:
        warnings.append("learnability: the episode return is identical in every random-play episode; there is no learning signal")
    if real_time and spec["observation_size"] > RAW_GRID_WARN:
        warnings.append(
            f"learnability: observation_size={spec['observation_size']} looks like a raw grid/pixel dump. Small networks "
            "learn far faster from egocentric/relative features (distance and direction to the goal, danger flags, "
            "own heading and velocity) than from absolute cell values"
        )
    cap = spec.get("max_frames") or 3000
    if real_time and rnd["mean_length"] >= 0.9 * cap:
        warnings.append(
            f"learnability: random play almost always runs into the {cap}-frame cap; the game has no easy failure state, "
            "so episodes are long and slow to learn from. Add a rule-based end condition or set max_frames/stall_frames lower"
        )
    return errors, warnings
