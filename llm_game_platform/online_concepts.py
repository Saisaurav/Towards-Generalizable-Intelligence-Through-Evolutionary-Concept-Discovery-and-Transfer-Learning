"""Discover bounded, game-specific action features from generated-game episodes."""

from __future__ import annotations

import importlib.util
import random
import re
import sys
from functools import lru_cache
from pathlib import Path
from typing import Any


_REPO_ROOT = Path(__file__).resolve().parents[1]
_UNAVAILABLE_PRIMITIVES = {
    "state_boundary_margin",
    "transition_delta_mean",
    "transition_delta_abs_mean",
    "transition_delta_max_abs",
    "transition_state_alignment",
    "board_own_open_lines",
    "board_opponent_open_lines",
    "board_own_immediate_wins",
    "board_opponent_immediate_wins",
    "board_center_control",
    "board_occupancy",
    "action_wins_now",
    "action_blocks_opponent",
}
_OUTCOME_VALUE = {"win": 1.0, "loss": -1.0, "draw": 0.0}
_WHITESPACE = re.compile(r"\s+")


@lru_cache(maxsize=1)
def _gp_module():
    module_path = _REPO_ROOT / "gp_transfer.py"
    spec = importlib.util.spec_from_file_location("_online_game_gp", module_path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"could not load GP implementation from {module_path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _episode_rows(episode: dict[str, Any], n_actions: int, gamma: float):
    gp = _gp_module()
    samples = episode.get("samples")
    if not isinstance(samples, list) or not samples:
        return [], []
    outcome = _OUTCOME_VALUE.get(episode.get("result"))
    rewards = [float(sample.get("reward", 0.0)) for sample in samples]
    include_outcome = outcome is not None and all(abs(reward) < 1e-8 for reward in rewards)
    running = outcome if include_outcome else 0.0
    targets = [0.0] * len(samples)
    for index in range(len(samples) - 1, -1, -1):
        running = rewards[index] + gamma * running
        targets[index] = running

    rows = []
    previous: list[float] | None = None
    for sample in samples:
        observation = [float(value) for value in sample["observation"]]
        action = int(sample["action"])
        legal = sample["legal"]
        delta = (
            [0.0] * len(observation)
            if previous is None
            else [current - before for current, before in zip(observation, previous)]
        )
        rows.append(gp.summaries(
            observation, action, n_actions, len(legal), "generated", delta
        ))
        previous = observation
    return rows, targets


def discover_concepts(
    episodes: list[dict[str, Any]],
    n_actions: int,
    existing: list[str],
    seed: int,
    max_concepts: int = 8,
) -> list[str]:
    """Evolve expressions that predict game returns and run in the current policy runtime."""
    gp = _gp_module()
    usable = [episode for episode in episodes if episode.get("samples")]
    if len(usable) < 6:
        return []

    random.Random(seed).shuffle(usable)
    split = max(1, min(len(usable) - 1, int(len(usable) * 0.8)))
    train_episodes, validation_episodes = usable[:split], usable[split:]

    def dataset(items):
        rows, targets = [], []
        for episode in items:
            episode_rows, episode_targets = _episode_rows(episode, n_actions, 0.97)
            rows.extend(episode_rows)
            targets.extend(episode_targets)
        if not rows:
            return None, None
        np = gp.np
        return np.asarray(rows, dtype=float).T, np.asarray(targets, dtype=float)

    train_x, train_y = dataset(train_episodes)
    validation_x, validation_y = dataset(validation_episodes)
    if (
        train_x is None
        or validation_x is None
        or len(train_y) < 32
        or len(validation_y) < 8
        or gp.np.std(train_y) < 1e-8
        or gp.np.std(validation_y) < 1e-8
    ):
        return []

    primitives = [
        name for name in gp.PRIMS if name not in _UNAVAILABLE_PRIMITIVES
    ]
    rng = random.Random(seed ^ 0x5EED)
    population_size = 48
    population = list(primitives[:population_size])
    population.extend(
        gp.randtree(rng, maxd=3, primitives=primitives)
        for _ in range(population_size - len(population))
    )
    _, archive, _, _ = gp.evolve_population(
        population,
        train_x,
        train_y,
        validation_x,
        validation_y,
        gp.NoveltyMonitor(validation_x, window=4),
        rng,
        primitives,
        generations=12,
        stagnation_window=4,
        max_unsuccessful_restarts=1,
        restart_mutation_generations=2,
    )

    existing_normalized = {_WHITESPACE.sub("", expression) for expression in existing}
    ranked = []
    seen = set()
    allowed_dependencies = set(primitives)
    for tree in primitives + archive + population:
        expression = gp.render(tree)
        canonical = gp.canon(tree)
        if canonical in seen:
            continue
        seen.add(canonical)
        dependencies = set(gp.dependencies(tree))
        if not dependencies.intersection(allowed_dependencies - {"action_norm", "legal_norm"}):
            continue
        if not dependencies.issubset(allowed_dependencies):
            continue
        if _WHITESPACE.sub("", expression) in existing_normalized or gp.depth(tree) > 5:
            continue
        train_values = gp.ev(tree, train_x)
        validation_values = gp.ev(tree, validation_x)
        if gp.np.std(train_values) < 0.01 or gp.np.std(validation_values) < 0.01:
            continue
        train_corr = gp.corr(train_values, train_y)
        validation_corr = gp.corr(validation_values, validation_y)
        robust_corr = min(train_corr, validation_corr)
        if robust_corr >= 0.02:
            ranked.append((robust_corr, validation_corr, -gp.node_size(tree), expression))

    ranked.sort(reverse=True)
    return [expression for _, _, _, expression in ranked[:max_concepts]]
