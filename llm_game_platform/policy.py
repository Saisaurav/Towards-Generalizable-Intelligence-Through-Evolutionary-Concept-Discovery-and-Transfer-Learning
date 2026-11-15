"""Compatible concept loading and inference for exported brain policies."""

import json
import math
import re
from collections import deque
from pathlib import Path
from typing import Any


_LEGACY_CONTRACTS = {"generic_state_action_v1", "generic_state_action_v2"}
_CONTRACTS = _LEGACY_CONTRACTS | {"generic_state_action_v3", "generic_state_action_v4", "generic_state_action_v5", "generic_state_action_v6", "generic_state_action_v7", "generic_state_action_v8"}
_LEGACY_PRIMITIVES = {
    "state_mean",
    "state_abs_mean",
    "state_var",
    "state_max_abs",
    "action_norm",
    "legal_norm",
    "state_magnitude",
    "state_dispersion",
    "state_boundary_margin",
}
_ADVANCED_PRIMITIVES = {
    "state_min",
    "state_max",
    "state_range",
    "state_positive_fraction",
    "state_negative_fraction",
    "state_sparsity",
    "state_median",
    "state_iqr",
    "state_skewness",
    "state_entropy",
    "state_position_moment",
    "state_position_spread",
    "state_adjacent_difference",
    "state_sign_change_fraction",
}
_TEMPORAL_PRIMITIVES = {
    "transition_delta_mean",
    "transition_delta_abs_mean",
    "transition_delta_max_abs",
    "transition_state_alignment",
}
_BOARD_PRIMITIVES = {
    "board_own_open_lines",
    "board_opponent_open_lines",
    "board_own_immediate_wins",
    "board_opponent_immediate_wins",
    "board_center_control",
    "board_occupancy",
}
_ACTION_TACTIC_PRIMITIVES = {"action_wins_now", "action_blocks_opponent"}
_PRIMITIVES = _LEGACY_PRIMITIVES | _ADVANCED_PRIMITIVES | _TEMPORAL_PRIMITIVES | _BOARD_PRIMITIVES | _ACTION_TACTIC_PRIMITIVES
_GRID_PRIMITIVES = _PRIMITIVES - {"state_boundary_margin"} - _TEMPORAL_PRIMITIVES - _BOARD_PRIMITIVES - _ACTION_TACTIC_PRIMITIVES
_LEGACY_OPERATIONS = {"add": 2, "sub": 2, "mul": 2, "div": 2, "min": 2, "max": 2}
_ADVANCED_OPERATIONS = {
    "abs": 1,
    "neg": 1,
    "square": 1,
    "sqrt": 1,
    "log1p": 1,
    "tanh": 1,
}
_OPERATIONS = _LEGACY_OPERATIONS | _ADVANCED_OPERATIONS
_TOKEN = re.compile(r"[A-Za-z_][A-Za-z0-9_]*|[(),]")
_DIRECTIONS = ((0, -1), (0, 1), (-1, 0), (1, 0))
_ACTION_NAMES = ("up", "down", "left", "right")


class _ExpressionParser:
    def __init__(self, expression: str, advanced: bool = True) -> None:
        if not expression or len(expression) > 4096:
            raise ValueError("concept expression must contain 1 to 4096 characters")
        self.tokens = _TOKEN.findall(expression)
        if "".join(self.tokens) != re.sub(r"\s+", "", expression):
            raise ValueError("concept expression contains unsupported characters")
        self.index = 0
        self.advanced = advanced

    def parse(self) -> Any:
        tree = self._expression(0)
        if self.index != len(self.tokens):
            raise ValueError("concept expression contains trailing tokens")
        return tree

    def _expression(self, depth: int) -> Any:
        if depth > 64 or self.index >= len(self.tokens):
            raise ValueError("concept expression is too deep or incomplete")
        name = self.tokens[self.index]
        self.index += 1
        operations = _OPERATIONS if self.advanced else _LEGACY_OPERATIONS
        if name in operations:
            self._expect("(")
            arguments = [self._expression(depth + 1)]
            if operations[name] == 2:
                self._expect(",")
                arguments.append(self._expression(depth + 1))
            self._expect(")")
            return (name, *arguments)
        primitives = _PRIMITIVES if self.advanced else _LEGACY_PRIMITIVES
        if name in primitives:
            return name
        raise ValueError(f"unsupported concept primitive or operation: {name}")

    def _expect(self, token: str) -> None:
        if self.index >= len(self.tokens) or self.tokens[self.index] != token:
            raise ValueError(f"expected {token!r} in concept expression")
        self.index += 1


def _dependencies(tree: Any) -> set[str]:
    if isinstance(tree, str):
        return {tree}
    return set().union(*(_dependencies(child) for child in tree[1:]))


def parse_expression(expression: str, advanced: bool = True) -> Any:
    return _ExpressionParser(expression, advanced).parse()


_GRID_STATE_DIM = 13  # 5 position/goal/time features + 4 directions x (blocked, hazard)


def probe_primitive_sets(seed: int = 2026, count: int = 400) -> list[dict[str, float]]:
    """Primitive dictionaries from synthetic grid states and every action, used to MEASURE concept outputs.

    The states follow `_grid_state_features` (same layout and normalisation) over several grid shapes, so a
    concept is judged on the distribution it will actually see in a generated grid game.
    """
    import random

    rng = random.Random(seed)
    probes: list[dict[str, float]] = []
    for _ in range(count):
        width, height = rng.choice([(4, 4), (5, 5), (6, 4), (8, 8), (10, 6)])
        x, y = rng.randrange(width), rng.randrange(height)
        state = [
            x / (width - 1),
            y / (height - 1),
            rng.randrange(width) / (width - 1),
            rng.randrange(height) / (height - 1),
            rng.random(),
        ]
        legal = 0
        for dx, dy in _DIRECTIONS:
            inside = 0 <= x + dx < width and 0 <= y + dy < height
            blocked = float(not inside or rng.random() < 0.15)
            state.extend([blocked, float(inside and not blocked and rng.random() < 0.1)])
            legal += int(not blocked)
        legal = max(legal, 1)
        base = _state_primitives(state, legal)
        for action in range(4):
            probes.append(dict(base, action_norm=2 * action / 3 - 1))
    return probes


def screen_concepts(
    bank_path: Path,
    limit: int = 32,
    min_output_std: float = 0.01,
    probes: list[dict[str, float]] | None = None,
    state_dim: int = _GRID_STATE_DIM,
) -> tuple[list[str], list[dict[str, str]]]:
    """Return (accepted expressions, rejections with reasons). Concepts are MEASURED, never trusted.

    Rejected: malformed formula; unknown primitive/operation; primitive dependencies unavailable in grid games;
    unsupported primitive contract; declared state dimension different from the target's; any evaluation
    error; non-finite raw output on any probe; squashed-output standard deviation below `min_output_std`.
    """
    try:
        bank = json.loads(bank_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise ValueError(f"could not read concept bank {bank_path}: {error}") from error
    if not isinstance(bank, dict) or not isinstance(bank.get("concepts"), list):
        raise ValueError("concept bank must contain a concepts list")
    if not math.isfinite(min_output_std) or min_output_std < 0:
        raise ValueError("min_output_std must be a finite non-negative number")
    probes = probe_primitive_sets() if probes is None else probes

    bank_contract = bank.get("primitive_contract")
    accepted: list[str] = []
    rejected: list[dict[str, str]] = []
    seen: set[str] = set()

    def reject(item: Any, reason: str) -> None:
        text = item.get("expression") if isinstance(item, dict) else None
        rejected.append({"expression": text if isinstance(text, str) else repr(item)[:80], "reason": reason})

    for concept in bank["concepts"]:
        if not isinstance(concept, dict):
            reject(concept, "entry is not an object")
            continue
        if concept.get("primitive_contract", bank_contract) not in _CONTRACTS:
            reject(concept, "unsupported primitive contract")
            continue
        expression = concept.get("expression")
        if not isinstance(expression, str):
            reject(concept, "missing expression")
            continue
        declared_dim = concept.get("state_dim", bank.get("state_dim"))
        if declared_dim is not None and (
            isinstance(declared_dim, bool) or not isinstance(declared_dim, int) or declared_dim != state_dim
        ):
            reject(concept, f"incompatible input dimension (declared {declared_dim!r}, target {state_dim})")
            continue
        stored_std = concept.get("target_output_std")
        if stored_std is not None and (
            isinstance(stored_std, bool) or not isinstance(stored_std, (int, float))
            or not math.isfinite(stored_std) or stored_std < min_output_std
        ):
            reject(concept, f"stored target_output_std {stored_std!r} is invalid or below {min_output_std}")
            continue
        try:
            tree = parse_expression(expression, advanced=concept.get("primitive_contract", bank_contract) in _CONTRACTS - _LEGACY_CONTRACTS)
        except ValueError as error:
            reject(concept, f"malformed or unsupported formula: {error}")
            continue
        missing = _dependencies(tree) - _GRID_PRIMITIVES
        if missing:
            reject(concept, "primitive dependencies unavailable in grid games: " + ",".join(sorted(missing)))
            continue
        try:
            raw = [_evaluate_expression(tree, probe) for probe in probes]
        except (KeyError, ArithmeticError, TypeError, ValueError, OverflowError) as error:
            reject(concept, f"cannot be evaluated: {error!r}")
            continue
        if not probes or any(not math.isfinite(value) for value in raw):
            reject(concept, "non-finite output on probe states")
            continue
        squashed = [math.tanh(max(-10.0, min(10.0, value))) for value in raw]
        mean = sum(squashed) / len(squashed)
        std = math.sqrt(sum((v - mean) ** 2 for v in squashed) / len(squashed))
        if not std >= min_output_std:
            reject(concept, f"output standard deviation {std:.3g} below {min_output_std}")
            continue
        if expression in seen:
            reject(concept, "duplicate expression")
            continue
        seen.add(expression)
        accepted.append(expression)
        if len(accepted) >= limit:
            break
    return accepted, rejected


def load_compatible_concepts(
    bank_path: Path, limit: int = 32, min_output_std: float = 0.01
) -> list[str]:
    """Load only grid-compatible concepts whose outputs were measured as finite and non-constant."""
    return screen_concepts(bank_path, limit=limit, min_output_std=min_output_std)[0]


def _evaluate_expression(tree: Any, primitives: dict[str, float]) -> float:
    if isinstance(tree, str):
        return primitives[tree]
    operation = tree[0]
    left = _evaluate_expression(tree[1], primitives)
    if operation == "abs":
        value = abs(left)
    elif operation == "neg":
        value = -left
    elif operation == "square":
        value = left * left
    elif operation == "sqrt":
        value = math.sqrt(abs(left))
    elif operation == "log1p":
        value = math.log1p(abs(left))
    elif operation == "tanh":
        value = math.tanh(left)
    else:
        right = _evaluate_expression(tree[2], primitives)
        if operation == "add":
            value = left + right
        elif operation == "sub":
            value = left - right
        elif operation == "mul":
            value = left * right
        elif operation == "div":
            denominator = right if abs(right) >= 1e-4 else (-1e-4 if right < 0 else 1e-4)
            value = left / denominator
        elif operation == "min":
            value = min(left, right)
        else:
            value = max(left, right)
    return value


def _concept_value(tree: Any, primitives: dict[str, float]) -> float:
    value = _evaluate_expression(tree, primitives)
    if not math.isfinite(value):
        return 0.0
    return math.tanh(max(-10.0, min(10.0, value)))



def _quantile(values: list[float], q: float) -> float:
    ordered = sorted(values)
    if not ordered:
        return 0.0
    position = q * (len(ordered) - 1)
    lower = int(position)
    upper = min(lower + 1, len(ordered) - 1)
    return ordered[lower] + (position - lower) * (ordered[upper] - ordered[lower])


def _normalized_abs_entropy(values: list[float]) -> float:
    mass = sum(abs(value) for value in values)
    if mass <= 1e-12 or len(values) <= 1:
        return 0.0
    return -sum((abs(value) / mass) * math.log(abs(value) / mass)
                for value in values if abs(value) > 0.0) / math.log(len(values))


def _state_primitives(state: list[float], n_legal: int) -> dict[str, float]:
    """All state-level primitives of the v7 contract available in grid games (action_norm is added per action)."""
    mean = sum(state) / len(state)
    variance = sum((value - mean) ** 2 for value in state) / len(state)
    return {
        "state_mean": mean,
        "state_abs_mean": sum(abs(value) for value in state) / len(state),
        "state_var": variance,
        "state_max_abs": max(abs(value) for value in state),
        "state_magnitude": math.sqrt(sum(value * value for value in state) / len(state)),
        "state_dispersion": math.sqrt(variance),
        "legal_norm": n_legal / 4,
        "state_min": min(state),
        "state_max": max(state),
        "state_range": max(state) - min(state),
        "state_positive_fraction": sum(value > 0 for value in state) / len(state),
        "state_negative_fraction": sum(value < 0 for value in state) / len(state),
        "state_sparsity": sum(abs(value) <= 1e-6 for value in state) / len(state),
        "state_median": sorted(state)[(len(state) - 1) // 2] if len(state) % 2 else (sorted(state)[len(state)//2-1] + sorted(state)[len(state)//2]) / 2,
        "state_iqr": _quantile(state, 0.75) - _quantile(state, 0.25),
        "state_skewness": (sum(((value - mean) / (math.sqrt(variance) + 1e-8)) ** 3 for value in state) / len(state)) if variance >= 1e-16 else 0.0,
        "state_entropy": _normalized_abs_entropy(state),
        "state_position_moment": sum(value * (2 * index / (len(state) - 1) - 1) for index, value in enumerate(state)) / len(state) if len(state) > 1 else 0.0,
        "state_position_spread": sum(abs(value) * (2 * index / (len(state) - 1) - 1) ** 2 for index, value in enumerate(state)) / (sum(abs(value) for value in state) + 1e-12) if len(state) > 1 else 0.0,
        "state_adjacent_difference": sum(abs(state[index] - state[index - 1]) for index in range(1, len(state))) / (len(state) - 1) if len(state) > 1 else 0.0,
        "state_sign_change_fraction": sum(state[index] * state[index - 1] < 0 for index in range(1, len(state))) / (len(state) - 1) if len(state) > 1 else 0.0,
    }


def _grid_state_features(environment: Any) -> list[float]:
    spec = environment.spec
    x, y = environment.position
    features = [
        x / (spec.width - 1),
        y / (spec.height - 1),
        spec.goal[0] / (spec.width - 1),
        spec.goal[1] / (spec.height - 1),
        (spec.max_turns - environment.turns) / spec.max_turns,
    ]
    walls = set(spec.walls)
    hazards = set(spec.hazards)
    for dx, dy in _DIRECTIONS:
        neighbor = (x + dx, y + dy)
        inside = 0 <= neighbor[0] < spec.width and 0 <= neighbor[1] < spec.height
        features.extend(
            [
                float(not inside or neighbor in walls),
                float(inside and neighbor in hazards),
            ]
        )
    return features


class GridPolicy:
    """Runs the exported NEAT phenotype over legal grid actions."""

    def __init__(self, model: dict[str, Any]) -> None:
        self.input_ids = model.get("input_ids")
        self.n_logit = model.get("n_logit")
        nodes = model.get("nodes")
        connections = model.get("connections")
        concepts = model.get("concepts", [])
        if (
            not isinstance(self.input_ids, list)
            or not self.input_ids
            or any(isinstance(item, bool) or not isinstance(item, int) for item in self.input_ids)
            or isinstance(self.n_logit, bool)
            or not isinstance(self.n_logit, int)
            or self.n_logit < 1
            or not isinstance(nodes, list)
            or not isinstance(connections, list)
            or not isinstance(concepts, list)
            or any(not isinstance(item, str) for item in concepts)
        ):
            raise ValueError("exported brain policy has an invalid schema")

        self.concept_expressions = list(concepts)
        if "action_norm" not in self.concept_expressions:
            self.concept_expressions.append("action_norm")
        self.concept_trees = [parse_expression(item) for item in self.concept_expressions]
        if any(not _dependencies(tree) <= _GRID_PRIMITIVES for tree in self.concept_trees):
            raise ValueError("exported policy uses concepts unavailable in grid games")

        self.nodes: dict[int, int] = {}
        for node in nodes:
            if (
                not isinstance(node, dict)
                or isinstance(node.get("id"), bool)
                or not isinstance(node.get("id"), int)
                or node.get("type") not in (0, 1, 2, 3)
                or node["id"] in self.nodes
            ):
                raise ValueError("exported brain policy contains an invalid node")
            self.nodes[node["id"]] = node["type"]
        input_nodes = [node_id for node_id, node_type in self.nodes.items() if node_type == 0]
        if input_nodes != self.input_ids:
            raise ValueError("exported brain policy input ordering is inconsistent")

        self.connections: list[tuple[int, int, float]] = []
        targets: dict[int, list[tuple[int, float]]] = {}
        target_nodes = {node_id for node_id, node_type in self.nodes.items() if node_type in (2, 3)}
        indegree = {node_id: 0 for node_id in target_nodes}
        outgoing = {node_id: [] for node_id in target_nodes}
        for connection in connections:
            if not isinstance(connection, dict) or not isinstance(connection.get("enabled"), bool):
                raise ValueError("exported brain policy contains an invalid connection")
            source, target, weight = (
                connection.get("from"),
                connection.get("to"),
                connection.get("weight"),
            )
            if (
                isinstance(source, bool)
                or not isinstance(source, int)
                or isinstance(target, bool)
                or not isinstance(target, int)
                or source not in self.nodes
                or target not in self.nodes
                or isinstance(weight, bool)
                or not isinstance(weight, (int, float))
                or not math.isfinite(weight)
            ):
                raise ValueError("exported brain policy contains an invalid connection")
            if not connection["enabled"]:
                continue
            if self.nodes[target] not in (2, 3):
                raise ValueError("exported brain policy has an input-targeting connection")
            self.connections.append((source, target, float(weight)))
            targets.setdefault(target, []).append((source, float(weight)))
            if source in target_nodes:
                indegree[target] += 1
                outgoing[source].append(target)

        ready = deque(node_id for node_id in target_nodes if indegree[node_id] == 0)
        self.order: list[int] = []
        while ready:
            node_id = ready.popleft()
            self.order.append(node_id)
            for target in outgoing[node_id]:
                indegree[target] -= 1
                if indegree[target] == 0:
                    ready.append(target)
        if len(self.order) != len(target_nodes):
            raise ValueError("exported brain policy graph contains a cycle")

        self.incoming = targets
        self.output_ids = [0x1000001 + index for index in range(self.n_logit + 1)]
        if any(self.nodes.get(node_id) != 2 for node_id in self.output_ids):
            raise ValueError("exported brain policy is missing output nodes")
        if any(node_type == 1 and node_id != 0x1000000 for node_id, node_type in self.nodes.items()):
            raise ValueError("exported brain policy has an unsupported bias node")

    @classmethod
    def from_file(cls, path: Path) -> "GridPolicy":
        try:
            model = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as error:
            raise ValueError(f"could not load exported brain policy: {error}") from error
        if not isinstance(model, dict):
            raise ValueError("exported brain policy must be a JSON object")
        return cls(model)

    def _scores(self, features: list[float]) -> list[float]:
        if len(features) != len(self.input_ids):
            raise ValueError(
                f"brain expected {len(self.input_ids)} inputs but received {len(features)}"
            )
        values = {node_id: float(value) for node_id, value in zip(self.input_ids, features)}
        for node_id, node_type in self.nodes.items():
            if node_type == 1:
                values[node_id] = 1.0
        for node_id in self.order:
            total = sum(weight * values[source] for source, weight in self.incoming.get(node_id, []))
            values[node_id] = math.tanh(total) if self.nodes[node_id] == 3 else total
        return [values.get(node_id, 0.0) for node_id in self.output_ids]

    def choose_action(self, environment: Any) -> str:
        legal = environment.legal_actions()
        if not legal:
            raise ValueError("brain cannot act because the game has no legal actions")
        state = _grid_state_features(environment)
        n_legal = len(legal)
        primitives = _state_primitives(state, n_legal)
        action_scores = []
        for action_name in legal:
            action = _ACTION_NAMES.index(action_name)
            action_features = dict(primitives)
            action_features["action_norm"] = 2 * action / 3 - 1
            concept_features = [
                _concept_value(tree, action_features) for tree in self.concept_trees
            ]
            logits = self._scores(state + concept_features)
            action_scores.append((logits[0], action))
        _, selected = max(action_scores, key=lambda score: score[0])
        return _ACTION_NAMES[selected]
