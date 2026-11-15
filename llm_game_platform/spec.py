"""Validated game specification and deterministic grid-game environment."""

from collections import deque
from dataclasses import dataclass
from typing import Any, Mapping


_SPEC_FIELDS = {
    "title",
    "description",
    "width",
    "height",
    "start",
    "goal",
    "walls",
    "hazards",
    "max_turns",
}
_DIRECTIONS = {
    "up": (0, -1),
    "down": (0, 1),
    "left": (-1, 0),
    "right": (1, 0),
}


def _integer(value: Any, name: str, minimum: int, maximum: int) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise ValueError(f"{name} must be an integer")
    if value < minimum or value > maximum:
        raise ValueError(f"{name} must be between {minimum} and {maximum}")
    return value


def _coordinate(value: Any, name: str, width: int, height: int) -> tuple[int, int]:
    if not isinstance(value, list) or len(value) != 2:
        raise ValueError(f"{name} must be a [x, y] coordinate")
    x = _integer(value[0], f"{name}[0]", 0, width - 1)
    y = _integer(value[1], f"{name}[1]", 0, height - 1)
    return x, y


@dataclass(frozen=True)
class GameSpec:
    title: str
    description: str
    width: int
    height: int
    start: tuple[int, int]
    goal: tuple[int, int]
    walls: tuple[tuple[int, int], ...]
    hazards: tuple[tuple[int, int], ...]
    max_turns: int

    @classmethod
    def from_mapping(cls, value: Mapping[str, Any]) -> "GameSpec":
        if not isinstance(value, Mapping):
            raise ValueError("game specification must be a JSON object")
        missing = _SPEC_FIELDS - set(value)
        extra = set(value) - _SPEC_FIELDS
        if missing:
            raise ValueError(f"missing fields: {', '.join(sorted(missing))}")
        if extra:
            raise ValueError(f"unsupported fields: {', '.join(sorted(extra))}")

        title = value["title"]
        description = value["description"]
        if not isinstance(title, str) or not title.strip() or len(title) > 80:
            raise ValueError("title must contain 1 to 80 characters")
        if not isinstance(description, str) or not description.strip() or len(description) > 500:
            raise ValueError("description must contain 1 to 500 characters")

        width = _integer(value["width"], "width", 4, 10)
        height = _integer(value["height"], "height", 4, 10)
        start = _coordinate(value["start"], "start", width, height)
        goal = _coordinate(value["goal"], "goal", width, height)
        max_turns = _integer(value["max_turns"], "max_turns", 1, 100)

        def coordinates(field: str) -> tuple[tuple[int, int], ...]:
            items = value[field]
            if not isinstance(items, list):
                raise ValueError(f"{field} must be a list of [x, y] coordinates")
            if len(items) > width * height:
                raise ValueError(f"{field} contains too many coordinates")
            parsed = tuple(
                _coordinate(item, f"{field}[{index}]", width, height)
                for index, item in enumerate(items)
            )
            if len(set(parsed)) != len(parsed):
                raise ValueError(f"{field} must not contain duplicate coordinates")
            return parsed

        walls = coordinates("walls")
        hazards = coordinates("hazards")
        blocked = set(walls) | set(hazards)
        if start in blocked:
            raise ValueError("start cannot overlap a wall or hazard")
        if goal in blocked:
            raise ValueError("goal cannot overlap a wall or hazard")
        if start == goal:
            raise ValueError("start and goal must be different")

        distance = cls._path_length(start, goal, width, height, blocked)
        if distance is None:
            raise ValueError("goal must be reachable from start without crossing walls or hazards")
        if distance > max_turns:
            raise ValueError("max_turns is too small to reach the goal")

        return cls(
            title=title.strip(),
            description=description.strip(),
            width=width,
            height=height,
            start=start,
            goal=goal,
            walls=walls,
            hazards=hazards,
            max_turns=max_turns,
        )

    @staticmethod
    def _path_length(
        start: tuple[int, int],
        goal: tuple[int, int],
        width: int,
        height: int,
        blocked: set[tuple[int, int]],
    ) -> int | None:
        pending = deque([(start, 0)])
        visited = {start}
        while pending:
            (x, y), distance = pending.popleft()
            if (x, y) == goal:
                return distance
            for dx, dy in _DIRECTIONS.values():
                neighbor = (x + dx, y + dy)
                if (
                    0 <= neighbor[0] < width
                    and 0 <= neighbor[1] < height
                    and neighbor not in blocked
                    and neighbor not in visited
                ):
                    visited.add(neighbor)
                    pending.append((neighbor, distance + 1))
        return None

    def to_dict(self) -> dict[str, Any]:
        return {
            "title": self.title,
            "description": self.description,
            "width": self.width,
            "height": self.height,
            "start": list(self.start),
            "goal": list(self.goal),
            "walls": [list(point) for point in self.walls],
            "hazards": [list(point) for point in self.hazards],
            "max_turns": self.max_turns,
        }


class GameEnvironment:
    """Single-player grid game implementing reset, legal_actions, and step."""

    def __init__(self, spec: GameSpec) -> None:
        self.spec = spec
        self.reset()

    def reset(self) -> dict[str, Any]:
        self.position = self.spec.start
        self.turns = 0
        self.done = False
        self.outcome = "playing"
        return self.state()

    def legal_actions(self) -> list[str]:
        x, y = self.position
        walls = set(self.spec.walls)
        return [
            action
            for action, (dx, dy) in _DIRECTIONS.items()
            if 0 <= x + dx < self.spec.width
            and 0 <= y + dy < self.spec.height
            and (x + dx, y + dy) not in walls
        ]

    def step(self, action: str) -> dict[str, Any]:
        if self.done:
            raise ValueError("game is already finished; reset before taking another action")
        if action not in self.legal_actions():
            raise ValueError(f"illegal action: {action}")

        dx, dy = _DIRECTIONS[action]
        self.position = (self.position[0] + dx, self.position[1] + dy)
        self.turns += 1
        reward = -0.01
        if self.position in self.spec.hazards:
            self.done = True
            self.outcome = "lost"
            reward = -1.0
        elif self.position == self.spec.goal:
            self.done = True
            self.outcome = "won"
            reward = 1.0
        elif self.turns >= self.spec.max_turns:
            self.done = True
            self.outcome = "out_of_turns"
            reward = -0.25
        return {"state": self.state(), "reward": reward}

    def state(self) -> dict[str, Any]:
        return {
            "position": list(self.position),
            "turns": self.turns,
            "done": self.done,
            "outcome": self.outcome,
            "legal_actions": self.legal_actions(),
        }
