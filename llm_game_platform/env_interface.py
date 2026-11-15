"""The standardised interface every LLM-generated environment implements.

A generated source file defines a module-level ``SPEC`` dict and a class named
``Environment``. The runtime (sandbox worker) owns episode management, step
limits, seeding policy checks, opponents and result bookkeeping; the generated
class only implements game logic.

    SPEC = {
        "title": "My Game",
        "description": "One sentence describing the game.",
        "agents": [0, 1],                 # integer agent ids
        "observation_size": 18,           # length of observe() vectors
        "action": {"type": "discrete", "n": 9},
        # or {"type": "continuous", "dim": 2, "low": -1.0, "high": 1.0}
        "turn_based": True,               # False => several agents act per step
        "real_time": False,               # True => each step is one frame; all active agents act together
        "max_steps": 20,                  # hard truncation limit (runtime enforced)
        # optional:
        "learner_agent": 0,               # default: first agent
        "symmetric_agents": True,         # randomise the learner's seat each episode
        "noop_action": 4,                 # discrete action used when a human sends none
        "tick_ms": 120,                   # real-time pacing hint for the browser
        "hold_last_action": True,         # browser keeps sending the last key's action (e.g. Snake heading)
        "controls": {"ArrowUp": 0},       # key -> action for human play
        "action_names": ["up", ...],
        "cooperative": False,
    }

    class Environment:
        def reset(self, seed=None): ...          # start a new episode; must be deterministic in seed
        def observe(self, agent_id): ...         # list[float], len == observation_size, from agent's view
        def legal_actions(self, agent_id): ...   # list[int]; [] once terminal (discrete only)
        def active_agents(self): ...             # agents that must act now ([] when terminal)
        def step(self, actions): ...             # actions: {agent_id: action}; returns {agent_id: reward}
        def is_terminal(self): ...               # True when the game ended by its own rules
        def result(self, agent_id): ...          # "win" | "loss" | "draw" (valid once terminal)
        def render(self): ...                    # scene dict, see render_protocol
        def render_text(self): ...               # str: plain-text picture for the CLI/text mode (strongly recommended)
        # optional:
        def is_truncated(self): ...              # env-defined time-out (runtime also truncates at max_steps)
        def scripted_action(self, agent_id): ... # hand-written rule-based opponent hook
        def clone(self): ...                     # independent copy (new lists/dicts; RNG may be shared); makes the rule-based opponent fast, deepcopy otherwise

Turn-based games return one agent from ``active_agents()``. For real-time games,
set ``real_time`` to True, set ``tick_ms`` to the desired frame interval, and
return all agents that act during each frame from ``active_agents()``. Each
``step()`` advances exactly one frame. ``real_time`` defaults to the inverse of
``turn_based`` for older environments. Continuous action spaces are supported
by the runtime and validator, but the NEAT/PPO learner currently only trains on
discrete action spaces (see ``EnvSpec.trainable``).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Mapping

REQUIRED_METHODS = (
    "reset",
    "observe",
    "legal_actions",
    "active_agents",
    "step",
    "is_terminal",
    "result",
    "render",
)
OPTIONAL_METHODS = ("is_truncated", "scripted_action", "clone", "render_text")
RESULT_VALUES = ("win", "loss", "draw")
MAX_DISCRETE_ACTIONS = 128  # C++ ConceptEnv buffers legal ids in int[128]
MAX_OBSERVATION_SIZE = 1024
MAX_AGENTS = 8

_SPEC_FIELDS = {
    "title", "description", "agents", "observation_size", "action", "turn_based", "real_time",
    "max_steps", "learner_agent", "symmetric_agents", "noop_action", "tick_ms",
    "controls", "action_names", "cooperative", "hold_last_action",
}
_REQUIRED_SPEC = {
    "title", "description", "agents", "observation_size", "action", "turn_based", "max_steps"
}


def _int(value: Any, name: str, lo: int, hi: int) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise ValueError(f"SPEC.{name} must be an integer")
    if value < lo or value > hi:
        raise ValueError(f"SPEC.{name} must be between {lo} and {hi}")
    return value


@dataclass(frozen=True)
class EnvSpec:
    title: str
    description: str
    agents: tuple[int, ...]
    observation_size: int
    action_type: str
    n_actions: int  # discrete: number of actions; continuous: dimension
    action_low: float
    action_high: float
    turn_based: bool
    real_time: bool
    max_steps: int
    learner_agent: int
    symmetric_agents: bool = False
    noop_action: int | None = None
    tick_ms: int = 150
    controls: dict[str, int] = field(default_factory=dict)
    action_names: tuple[str, ...] = ()
    cooperative: bool = False
    hold_last_action: bool = False

    @property
    def discrete(self) -> bool:
        return self.action_type == "discrete"

    @property
    def trainable(self) -> bool:
        """The current NEAT/PPO learner only supports discrete actions."""
        return self.discrete

    @property
    def untrainable_reason(self) -> str | None:
        if self.discrete:
            return None
        return (
            "continuous action spaces are executable and validated, but the "
            "NEAT/PPO learner only trains on discrete actions"
        )

    @classmethod
    def from_mapping(cls, value: Mapping[str, Any]) -> "EnvSpec":
        if not isinstance(value, Mapping):
            raise ValueError("SPEC must be a dict")
        missing = _REQUIRED_SPEC - set(value)
        extra = set(value) - _SPEC_FIELDS
        if missing:
            raise ValueError(f"SPEC is missing fields: {', '.join(sorted(missing))}")
        if extra:
            raise ValueError(f"SPEC has unsupported fields: {', '.join(sorted(extra))}")
        title, description = value["title"], value["description"]
        if not isinstance(title, str) or not title.strip() or len(title) > 80:
            raise ValueError("SPEC.title must contain 1 to 80 characters")
        if not isinstance(description, str) or not description.strip() or len(description) > 600:
            raise ValueError("SPEC.description must contain 1 to 600 characters")
        agents = value["agents"]
        if (
            not isinstance(agents, (list, tuple))
            or not 1 <= len(agents) <= MAX_AGENTS
            or any(isinstance(a, bool) or not isinstance(a, int) or a < 0 for a in agents)
            or len(set(agents)) != len(agents)
        ):
            raise ValueError(f"SPEC.agents must list 1 to {MAX_AGENTS} distinct non-negative integer ids")
        observation_size = _int(value["observation_size"], "observation_size", 1, MAX_OBSERVATION_SIZE)
        action = value["action"]
        if not isinstance(action, Mapping) or action.get("type") not in ("discrete", "continuous"):
            raise ValueError('SPEC.action must be {"type": "discrete", "n": N} or a continuous spec')
        low = high = 0.0
        if action["type"] == "discrete":
            if set(action) - {"type", "n"}:
                raise ValueError("SPEC.action (discrete) only supports 'type' and 'n'")
            n_actions = _int(action.get("n"), "action.n", 1, MAX_DISCRETE_ACTIONS)
        else:
            if set(action) - {"type", "dim", "low", "high"}:
                raise ValueError("SPEC.action (continuous) only supports type, dim, low, high")
            n_actions = _int(action.get("dim"), "action.dim", 1, 16)
            low, high = action.get("low"), action.get("high")
            if any(isinstance(v, bool) or not isinstance(v, (int, float)) for v in (low, high)) or not low < high:
                raise ValueError("SPEC.action low/high must be numbers with low < high")
            low, high = float(low), float(high)
        if not isinstance(value["turn_based"], bool):
            raise ValueError("SPEC.turn_based must be a boolean")
        real_time = value.get("real_time", not value["turn_based"])
        if not isinstance(real_time, bool):
            raise ValueError("SPEC.real_time must be a boolean")
        if real_time and value["turn_based"]:
            raise ValueError("SPEC.real_time=True requires SPEC.turn_based=False")
        max_steps = _int(value["max_steps"], "max_steps", 1, 10000)
        learner = value.get("learner_agent", agents[0])
        if learner not in agents:
            raise ValueError("SPEC.learner_agent must be one of SPEC.agents")
        symmetric = value.get("symmetric_agents", False)
        cooperative = value.get("cooperative", False)
        hold_last = value.get("hold_last_action", False)
        if not all(isinstance(v, bool) for v in (symmetric, cooperative, hold_last)):
            raise ValueError("SPEC.symmetric_agents, SPEC.cooperative and SPEC.hold_last_action must be booleans")
        noop = value.get("noop_action")
        if noop is not None:
            if action["type"] != "discrete":
                raise ValueError("SPEC.noop_action requires a discrete action space")
            noop = _int(noop, "noop_action", 0, n_actions - 1)
        tick_ms = _int(value.get("tick_ms", 150), "tick_ms", 20, 5000)
        controls_in = value.get("controls", {})
        controls: dict[str, int] = {}
        if not isinstance(controls_in, Mapping) or len(controls_in) > 32:
            raise ValueError("SPEC.controls must map key names to actions")
        for key, act in controls_in.items():
            if not isinstance(key, str) or not key or len(key) > 24:
                raise ValueError("SPEC.controls keys must be short key names such as 'ArrowUp'")
            controls[key] = _int(act, f"controls[{key}]", 0, max(n_actions - 1, 0))
        names_in = value.get("action_names", [])
        if names_in:
            if (
                action["type"] != "discrete"
                or not isinstance(names_in, (list, tuple))
                or len(names_in) != n_actions
                or any(not isinstance(n, str) or len(n) > 24 for n in names_in)
            ):
                raise ValueError("SPEC.action_names must list one short string per discrete action")
        return cls(
            title=title.strip(), description=description.strip(), agents=tuple(agents),
            observation_size=observation_size, action_type=action["type"], n_actions=n_actions,
            action_low=low, action_high=high, turn_based=value["turn_based"], real_time=real_time,
            max_steps=max_steps, learner_agent=learner, symmetric_agents=symmetric,
            noop_action=noop, tick_ms=tick_ms, controls=controls,
            action_names=tuple(names_in or ()), cooperative=cooperative, hold_last_action=hold_last,
        )

    def to_dict(self) -> dict[str, Any]:
        action: dict[str, Any] = (
            {"type": "discrete", "n": self.n_actions}
            if self.discrete
            else {"type": "continuous", "dim": self.n_actions, "low": self.action_low, "high": self.action_high}
        )
        return {
            "title": self.title, "description": self.description, "agents": list(self.agents),
            "observation_size": self.observation_size, "action": action,
            "turn_based": self.turn_based, "real_time": self.real_time, "max_steps": self.max_steps,
            "learner_agent": self.learner_agent, "symmetric_agents": self.symmetric_agents,
            "noop_action": self.noop_action, "tick_ms": self.tick_ms,
            "controls": dict(self.controls), "action_names": list(self.action_names),
            "cooperative": self.cooperative, "hold_last_action": self.hold_last_action, "trainable": self.trainable,
            "untrainable_reason": self.untrainable_reason,
        }
