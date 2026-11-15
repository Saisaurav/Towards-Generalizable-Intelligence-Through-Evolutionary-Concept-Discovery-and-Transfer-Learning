"""Generic exported-brain inference for any discrete environment.

Reuses the validated NEAT phenotype evaluation in ``policy.GridPolicy`` but
builds inputs from a generic observation vector and legal-action list, matching
the C++ ``ConceptEnv`` primitives (action_norm = 2a/(n-1)-1, legal_norm =
#legal/n_actions). Pure stdlib so the sandbox worker can import it.
"""

from __future__ import annotations

import math
import random
from typing import Any, Sequence

try:  # package import
    from .policy import GridPolicy, _concept_value, _state_primitives
except ImportError:  # imported by path inside the sandbox worker
    from policy import GridPolicy, _concept_value, _state_primitives  # type: ignore


def concept_features(trees: Sequence[Any], observation: Sequence[float], legal: Sequence[int],
                     n_actions: int) -> list[list[float]]:
    """Concept values per legal action for a generic environment (one row per legal action).

    INTEGRATION ADAPTER: uses V8's complete state-primitive set (the Python mirror of the C++
    generic_state_action_v8 contract that ConceptEnv evaluates during training) instead of a
    hand-copied subset. The only generic-environment difference is legal_norm, which the C++
    ConceptEnv defines as #legal / n_actions (1.0 when n_actions == 1), not the grid's fixed /4.
    """
    state = [float(v) for v in observation]
    base = _state_primitives(state, len(legal))
    base["legal_norm"] = len(legal) / n_actions if n_actions > 1 else 1.0
    rows = []
    for action in legal:
        prims = dict(base)
        prims["action_norm"] = 2.0 * action / (n_actions - 1) - 1.0 if n_actions > 1 else 0.0
        rows.append([_concept_value(tree, prims) for tree in trees])
    return rows


class NeatPolicy(GridPolicy):
    def __init__(self, model: dict[str, Any], n_actions: int) -> None:
        super().__init__(model)
        if n_actions < 1:
            raise ValueError("n_actions must be positive")
        self.n_actions = n_actions

    @classmethod
    def from_file(cls, path, n_actions: int) -> "NeatPolicy":  # type: ignore[override]
        import json
        from pathlib import Path

        model = json.loads(Path(path).read_text(encoding="utf-8"))
        if "concepts" not in model:
            sidecar = Path(str(path) + ".concepts")
            if sidecar.is_file():
                model["concepts"] = [l for l in sidecar.read_text().splitlines() if l and l[0] != "#"]
        return cls(model, n_actions)

    def scores(self, observation: Sequence[float], legal: Sequence[int]) -> list[float]:
        if not legal:
            raise ValueError("no legal actions to score")
        state = [float(v) for v in observation]
        rows = concept_features(self.concept_trees, state, legal, self.n_actions)
        return [self._scores(state + concepts)[0] for concepts in rows]

    def choose(self, observation: Sequence[float], legal: Sequence[int],
               rng: random.Random | None = None, greedy: bool = True) -> int:
        logits = self.scores(observation, legal)
        if greedy or rng is None:
            return legal[max(range(len(legal)), key=lambda i: logits[i])]
        top = max(logits)
        weights = [math.exp(l - top) for l in logits]
        return rng.choices(list(legal), weights=weights)[0]
