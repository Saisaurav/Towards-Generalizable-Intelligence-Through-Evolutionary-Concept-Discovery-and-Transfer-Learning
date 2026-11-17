"""Small deterministic evaluator for FrameForge's exported NEAT policy JSON."""
from __future__ import annotations
import json
import math
from pathlib import Path

OUT_BASE = (1 << 24) + 1
HID_BASE = 1 << 25

class NeatPolicy:
    def __init__(self, path: str | Path):
        data = json.loads(Path(path).read_text(encoding="utf-8"))
        self.data = data
        self.input_ids = list(data["input_ids"])
        self.act_dim = int(data.get("act_dim", data.get("action_dim", 0)))
        if self.act_dim < 1:
            raise ValueError("policy does not contain a continuous action head")
        self.nodes = {int(n["id"]): n for n in data["nodes"]}
        compute = [int(n["id"]) for n in data["nodes"] if int(n["type"]) in (2, 3)]
        compute_set = set(compute)
        self.incoming = {node: [] for node in compute}
        for edge in data["connections"]:
            if edge.get("enabled", False) and int(edge["to"]) in compute_set:
                self.incoming[int(edge["to"])].append((int(edge["from"]), float(edge["weight"])))
        live, stack = set(), [OUT_BASE + i for i in range(self.act_dim)]
        for node in stack:
            if node not in compute_set:
                raise ValueError(f"missing output node {node}")
        while stack:
            node = stack.pop()
            if node in live:
                continue
            live.add(node)
            for source, _ in self.incoming[node]:
                if source in compute_set and source not in live:
                    stack.append(source)
        indegree = {node: 0 for node in live}
        adjacency = {node: [] for node in live}
        for node in live:
            for source, _ in self.incoming[node]:
                if source in live:
                    indegree[node] += 1
                    adjacency[source].append(node)
        queue = [node for node in compute if node in live and indegree[node] == 0]
        self.order = []
        while queue:
            node = queue.pop(0)
            self.order.append(node)
            for target in adjacency[node]:
                indegree[target] -= 1
                if indegree[target] == 0:
                    queue.append(target)
        if len(self.order) != len(live):
            raise ValueError("active policy topology is cyclic")
        self.obs_names = data.get("obs_names", [])

    def act(self, observation: list[float]) -> list[float]:
        if len(observation) != len(self.input_ids):
            raise ValueError(f"expected {len(self.input_ids)} observation values, got {len(observation)}")
        values = {node: float(observation[i]) for i, node in enumerate(self.input_ids)}
        values[1 << 24] = 1.0
        for node in self.order:
            total = sum(values.get(source, 0.0) * weight for source, weight in self.incoming[node])
            values[node] = math.tanh(total) if int(self.nodes[node]["type"]) == 3 else total
        return [max(-1.0, min(1.0, values.get(OUT_BASE + i, 0.0))) for i in range(self.act_dim)]
