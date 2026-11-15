import random

SPEC = {
    "title": "Snake",
    "description": "Steer a growing snake around an 8x8 board, eat food, and avoid walls and your own body.",
    "agents": [0],
    "observation_size": 11,
    "action": {"type": "discrete", "n": 4},
    "turn_based": False,
    "max_steps": 300,
    "noop_action": 1,
    "tick_ms": 180,
    "hold_last_action": True,
    "controls": {"ArrowUp": 0, "ArrowRight": 1, "ArrowDown": 2, "ArrowLeft": 3, "w": 0, "d": 1, "s": 2, "a": 3},
    "action_names": ["up", "right", "down", "left"],
}

W = H = 8
DIRS = ((0, -1), (1, 0), (0, 1), (-1, 0))


class Environment:
    def reset(self, seed=None):
        self.rng = random.Random(seed)
        self.body = [(W // 2, H // 2), (W // 2 - 1, H // 2)]   # head first
        self.heading = 1
        self.dead = False
        self.score = 0
        self.food = None
        self._spawn_food()

    def _spawn_food(self):
        free = [(x, y) for x in range(W) for y in range(H) if (x, y) not in self.body]
        self.food = self.rng.choice(free) if free else None

    def _blocked(self, cell):
        x, y = cell
        return not (0 <= x < W and 0 <= y < H) or cell in self.body[:-1]

    def observe(self, agent_id):
        hx, hy = self.body[0]
        danger = [1.0 if self._blocked((hx + dx, hy + dy)) else 0.0 for dx, dy in DIRS]
        heading = [1.0 if i == self.heading else 0.0 for i in range(4)]
        fx, fy = self.food if self.food else (hx, hy)
        return danger + heading + [(fx - hx) / W, (fy - hy) / H, len(self.body) / (W * H)]

    def legal_actions(self, agent_id):
        if self.is_terminal():
            return []
        back = (self.heading + 2) % 4
        return [a for a in range(4) if not (a == back and len(self.body) > 1)]

    def active_agents(self):
        return [] if self.is_terminal() else [0]

    def step(self, actions):
        if self.is_terminal():
            raise ValueError("game is over")
        action = actions[0]
        if action not in self.legal_actions(0):
            raise ValueError("illegal action %s" % action)
        self.heading = action
        hx, hy = self.body[0]
        dx, dy = DIRS[action]
        new = (hx + dx, hy + dy)
        if self._blocked(new):
            self.dead = True
            return {0: -1.0}
        old_dist = abs(self.food[0] - hx) + abs(self.food[1] - hy) if self.food else 0
        self.body.insert(0, new)
        reward = -0.01
        if new == self.food:
            self.score += 1
            reward += 1.0
            self._spawn_food()
        else:
            self.body.pop()
            if self.food:
                new_dist = abs(self.food[0] - new[0]) + abs(self.food[1] - new[1])
                reward += 0.02 if new_dist < old_dist else -0.02
        return {0: reward}

    def is_terminal(self):
        return self.dead or self.food is None

    def result(self, agent_id):
        if self.dead:
            return "loss"
        return "win"

    def render_text(self):
        rows = []
        for y in range(H):
            row = ""
            for x in range(W):
                row += "H" if (x, y) == self.body[0] else "s" if (x, y) in self.body else "*" if (x, y) == self.food else "."
            rows.append(row)
        return "\n".join(rows) + "\nscore %d  length %d\nMoves: 0=up 1=right 2=down 3=left (Enter = keep going)" % (self.score, len(self.body))

    def render(self):
        cell = 36
        items = [{"type": "rect", "x": 0, "y": 0, "w": W * cell, "h": H * cell, "fill": "#18232f"}]
        if self.food:
            items.append({"type": "circle", "x": self.food[0] * cell + cell / 2, "y": self.food[1] * cell + cell / 2, "r": cell * 0.3, "fill": "#ff9a9a", "effect": "pulse"})
        for i, (x, y) in enumerate(self.body):
            items.append({"type": "rect", "x": x * cell + 2, "y": y * cell + 2, "w": cell - 4, "h": cell - 4, "radius": 8,
                          "fill": "#82d5bb" if i == 0 else "#3a795a"})
        hud = "Game over - score %d" % self.score if self.is_terminal() else "Score %d  length %d" % (self.score, len(self.body))
        return {"width": W * cell, "height": H * cell, "items": items, "hud": hud}
