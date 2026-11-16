import random

SPEC = {
    "title": "Snake (real-time)",
    "description": "Steer a growing snake around a 10x10 board in real time. Eat food, avoid the walls and your own body; fill the board to win.",
    "agents": [0],
    "observation_size": 15,
    "action": {"type": "discrete", "n": 4},
    "turn_based": False,
    "real_time": True,
    "max_steps": 0,
    "noop_action": 1,
    "tick_ms": 150,
    "max_frames": 2000,
    "stall_frames": 200,
    "hold_last_action": True,
    "controls": {"ArrowUp": 0, "ArrowRight": 1, "ArrowDown": 2, "ArrowLeft": 3,
                 "w": 0, "d": 1, "s": 2, "a": 3},
    "action_names": ["up", "right", "down", "left"],
}

W = H = 10
DIRS = ((0, -1), (1, 0), (0, 1), (-1, 0))   # up, right, down, left


class Environment:
    def reset(self, seed=None):
        self.rng = random.Random(seed)
        cx, cy = W // 2, H // 2
        self.body = [(cx, cy), (cx - 1, cy), (cx - 2, cy)]   # head first
        self.heading = 1
        self.dead = False
        self.score = 0
        self.food = None
        self._spawn_food()

    def _spawn_food(self):
        taken = set(self.body)
        free = [(x, y) for x in range(W) for y in range(H) if (x, y) not in taken]
        self.food = self.rng.choice(free) if free else None

    def _blocked(self, cell):
        x, y = cell
        if not (0 <= x < W and 0 <= y < H):
            return True
        return cell in self.body[:-1]     # the tail cell frees up as the head advances

    def _dist(self, cell):
        return abs(self.food[0] - cell[0]) + abs(self.food[1] - cell[1]) if self.food else 0

    def observe(self, agent_id):
        hx, hy = self.body[0]
        danger = [1.0 if self._blocked((hx + dx, hy + dy)) else 0.0 for dx, dy in DIRS]
        heading = [1.0 if i == self.heading else 0.0 for i in range(4)]
        fx, fy = self.food if self.food else (hx, hy)
        food_dir = [1.0 if fy < hy else 0.0, 1.0 if fx > hx else 0.0,
                    1.0 if fy > hy else 0.0, 1.0 if fx < hx else 0.0]
        return (danger + heading + food_dir
                + [(fx - hx) / W, (fy - hy) / H, len(self.body) / (W * H)])

    def legal_actions(self, agent_id):
        if self.is_terminal():
            return []
        back = (self.heading + 2) % 4
        return [a for a in range(4) if a != back]     # a snake cannot reverse into its neck

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
        before = self._dist((hx, hy))
        self.body.insert(0, new)
        if new == self.food:
            self.score += 1
            self._spawn_food()
            return {0: 1.0}
        self.body.pop()
        after = self._dist(new)
        return {0: 0.05 if after < before else -0.05}

    def scripted_action(self, agent_id):
        """Greedy: step to the free neighbour closest to the food."""
        hx, hy = self.body[0]
        best, best_dist = None, None
        for a in self.legal_actions(agent_id):
            dx, dy = DIRS[a]
            cell = (hx + dx, hy + dy)
            if self._blocked(cell):
                continue
            d = self._dist(cell)
            if best is None or d < best_dist:
                best, best_dist = a, d
        return best if best is not None else self.heading

    def is_terminal(self):
        return self.dead or self.food is None

    def result(self, agent_id):
        return "loss" if self.dead else "win"      # only a full board ends the game without dying

    def render_text(self):
        rows = []
        for y in range(H):
            row = ""
            for x in range(W):
                cell = (x, y)
                row += ("H" if cell == self.body[0] else "s" if cell in self.body
                        else "*" if cell == self.food else ".")
            rows.append(row)
        return ("\n".join(rows) + "\nscore %d  length %d\n" % (self.score, len(self.body))
                + "Moves: 0=up 1=right 2=down 3=left (Enter = keep going)")

    def render(self):
        cell = 36
        items = [{"type": "rect", "x": 0, "y": 0, "w": W * cell, "h": H * cell, "fill": "#18232f"}]
        if self.food:
            items.append({"type": "circle", "x": self.food[0] * cell + cell / 2,
                          "y": self.food[1] * cell + cell / 2, "r": cell * 0.3,
                          "fill": "#ff9a9a", "effect": "pulse"})
        for i, (x, y) in enumerate(self.body):
            items.append({"type": "rect", "x": x * cell + 2, "y": y * cell + 2, "w": cell - 4, "h": cell - 4,
                          "radius": 8, "fill": "#82d5bb" if i == 0 else "#3a795a"})
        hud = ("Game over - score %d" if self.is_terminal() else "Score %d") % self.score
        return {"width": W * cell, "height": H * cell, "items": items, "hud": hud}
