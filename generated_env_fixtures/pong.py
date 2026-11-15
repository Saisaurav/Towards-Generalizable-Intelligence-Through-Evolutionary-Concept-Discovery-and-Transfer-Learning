import math
import random

SPEC = {
    "title": "Pong",
    "description": "Two paddles, one ball. Return the ball, score when the opponent misses; first to 3 points wins.",
    "agents": [0, 1],
    "observation_size": 7,
    "action": {"type": "discrete", "n": 3},
    "turn_based": False,
    "max_steps": 1500,
    "symmetric_agents": True,
    "noop_action": 1,
    "tick_ms": 40,
    "controls": {"ArrowUp": 0, "ArrowDown": 2, "w": 0, "s": 2},
    "action_names": ["up", "stay", "down"],
}

W, H = 100.0, 60.0
PADDLE_H, PADDLE_W, PADDLE_SPEED = 14.0, 2.0, 2.5
LEFT_X, RIGHT_X = 4.0, W - 4.0
BALL_R = 1.5
WIN_SCORE = 3
MAX_SPEED = 4.0


class Environment:
    def reset(self, seed=None):
        self.rng = random.Random(seed)
        self.paddle = [H / 2, H / 2]       # paddle centres (agent 0 left, agent 1 right)
        self.score = [0, 0]
        self.hits = 0
        self._serve(self.rng.choice([-1, 1]))

    def _serve(self, direction):
        self.bx, self.by = W / 2, H / 2
        angle = self.rng.uniform(-0.6, 0.6)
        speed = 1.8
        self.vx, self.vy = direction * speed * math.cos(angle), speed * math.sin(angle)

    def observe(self, agent_id):
        own, opp = self.paddle[agent_id], self.paddle[1 - agent_id]
        sign = 1.0 if agent_id == 0 else -1.0     # mirror so "forward" is always +x
        bx = self.bx / W if agent_id == 0 else 1.0 - self.bx / W
        return [own / H, opp / H, bx, self.by / H, sign * self.vx / MAX_SPEED, self.vy / MAX_SPEED, (self.by - own) / H]

    def legal_actions(self, agent_id):
        return [] if self.is_terminal() else [0, 1, 2]

    def active_agents(self):
        return [] if self.is_terminal() else [0, 1]

    def scripted_action(self, agent_id):
        target, own = self.by, self.paddle[agent_id]
        if target < own - 2.0:
            return 0
        if target > own + 2.0:
            return 2
        return 1

    def step(self, actions):
        if self.is_terminal():
            raise ValueError("game is over")
        rewards = {0: 0.0, 1: 0.0}
        for agent in (0, 1):
            move = (actions[agent] - 1) * PADDLE_SPEED
            self.paddle[agent] = max(PADDLE_H / 2, min(H - PADDLE_H / 2, self.paddle[agent] + move))
        px, py = self.bx, self.by
        self.bx += self.vx
        self.by += self.vy
        if self.by < BALL_R or self.by > H - BALL_R:
            self.by = max(BALL_R, min(H - BALL_R, self.by))
            self.vy = -self.vy
        # paddle collisions: test that the ball's leading edge crossed the paddle face this step
        for agent, face, going in ((0, LEFT_X + PADDLE_W, -1), (1, RIGHT_X - PADDLE_W, 1)):
            if self.vx * going <= 0:
                continue
            edge_prev = px + going * BALL_R
            edge_new = self.bx + going * BALL_R
            crossed = (edge_prev - face) * going <= 0 < (edge_new - face) * going
            if crossed and abs(self.by - self.paddle[agent]) <= PADDLE_H / 2 + BALL_R:
                offset = (self.by - self.paddle[agent]) / (PADDLE_H / 2)
                speed = min(MAX_SPEED, math.hypot(self.vx, self.vy) * 1.06)
                self.vx = -going * abs(speed * math.cos(offset * 0.9))
                self.vy = speed * math.sin(offset * 0.9)
                self.bx = face - going * BALL_R
                rewards[agent] += 0.1
                self.hits += 1
        if self.bx < 0 or self.bx > W:
            scorer = 1 if self.bx < 0 else 0
            self.score[scorer] += 1
            rewards[scorer] += 1.0
            rewards[1 - scorer] -= 1.0
            if not self.is_terminal():
                self._serve(-1 if scorer == 1 else 1)
        return rewards

    def is_terminal(self):
        return max(self.score) >= WIN_SCORE

    def result(self, agent_id):
        mine, theirs = self.score[agent_id], self.score[1 - agent_id]
        return "win" if mine > theirs else "loss" if mine < theirs else "draw"

    def render_text(self):
        cols, rows = 50, 15
        grid = [[" "] * cols for _ in range(rows)]
        for agent, x in ((0, 0), (1, cols - 1)):
            top = int((self.paddle[agent] - PADDLE_H / 2) / H * rows)
            for y in range(max(0, top), min(rows, top + max(1, int(PADDLE_H / H * rows)))):
                grid[y][x] = "|"
        bx, by = int(min(cols - 1, max(0, self.bx / W * cols))), int(min(rows - 1, max(0, self.by / H * rows)))
        grid[by][bx] = "O"
        body = "\n".join("".join(r) for r in grid)
        return "%s\nscore  you(left) %d : %d (right)\nMoves: 0=up 1=stay 2=down (Enter = stay)" % (body, self.score[0], self.score[1])

    def render(self):
        s = 4.0
        items = [{"type": "rect", "x": 0, "y": 0, "w": W * s, "h": H * s, "fill": "#101721"},
                 {"type": "line", "x1": W * s / 2, "y1": 0, "x2": W * s / 2, "y2": H * s, "stroke": "#3b4d60", "stroke_width": 2}]
        for agent, x in ((0, LEFT_X), (1, RIGHT_X)):
            items.append({"type": "rect", "x": (x - PADDLE_W) * s if agent == 1 else x * s - PADDLE_W * s, "y": (self.paddle[agent] - PADDLE_H / 2) * s,
                          "w": PADDLE_W * s, "h": PADDLE_H * s, "radius": 3, "fill": "#82d5bb" if agent == 0 else "#f0cf75"})
        items.append({"type": "circle", "x": self.bx * s, "y": self.by * s, "r": BALL_R * s, "fill": "#e8eef5"})
        items.append({"type": "text", "x": W * s * 0.25, "y": 34, "text": str(self.score[0]), "size": 28, "fill": "#82d5bb"})
        items.append({"type": "text", "x": W * s * 0.75, "y": 34, "text": str(self.score[1]), "size": 28, "fill": "#f0cf75"})
        hud = ("Left wins" if self.score[0] > self.score[1] else "Right wins") if self.is_terminal() else "First to %d" % WIN_SCORE
        return {"width": W * s, "height": H * s, "items": items, "hud": hud}
