import math
import random

SPEC = {
    "title": "Pong (real-time)",
    "description": "Two paddles, one ball, played in real time. Return the ball and score when the opponent misses; first to 5 points wins.",
    "agents": [0, 1],
    "observation_size": 8,
    "action": {"type": "discrete", "n": 3},
    "turn_based": False,
    "real_time": True,
    "max_steps": 0,
    "symmetric_agents": True,
    "noop_action": 1,
    "tick_ms": 40,
    "max_frames": 4000,
    "stall_frames": 500,
    "controls": {"ArrowUp": 0, "ArrowDown": 2, "w": 0, "s": 2},
    "action_names": ["up", "stay", "down"],
}

W, H = 100.0, 60.0
PADDLE_H, PADDLE_W, PADDLE_SPEED = 12.0, 2.0, 2.2
LEFT_X, RIGHT_X = 4.0, W - 4.0      # paddle faces are at LEFT_X + PADDLE_W and RIGHT_X - PADDLE_W
BALL_R = 1.5
WIN_SCORE = 5
MAX_SPEED = 4.0
SERVE_SPEED = 1.8


class Environment:
    def reset(self, seed=None):
        self.rng = random.Random(seed)
        self.paddle = [H / 2, H / 2]      # paddle centres: agent 0 left, agent 1 right
        self.score = [0, 0]
        self.hits = 0
        self._serve(self.rng.choice([-1, 1]))

    def _serve(self, direction):
        self.bx, self.by = W / 2, H / 2
        angle = self.rng.uniform(-0.6, 0.6)
        self.vx = direction * SERVE_SPEED * math.cos(angle)
        self.vy = SERVE_SPEED * math.sin(angle)

    def observe(self, agent_id):
        # Mirrored so that "toward the opponent" is always +x for whichever seat the agent has.
        own, opp = self.paddle[agent_id], self.paddle[1 - agent_id]
        sign = 1.0 if agent_id == 0 else -1.0
        bx = self.bx / W if agent_id == 0 else 1.0 - self.bx / W
        incoming = 1.0 if self.vx * sign < 0 else 0.0
        return [own / H, opp / H, bx, self.by / H,
                sign * self.vx / MAX_SPEED, self.vy / MAX_SPEED, (self.by - own) / H, incoming]

    def legal_actions(self, agent_id):
        return [] if self.is_terminal() else [0, 1, 2]

    def active_agents(self):
        return [] if self.is_terminal() else [0, 1]

    def scripted_action(self, agent_id):
        """Track the ball's height with a small dead zone."""
        diff = self.by - self.paddle[agent_id]
        if diff < -2.0:
            return 0
        if diff > 2.0:
            return 2
        return 1

    def step(self, actions):
        if self.is_terminal():
            raise ValueError("game is over")
        rewards = {0: 0.0, 1: 0.0}
        for agent in (0, 1):
            move = (actions[agent] - 1) * PADDLE_SPEED
            self.paddle[agent] = max(PADDLE_H / 2, min(H - PADDLE_H / 2, self.paddle[agent] + move))
        px = self.bx
        self.bx += self.vx
        self.by += self.vy
        if self.by < BALL_R or self.by > H - BALL_R:
            self.by = max(BALL_R, min(H - BALL_R, self.by))
            self.vy = -self.vy
        # a paddle returns the ball when the ball's leading edge crosses its face this frame
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
        span = max(1, int(PADDLE_H / H * rows))
        for agent, x in ((0, 0), (1, cols - 1)):
            top = int((self.paddle[agent] - PADDLE_H / 2) / H * rows)
            for y in range(max(0, top), min(rows, top + span)):
                grid[y][x] = "|"
        bx = int(min(cols - 1, max(0, self.bx / W * cols)))
        by = int(min(rows - 1, max(0, self.by / H * rows)))
        grid[by][bx] = "O"
        body = "\n".join("".join(r) for r in grid)
        return "%s\nscore  left %d : %d right\nMoves: 0=up 1=stay 2=down (Enter = stay)" % (body, self.score[0], self.score[1])

    def render(self):
        s = 4.0
        items = [{"type": "rect", "x": 0, "y": 0, "w": W * s, "h": H * s, "fill": "#101721"},
                 {"type": "line", "x1": W * s / 2, "y1": 0, "x2": W * s / 2, "y2": H * s,
                  "stroke": "#3b4d60", "stroke_width": 2}]
        for agent, face in ((0, LEFT_X + PADDLE_W), (1, RIGHT_X - PADDLE_W)):
            left_edge = (face - PADDLE_W) if agent == 0 else face
            items.append({"type": "rect", "x": left_edge * s, "y": (self.paddle[agent] - PADDLE_H / 2) * s,
                          "w": PADDLE_W * s, "h": PADDLE_H * s, "radius": 3,
                          "fill": "#82d5bb" if agent == 0 else "#f0cf75"})
        items.append({"type": "circle", "x": self.bx * s, "y": self.by * s, "r": BALL_R * s, "fill": "#e8eef5"})
        items.append({"type": "text", "x": W * s * 0.25, "y": 34, "text": str(self.score[0]), "size": 28, "fill": "#82d5bb"})
        items.append({"type": "text", "x": W * s * 0.75, "y": 34, "text": str(self.score[1]), "size": 28, "fill": "#f0cf75"})
        if self.is_terminal():
            hud = "Left wins" if self.score[0] > self.score[1] else "Right wins"
        else:
            hud = "First to %d" % WIN_SCORE
        return {"width": W * s, "height": H * s, "items": items, "hud": hud}
