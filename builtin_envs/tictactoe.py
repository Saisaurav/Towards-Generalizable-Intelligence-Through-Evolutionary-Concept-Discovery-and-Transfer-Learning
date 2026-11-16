import random

SPEC = {
    "title": "Tic-Tac-Toe",
    "description": "Two players alternate placing marks on a 3x3 board; three in a row wins, a full board is a draw.",
    "agents": [0, 1],
    "observation_size": 18,
    "action": {"type": "discrete", "n": 9},
    "turn_based": True,
    "max_steps": 10,
    "symmetric_agents": True,
    "action_names": ["top-left", "top", "top-right", "left", "center", "right",
                     "bottom-left", "bottom", "bottom-right"],
}

LINES = ((0, 1, 2), (3, 4, 5), (6, 7, 8), (0, 3, 6), (1, 4, 7), (2, 5, 8), (0, 4, 8), (2, 4, 6))


def _wins(board, mark):
    return any(all(board[i] == mark for i in line) for line in LINES)


class Environment:
    def reset(self, seed=None):
        self.rng = random.Random(seed)
        self.board = [0] * 9      # 0 empty, 1 = agent 0 (X), 2 = agent 1 (O)
        self.current = 0
        self.winner = None
        self.last_move = None

    def clone(self):
        other = Environment()
        other.rng = self.rng
        other.board = list(self.board)
        other.current, other.winner, other.last_move = self.current, self.winner, self.last_move
        return other

    def observe(self, agent_id):
        own, opp = agent_id + 1, 2 - agent_id
        return ([1.0 if c == own else 0.0 for c in self.board]
                + [1.0 if c == opp else 0.0 for c in self.board])

    def legal_actions(self, agent_id):
        if self.is_terminal() or agent_id != self.current:
            return []
        return [i for i, c in enumerate(self.board) if c == 0]

    def active_agents(self):
        return [] if self.is_terminal() else [self.current]

    def step(self, actions):
        if self.is_terminal():
            raise ValueError("game is over")
        agent = self.current
        if agent not in actions:
            raise ValueError("it is not that player's turn")
        cell = actions[agent]
        if not 0 <= cell < 9 or self.board[cell] != 0:
            raise ValueError("illegal move: cell %s is not empty" % cell)
        self.board[cell] = agent + 1
        self.last_move = cell
        rewards = {0: 0.0, 1: 0.0}
        if _wins(self.board, agent + 1):
            self.winner = agent
            rewards[agent], rewards[1 - agent] = 1.0, -1.0
        else:
            self.current = 1 - agent
        return rewards

    def scripted_action(self, agent_id):
        """Win if possible, else block, else centre, else a corner, else any cell."""
        legal = self.legal_actions(agent_id)
        if not legal:
            return 0
        me, them = agent_id + 1, 2 - agent_id
        for mark in (me, them):
            for cell in legal:
                trial = list(self.board)
                trial[cell] = mark
                if _wins(trial, mark):
                    return cell
        if 4 in legal:
            return 4
        corners = [c for c in (0, 2, 6, 8) if c in legal]
        return self.rng.choice(corners or legal)

    def is_terminal(self):
        return self.winner is not None or all(c != 0 for c in self.board)

    def result(self, agent_id):
        if self.winner is None:
            return "draw"
        return "win" if self.winner == agent_id else "loss"

    def render_text(self):
        marks = {0: None, 1: "X", 2: "O"}
        rows = [" " + " | ".join(marks[self.board[r * 3 + c]] or str(r * 3 + c) for c in range(3))
                for r in range(3)]
        if self.is_terminal():
            status = "Draw." if self.winner is None else "%s wins!" % ("X" if self.winner == 0 else "O")
        else:
            status = "%s to move - type a cell number" % ("X" if self.current == 0 else "O")
        return "\n---+---+---\n".join(rows) + "\n\n" + status

    def render(self):
        size, pad = 90, 10
        items = []
        legal = self.legal_actions(self.current) if not self.is_terminal() else []
        for i, c in enumerate(self.board):
            x, y = pad + (i % 3) * size, pad + (i // 3) * size
            cell = {"type": "rect", "x": x, "y": y, "w": size - 6, "h": size - 6, "radius": 8,
                    "fill": "#263545", "stroke": "#82d5bb" if i == self.last_move else "#3b4d60"}
            if i in legal:
                cell["action"] = i
            items.append(cell)
            cx, cy = x + (size - 6) / 2, y + (size - 6) / 2
            if c == 1:
                items.append({"type": "line", "x1": cx - 22, "y1": cy - 22, "x2": cx + 22, "y2": cy + 22,
                              "stroke": "#82d5bb", "stroke_width": 7})
                items.append({"type": "line", "x1": cx - 22, "y1": cy + 22, "x2": cx + 22, "y2": cy - 22,
                              "stroke": "#82d5bb", "stroke_width": 7})
            elif c == 2:
                items.append({"type": "circle", "x": cx, "y": cy, "r": 24, "stroke": "#f0cf75",
                              "stroke_width": 7, "fill": "none"})
        if self.winner is not None:
            mark = self.winner + 1
            for line in LINES:
                if all(self.board[i] == mark for i in line):
                    a, b = line[0], line[2]
                    items.append({"type": "line",
                                  "x1": pad + (a % 3) * size + (size - 6) / 2, "y1": pad + (a // 3) * size + (size - 6) / 2,
                                  "x2": pad + (b % 3) * size + (size - 6) / 2, "y2": pad + (b // 3) * size + (size - 6) / 2,
                                  "stroke": "#ff9a9a", "stroke_width": 5, "effect": "pulse"})
        if self.is_terminal():
            hud = "Draw" if self.winner is None else ("X wins" if self.winner == 0 else "O wins")
        else:
            hud = "X to move" if self.current == 0 else "O to move"
        return {"width": 3 * size + 2 * pad, "height": 3 * size + 2 * pad, "items": items, "hud": hud}
