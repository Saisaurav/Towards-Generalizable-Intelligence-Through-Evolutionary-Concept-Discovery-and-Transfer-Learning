import random

SPEC = {
    "title": "Connect 4",
    "description": "Two players drop discs into a 7-column, 6-row grid; the first to connect four in a row, column or diagonal wins.",
    "agents": [0, 1],
    "observation_size": 84,
    "action": {"type": "discrete", "n": 7},
    "turn_based": True,
    "max_steps": 45,
    "symmetric_agents": True,
    "action_names": ["col 1", "col 2", "col 3", "col 4", "col 5", "col 6", "col 7"],
}

ROWS, COLS = 6, 7


def _four_at(board, r, c, mark):
    """True if the disc just placed at (r, c) completes four in a row."""
    for dr, dc in ((0, 1), (1, 0), (1, 1), (1, -1)):
        count = 1
        for sign in (1, -1):
            rr, cc = r + sign * dr, c + sign * dc
            while 0 <= rr < ROWS and 0 <= cc < COLS and board[rr][cc] == mark:
                count += 1
                rr += sign * dr
                cc += sign * dc
        if count >= 4:
            return True
    return False


class Environment:
    def reset(self, seed=None):
        self.rng = random.Random(seed)
        self.board = [[0] * COLS for _ in range(ROWS)]   # row 0 is the top; 1 = agent 0, 2 = agent 1
        self.heights = [0] * COLS
        self.current = 0
        self.winner = None
        self.moves = 0
        self.last = None

    def clone(self):
        other = Environment()
        other.rng = self.rng
        other.board = [list(row) for row in self.board]
        other.heights = list(self.heights)
        other.current, other.winner, other.moves, other.last = self.current, self.winner, self.moves, self.last
        return other

    def _drop_row(self, col):
        return ROWS - 1 - self.heights[col]

    def observe(self, agent_id):
        own, opp = agent_id + 1, 2 - agent_id
        flat = [cell for row in self.board for cell in row]
        return [1.0 if v == own else 0.0 for v in flat] + [1.0 if v == opp else 0.0 for v in flat]

    def legal_actions(self, agent_id):
        if self.is_terminal() or agent_id != self.current:
            return []
        return [c for c in range(COLS) if self.heights[c] < ROWS]

    def active_agents(self):
        return [] if self.is_terminal() else [self.current]

    def step(self, actions):
        if self.is_terminal():
            raise ValueError("game is over")
        agent = self.current
        if agent not in actions:
            raise ValueError("it is not that player's turn")
        col = actions[agent]
        if not 0 <= col < COLS or self.heights[col] >= ROWS:
            raise ValueError("illegal move: column %s is full" % col)
        row = self._drop_row(col)
        self.board[row][col] = agent + 1
        self.heights[col] += 1
        self.moves += 1
        self.last = (row, col)
        rewards = {0: 0.0, 1: 0.0}
        if _four_at(self.board, row, col, agent + 1):
            self.winner = agent
            rewards[agent], rewards[1 - agent] = 1.0, -1.0
        else:
            self.current = 1 - agent
        return rewards

    def _wins_if(self, col, mark):
        row = self._drop_row(col)
        self.board[row][col] = mark
        won = _four_at(self.board, row, col, mark)
        self.board[row][col] = 0
        return won

    def scripted_action(self, agent_id):
        """Win now, block a win, avoid handing over a win, prefer the centre columns."""
        legal = self.legal_actions(agent_id)
        if not legal:
            return 0
        me, them = agent_id + 1, 2 - agent_id
        for mark in (me, them):
            for col in legal:
                if self._wins_if(col, mark):
                    return col
        safe = []
        for col in legal:
            if self.heights[col] + 1 < ROWS:
                row = self._drop_row(col)
                self.board[row][col] = me
                self.heights[col] += 1
                gives = self._wins_if(col, them)
                self.heights[col] -= 1
                self.board[row][col] = 0
                if gives:
                    continue
            safe.append(col)
        pool = safe or legal
        best = min(abs(c - 3) for c in pool)
        return self.rng.choice([c for c in pool if abs(c - 3) == best])

    def is_terminal(self):
        return self.winner is not None or self.moves >= ROWS * COLS

    def result(self, agent_id):
        if self.winner is None:
            return "draw"
        return "win" if self.winner == agent_id else "loss"

    def render_text(self):
        sym = {0: ".", 1: "X", 2: "O"}
        rows = [" ".join(sym[v] for v in row) for row in self.board]
        header = " ".join(str(c) for c in range(COLS))
        if self.is_terminal():
            status = "Draw." if self.winner is None else "%s wins!" % ("X" if self.winner == 0 else "O")
        else:
            status = "%s to move - type a column number (0-6)" % ("X" if self.current == 0 else "O")
        return "\n".join(rows) + "\n" + header + "\n\n" + status

    def render(self):
        cell, pad = 56, 12
        width, height = COLS * cell + 2 * pad, ROWS * cell + 2 * pad + 8
        legal = self.legal_actions(self.current) if not self.is_terminal() else []
        items = [{"type": "rect", "x": 0, "y": 8, "w": width, "h": height - 8, "radius": 12, "fill": "#1d4f91"}]
        for r in range(ROWS):
            for c in range(COLS):
                v = self.board[r][c]
                fill = "#101721" if v == 0 else "#f0cf75" if v == 1 else "#ff9a9a"
                item = {"type": "circle", "x": pad + c * cell + cell / 2, "y": 8 + pad + r * cell + cell / 2,
                        "r": cell * 0.4, "fill": fill}
                if self.last == (r, c):
                    item["stroke"] = "#ffffff"
                    item["stroke_width"] = 3
                items.append(item)
        for c in legal:   # clickable column strips drawn on top of the board
            items.append({"type": "rect", "x": pad + c * cell, "y": 8, "w": cell, "h": ROWS * cell + 2 * pad,
                          "fill": "#ffffff", "opacity": 0.001, "action": c})
        if self.is_terminal():
            hud = "Draw" if self.winner is None else ("Yellow wins" if self.winner == 0 else "Red wins")
        else:
            hud = "Yellow to move" if self.current == 0 else "Red to move"
        return {"width": width, "height": height, "items": items, "hud": hud}
