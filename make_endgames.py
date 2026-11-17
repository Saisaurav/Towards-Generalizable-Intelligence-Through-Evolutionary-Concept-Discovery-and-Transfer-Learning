"""Generate builtin_envs/chess_kqk.py and chess_krk.py (king+queen vs king, king+rook vs king) from chess.py's move generator."""
from pathlib import Path
ROOT = Path(__file__).resolve().parent
src = (ROOT / "builtin_envs" / "chess.py").read_text(encoding="utf-8")
pos_code = src[src.index("def _ray"):src.index("class Environment:")]
display = src[src.index("    # ---- display"):]

HEAD = '''import random

# {title}. Agent 0 (White, the learner) has a king and a {pname}; agent 1 (Black) has only a king.
# White must checkmate before the move limit; stalemate, losing the {pname}, repetition or the move limit are draws.
# One chess move is two actions: pick the piece's square, then its destination (the piece's own square cancels).
# Squares are numbered from the mover's own side (id 0 = a8 for White; Black sees the board mirrored).
# Rewards: +1 checkmate, -1 for being mated, 0 draw, plus small zero-sum shaping: pushing Black's king toward
# the edge and a capture bonus (taking the {pname} costs White 0.02 x its value).

SPEC = {{
    "title": "{title}",
    "description": "{desc}",
    "agents": [0, 1],
    "observation_size": 322,
    "action": {{"type": "discrete", "n": 64}},
    "turn_based": True,
    "max_steps": {max_steps},
    "symmetric_agents": False,
    "action_names": [
                     "a8", "b8", "c8", "d8", "e8", "f8", "g8", "h8",
                     "a7", "b7", "c7", "d7", "e7", "f7", "g7", "h7",
                     "a6", "b6", "c6", "d6", "e6", "f6", "g6", "h6",
                     "a5", "b5", "c5", "d5", "e5", "f5", "g5", "h5",
                     "a4", "b4", "c4", "d4", "e4", "f4", "g4", "h4",
                     "a3", "b3", "c3", "d3", "e3", "f3", "g3", "h3",
                     "a2", "b2", "c2", "d2", "e2", "f2", "g2", "h2",
                     "a1", "b1", "c1", "d1", "e1", "f1", "g1", "h1",
    ],
}}

VALUE = (0, 1, 3, 3, 5, 9, 0)
GLYPH = {{1: "\\u265f\\ufe0e", 2: "\\u265e", 3: "\\u265d", 4: "\\u265c", 5: "\\u265b", 6: "\\u265a"}}
LETTER = {{1: "P", 2: "N", 3: "B", 4: "R", 5: "Q", 6: "K"}}
SHAPING = 0.02
EDGE_SHAPING = 0.1
IDLE_LIMIT = 2
PIECE = {piece}
MAX_PLIES = {plies}


'''

ENV = '''class Environment:
    def reset(self, seed=None):
        self.rng = random.Random(seed)
        pos = None
        for _ in range(2000):
            wk, bk, pc = self.rng.sample(range(64), 3)
            if max(abs(wk // 8 - bk // 8), abs(wk % 8 - bk % 8)) <= 1:
                continue
            b = [0] * 64
            b[wk], b[bk], b[pc] = 6, -6, PIECE
            cand = Pos(b, 0, [False, False, False, False], -1, 0)
            if cand.attacked(bk, 0) or not cand.has_legal():
                continue                      # Black may not start in check; White needs a move
            pos = cand
            break
        self.pos = pos
        self.seen = {self.pos.key(): 1}
        self.phase = 0
        self.sel = -1
        self.idle = 0
        self.plies = 0
        self.winner = None
        self.reason = None
        self.last = None
        self._moves_cache = None
        self._best = None

    def clone(self):
        other = Environment()
        other.rng = self.rng
        other.pos = self.pos.copy()
        other.seen = dict(self.seen)
        other.phase, other.sel, other.idle, other.plies = self.phase, self.sel, self.idle, self.plies
        other.winner, other.reason, other.last = self.winner, self.reason, self.last
        other._moves_cache = None
        other._best = None
        return other

    @staticmethod
    def _rel(agent_id, sq):
        return sq if agent_id == 0 else sq ^ 56

    def _moves(self):
        if self._moves_cache is None:
            self._moves_cache = self.pos.legal()
        return self._moves_cache

    # ---- interface -----------------------------------------------------------------------
    def observe(self, agent_id):
        # 4 planes x 64: own king, own rook/queen, opposing king, opposing rook/queen; then the selected square
        # plane and two flags (choosing a destination, in check). 322 inputs.
        own = 1 if agent_id == 0 else -1
        planes = [0.0] * 256
        for s in range(64):
            v = self.pos.b[s]
            if v:
                rs = s if agent_id == 0 else s ^ 56
                mine = v * own > 0
                planes[((0 if mine else 2) + (0 if abs(v) == 6 else 1)) * 64 + rs] = 1.0
        sel = [0.0] * 64
        if self.sel >= 0 and agent_id == self.pos.turn:
            sel[self._rel(agent_id, self.sel)] = 1.0
        flags = [1.0 if (self.phase == 1 and agent_id == self.pos.turn) else 0.0,
                 1.0 if (agent_id == self.pos.turn and self.pos.in_check()) else 0.0]
        return planes + sel + flags

    def legal_actions(self, agent_id):
        if self.is_terminal() or agent_id != self.pos.turn:
            return []
        moves = self._moves()
        if self.phase == 0:
            return sorted(set(self._rel(agent_id, m[0]) for m in moves))
        out = set(self._rel(agent_id, m[1]) for m in moves if m[0] == self.sel)
        if self.idle < IDLE_LIMIT:
            out.add(self._rel(agent_id, self.sel))
        return sorted(out)

    def active_agents(self):
        return [] if self.is_terminal() else [self.pos.turn]

    @staticmethod
    def _edge(pos):
        """White's potential: how far Black's king is from the centre (0..1)."""
        bk = pos.b.index(-6)
        return abs(bk // 8 - 3.5) + abs(bk % 8 - 3.5)

    def step(self, actions):
        if self.is_terminal():
            raise ValueError("game is over")
        agent = self.pos.turn
        if agent not in actions:
            raise ValueError("it is not that player's turn")
        action = actions[agent]
        if action not in self.legal_actions(agent):
            raise ValueError("illegal action %s for the current phase" % action)
        sq = self._rel(agent, action)
        rewards = {0: 0.0, 1: 0.0}
        if self.phase == 0:
            self.sel, self.phase = sq, 1
            return rewards
        if sq == self.sel:
            self.sel, self.phase = -1, 0
            self.idle += 1
            return rewards
        move = None
        for m in self._moves():
            if m[0] == self.sel and m[1] == sq:
                move = m
                break
        before = self._edge(self.pos)
        self.pos, captured = self.pos.child(move)
        self.last = (move[0], move[1])
        self.sel, self.phase, self.idle = -1, 0, 0
        self.plies += 1
        self._moves_cache = None
        self._best = None
        bonus = SHAPING * VALUE[captured]
        rewards[agent] += bonus
        rewards[1 - agent] -= bonus
        if self.pos.has_legal() and PIECE in [abs(v) for v in self.pos.b]:
            shape = EDGE_SHAPING * (self._edge(self.pos) - before) / 7.0
            rewards[0] += shape
            rewards[1] -= shape
        key = self.pos.key()
        self.seen[key] = self.seen.get(key, 0) + 1
        if not self.pos.has_legal():
            if self.pos.in_check():
                self.winner, self.reason = agent, "checkmate"
                rewards[agent] += 1.0
                rewards[1 - agent] -= 1.0
            else:
                self.reason = "stalemate"
        elif self.pos.insufficient():
            self.reason = "insufficient material"
        elif self.seen[key] >= 3:
            self.reason = "threefold repetition"
        elif self.plies >= MAX_PLIES:
            self.reason = "move limit"
        return rewards

    def is_terminal(self):
        return self.reason is not None

    def result(self, agent_id):
        if self.winner is None:
            return "draw"
        return "win" if self.winner == agent_id else "loss"

    # ---- scripted players (2-ply search with a mating heuristic) ---------------------------
    @staticmethod
    def _eval(pos, att):
        """Score for the attacking side `att` of a position (higher is better for the attacker)."""
        s = 1 if att == 0 else -1
        has_piece = False
        for v in pos.b:
            if v * s == PIECE:
                has_piece = True
                break
        if not has_piece:
            return -800.0
        if pos.turn != att and not pos.has_legal():
            if pos.in_check():
                return 1000.0
            return -500.0
        dk, ak = pos.b.index(-6 * s), pos.b.index(6 * s)
        centre = abs(dk // 8 - 3.5) + abs(dk % 8 - 3.5)
        kd = abs(dk // 8 - ak // 8) + abs(dk % 8 - ak % 8)
        return 10.0 * centre + 4.0 * (14 - kd)

    def _best_move(self):
        if self._best is not None:
            return self._best
        pos = self.pos
        me = pos.turn
        s = 1 if me == 0 else -1
        attacker = any(v * s == PIECE for v in pos.b)
        att = me if attacker else 1 - me
        best, best_score = None, None
        for m in self._moves():
            ch = pos.child(m)[0]
            replies = ch.legal()
            if not replies:
                score = self._eval(ch, att)
            elif attacker:
                score = min(self._eval(ch.child(r)[0], att) for r in replies)
            else:
                score = max(self._eval(ch.child(r)[0], att) for r in replies)
            if not attacker:
                score = -score                      # the defender wants the smallest attacker score
            if best_score is None or score > best_score:
                best, best_score = m, score
        self._best = best
        return best

    def scripted_action(self, agent_id):
        legal = self.legal_actions(agent_id)
        if not legal:
            return 0
        best = self._best_move()
        if best is None:
            return legal[0]
        if self.phase == 1:
            if best[0] != self.sel:
                return self._rel(agent_id, self.sel)  # cancel this selection, then select the expert's piece
            return self._rel(agent_id, best[1])
        return self._rel(agent_id, best[0])

'''

VARIANTS = {
    "chess_kqk.py": dict(title="Chess endgame: King and Queen vs King", pname="queen", piece=5, plies=60, max_steps=160,
        desc="Chess endgame. White has a king and a queen against Black's lone king and must checkmate. A move is two actions: pick a piece, then its destination square. Stalemate, losing the queen, repetition or running out of moves is a draw."),
    "chess_krk.py": dict(title="Chess endgame: King and Rook vs King", pname="rook", piece=4, plies=100, max_steps=260,
        desc="Chess endgame. White has a king and a rook against Black's lone king and must checkmate. A move is two actions: pick a piece, then its destination square. Stalemate, losing the rook, repetition or running out of moves is a draw."),
}
for name, v in VARIANTS.items():
    (ROOT / "builtin_envs" / name).write_text(HEAD.format(**v) + pos_code + ENV + display, encoding="utf-8")
    print("wrote", name)