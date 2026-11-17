import random

# Chess for two agents (agent 0 = White, agent 1 = Black).
#
# One chess move is two actions, because the learner's action space is capped at 128 ids:
#   1. choose the square of the piece to move  (64 ids, one per square)
#   2. choose the square to move it to         (same 64 ids; the piece's own square cancels the choice)
# Squares are numbered from the mover's own side (id 0 = a8 for White; Black sees the board
# mirrored top-to-bottom), so one policy can play either colour. Pawns promote to a queen.
# Rewards: +1 for checkmate, -1 for being mated, 0 for a draw (stalemate, 50-move rule,
# threefold repetition, insufficient material, or the step cap). Captures add a small zero-sum
# bonus (0.02 x piece value) so random play already produces a learning signal.

SPEC = {
    "title": "Chess",
    "description": "Standard chess for two players. A move is two actions: pick a piece, then pick its destination square. Checkmate wins; stalemate, repetition, the 50-move rule and insufficient material are draws.",
    "agents": [0, 1],
    "observation_size": 838,
    "action": {"type": "discrete", "n": 64},
    "turn_based": True,
    "max_steps": 400,
    "symmetric_agents": True,
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
}

VALUE = (0, 1, 3, 3, 5, 9, 0)          # pawn, knight, bishop, rook, queen, king (index = piece type)
GLYPH = {1: "\u265f\ufe0e", 2: "\u265e", 3: "\u265d", 4: "\u265c", 5: "\u265b", 6: "\u265a"}
LETTER = {1: "P", 2: "N", 3: "B", 4: "R", 5: "Q", 6: "K"}
SHAPING = 0.02
IDLE_LIMIT = 2


def _ray(r, c, dr, dc):
    out = []
    r, c = r + dr, c + dc
    while 0 <= r < 8 and 0 <= c < 8:
        out.append(r * 8 + c)
        r, c = r + dr, c + dc
    return tuple(out)


def _jumps(s, deltas):
    r, c = divmod(s, 8)
    return tuple((r + dr) * 8 + (c + dc) for dr, dc in deltas if 0 <= r + dr < 8 and 0 <= c + dc < 8)


ROOK_RAYS = [[_ray(s // 8, s % 8, dr, dc) for dr, dc in ((-1, 0), (1, 0), (0, -1), (0, 1))] for s in range(64)]
BISHOP_RAYS = [[_ray(s // 8, s % 8, dr, dc) for dr, dc in ((-1, -1), (-1, 1), (1, -1), (1, 1))] for s in range(64)]
KNIGHT_T = [_jumps(s, ((-2, -1), (-2, 1), (-1, -2), (-1, 2), (1, -2), (1, 2), (2, -1), (2, 1))) for s in range(64)]
KING_T = [_jumps(s, ((-1, -1), (-1, 0), (-1, 1), (0, -1), (0, 1), (1, -1), (1, 0), (1, 1))) for s in range(64)]
PAWN_TO = [[_jumps(s, ((-1, -1), (-1, 1))) for s in range(64)],      # squares a White pawn on s attacks
           [_jumps(s, ((1, -1), (1, 1))) for s in range(64)]]        # squares a Black pawn on s attacks
PAWN_FROM = [[_jumps(s, ((1, -1), (1, 1))) for s in range(64)],      # squares a White pawn attacks s from
             [_jumps(s, ((-1, -1), (-1, 1))) for s in range(64)]]    # squares a Black pawn attacks s from
CASTLE_ROOKS = ((63, 0), (56, 1), (7, 2), (0, 3))                    # (rook square, castling-right index)


class Pos:
    """Board state. b[s]: 0 empty, +1..+6 White P N B R Q K, -1..-6 Black. s = 0 is a8, s = 63 is h1."""

    def __init__(self, b, turn, castle, ep, half):
        self.b, self.turn, self.castle, self.ep, self.half = b, turn, castle, ep, half

    @staticmethod
    def start():
        b = [0] * 64
        back = (4, 2, 3, 5, 6, 3, 2, 4)
        for c in range(8):
            b[c], b[8 + c], b[48 + c], b[56 + c] = -back[c], -1, 1, back[c]
        return Pos(b, 0, [True, True, True, True], -1, 0)

    def copy(self):
        return Pos(self.b[:], self.turn, self.castle[:], self.ep, self.half)

    def key(self):
        return (tuple(self.b), self.turn, tuple(self.castle), self.ep)

    def attacked(self, sq, by):
        b = self.b
        s = 1 if by == 0 else -1
        for p in PAWN_FROM[by][sq]:
            if b[p] == s:
                return True
        for t in KNIGHT_T[sq]:
            if b[t] == 2 * s:
                return True
        for t in KING_T[sq]:
            if b[t] == 6 * s:
                return True
        for ray in ROOK_RAYS[sq]:
            for t in ray:
                v = b[t]
                if v:
                    if v == 4 * s or v == 5 * s:
                        return True
                    break
        for ray in BISHOP_RAYS[sq]:
            for t in ray:
                v = b[t]
                if v:
                    if v == 3 * s or v == 5 * s:
                        return True
                    break
        return False

    def in_check(self):
        s = 6 if self.turn == 0 else -6
        return self.attacked(self.b.index(s), 1 - self.turn)

    def pseudo(self):
        """Moves as (from, to, flag). flag: 0 normal, 1 en passant, 2 O-O, 3 O-O-O, 4 double push, 5 promotion."""
        b, turn = self.b, self.turn
        s = 1 if turn == 0 else -1
        moves = []
        for f in range(64):
            t = b[f] * s
            if t <= 0:
                continue
            if t == 1:
                fw = -8 if turn == 0 else 8
                promo_row = 0 if turn == 0 else 7
                to = f + fw
                if 0 <= to < 64 and b[to] == 0:
                    if to >> 3 == promo_row:
                        moves.append((f, to, 5))
                    else:
                        moves.append((f, to, 0))
                        if (f >> 3) == (6 if turn == 0 else 1) and b[to + fw] == 0:
                            moves.append((f, to + fw, 4))
                for to in PAWN_TO[turn][f]:
                    target = b[to]
                    if target * s < 0:
                        moves.append((f, to, 5 if to >> 3 == promo_row else 0))
                    elif target == 0 and to == self.ep:
                        moves.append((f, to, 1))
            elif t == 2:
                for to in KNIGHT_T[f]:
                    if b[to] * s <= 0:
                        moves.append((f, to, 0))
            elif t == 6:
                for to in KING_T[f]:
                    if b[to] * s <= 0:
                        moves.append((f, to, 0))
                if f == (60 if turn == 0 else 4) and (self.castle[2 * turn] or self.castle[2 * turn + 1]):
                    if not self.attacked(f, 1 - turn):
                        if (self.castle[2 * turn] and b[f + 1] == 0 and b[f + 2] == 0 and b[f + 3] == 4 * s
                                and not self.attacked(f + 1, 1 - turn)):
                            moves.append((f, f + 2, 2))
                        if (self.castle[2 * turn + 1] and b[f - 1] == 0 and b[f - 2] == 0 and b[f - 3] == 0
                                and b[f - 4] == 4 * s and not self.attacked(f - 1, 1 - turn)):
                            moves.append((f, f - 2, 3))
            else:
                if t != 3:      # rook or queen
                    for ray in ROOK_RAYS[f]:
                        for to in ray:
                            v = b[to] * s
                            if v <= 0:
                                moves.append((f, to, 0))
                            if v != 0:
                                break
                if t != 4:      # bishop or queen
                    for ray in BISHOP_RAYS[f]:
                        for to in ray:
                            v = b[to] * s
                            if v <= 0:
                                moves.append((f, to, 0))
                            if v != 0:
                                break
        return moves

    def child(self, m):
        """(position after move m, absolute value of the captured piece type)."""
        f, t, flag = m
        turn = self.turn
        b = self.b[:]
        v = b[f]
        captured = abs(b[t])
        b[t], b[f] = v, 0
        ep = -1
        half = 0 if (captured or abs(v) == 1) else self.half + 1
        if flag == 1:
            cap_sq = t + 8 if turn == 0 else t - 8
            captured = abs(b[cap_sq])
            b[cap_sq] = 0
        elif flag == 2:
            b[f + 1], b[f + 3] = b[f + 3], 0
        elif flag == 3:
            b[f - 1], b[f - 4] = b[f - 4], 0
        elif flag == 4:
            ep = (f + t) // 2
        elif flag == 5:
            b[t] = 5 if turn == 0 else -5
        castle = self.castle[:]
        if abs(v) == 6:
            castle[2 * turn] = castle[2 * turn + 1] = False
        for sq, idx in CASTLE_ROOKS:
            if f == sq or t == sq:
                castle[idx] = False
        return Pos(b, 1 - turn, castle, ep, half), captured

    def legal(self):
        turn = self.turn
        king = 6 if turn == 0 else -6
        out = []
        for m in self.pseudo():
            ch = self.child(m)[0]
            if not ch.attacked(ch.b.index(king), 1 - turn):
                out.append(m)
        return out

    def has_legal(self):
        turn = self.turn
        king = 6 if turn == 0 else -6
        for m in self.pseudo():
            ch = self.child(m)[0]
            if not ch.attacked(ch.b.index(king), 1 - turn):
                return True
        return False

    def insufficient(self):
        minors = 0
        for v in self.b:
            a = abs(v)
            if a in (1, 4, 5):
                return False
            if a in (2, 3):
                minors += 1
                if minors > 1:
                    return False
        return True


class Environment:
    def reset(self, seed=None):
        self.rng = random.Random(seed)
        self.pos = Pos.start()
        self.seen = {self.pos.key(): 1}
        self.phase = 0              # 0: choose a piece, 1: choose its destination
        self.sel = -1               # absolute square of the chosen piece
        self.idle = 0               # consecutive cancelled selections
        self.plies = 0
        self.winner = None
        self.reason = None
        self.last = None            # (from, to) of the latest move, absolute squares
        self._moves_cache = None

    def clone(self):
        other = Environment()
        other.rng = self.rng
        other.pos = self.pos.copy()
        other.seen = dict(self.seen)
        other.phase, other.sel, other.idle, other.plies = self.phase, self.sel, self.idle, self.plies
        other.winner, other.reason, other.last = self.winner, self.reason, self.last
        other._moves_cache = None
        return other

    # ---- helpers -------------------------------------------------------------------------
    @staticmethod
    def _rel(agent_id, sq):
        return sq if agent_id == 0 else sq ^ 56

    def _moves(self):
        if self._moves_cache is None:
            self._moves_cache = self.pos.legal()
        return self._moves_cache

    # ---- interface -----------------------------------------------------------------------
    def observe(self, agent_id):
        b = self.pos.b
        own = 1 if agent_id == 0 else -1
        planes = [0.0] * 768
        for s in range(64):
            v = b[s]
            if v:
                rs = s if agent_id == 0 else s ^ 56
                kind = abs(v) - 1
                planes[(kind if v * own > 0 else 6 + kind) * 64 + rs] = 1.0
        sel = [0.0] * 64
        if self.sel >= 0 and agent_id == self.pos.turn:
            sel[self._rel(agent_id, self.sel)] = 1.0
        c = self.pos.castle
        me, them = 2 * agent_id, 2 * (1 - agent_id)
        flags = [1.0 if (self.phase == 1 and agent_id == self.pos.turn) else 0.0,
                 1.0 if c[me] else 0.0, 1.0 if c[me + 1] else 0.0,
                 1.0 if c[them] else 0.0, 1.0 if c[them + 1] else 0.0,
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
            out.add(self._rel(agent_id, self.sel))      # cancel and pick another piece
        return sorted(out)

    def active_agents(self):
        return [] if self.is_terminal() else [self.pos.turn]

    def step(self, actions):
        if self.is_terminal():
            raise ValueError("game is over")
        agent = self.pos.turn
        if agent not in actions:
            raise ValueError("it is not that player's turn")
        action = actions[agent]
        if action not in self.legal_actions(agent):
            raise ValueError("illegal action %s for the current phase" % action)
        sq = self._rel(agent, action)          # the mirror is its own inverse
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
        self.pos, captured = self.pos.child(move)
        self.last = (move[0], move[1])
        self.sel, self.phase, self.idle = -1, 0, 0
        self.plies += 1
        self._moves_cache = None
        bonus = SHAPING * VALUE[captured]
        rewards[agent] += bonus
        rewards[1 - agent] -= bonus
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
        elif self.pos.half >= 100:
            self.reason = "50-move rule"
        elif self.seen[key] >= 3:
            self.reason = "threefold repetition"
        return rewards

    def is_terminal(self):
        return self.reason is not None

    def result(self, agent_id):
        if self.winner is None:
            return "draw"
        return "win" if self.winner == agent_id else "loss"

    # ---- scripted opponent ----------------------------------------------------------------
    def _score(self, pos, m, noise):
        f, t, flag = m
        me = pos.turn
        piece = abs(pos.b[f])
        captured = 1 if flag == 1 else abs(pos.b[t])
        ch = pos.child(m)[0]
        score = float(VALUE[captured])
        moved = 5 if flag == 5 else piece
        if flag == 5:
            score += 8.0
        if ch.attacked(t, 1 - me):                  # the piece can be taken on arrival
            if ch.attacked(t, me):
                score -= max(0, VALUE[moved] - VALUE[captured]) * 0.9
            else:
                score -= VALUE[moved]
        gives_check = ch.in_check()
        if gives_check:
            score += 0.4
        if gives_check or self.plies > 80:
            if not ch.has_legal():
                score += 1000.0 if gives_check else -3.0
        if piece == 1:
            score += 0.05 * ((6 - (f >> 3)) if me == 0 else ((f >> 3) - 1))
        if t in (27, 28, 35, 36):
            score += 0.15
        if flag in (2, 3):
            score += 0.6
        elif piece == 6:
            score -= 0.3
        elif piece in (2, 3) and (f >> 3) == (7 if me == 0 else 0):
            score += 0.15
        return score + noise * self.rng.random()

    def scripted_action(self, agent_id):
        """Mate in one, then win material safely, give checks, develop and castle."""
        legal = self.legal_actions(agent_id)
        if not legal:
            return 0
        moves = self._moves()
        if self.phase == 1:
            moves = [m for m in moves if m[0] == self.sel]
        if not moves:
            return legal[0]
        noise = 0.05 if self.phase == 0 else 0.0
        best, best_score = moves[0], None
        for m in moves:
            sc = self._score(self.pos, m, noise)
            if best_score is None or sc > best_score:
                best, best_score = m, sc
        return self._rel(agent_id, best[0] if self.phase == 0 else best[1])

    # ---- display -------------------------------------------------------------------------
    def _status(self):
        names = ("White", "Black")
        if self.is_terminal():
            if self.winner is None:
                return "Draw by %s." % self.reason
            return "Checkmate. %s wins." % names[self.winner]
        mover = names[self.pos.turn]
        if self.phase == 0:
            return "%s to move: choose a piece." % mover + (" Check!" if self.pos.in_check() else "")
        return "%s: choose where to move the piece (or pick its square again to cancel)." % mover

    def render_text(self):
        turn = self.pos.turn
        rows = []
        for r in range(8):
            rr = r if turn == 0 else 7 - r
            cells = []
            for c in range(8):
                v = self.pos.b[rr * 8 + c]
                cells.append("." if v == 0 else (LETTER[abs(v)] if v > 0 else LETTER[abs(v)].lower()))
            rows.append("%d  %s" % (8 - r, " ".join(cells)))
        text = "\n".join(rows) + "\n\n   a b c d e f g h\n\nWhite = capitals, Black = lower case.\n"
        if not self.is_terminal():
            text += "Squares are named from the mover's side (rank 1 is the mover's own back rank). Type a square, e.g. e2.\n"
        return text + self._status()

    def render(self):
        cell, pad = 54, 26
        size = 8 * cell + 2 * pad
        turn = self.pos.turn
        over = self.is_terminal()
        items = [{"type": "rect", "x": pad - 4, "y": pad - 4, "w": 8 * cell + 8, "h": 8 * cell + 8,
                  "radius": 3, "fill": "#0d110b"}]
        king_sq = -1
        if not over and self.pos.in_check():
            king_sq = self.pos.b.index(6 if turn == 0 else -6)
        for s in range(64):
            r, c = divmod(s, 8)
            light = (r + c) % 2 == 0
            items.append({"type": "rect", "x": pad + c * cell, "y": pad + r * cell, "w": cell, "h": cell,
                          "fill": "#cfd6b8" if light else "#6f7f54"})
            tint = None
            if self.last and s in self.last:
                tint, alpha = "#c7f36a", 0.38
            if s == self.sel:
                tint, alpha = "#c7f36a", 0.6
            if s == king_sq:
                tint, alpha = "#ff5c5c", 0.55
            if tint:
                items.append({"type": "rect", "x": pad + c * cell, "y": pad + r * cell, "w": cell, "h": cell,
                              "fill": tint, "opacity": alpha})
        for i in range(8):
            items.append({"type": "text", "x": pad + i * cell + cell / 2, "y": size - pad / 2 + 1,
                          "text": "abcdefgh"[i], "size": 11, "fill": "#8a9378"})
            items.append({"type": "text", "x": pad / 2, "y": pad + i * cell + cell / 2,
                          "text": str(8 - i), "size": 11, "fill": "#8a9378"})
        for s in range(64):
            v = self.pos.b[s]
            if not v:
                continue
            r, c = divmod(s, 8)
            white = v > 0
            items.append({"type": "text", "x": pad + c * cell + cell / 2, "y": pad + r * cell + cell / 2 + 2,
                          "text": GLYPH[abs(v)], "size": 42,
                          "fill": "#f7f4ea" if white else "#1b2016",
                          "stroke": "#1b2016" if white else "#e3ebcf", "stroke_width": 1.1})
        if not over:
            legal = self.legal_actions(turn)
            for a in legal:
                s = self._rel(turn, a)
                r, c = divmod(s, 8)
                cx, cy = pad + c * cell + cell / 2, pad + r * cell + cell / 2
                if self.phase == 1 and s != self.sel:
                    occupied = self.pos.b[s] != 0
                    items.append({"type": "circle", "x": cx, "y": cy, "r": 21 if occupied else 8,
                                  "fill": "#c7f36a", "opacity": 0.35 if occupied else 0.75})
            for a in legal:                    # transparent click targets, drawn last
                s = self._rel(turn, a)
                r, c = divmod(s, 8)
                items.append({"type": "rect", "x": pad + c * cell, "y": pad + r * cell, "w": cell, "h": cell,
                              "fill": "#ffffff", "opacity": 0.001, "action": a})
        return {"width": size, "height": size, "background": "#171c14", "items": items, "hud": self._status()}
