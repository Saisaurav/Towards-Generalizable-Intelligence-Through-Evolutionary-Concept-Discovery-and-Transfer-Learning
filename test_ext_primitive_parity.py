"""Integration-boundary test: Python NeatPolicy concept features == C++ ConceptEnv values.

A trained policy is exported from the C++ learner but executed in Python (human play, evaluation, self-play
opponents). If the Python primitive computation drifts from the C++ one, the exported brain silently behaves
differently from the brain that was trained. This test drives V8's real C++ ConceptEnv (via a mock Env, exactly
how a generated game is wrapped) and compares every legal-action concept value with
``llm_game_platform.neat_policy.concept_features`` over randomized observations.
"""
import math
import random
import shutil
import subprocess
import tempfile
import unittest
from pathlib import Path

from llm_game_platform.neat_policy import concept_features
from llm_game_platform.policy import (_GRID_PRIMITIVES, _concept_value, _state_primitives, parse_expression,
                                      screen_concepts)

ROOT = Path(__file__).resolve().parent


def _cases(rng, n=300):
    out = []
    for _ in range(n):
        n_state = rng.choice([1, 2, 3, 4, 9, 16, 25, 42, 50])
        kind = rng.choice(["uniform", "board", "sparse", "constant", "onehot", "ramp"])
        if kind == "uniform": s = [rng.uniform(-1, 1) for _ in range(n_state)]
        elif kind == "board": s = [rng.choice([-1.0, 0.0, 1.0]) for _ in range(n_state)]
        elif kind == "sparse": s = [rng.uniform(-1, 1) if rng.random() < 0.15 else 0.0 for _ in range(n_state)]
        elif kind == "constant": s = [rng.choice([0.0, 0.5, -0.25])] * n_state
        elif kind == "onehot": s = [0.0] * n_state; s[rng.randrange(n_state)] = 1.0
        else: s = [i / max(1, n_state - 1) for i in range(n_state)]
        s = [float(f"{v:.6f}") for v in s]           # exactly representable text for both sides
        n_actions = rng.choice([1, 2, 3, 4, 9, 42, 100])
        legal = sorted(rng.sample(range(n_actions), rng.randint(1, min(n_actions, 7))))
        out.append((n_actions, legal, s))
    return out


@unittest.skipUnless(shutil.which("g++"), "g++ is required to build the C++ parity harness")
class ExtPrimitiveParity(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tmp = tempfile.TemporaryDirectory()
        cls.exe = Path(cls.tmp.name) / "parity"
        subprocess.run(["g++", "-std=c++17", "-O2", "-I", str(ROOT / "src"),
                        str(ROOT / "src" / "ext_primitive_parity_test.cpp"), "-o", str(cls.exe)],
                       check=True, capture_output=True)

    @classmethod
    def tearDownClass(cls):
        cls.tmp.cleanup()

    @staticmethod
    def _ill_conditioned(tree, s, lg, na, action, tol, rng):
        """True if the *Python* value is unstable under float32-sized perturbations of its own inputs.

        C++ evaluates primitives in float32, Python in float64. Where an expression divides by a quantity that
        cancels to rounding noise (e.g. ``action_norm + sqrt(state_entropy)`` = -1 + 1), the clamped division
        flips sign between the two precisions. That is a property of the concept expression at that point, not
        of the adapter, so such points are reported and bounded rather than compared.
        """
        prims = _state_primitives(s, len(lg))
        prims["legal_norm"] = len(lg) / na if na > 1 else 1.0
        prims["action_norm"] = 2.0 * action / (na - 1) - 1.0 if na > 1 else 0.0
        base = _concept_value(tree, prims)
        for _ in range(8):
            noisy = {k: v * (1.0 + rng.choice([-1, 1]) * 4e-7) for k, v in prims.items()}
            if abs(_concept_value(tree, noisy) - base) > tol:
                return True
        return False

    def _compare(self, expressions, tol):
        trees = [parse_expression(e) for e in expressions]
        expr_file = Path(self.tmp.name) / "expr.txt"
        expr_file.write_text("\n".join(expressions) + "\n")
        cases = _cases(random.Random(20260514))
        stdin = "".join(f"{na} {len(lg)} {' '.join(map(str, lg))} {len(s)} {' '.join(repr(v) for v in s)}\n"
                        for na, lg, s in cases)
        res = subprocess.run([str(self.exe), str(expr_file)], input=stdin, capture_output=True, text=True, check=True)
        lines = res.stdout.strip().split("\n")
        i = 0; total = 0; ill = 0; worst = (0.0, None); noise = random.Random(7)
        for na, lg, s in cases:
            rows = concept_features(trees, s, lg, na)
            for action, row in zip(lg, rows):
                cpp = [float(x) for x in lines[i].split()]; i += 1
                for e, tree, a, b in zip(expressions, trees, row, cpp):
                    total += 1
                    d = abs(a - b)
                    if d > tol and self._ill_conditioned(tree, s, lg, na, action, tol, noise):
                        ill += 1
                        continue
                    if d > worst[0]: worst = (d, (e, na, lg, s[:6], a, b))
        self.assertEqual(i, len(lines))
        self.assertLessEqual(worst[0], tol, f"max |python-cpp| = {worst[0]:.3g} at {worst[1]}")
        self.assertLessEqual(ill / total, 0.005, f"{ill}/{total} comparisons were ill-conditioned: too many to excuse")
        return worst[0], ill, total

    def test_every_screened_primitive_matches_cpp(self):
        prims = sorted(_GRID_PRIMITIVES)
        worst, ill, total = self._compare(prims, tol=2e-4)
        print(f"\n  [parity] {len(prims)} primitives, {total} comparisons, max |py-cpp| = {worst:.3g}, ill-conditioned excused = {ill}")

    def test_one_hot_action_identity_follows_concept_features(self):
        trees = [parse_expression("action_norm")]
        rows = concept_features(trees, [0.0], [1, 3], 4, action_onehot=True)
        self.assertEqual(rows, [[math.tanh(2.0 / 3.0 - 1.0), 0.0, 1.0, 0.0, 0.0],
                                [math.tanh(1.0), 0.0, 0.0, 0.0, 1.0]])

    def test_all_accepted_bank_concepts_match_cpp(self):
        accepted, _ = screen_concepts(ROOT / "results" / "concept_bank.json")
        self.assertTrue(accepted)
        worst, ill, total = self._compare(accepted, tol=5e-3)     # nested div() amplifies float32 vs float64 rounding
        print(f"\n  [parity] {len(accepted)} bank concepts, {total} comparisons, max |py-cpp| = {worst:.3g}, ill-conditioned excused = {ill}")


if __name__ == "__main__":
    unittest.main()
