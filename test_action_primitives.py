"""Cross-language parity for the action-conditioned primitives (action_wins_now, action_blocks_opponent), plus hand-built boards."""
import math, subprocess, tempfile, unittest, os
import gp_transfer as G

class ActionPrimitiveTests(unittest.TestCase):
    def test_ttt_hand_boards(self):
        own = [1,1,0, 0,0,0, 0,0,0]; opp = [0,0,0, 1,1,0, 0,0,0]       # own can win at cell 2; opponent can win at cell 5
        state = [float(v) for v in own] + [float(v) for v in opp]
        self.assertEqual(G.board_action_features(state, 2, 'ttt'), (1.0, 0.0))
        self.assertEqual(G.board_action_features(state, 5, 'ttt'), (0.0, 1.0))
        self.assertEqual(G.board_action_features(state, 8, 'ttt'), (0.0, 0.0))
        self.assertEqual(G.board_action_features(state, 0, 'ttt'), (0.0, 0.0))      # occupied cell
    def test_connect_four_hand_boards(self):
        rows, cols = 6, 7; own = [[0]*cols for _ in range(rows)]; opp = [[0]*cols for _ in range(rows)]
        for c in range(3): own[5][c] = 1                                  # own three in the bottom row, column 3 wins
        for r in (5, 4, 3): opp[r][6] = 1                                 # opponent three stacked in column 6; playing column 6 blocks (not for own)
        state = [float(v) for row in own for v in row] + [float(v) for row in opp for v in row]
        self.assertEqual(G.board_action_features(state, 3, 'connect_four'), (1.0, 0.0))
        self.assertEqual(G.board_action_features(state, 6, 'connect_four'), (0.0, 1.0))
        self.assertEqual(G.board_action_features(state, 0, 'connect_four'), (0.0, 0.0))
    def test_python_matches_cpp(self):
        src = os.path.join(os.path.dirname(__file__), 'src')
        with tempfile.TemporaryDirectory() as d:
            exe = os.path.join(d, 'probe')
            subprocess.run(['g++', '-std=c++17', '-O2', '-I', src, os.path.join(src, 'action_primitive_probe.cpp'), '-o', exe], check=True)
            out = subprocess.run([exe], check=True, capture_output=True, text=True).stdout.splitlines()
        nonzero = 0
        for line in out:
            head, tail = line.split('|'); t = head.split(); env = 'ttt' if t[0] == 'ttt' else 'connect_four'
            w, b = G.board_action_features([float(x) for x in t[2:]], int(t[1]), env); cw, cb = [float(x) for x in tail.split()]
            self.assertAlmostEqual(math.tanh(w), cw, places=5); self.assertAlmostEqual(math.tanh(b), cb, places=5); nonzero += (w > 0) + (b > 0)
        self.assertGreater(len(out), 500); self.assertGreater(nonzero, 30)
    def test_vocabulary_registered(self):
        for name in ('action_wins_now', 'action_blocks_opponent'):
            self.assertIn(name, G.PRIMS); self.assertEqual(G.PRIMITIVE_METADATA[name]['available_in'], ('ttt', 'connect_four'))
        self.assertEqual(G.PRIMITIVE_CONTRACT, 'generic_state_action_v8')
        self.assertEqual(len(G.summaries([0.0]*18, 0, 9, 9, 'ttt')), len(G.PRIMS))
if __name__ == '__main__': unittest.main()
