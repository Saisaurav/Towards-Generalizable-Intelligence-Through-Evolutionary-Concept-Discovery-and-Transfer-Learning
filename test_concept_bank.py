"""Tests for concept-bank screening: concepts are measured on probe states, not trusted from metadata."""
import json, tempfile, unittest
from pathlib import Path
from llm_game_platform.policy import load_compatible_concepts, screen_concepts

def screen(concepts, **kw):
    bank = {"primitive_contract": "generic_state_action_v2", "concepts": concepts}
    with tempfile.TemporaryDirectory() as d:
        p = Path(d) / "bank.json"; p.write_text(json.dumps(bank), encoding="utf-8"); return screen_concepts(p, **kw)

def reasons(rej): return " | ".join(r["reason"] for r in rej)

class ConceptBankTests(unittest.TestCase):
    def test_valid_concept_accepted(self):
        ok, rej = screen([{"expression": "add(state_mean,action_norm)"}]); self.assertEqual(ok, ["add(state_mean,action_norm)"]); self.assertEqual(rej, [])
    def test_constant_concepts_rejected_even_if_metadata_claims_variance(self):
        ok, rej = screen([{"expression": "div(action_norm,action_norm)", "target_output_std": 0.9},
                          {"expression": "sub(state_mean,state_mean)", "target_output_std": 0.9}])
        self.assertEqual(ok, []); self.assertIn("standard deviation", reasons(rej)); self.assertEqual(len(rej), 2)
    def test_near_constant_concept_rejected(self):
        from llm_game_platform.policy import probe_primitive_sets, parse_expression, _evaluate_expression
        import math, statistics
        near = "mul(state_var,mul(state_var,state_var))"   # valid grammar, tiny measured spread
        sd = statistics.pstdev(math.tanh(_evaluate_expression(parse_expression(near), p)) for p in probe_primitive_sets())
        self.assertLess(sd, 0.01)
        ok, rej = screen([{"expression": near, "target_output_std": 0.5}]); self.assertEqual(ok, []); self.assertIn("standard deviation", reasons(rej))
        ok, rej = screen([{"expression": "div(state_mean,add(state_max_abs,action_norm))"}], min_output_std=5.0); self.assertEqual(ok, []); self.assertIn("below", reasons(rej))
    def test_nonfinite_output_rejected(self):
        ok, rej = screen([{"expression": "add(state_mean,action_norm)"}], probes=[{"state_mean": float("nan"), "action_norm": 0.0}] * 5)
        self.assertEqual(ok, []); self.assertIn("non-finite", reasons(rej))
        ok, rej = screen([{"expression": "mul(state_mean,action_norm)"}], probes=[{"state_mean": float("inf"), "action_norm": 1.0}] * 5); self.assertEqual(ok, [])
    def test_missing_or_unavailable_primitives_rejected(self):
        ok, rej = screen([{"expression": "state_boundary_margin"}, {"expression": "add(state_mean,not_a_primitive)"}])
        self.assertEqual(ok, []); self.assertIn("unavailable", reasons(rej)); self.assertIn("malformed", reasons(rej))
    def test_malformed_formulas_rejected_not_fatal(self):
        bad = ["add(state_mean", "add(state_mean,)", "", "state_mean)", "add(state_mean,action_norm,legal_norm)", "__import__('os')", "div(state_mean,action_norm"]
        ok, rej = screen([{"expression": b} for b in bad] + [{"expression": "state_mean"}, {"no_expression": 1}, 5])
        self.assertEqual(ok, ["state_mean"]); self.assertEqual(len(rej), len(bad) + 2)
    def test_incompatible_input_dimension_rejected(self):
        ok, rej = screen([{"expression": "state_mean", "state_dim": 4}, {"expression": "state_abs_mean", "state_dim": 13}, {"expression": "state_var", "state_dim": "x"}])
        self.assertEqual(ok, ["state_abs_mean"]); self.assertIn("incompatible input dimension", reasons(rej))
    def test_unsupported_contract_and_duplicates(self):
        ok, rej = screen([{"expression": "state_mean", "primitive_contract": "old"}, {"expression": "state_var"}, {"expression": "state_var"}])
        self.assertEqual(ok, ["state_var"]); self.assertIn("contract", reasons(rej)); self.assertIn("duplicate", reasons(rej))
    def test_historical_bank_constants_removed(self):
        # pre-repair bank: includes concepts that are exactly constant (std ~2e-16) on the stated target.
        path = Path(__file__).parent / "results" / "concept_bank_v1_backup.json"
        if not path.exists(): self.skipTest("no backup bank")
        ok, rej = screen_concepts(path)
        self.assertNotIn("div(action_norm,action_norm)", ok)
        self.assertTrue(any(r["expression"] == "div(action_norm,action_norm)" and ("standard deviation" in r["reason"] or "stored target_output_std" in r["reason"]) for r in rej))
        self.assertGreater(len(rej), 0)
    def test_unreadable_bank_raises(self):
        with tempfile.TemporaryDirectory() as d:
            p = Path(d) / "bank.json"; p.write_text("{not json"); 
            with self.assertRaises(ValueError): screen_concepts(p)
if __name__ == "__main__": unittest.main()
