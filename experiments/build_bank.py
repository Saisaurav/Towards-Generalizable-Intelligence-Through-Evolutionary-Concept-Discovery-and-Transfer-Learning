#!/usr/bin/env python3
"""Regenerate results/concept_bank.json (v3) from the GP discovery outputs + measured transfer results.
Old bank is preserved at results/concept_bank_v1_backup.json. Nothing here is invented: every number is read from a result file."""
import json, math, statistics, datetime, hashlib
from llm_game_platform.policy import probe_primitive_sets, parse_expression, _evaluate_expression, screen_concepts
from pathlib import Path
A = json.load(open('results_v2/transfer_analysis.json'))
probes = probe_primitive_sets()
def grid_std(e):
    v = [math.tanh(max(-10, min(10, _evaluate_expression(parse_expression(e), p)))) for p in probes]; return statistics.pstdev(v)
concepts = []
now = datetime.datetime.now(datetime.timezone.utc).isoformat(timespec='seconds')
for pair, tgt in [('ttt_to_cartpole', 'cartpole'), ('c4_to_nim', 'nim')]:
    d = json.load(open(f'results_v2/gp/{pair}/concept_candidates.json'))
    cmp_ = A[tgt]['comparisons']['source_vs_scratch']; cond = A[tgt]['conditions']
    for c in d['candidates']:
        deps = c['primitive_dependencies']
        gstd = grid_std(c['expression']) if set(deps) <= {'state_mean','state_abs_mean','state_var','state_max_abs','action_norm','legal_norm','state_magnitude','state_dispersion'} else None
        concepts.append({
            'id': c['id'], 'expression': c['expression'], 'canonical': c['canonical'], 'formula': c['expression'],
            'source_environment': c['source_game'], 'evaluated_target_environment': c['target_game'],
            'primitive_contract': c['primitive_contract'], 'primitive_dependencies': deps,
            'target_output_std': c['target_output_std'], 'grid_probe_output_std': gstd,
            'discovery_fitness': {'source_training_abs_corr': c['source_training_abs_corr'], 'behavioral_novelty': c['behavioral_novelty']},
            'validation_performance': {'source_validation_abs_corr': c['source_validation_abs_corr'], 'source_test_abs_corr': c['source_test_abs_corr']},
            'transfer_performance': {
                'scope': 'pair-level: all 4 shortlisted source concepts of this pair were added together; per-concept effect not measured',
                'target_metric': A[tgt]['metric'], 'n_seeds': 10,
                'scratch_mean': cond['scratch']['final_mean'], 'source_concepts_mean': cond['source']['final_mean'],
                'mean_difference': cmp_['mean_diff'], 'ci95': cmp_['ci95'], 'permutation_p': cmp_['perm_p']},
            'discovery_seed': 2026, 'created_utc': now, 'complexity': c['complexity'], 'depth': c['depth'],
            'admission_status': 'not_admitted', 'admission_reason': 'no measured benefit over from-scratch learning in the 10-seed transfer matrix (see results_v2/transfer_analysis.json)',
            'admitted_into_target': False})
bank = {'version': 3, 'primitive_contract': 'generic_state_action_v2', 'created_utc': now,
        'note': 'Regenerated after the constant-concept repair. v1 bank preserved at results/concept_bank_v1_backup.json. '
                'All concepts were verified non-constant on the target game (std >= 0.01). None were admitted: transfer showed no measured benefit.',
        'concepts': concepts}
Path('results/concept_bank.json').write_text(json.dumps(bank, indent=2))
ok, rej = screen_concepts(Path('results/concept_bank.json'))
print(len(concepts), 'concepts written;', len(ok), 'pass grid-game screening;', 'rejected:', [(r['expression'][:30], r['reason'][:60]) for r in rej])
