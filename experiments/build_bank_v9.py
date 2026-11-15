#!/usr/bin/env python3
"""Concept bank v4 from the v9 generator runs (results_v9/gp) + measured transfer results (results_v9/transfer_analysis.json).
Previous bank (v3, audited, v2-contract) is preserved as results/concept_bank_v3_audit.json."""
import json, math, statistics, datetime, shutil, os
from pathlib import Path
from llm_game_platform.policy import probe_primitive_sets, parse_expression, _evaluate_expression, screen_concepts, _GRID_PRIMITIVES, _dependencies
A = json.load(open('results_v9/transfer_analysis.json')); probes = probe_primitive_sets(); now = datetime.datetime.now(datetime.timezone.utc).isoformat(timespec='seconds')
if not os.path.exists('results/concept_bank_v3_audit.json') and json.load(open('results/concept_bank.json')).get('version') == 3: shutil.copy('results/concept_bank.json', 'results/concept_bank_v3_audit.json')
PAIRS = [('ttt_to_cartpole_rollout', 'cartpole', 'source_rollout', 'rollout'), ('ttt_to_cartpole_oracle', 'cartpole', 'source_oracle', 'oracle'),
         ('ttt_to_c4_rollout', 'connect_four', 'source_rollout', 'rollout'), ('ttt_to_c4_oracle', 'connect_four', 'source_oracle', 'oracle'), ('c4_to_nim_rollout', 'nim', 'source_rollout', 'rollout')]
def grid_std(e, deps):
    if not set(deps) <= _GRID_PRIMITIVES: return None
    v = [math.tanh(max(-10, min(10, _evaluate_expression(parse_expression(e), p)))) for p in probes]; return statistics.pstdev(v)
concepts = []
for pair, tgt, cond, teacher in PAIRS:
    cands = json.load(open(f'results_v9/gp/{pair}/concept_candidates.json'))['candidates']; cmp = A[tgt]['comparisons'][f'{cond}_vs_scratch']; cc = A[tgt]['conditions']
    for c in cands:
        concepts.append({'id': f'{pair}:{c["id"]}', 'expression': c['expression'], 'canonical': c['canonical'], 'formula': c['expression'], 'source_environment': c['source_game'],
            'evaluated_target_environment': c['target_game'], 'teacher_policy': c.get('teacher_policy', teacher), 'primitive_contract': c['primitive_contract'], 'primitive_dependencies': c['primitive_dependencies'],
            'target_output_std': c['target_output_std'], 'grid_probe_output_std': grid_std(c['expression'], c['primitive_dependencies']),
            'discovery_fitness': {'source_robust_signed_corr': c['source_robust_signed_corr'], 'source_generalization_consistency': c['source_generalization_consistency'], 'behavioral_novelty': c['behavioral_novelty']},
            'validation_performance': {'source_validation_signed_corr': c['source_validation_signed_corr'], 'source_test_abs_corr': c['source_test_abs_corr']},
            'transfer_performance': {'scope': 'pair-level: all 4 shortlisted concepts of this pair added together; per-concept effect not measured', 'target_metric': A[tgt]['unit'] + ' (greedy, 1000 held-out episodes)',
               'n_seeds': 10, 'scratch_mean': cc['scratch']['mean'], 'source_concepts_mean': cc[cond]['mean'], 'mean_difference': cmp['mean_diff'], 'ci95': cmp['ci95'], 'permutation_p': cmp['perm_p'], 'holm_p': cmp['holm_p']},
            'discovery_seed': 2026, 'created_utc': now, 'complexity': c['complexity'], 'depth': c['depth'], 'admission_status': 'not_admitted',
            'admission_reason': 'no benefit over scratch, random or raw state-action features after Holm correction (results_v9/transfer_analysis.json)', 'admitted_into_target': False,
            'curve_section': 'v9_transfer', 'with_concepts_group': f'{tgt}_{cond}', 'without_concepts_group': f'{tgt}_scratch', 'random_control_group': f'{tgt}_random'})
bank = {'version': 4, 'primitive_contract': 'generic_state_action_v7', 'created_utc': now, 'concepts': concepts,
        'note': 'v9 generator (v7 primitive contract, rollout and oracle teachers). All concepts non-constant on their target; none admitted: no measured transfer benefit. v3 bank preserved at results/concept_bank_v3_audit.json; v1 at results/concept_bank_v1_backup.json.'}
Path('results/concept_bank.json').write_text(json.dumps(bank, indent=2))
ok, rej = screen_concepts(Path('results/concept_bank.json'), limit=100)
print(len(concepts), 'concepts; pass grid-game screening:', len(ok)); print('rejected for grid games:', sorted({r['reason'][:55] for r in rej}))
