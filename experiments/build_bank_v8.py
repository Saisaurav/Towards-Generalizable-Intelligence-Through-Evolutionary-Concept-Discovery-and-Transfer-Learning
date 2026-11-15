#!/usr/bin/env python3
"""Concept bank v5 = v4 (v9 generator, v7 contract) + v8-vocabulary concepts (action-conditioned primitives). Numbers come from result files."""
import json, datetime, math, statistics
from pathlib import Path
from llm_game_platform.policy import screen_concepts
A8 = json.load(open('results_v8/transfer_analysis.json')); bank = json.load(open('results/concept_bank.json'))
assert bank['version'] == 4, 'run build_bank_v9.py first'
now = datetime.datetime.now(datetime.timezone.utc).isoformat(timespec='seconds'); C = A8['conditions']; K = A8['comparisons']
for pair, cond, teacher in [('ttt_to_c4_rollout', 'source_rollout', 'rollout'), ('ttt_to_c4_oracle', 'source_oracle', 'oracle')]:
    for c in json.load(open(f'results_v8/gp/{pair}/concept_candidates.json'))['candidates']:
        bank['concepts'].append({'id': f'v8_{pair}:{c["id"]}', 'expression': c['expression'], 'canonical': c['canonical'], 'formula': c['expression'], 'source_environment': c['source_game'],
          'evaluated_target_environment': c['target_game'], 'teacher_policy': c.get('teacher_policy', teacher), 'primitive_contract': c['primitive_contract'], 'primitive_dependencies': c['primitive_dependencies'],
          'target_output_std': c['target_output_std'], 'grid_probe_output_std': None,
          'discovery_fitness': {'source_robust_signed_corr': c['source_robust_signed_corr'], 'source_generalization_consistency': c['source_generalization_consistency'], 'behavioral_novelty': c['behavioral_novelty']},
          'validation_performance': {'source_validation_signed_corr': c['source_validation_signed_corr'], 'source_test_abs_corr': c['source_test_abs_corr']},
          'transfer_performance': {'scope': 'pair-level: all 4 shortlisted concepts added together; per-concept effect not measured', 'target_metric': 'mean return vs random (greedy, 1000 held-out episodes)', 'n_seeds': 10,
             'scratch_mean': C['scratch']['mean'], 'source_concepts_mean': C[cond]['mean'], 'mean_difference': K[f'{cond}_vs_scratch']['mean_diff'], 'ci95': K[f'{cond}_vs_scratch']['ci95'],
             'permutation_p': K[f'{cond}_vs_scratch']['perm_p'], 'holm_p': K[f'{cond}_vs_scratch']['holm_p'],
             'vs_random_vocabulary_features': K[f'{cond}_vs_random']['mean_diff'], 'vs_primitives_only_control': K[f'{cond}_vs_primitives_only']['mean_diff']},
          'discovery_seed': 2026, 'created_utc': now, 'complexity': c['complexity'], 'depth': c['depth'],
          'admission_status': 'admitted_with_caveat', 'admitted_into_target': True,
          'admission_reason': 'Beats scratch, random v8-vocabulary features and raw state-action after Holm correction on Connect Four, but does not beat the primitives-only control: the benefit is attributable to the supplied action_wins_now/action_blocks_opponent primitives, not to the discovered combination.',
          'curve_section': 'v8_transfer', 'with_concepts_group': f'connect_four_{cond}', 'without_concepts_group': 'connect_four_scratch', 'random_control_group': 'connect_four_random'})
bank['version'] = 5; bank['primitive_contract'] = 'generic_state_action_v8'; bank['created_utc'] = now
bank['note'] = ('v5: v4 concepts (v9 generator, no transfer benefit, not admitted) + 8 concepts from the v8 vocabulary (action-conditioned primitives, supplied knowledge) admitted for Connect Four with the caveat that a primitives-only control '
                'matches them. v3: results/concept_bank_v3_audit.json; v1: results/concept_bank_v1_backup.json.')
Path('results/concept_bank.json').write_text(json.dumps(bank, indent=2))
ok, rej = screen_concepts(Path('results/concept_bank.json'), limit=200)
print(len(bank['concepts']), 'concepts; grid-screen pass', len(ok))
