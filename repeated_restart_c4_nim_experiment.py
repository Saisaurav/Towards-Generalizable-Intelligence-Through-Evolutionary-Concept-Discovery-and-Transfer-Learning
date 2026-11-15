import json, random
from pathlib import Path
import numpy as np
import gp_transfer as gp

OUT=Path('restart_experiment_c4_nim'); OUT.mkdir(exist_ok=True)
SOURCE='connect_four'; TARGET='nim'; GP_SEEDS=[731,732,733]
POP_SIZE=20; GENERATIONS=14
TARGET_SEEDS=[11,22,33]
BUDGET=3000; DOWNSTREAM_POP=8
zero='sub(state_mean,state_mean)'

class CaptureMonitor(gp.NoveltyMonitor):
    def __init__(self,*a,**kw): super().__init__(*a,**kw); self.populations=[]
    def observe(self,pop):
        self.populations.append(list(pop)); return super().observe(pop)

def norm(v):
    sd=float(np.std(v)); return None if sd<1e-8 else (v-v.mean())/(sd+1e-8)

def run_condition(seed, restart_enabled):
    X,y=gp.make_dataset(SOURCE,seed+410,nstates=60,rollouts=4)
    Xv,yv=gp.make_dataset(SOURCE,seed+441,nstates=45,rollouts=4)
    Xtest,ytest=gp.make_dataset(SOURCE,seed+442,nstates=45,rollouts=4)
    Xt,_=gp.make_dataset(TARGET,seed+912,nstates=35,rollouts=3)
    prims=gp.available_primitives(SOURCE); target_prims=set(gp.available_primitives(TARGET))
    initial=['action_norm']*POP_SIZE
    monitor=CaptureMonitor(Xv,window=(1 if restart_enabled else 999))
    monitor.observe(initial) # pre-seed archive to represent already-exhausted behaviour
    final,archive,logs,restarts=gp.evolve_population(
        list(initial),X,y,Xv,yv,monitor,random.Random(seed),prims,GENERATIONS,
        stagnation_window=(1 if restart_enabled else 999),max_unsuccessful_restarts=3,
        restart_random_fraction=.65,restart_mutation_generations=3)
    initial_fps=[]
    for t in initial:
        v=norm(gp.ev(t,Xv))
        if v is not None and all(abs(gp.corr(v,z))<.995 for z in initial_fps): initial_fps.append(v)
    # Compare the final population plus Pareto archive in both conditions.
    # Only concepts behaviorally novel relative to the initial exhausted bank qualify.
    pool=final+archive
    candidates=[]; seen=set()
    for t in pool:
        key=gp.canon(t)
        if key in seen: continue
        seen.add(key)
        v=norm(gp.ev(t,Xv))
        if v is None or not all(abs(gp.corr(v,z))<.995 for z in initial_fps): continue
        req=set(gp.dependencies(t))
        if not req.issubset(target_prims) or gp.depth(t)>5: continue
        target_values=gp.ev(t,Xt)
        if np.std(target_values)<.01: continue
        candidates.append({'tree':t,'expression':gp.render(t),
                           'source_validation_abs_corr':float(abs(gp.corr(gp.ev(t,Xv),yv))),
                           'source_test_abs_corr':float(abs(gp.corr(gp.ev(t,Xtest),ytest))),
                           'target_output_std':float(np.std(target_values)),
                           'dependencies':sorted(req)})
    candidates.sort(key=lambda c:c['source_validation_abs_corr'],reverse=True)
    chosen=candidates[0] if candidates else None
    return {
      'seed':seed,'generations_run':len(logs),'restart_summary':restarts,
      'unique_structures':len(monitor.struct),'unique_behaviours':len(monitor.behaviors),
      'candidate_count':len(candidates),'candidate':({k:v for k,v in chosen.items() if k!='tree'} if chosen else None),
      'logs':logs,'chosen_tree':chosen['tree'] if chosen else None,
    }

runs=[]
for seed in GP_SEEDS:
    print(f'GP discovery seed {seed}...',flush=True)
    runs.append({'restart':run_condition(seed,True),'control':run_condition(seed,False)})

# Baseline once per target seed; every condition uses identical paired seeds/budget.
brain=gp.resolve_brain_path('src/brain')
baseline={s:gp.run_brain_trial(brain,TARGET,['action_norm',zero,zero,zero,zero],BUDGET,s,DOWNSTREAM_POP) for s in TARGET_SEEDS}
downstream=[]
for run in runs:
    seed=run['restart']['seed']
    for condition in ('restart','control'):
        candidate=run[condition]['candidate']
        if candidate is None: continue
        expr=['action_norm',candidate['expression'],zero,zero,zero]
        returns={s:gp.run_brain_trial(brain,TARGET,expr,BUDGET,s,DOWNSTREAM_POP) for s in TARGET_SEEDS}
        diffs={str(s):returns[s]-baseline[s] for s in TARGET_SEEDS}
        downstream.append({'gp_seed':seed,'condition':condition,'expression':candidate['expression'],
                           'source_validation_abs_corr':candidate['source_validation_abs_corr'],
                           'source_test_abs_corr':candidate['source_test_abs_corr'],
                           'baseline_returns':{str(s):baseline[s] for s in TARGET_SEEDS},
                           'candidate_returns':{str(s):returns[s] for s in TARGET_SEEDS},
                           'paired_differences':diffs,
                           'mean_paired_difference':float(np.mean(list(diffs.values())))})

compact=[]
for run in runs:
    row={'seed':run['restart']['seed']}
    for c in ('restart','control'):
        r=run[c]
        row[c]={'restarts_triggered':r['restart_summary']['restarts_triggered'],
                'successful_restarts':r['restart_summary']['successful_restarts'],
                'unsuccessful_restarts':r['restart_summary']['unsuccessful_restarts'],
                'unique_structures':r['unique_structures'],'unique_behaviours':r['unique_behaviours'],
                'candidate_count':r['candidate_count'],'candidate':r['candidate']}
    compact.append(row)

by_condition={}
for condition in ('restart','control'):
    vals=[x['mean_paired_difference'] for x in downstream if x['condition']==condition]
    by_condition[condition]={'n_gp_seeds_with_candidates':len(vals),
                             'mean_of_gp_seed_mean_improvements':float(np.mean(vals)) if vals else None,
                             'sd_across_gp_seed_mean_improvements':float(np.std(vals,ddof=1)) if len(vals)>1 else None,
                             'all_gp_seed_improvements':vals}
summary={'experiment':'three_seed_final_bank_forced_stagnation_restart_vs_no_restart',
         'source':SOURCE,'target':TARGET,'gp_seeds':GP_SEEDS,'target_paired_seeds':TARGET_SEEDS,
         'population':POP_SIZE,'generations':GENERATIONS,'downstream_budget_per_learner':BUDGET,
         'downstream_population':DOWNSTREAM_POP,
         'forced_stagnation_protocol':'Initial population is 20 copies of action_norm; novelty monitor is pre-seeded with its fingerprint. Restart arm uses a 1-generation stagnation threshold; control uses 999.',
         'target_baseline_returns':{str(s):baseline[s] for s in TARGET_SEEDS},
         'discovery_by_seed':compact,'downstream_by_candidate':downstream,
         'downstream_aggregate':by_condition}
(OUT/'repeated_experiment_summary.json').write_text(json.dumps(summary,indent=2))
print(json.dumps(summary,indent=2))
