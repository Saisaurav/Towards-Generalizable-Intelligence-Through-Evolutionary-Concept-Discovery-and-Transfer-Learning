import json, random, math
from pathlib import Path
import numpy as np
import gp_transfer as gp

OUT=Path('restart_experiment')
OUT.mkdir(exist_ok=True)
SOURCE='nim'; TARGET='lineworld'; SEED=731
POP_SIZE=40; GENERATIONS=24
X,y=gp.make_dataset(SOURCE, SEED+410, nstates=60, rollouts=4)
Xv,yv=gp.make_dataset(SOURCE, SEED+441, nstates=45, rollouts=4)
Xt,_=gp.make_dataset(TARGET, SEED+912, nstates=35, rollouts=3)
prims=gp.available_primitives(SOURCE)
target_prims=set(gp.available_primitives(TARGET))
initial=['action_norm']*POP_SIZE

class CaptureMonitor(gp.NoveltyMonitor):
    def __init__(self,*a,**kw):
        super().__init__(*a,**kw); self.populations=[]
    def observe(self,pop):
        self.populations.append(list(pop))
        return super().observe(pop)

def run(label, window):
    rng=random.Random(SEED)
    monitor=CaptureMonitor(Xv,window=window)
    # Prime the behavioural archive with the exhausted population. The first
    # in-loop observation therefore has no new behaviour: a controlled,
    # reproducible forced-stagnation condition.
    monitor.observe(initial)
    final,archive,logs,restarts=gp.evolve_population(
        list(initial),X,y,Xv,yv,monitor,rng,prims,GENERATIONS,
        stagnation_window=window,max_unsuccessful_restarts=3,
        restart_random_fraction=.65,restart_mutation_generations=3)
    return {'final':final,'archive':archive,'logs':logs,'restarts':restarts,'monitor':monitor}

restart=run('restart',1)
control=run('control',999)

def normalized(v):
    sd=float(np.std(v))
    return None if sd<1e-8 else (v-v.mean())/(sd+1e-8)

def new_vs_initial(tree, initial_fps):
    v=normalized(gp.ev(tree,Xv))
    if v is None:return False
    return all(abs(gp.corr(v,z)) < .995 for z in initial_fps)

initial_fps=[]
for t in initial:
    v=normalized(gp.ev(t,Xv))
    if v is not None and all(abs(gp.corr(v,z))<.995 for z in initial_fps): initial_fps.append(v)

# Identify expressions present immediately after the restart, and classify
# whether their output behaviour is globally new relative to the pre-restart bank.
restart_pop=restart['monitor'].populations[2] if len(restart['monitor'].populations)>2 else []
# populations[0] is priming; [1] is generation 0 pre-restart; [2] is generation 1 post-restart
post_candidates=[]; seen=set()
for t in restart_pop:
    key=gp.canon(t)
    if key in seen: continue
    seen.add(key)
    if not new_vs_initial(t,initial_fps): continue
    req=set(gp.dependencies(t))
    if not req.issubset(target_prims): continue
    vals=gp.ev(t,Xt)
    if np.std(vals)<.01 or gp.depth(t)>5: continue
    post_candidates.append({'tree':t,'expression':gp.render(t),'source_val_corr':abs(gp.corr(gp.ev(t,Xv),yv)),
                            'source_test_corr':None,'target_std':float(np.std(vals)),
                            'dependencies':sorted(req)})

# Choose by validation correlation, but compute source test score only after selection.
post_candidates.sort(key=lambda x:x['source_val_corr'],reverse=True)
for c in post_candidates[:5]:
    c['source_test_corr']=abs(gp.corr(gp.ev(c['tree'],X),y))

# The no-restart control is searched for candidates that became novel through ordinary evolution.
control_candidates=[]; seen=set()
for t in control['final']+control['archive']:
    key=gp.canon(t)
    if key in seen: continue
    seen.add(key)
    if not new_vs_initial(t,initial_fps): continue
    req=set(gp.dependencies(t))
    if not req.issubset(target_prims): continue
    vals=gp.ev(t,Xt)
    if np.std(vals)<.01 or gp.depth(t)>5: continue
    control_candidates.append({'tree':t,'expression':gp.render(t),'source_val_corr':abs(gp.corr(gp.ev(t,Xv),yv)),
                               'source_test_corr':abs(gp.corr(gp.ev(t,X),y)), 'target_std':float(np.std(vals)),
                               'dependencies':sorted(req)})
control_candidates.sort(key=lambda x:x['source_val_corr'],reverse=True)

summary={
 'experiment':'forced_stagnation_restart_vs_no_restart',
 'source':SOURCE,'target':TARGET,'seed':SEED,'population':POP_SIZE,'generations_requested':GENERATIONS,
 'initial_population':'40 copies of action_norm; novelty archive pre-seeded with that behaviour',
 'restart_condition':{'generations_run':len(restart['logs']),'restart_summary':restart['restarts'],
                      'unique_structures':len(restart['monitor'].struct),'unique_behaviours':len(restart['monitor'].behaviors),
                      'post_restart_generation_population_size':len(restart_pop),
                      'post_restart_behaviourally_new_target_compatible_candidates':len(post_candidates),
                      'top_candidates':[{k:v for k,v in c.items() if k!='tree'} for c in post_candidates[:5]]},
 'no_restart_control':{'generations_run':len(control['logs']),'restart_summary':control['restarts'],
                       'unique_structures':len(control['monitor'].struct),'unique_behaviours':len(control['monitor'].behaviors),
                       'novel_target_compatible_candidates':len(control_candidates),
                       'top_candidates':[{k:v for k,v in c.items() if k!='tree'} for c in control_candidates[:5]]},
 'generation_logs':{'restart':restart['logs'],'control':control['logs']}
}
(OUT/'discovery_summary.json').write_text(json.dumps(summary,indent=2))
print(json.dumps({k:v for k,v in summary.items() if k!='generation_logs'},indent=2))

# Run a small paired downstream pilot if restart found a transferable concept.
if post_candidates:
    brain=gp.resolve_brain_path('src/brain')
    seeds=[11,22,33]
    budget=3000; pop=8
    zero='sub(state_mean,state_mean)'
    baseline_expr=['action_norm',zero,zero,zero,zero]
    baselines={s:gp.run_brain_trial(brain,TARGET,baseline_expr,budget,s,pop) for s in seeds}
    trial_results=[]
    candidates=[('restart',post_candidates[0])]
    if control_candidates: candidates.append(('no_restart',control_candidates[0]))
    for label,c in candidates:
        expr=['action_norm',c['expression'],zero,zero,zero]
        returns={s:gp.run_brain_trial(brain,TARGET,expr,budget,s,pop) for s in seeds}
        diffs={str(s):returns[s]-baselines[s] for s in seeds}
        trial_results.append({'condition':label,'expression':c['expression'],'baseline_returns':{str(s):baselines[s] for s in seeds},
                              'candidate_returns':{str(s):returns[s] for s in seeds},'paired_differences':diffs,
                              'mean_paired_difference':float(np.mean(list(diffs.values())))})
    downstream={'target':TARGET,'budget_per_learner':budget,'population':pop,'paired_seeds':seeds,'results':trial_results}
    (OUT/'downstream_pilot.json').write_text(json.dumps(downstream,indent=2))
    print('\nDOWNSTREAM PILOT\n'+json.dumps(downstream,indent=2))
else:
    print('\nNo post-restart candidate met target-compatibility screens; downstream pilot skipped.')
