import sys, random, math, statistics, os
sys.path.insert(0, '.')
import gp_transfer as G
from llm_game_platform.policy import probe_primitive_sets, parse_expression, _evaluate_expression
probes = probe_primitive_sets(); prims = [p for p in G.PRIMS if p != 'state_boundary_margin']
os.makedirs('results_v2/features', exist_ok=True)
for s in [11,22,33,44,55,66,77,88,99,111]:
    rng = random.Random(555000 + s); chosen = []
    while len(chosen) < 8:
        e = G.render(G.randtree(rng, primitives=prims))
        v = [math.tanh(max(-10, min(10, _evaluate_expression(parse_expression(e), p)))) for p in probes]
        if statistics.pstdev(v) >= .01 and e not in chosen: chosen.append(e)
    open(f'results_v2/features/random_grid_s{s}.txt', 'w').write('# random grid features (control)\n' + ''.join(e + '\n' for e in chosen))
