#!/usr/bin/env python3
"""Reference values from exact expectimax (computed, NOT used as a training signal or hand-coded strategy).
TTT: agent is X (moves 'second' half the time because the random opponent opens with p=0.5). Nim: start uniform 10..21."""
import functools, json
L=[(0,1,2),(3,4,5),(6,7,8),(0,3,6),(1,4,7),(2,5,8),(0,4,8),(2,4,6)]
def win(b,p): return any(b[a]==b[c]==b[d]==p for a,c,d in L)
@functools.lru_cache(None)
def ttt(b):  # agent to move; returns (expected return, P(win or draw))
    best=None
    for a in range(9):
        if b[a]: continue
        nb=list(b); nb[a]=1
        if win(nb,1): v=(1.0,1.0)
        elif all(nb): v=(0.0,1.0)
        else:
            m=[i for i in range(9) if not nb[i]]; er=ew=0
            for o in m:
                ob=list(nb); ob[o]=2
                if win(ob,2): er+=-1; ew+=0
                elif all(ob): er+=0; ew+=1
                else: r=ttt(tuple(ob)); er+=r[0]; ew+=r[1]
            v=(er/len(m),ew/len(m))
        if best is None or v[0]>best[0]: best=v
    return best
e=[0]*9; v_first=ttt(tuple(e))
vs=[]
for o in range(9): b=[0]*9; b[o]=2; vs.append(ttt(tuple(b)))
ttt_ret=0.5*v_first[0]+0.5*sum(x[0] for x in vs)/9; ttt_wd=0.5*v_first[1]+0.5*sum(x[1] for x in vs)/9
@functools.lru_cache(None)
def nim_rand(n): # agent to move, expected return vs random opponent
    best=-2
    for t in (1,2,3):
        if t>n: continue
        r=n-t
        if r==0: v=1.0
        else:
            ks=list(range(1,min(3,r)+1)); v=sum((-1.0 if r-k==0 else nim_rand(r-k)) for k in ks)/len(ks)
        best=max(best,v)
    return best
nim_ret=sum(nim_rand(n) for n in range(10,22))/12
nim_opt_vs_opt=sum(1 for n in range(10,22) if n%4!=0)/12*2-1
out=dict(ttt_optimal_vs_random=dict(expected_return=ttt_ret,win_or_draw=ttt_wd),
         nim_optimal_vs_random=dict(expected_return=nim_ret), nim_optimal_vs_optimal_opponent=dict(expected_return=nim_opt_vs_opt))
print(json.dumps(out,indent=1)); json.dump(out,open('results_v2/reference_optimal.json','w'),indent=1)
