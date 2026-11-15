#!/usr/bin/env python3
"""Typed-by-contract GP experiments across the supported game environments."""
import argparse, json, math, os, random, re, subprocess, tempfile
from pathlib import Path
from functools import lru_cache
import numpy as np

PRIMS = [
    'state_mean', 'state_abs_mean', 'state_var', 'state_max_abs',
    'action_norm', 'legal_norm', 'state_magnitude', 'state_dispersion',
    'state_boundary_margin', 'state_min', 'state_max', 'state_range',
    'state_positive_fraction', 'state_negative_fraction', 'state_sparsity',
    'state_median', 'state_iqr', 'state_skewness', 'state_entropy',
    'state_position_moment', 'state_position_spread',
    'state_adjacent_difference', 'state_sign_change_fraction',
    'transition_delta_mean', 'transition_delta_abs_mean',
    'transition_delta_max_abs', 'transition_state_alignment',
    'board_own_open_lines', 'board_opponent_open_lines',
    'board_own_immediate_wins', 'board_opponent_immediate_wins',
    'board_center_control', 'board_occupancy',
    'action_wins_now', 'action_blocks_opponent',
]
PRIMITIVE_CONTRACT = 'generic_state_action_v8'
DATASET_PROTOCOL = 'game_calibrated_rollout_seeds_v3'
ENVIRONMENTS = ('ttt', 'cartpole', 'connect_four', 'nim', 'lineworld')
GAME_NAMES = {
    'ttt': 'TicTacToe',
    'cartpole': 'CartPole',
    'connect_four': 'ConnectFour',
    'nim': 'Nim',
    'lineworld': 'LineWorld',
}
DEFAULT_TARGETS = {
    'ttt': 'cartpole',
    'cartpole': 'ttt',
    'connect_four': 'nim',
    'nim': 'lineworld',
    'lineworld': 'connect_four',
}
PRIMITIVE_METADATA = {
    'state_mean': {'available_in': ENVIRONMENTS, 'definition': 'Mean of normalized state values.'},
    'state_abs_mean': {'available_in': ENVIRONMENTS, 'definition': 'Mean absolute normalized state value.'},
    'state_var': {'available_in': ENVIRONMENTS, 'definition': 'Variance of normalized state values.'},
    'state_max_abs': {'available_in': ENVIRONMENTS, 'definition': 'Maximum absolute normalized state value.'},
    'action_norm': {'available_in': ENVIRONMENTS, 'definition': 'Action index mapped to [-1, 1].'},
    'legal_norm': {'available_in': ENVIRONMENTS, 'definition': 'Fraction of available actions that are legal.'},
    'state_magnitude': {'available_in': ENVIRONMENTS, 'definition': 'Root mean square of normalized state values.'},
    'state_dispersion': {'available_in': ENVIRONMENTS, 'definition': 'Standard deviation of normalized state values.'},
    'state_boundary_margin': {'available_in': ('cartpole',), 'definition': 'One minus the larger normalized CartPole position/angle magnitude.'},
    'state_min': {'available_in': ENVIRONMENTS, 'definition': 'Minimum normalized state value.'},
    'state_max': {'available_in': ENVIRONMENTS, 'definition': 'Maximum normalized state value.'},
    'state_range': {'available_in': ENVIRONMENTS, 'definition': 'Maximum minus minimum normalized state value.'},
    'state_positive_fraction': {'available_in': ENVIRONMENTS, 'definition': 'Fraction of normalized state values greater than zero.'},
    'state_negative_fraction': {'available_in': ENVIRONMENTS, 'definition': 'Fraction of normalized state values less than zero.'},
    'state_sparsity': {'available_in': ENVIRONMENTS, 'definition': 'Fraction of normalized state values with absolute value at most 1e-6.'},
    'state_median': {'available_in': ENVIRONMENTS, 'definition': 'Median of normalized state values.'},
    'state_iqr': {'available_in': ENVIRONMENTS, 'definition': '75th minus 25th percentile of normalized state values.'},
    'state_skewness': {'available_in': ENVIRONMENTS, 'definition': 'Standardized third central moment; zero for near-constant states.'},
    'state_entropy': {'available_in': ENVIRONMENTS, 'definition': 'Normalized Shannon entropy of absolute state-value mass; zero for all-zero states.'},
    'state_position_moment': {'available_in': ENVIRONMENTS, 'definition': 'Signed first moment of values over normalized feature index; order-sensitive.'},
    'state_position_spread': {'available_in': ENVIRONMENTS, 'definition': 'Absolute-mass-weighted spread over normalized feature index; order-sensitive.'},
    'state_adjacent_difference': {'available_in': ENVIRONMENTS, 'definition': 'Mean absolute difference between adjacent normalized state features; order-sensitive.'},
    'state_sign_change_fraction': {'available_in': ENVIRONMENTS, 'definition': 'Fraction of adjacent feature pairs with opposite nonzero signs; order-sensitive.'},
    'transition_delta_mean': {'available_in': ('cartpole','lineworld'), 'definition': 'Mean normalized observation change from the previous environment step; zero before the first transition.'},
    'transition_delta_abs_mean': {'available_in': ('cartpole','lineworld'), 'definition': 'Mean absolute normalized observation change from the previous environment step.'},
    'transition_delta_max_abs': {'available_in': ('cartpole','lineworld'), 'definition': 'Maximum absolute normalized observation change from the previous environment step.'},
    'transition_state_alignment': {'available_in': ('cartpole','lineworld'), 'definition': 'Cosine alignment between current normalized observation and its previous-step change; zero when either vector has negligible norm.'},
    'board_own_open_lines': {'available_in': ('ttt','connect_four'), 'definition': 'Fraction of winning lines containing own marks and no opponent marks.'},
    'board_opponent_open_lines': {'available_in': ('ttt','connect_four'), 'definition': 'Fraction of winning lines containing opponent marks and no own marks.'},
    'board_own_immediate_wins': {'available_in': ('ttt','connect_four'), 'definition': 'Fraction of distinct empty cells that complete an own winning line in one move.'},
    'board_opponent_immediate_wins': {'available_in': ('ttt','connect_four'), 'definition': 'Fraction of distinct empty cells that complete an opponent winning line in one move.'},
    'board_center_control': {'available_in': ('ttt','connect_four'), 'definition': 'Normalized own-minus-opponent occupancy in the center cell/columns.'},
    'board_occupancy': {'available_in': ('ttt','connect_four'), 'definition': 'Fraction of board cells occupied by either player.'},
    'action_wins_now': {'available_in': ('ttt','connect_four'), 'definition': 'Afterstate lookahead (supplied knowledge): 1 if this action completes an own winning line immediately, else 0.'},
    'action_blocks_opponent': {'available_in': ('ttt','connect_four'), 'definition': 'Afterstate lookahead (supplied knowledge): 1 if the opponent would win by playing this action, so playing it blocks that win, else 0.'},
}
UNARY_OPS = ['abs', 'neg', 'square', 'sqrt', 'log1p', 'tanh']
BINARY_OPS = ['add', 'sub', 'mul', 'div', 'min', 'max']
OPS = UNARY_OPS + BINARY_OPS
ARITY = {**{x: 1 for x in UNARY_OPS}, **{x: 2 for x in BINARY_OPS}}


def available_primitives(environment):
    return [name for name in PRIMS if environment in PRIMITIVE_METADATA[name]['available_in']]


def node_size(t): return 1 if isinstance(t,str) else 1+sum(node_size(x) for x in t[1:])
def depth(t): return 1 if isinstance(t,str) else 1+max(depth(x) for x in t[1:])
def render(t): return t if isinstance(t,str) else t[0]+'('+','.join(render(x) for x in t[1:])+')'
def canon(t):
    if isinstance(t,str): return t
    ch=[canon(x) for x in t[1:]]
    if t[0] in ('add','mul','min','max'): ch=sorted(ch)
    return t[0]+'('+','.join(ch)+')'
def _eval_raw(t, X):
    if isinstance(t,str): return X[PRIMS.index(t)]
    a=_eval_raw(t[1],X)
    with np.errstate(all='ignore'):
        if ARITY[t[0]] == 1:
            if t[0]=='abs': z=np.abs(a)
            elif t[0]=='neg': z=-a
            elif t[0]=='square': z=a*a
            elif t[0]=='sqrt': z=np.sqrt(np.abs(a))
            elif t[0]=='log1p': z=np.log1p(np.abs(a))
            else: z=np.tanh(a)
        else:
            b=_eval_raw(t[2],X)
            if t[0]=='add': z=a+b
            elif t[0]=='sub': z=a-b
            elif t[0]=='mul': z=a*b
            elif t[0]=='div': z=a/np.where(np.abs(b)<1e-4,np.where(b<0,-1e-4,1e-4),b)
            elif t[0]=='min': z=np.minimum(a,b)
            else: z=np.maximum(a,b)
    return np.nan_to_num(z,nan=0,posinf=10,neginf=-10)
def ev(t, X):
    # Match the C++ runtime exactly: evaluate the raw tree, then clip/tanh once at the root.
    return np.tanh(np.clip(_eval_raw(t,X),-10,10))

def randtree(rng, d=0, maxd=3, primitives=None):
    primitives = PRIMS if primitives is None else primitives
    if d>=maxd or (d>0 and rng.random()<.35): return rng.choice(primitives)
    op=rng.choice(UNARY_OPS if rng.random()<.25 else BINARY_OPS)
    return (op, *(randtree(rng,d+1,maxd,primitives) for _ in range(ARITY[op])))
def walk_paths(t, path=()):
    yield path,t
    if not isinstance(t,str):
        for i,c in enumerate(t[1:]): yield from walk_paths(c,path+(i,))
def replace(t,path,new):
    if not path:return new
    i=path[0]+1; a=list(t);a[i]=replace(a[i],path[1:],new);return tuple(a)
def mutate(t,rng,primitives=None,intensity=1,subtree_depth=2):
    child=t
    for _ in range(intensity):
        paths=list(walk_paths(child)); p,_=rng.choice(paths)
        child=replace(child,p,randtree(rng,maxd=subtree_depth,primitives=primitives))
    return child
def crossover(a,b,rng):
    pa,_=rng.choice(list(walk_paths(a))); _,sub=rng.choice(list(walk_paths(b))); return replace(a,pa,sub)
def corr(a,b):
    if np.std(a)<1e-9 or np.std(b)<1e-9:return 0.
    value=float(np.corrcoef(a,b)[0,1])
    return value if math.isfinite(value) else 0.
def dependencies(t):
    if isinstance(t,str): return [t]
    return sorted(set().union(*(set(dependencies(x)) for x in t[1:])))
def board_structural_features(state, environment):
    values=np.asarray(state,dtype=np.float64)
    if environment=='ttt':
        rows,cols=3,3
    elif environment=='connect_four':
        rows,cols=6,7
    else:
        raise ValueError(f'board features unavailable for {environment}')
    cells=rows*cols
    if len(values)!=2*cells:
        raise ValueError(f'{environment} expects {2*cells} board features, got {len(values)}')
    own=values[:cells].reshape(rows,cols)>0.5
    opp=values[cells:].reshape(rows,cols)>0.5
    win_length=3 if environment=='ttt' else 4
    lines=[]
    for r in range(rows):
        for c in range(cols):
            for dr,dc in ((0,1),(1,0),(1,1),(1,-1)):
                er,ec=r+(win_length-1)*dr,c+(win_length-1)*dc
                if 0<=er<rows and 0<=ec<cols:
                    lines.append([(r+i*dr,c+i*dc) for i in range(win_length)])
    own_open=opp_open=0
    own_wins=set();opp_wins=set()
    for line in lines:
        own_count=sum(bool(own[r,c]) for r,c in line)
        opp_count=sum(bool(opp[r,c]) for r,c in line)
        empty=[(r,c) for r,c in line if not own[r,c] and not opp[r,c]]
        if own_count>0 and opp_count==0: own_open+=1
        if opp_count>0 and own_count==0: opp_open+=1
        if own_count==win_length-1 and opp_count==0 and len(empty)==1: own_wins.add(empty[0])
        if opp_count==win_length-1 and own_count==0 and len(empty)==1: opp_wins.add(empty[0])
    if environment=='ttt':
        center_own=float(own[1,1]); center_opp=float(opp[1,1])
    else:
        center_own=float(own[:,2:5].sum())
        center_opp=float(opp[:,2:5].sum())
        center_own/=rows*3; center_opp/=rows*3
    return {
        'board_own_open_lines':own_open/max(1,len(lines)),
        'board_opponent_open_lines':opp_open/max(1,len(lines)),
        'board_own_immediate_wins':len(own_wins)/cells,
        'board_opponent_immediate_wins':len(opp_wins)/cells,
        'board_center_control':center_own-center_opp,
        'board_occupancy':float(np.mean(own|opp)),
    }


def board_action_features(state, a, environment):
    """(wins_now, blocks_opponent) for placing/dropping at action a. TTT action=cell; Connect Four action=column (drops to lowest empty row)."""
    values=np.asarray(state,dtype=np.float64)
    rows,cols,win_length=(3,3,3) if environment=='ttt' else (6,7,4)
    cells=rows*cols
    own=(values[:cells].reshape(rows,cols)>0.5).copy(); opp=(values[cells:].reshape(rows,cols)>0.5).copy()
    if environment=='ttt':
        r,c=divmod(int(a),cols)
        if own[r,c] or opp[r,c]: return 0.0,0.0
    else:
        c=int(a); free=[r for r in range(rows-1,-1,-1) if not own[r,c] and not opp[r,c]]
        if not free: return 0.0,0.0
        r=free[0]
    def completes(plane):
        for dr,dc in ((0,1),(1,0),(1,1),(1,-1)):
            count=1
            for sign in (1,-1):
                rr,cc=r+sign*dr,c+sign*dc
                while 0<=rr<rows and 0<=cc<cols and plane[rr,cc]:
                    count+=1; rr+=sign*dr; cc+=sign*dc
            if count>=win_length: return True
        return False
    return float(completes(own)),float(completes(opp))


def summaries(s,a,nact,nlegal,environment,state_delta=None):
    s=np.asarray(s,dtype=np.float64); mean=float(s.mean())
    magnitude=float(np.sqrt(np.mean(s*s)))
    dispersion=float(np.sqrt(np.mean((s-mean)**2)))
    boundary_margin=float('nan')
    if environment=='cartpole':
        boundary_margin=1.-max(abs(float(s[0])),abs(float(s[2])))
    state_min=float(s.min())
    state_max=float(s.max())
    state_range=state_max-state_min
    positive_fraction=float(np.mean(s>0))
    negative_fraction=float(np.mean(s<0))
    sparsity=float(np.mean(np.abs(s)<=1e-6))
    median=float(np.median(s))
    q25,q75=np.quantile(s,[0.25,0.75])
    iqr=float(q75-q25)
    skewness=float(np.mean(((s-mean)/(dispersion+1e-8))**3)) if dispersion>=1e-8 else 0.0
    mass=np.abs(s); mass_sum=float(mass.sum())
    if mass_sum<=1e-12 or len(s)<=1:
        entropy=0.0
    else:
        probabilities=mass[mass>0]/mass_sum
        entropy=float(-np.sum(probabilities*np.log(probabilities))/np.log(len(s)))
    positions=np.linspace(-1.0,1.0,len(s)) if len(s)>1 else np.zeros(len(s))
    abs_mass=np.abs(s); abs_total=float(abs_mass.sum())
    position_moment=float(np.mean(s*positions)) if len(s) else 0.0
    position_spread=float(np.sum(abs_mass*(positions**2))/abs_total) if abs_total>1e-12 else 0.0
    adjacent_difference=float(np.mean(np.abs(np.diff(s)))) if len(s)>1 else 0.0
    signs=np.sign(s)
    sign_change_fraction=float(np.mean((signs[:-1]*signs[1:])<0)) if len(s)>1 else 0.0
    if environment in ('cartpole','lineworld'):
        delta=np.zeros_like(s) if state_delta is None else np.asarray(state_delta,dtype=np.float64)
        delta_mean=float(np.mean(delta)) if len(delta) else 0.0
        delta_abs_mean=float(np.mean(np.abs(delta))) if len(delta) else 0.0
        delta_max_abs=float(np.max(np.abs(delta))) if len(delta) else 0.0
        denom=float(np.linalg.norm(s)*np.linalg.norm(delta))
        delta_alignment=float(np.dot(s,delta)/denom) if denom>1e-12 else 0.0
    else:
        delta_mean=delta_abs_mean=delta_max_abs=delta_alignment=float('nan')
    if environment in ('ttt','connect_four'):
        board_features=board_structural_features(s,environment)
    else:
        board_features={name:float('nan') for name in ('board_own_open_lines','board_opponent_open_lines','board_own_immediate_wins','board_opponent_immediate_wins','board_center_control','board_occupancy')}
    return [
        mean,float(np.abs(s).mean()),float(np.mean((s-mean)**2)),float(np.abs(s).max()),
        (2*a/max(1,nact-1)-1),nlegal/max(1,nact),magnitude,dispersion,boundary_margin,
        state_min,state_max,state_range,positive_fraction,negative_fraction,sparsity,
        median,iqr,skewness,entropy,position_moment,position_spread,
        adjacent_difference,sign_change_fraction,
        delta_mean,delta_abs_mean,delta_max_abs,delta_alignment,
        *(board_features[name] for name in ('board_own_open_lines','board_opponent_open_lines','board_own_immediate_wins','board_opponent_immediate_wins','board_center_control','board_occupancy')),
        *(board_action_features(s,a,environment) if environment in ('ttt','connect_four') else (float('nan'),float('nan'))),
    ]

def paired_action_values(actions, rollout_fn, rollouts, rng, common_random_numbers=True):
    """Estimate action values with a configurable rollout-seed protocol.

    CRN can reduce variance when action outcomes are positively correlated,
    but can increase it when they are negatively correlated. The caller must
    select the mode based on environment-specific calibration.
    Returns (means, per_action_samples) in action order.
    """
    if rollouts < 1:
        raise ValueError('rollouts must be at least one')
    shared_seeds=[rng.randrange(0, 2**32) for _ in range(rollouts)] if common_random_numbers else None
    samples=[]
    for action in actions:
        seeds=shared_seeds if shared_seeds is not None else [rng.randrange(0, 2**32) for _ in range(rollouts)]
        values=[float(rollout_fn(action, random.Random(seed))) for seed in seeds]
        samples.append(values)
    means=np.asarray([np.mean(values) for values in samples],dtype=float)
    return means, samples


def ttt_win(b,p):
    return any(all(b[i]==p for i in line) for line in ((0,1,2),(3,4,5),(6,7,8),(0,3,6),(1,4,7),(2,5,8),(0,4,8),(2,4,6)))
def ttt_rollout(b, first_action, rng):
    b=b.copy();b[first_action]=1
    if ttt_win(b,1):return 1.
    if all(b):return 0.
    while True:
        legal=[i for i,x in enumerate(b) if x==0]
        if not legal:return 0.
        b[rng.choice(legal)]=2
        if ttt_win(b,2):return -1.
        legal=[i for i,x in enumerate(b) if x==0]
        if not legal:return 0.
        b[rng.choice(legal)]=1
        if ttt_win(b,1):return 1.

@lru_cache(maxsize=None)
def _ttt_minimax_cached(board_tuple, player):
    board=list(board_tuple)
    if ttt_win(board,1): return 1.0
    if ttt_win(board,2): return -1.0
    legal=[i for i,value in enumerate(board) if value==0]
    if not legal: return 0.0
    values=[]
    for action in legal:
        child=board.copy(); child[action]=player
        values.append(_ttt_minimax_cached(tuple(child),3-player))
    return max(values) if player==1 else min(values)


def ttt_oracle_action_values(board):
    """Exact minimax values for X-to-move TicTacToe states."""
    legal=[i for i,value in enumerate(board) if value==0]
    values=[]
    for action in legal:
        child=list(board); child[action]=1
        values.append(_ttt_minimax_cached(tuple(child),2))
    return legal,np.asarray(values,dtype=float)


def nim_oracle_action_values(remaining):
    """Exact normal-play values for a one-heap take-1-to-3 Nim state."""
    @lru_cache(maxsize=None)
    def value(n):
        if n<=0: return -1.0
        return max(-value(n-take) for take in range(1,min(3,n)+1))
    legal=list(range(min(3,remaining)))
    values=np.asarray([1.0 if remaining-action-1==0 else -value(remaining-action-1) for action in legal],dtype=float)
    return legal,values


def lineworld_oracle_action_values(position, steps):
    """Exact deterministic returns when the continuation follows the shortest path."""
    def greedy_return(first_action):
        pos=position; count=steps; total=0.0
        pos,count,reward,done=lineworld_step(pos,count,first_action); total+=reward
        while not done:
            action=1 if pos<10 else 0
            pos,count,reward,done=lineworld_step(pos,count,action); total+=reward
        return total
    legal=lineworld_legal(position)
    return legal,np.asarray([greedy_return(action) for action in legal],dtype=float)


def make_ttt_data(seed=410, nstates=100, rollouts=5, teacher_policy="rollout"):
    rng=random.Random(seed); X=[];Y=[]
    for _ in range(nstates):
        b=[0]*9
        # Reachable states with X to move, occasionally with O having opened.
        if rng.random()<.5:b[rng.randrange(9)]=2
        for _ in range(rng.randrange(0,5)):
            legal=[i for i,x in enumerate(b) if x==0]
            if not legal:break
            b[rng.choice(legal)]=1
            legal=[i for i,x in enumerate(b) if x==0]
            if not legal:break
            b[rng.choice(legal)]=2
        if ttt_win(b,1) or ttt_win(b,2):continue
        legal=[i for i,x in enumerate(b) if x==0]
        if len(legal)<2:continue
        sf=[float(x==1) for x in b]+[float(x==2) for x in b]
        if teacher_policy=='oracle':
            _,vals=ttt_oracle_action_values(b)
        else:
            vals,_=paired_action_values(
                legal,lambda action,rollout_rng:ttt_rollout(b,action,rollout_rng),rollouts,rng
            )
        adv=vals-np.mean(vals)
        for a,y in zip(legal,adv): X.append(summaries(sf,a,9,len(legal),'ttt'));Y.append(float(y))
    return np.asarray(X,dtype=float).T,np.asarray(Y,dtype=float)

# Classic CartPole dynamics, same normalised 4-state observation used by the C++ env.
def cp_step(s,a):
    x,xd,th,thd=s;g=9.8;mc=1.;mp=.1;mt=mc+mp;l=.5;pml=mp*l;tau=.02;f=10. if a else -10.;c=math.cos(th);sn=math.sin(th)
    tmp=(f+pml*thd*thd*sn)/mt; tha=(g*sn-c*tmp)/(l*(4./3.-mp*c*c/mt)); xa=tmp-pml*tha*c/mt
    ns=[x+tau*xd,xd+tau*xa,th+tau*thd,thd+tau*tha]
    done=abs(ns[0])>2.4 or abs(ns[2])>12*math.pi/180
    return ns,done

def cp_episode(s,first,rng):
    s=list(s);a=first
    for t in range(500):
        s,done=cp_step(s,a)
        if done:return t+1
        a=rng.randrange(2)
    return 500

def make_cp_data(seed=911,nstates=45,rollouts=2,teacher_policy="rollout"):
    rng=random.Random(seed); X=[];Y=[]
    for _ in range(nstates):
        s=[rng.uniform(-.05,.05),rng.uniform(-.05,.05),rng.uniform(-.05,.05),rng.uniform(-.05,.05)]
        # Random walk to sample nontrivial states; retain the last observed transition.
        previous_delta=[0.0]*4
        for __ in range(rng.randrange(0,80)):
            old=list(s)
            s,d=cp_step(s,rng.randrange(2))
            if d:
                s=[rng.uniform(-.05,.05),rng.uniform(-.05,.05),rng.uniform(-.05,.05),rng.uniform(-.05,.05)]
                previous_delta=[0.0]*4
            else:
                oldf=[old[0]/2.4,old[1]/2.,old[2]/.21,old[3]/2.]
                newf=[s[0]/2.4,s[1]/2.,s[2]/.21,s[3]/2.]
                previous_delta=[n-o for n,o in zip(newf,oldf)]
        sf=[s[0]/2.4,s[1]/2.,s[2]/.21,s[3]/2.]
        delta=previous_delta
        actions=[0,1]
        vals,_=paired_action_values(
            actions,lambda action,rollout_rng:cp_episode(s,action,rollout_rng),rollouts,rng,common_random_numbers=False
        )
        adv=vals-np.mean(vals)
        for a,y in enumerate(adv): X.append(summaries(sf,a,2,2,'cartpole',delta));Y.append(float(y))
    return np.asarray(X,dtype=float).T,np.asarray(Y,dtype=float)


def connect_four_wins(board,player):
    rows,cols=6,7
    for row in range(rows):
        for col in range(cols):
            if board[row][col]!=player:
                continue
            for dr,dc in ((0,1),(1,0),(1,1),(1,-1)):
                end_row,end_col=row+3*dr,col+3*dc
                if 0<=end_row<rows and 0<=end_col<cols and all(
                    board[row+step*dr][col+step*dc]==player for step in range(1,4)
                ):
                    return True
    return False


def connect_four_legal(board):
    return [col for col in range(7) if board[0][col]==0]


def connect_four_drop(board,col,player):
    result=[row[:] for row in board]
    for row in range(5,-1,-1):
        if result[row][col]==0:
            result[row][col]=player
            return result
    raise ValueError(f'illegal Connect Four column: {col}')


def connect_four_rollout(board,first_action,rng):
    board=connect_four_drop(board,first_action,1)
    if connect_four_wins(board,1):
        return 1.
    while True:
        legal=connect_four_legal(board)
        if not legal:
            return 0.
        board=connect_four_drop(board,rng.choice(legal),2)
        if connect_four_wins(board,2):
            return -1.
        legal=connect_four_legal(board)
        if not legal:
            return 0.
        board=connect_four_drop(board,rng.choice(legal),1)
        if connect_four_wins(board,1):
            return 1.


def make_connect_four_data(seed=1201,nstates=60,rollouts=2,teacher_policy="rollout"):
    rng=random.Random(seed); X=[];Y=[]
    while len(Y)<nstates:
        board=[[0]*7 for _ in range(6)]
        if rng.random()<.5:
            board=connect_four_drop(board,rng.randrange(7),2)
        for _ in range(42):
            legal=connect_four_legal(board)
            if len(legal)<2:
                break
            state=[float(board[row][col]==1) for row in range(6) for col in range(7)]
            state.extend(float(board[row][col]==2) for row in range(6) for col in range(7))
            values,_=paired_action_values(
                legal,lambda action,rollout_rng:connect_four_rollout(board,action,rollout_rng),rollouts,rng
            )
            advantage=values-np.mean(values)
            for col,value in zip(legal,advantage):
                X.append(summaries(state,col,7,len(legal),'connect_four'))
                Y.append(float(value))
            own_action=rng.choice(legal)
            board=connect_four_drop(board,own_action,1)
            if connect_four_wins(board,1):
                break
            legal=connect_four_legal(board)
            if not legal:
                break
            board=connect_four_drop(board,rng.choice(legal),2)
            if connect_four_wins(board,2):
                break
    return np.asarray(X,dtype=float).T,np.asarray(Y,dtype=float)


def nim_rollout(remaining,first_action,rng):
    remaining-=first_action+1
    if remaining<=0:
        return 1.
    while True:
        remaining-=rng.randint(1,min(3,remaining))
        if remaining<=0:
            return -1.
        remaining-=rng.randint(1,min(3,remaining))
        if remaining<=0:
            return 1.


def make_nim_data(seed=1301,nstates=60,rollouts=4,teacher_policy="rollout"):
    rng=random.Random(seed); X=[];Y=[]
    for _ in range(nstates):
        remaining=rng.randint(1,21)
        legal=list(range(min(3,remaining)))
        if teacher_policy=='oracle':
            _,values=nim_oracle_action_values(remaining)
        else:
            values,_=paired_action_values(
                legal,lambda action,rollout_rng:nim_rollout(remaining,action,rollout_rng),rollouts,rng,common_random_numbers=False
            )
        advantage=values-np.mean(values)
        for action,value in zip(legal,advantage):
            X.append(summaries([remaining/21.],action,3,len(legal),'nim'))
            Y.append(float(value))
    return np.asarray(X,dtype=float).T,np.asarray(Y,dtype=float)


def lineworld_legal(position):
    legal=[]
    if position>0:
        legal.append(0)
    if position<10:
        legal.append(1)
    return legal


def lineworld_step(position,steps,action):
    if action not in lineworld_legal(position):
        raise ValueError(f'illegal LineWorld action: {action}')
    position+=1 if action==1 else -1
    steps+=1
    done=position==10 or steps>=30
    reward=1. if position==10 else (-0.1 if done else -0.01)
    return position,steps,reward,done


def lineworld_rollout(position,steps,first_action,rng):
    total=0.
    position,steps,reward,done=lineworld_step(position,steps,first_action)
    total+=reward
    while not done:
        position,steps,reward,done=lineworld_step(position,steps,rng.choice(lineworld_legal(position)))
        total+=reward
    return total


def make_lineworld_data(seed=1401,nstates=60,rollouts=4,teacher_policy="rollout"):
    rng=random.Random(seed); X=[];Y=[]
    for _ in range(nstates):
        position=rng.randrange(10)
        steps=rng.randrange(0,25)
        legal=lineworld_legal(position)
        if teacher_policy=='oracle':
            _,values=lineworld_oracle_action_values(position,steps)
        else:
            values,_=paired_action_values(
                legal,lambda action,rollout_rng:lineworld_rollout(position,steps,action,rollout_rng),rollouts,rng
            )
        advantage=values-np.mean(values)
        state=[position/10.,1.-steps/30.]
        if position>0 and (position==10 or rng.random()<0.5):
            prev_position=position-1
        elif position<10:
            prev_position=position+1
        else:
            prev_position=position
        prev_steps=max(0,steps-1)
        previous=[prev_position/10.,1.-prev_steps/30.]
        delta=[state[i]-previous[i] for i in range(2)]
        for action,value in zip(legal,advantage):
            X.append(summaries(state,action,2,len(legal),'lineworld',delta))
            Y.append(float(value))
    return np.asarray(X,dtype=float).T,np.asarray(Y,dtype=float)


def make_dataset(environment,seed,nstates=None,rollouts=None,teacher_policy="rollout"):
    builders={
        'ttt':(make_ttt_data,(100,5)),
        'cartpole':(make_cp_data,(45,2)),
        'connect_four':(make_connect_four_data,(60,2)),
        'nim':(make_nim_data,(60,4)),
        'lineworld':(make_lineworld_data,(60,4)),
    }
    if environment not in builders:
        raise ValueError(f'unsupported game: {environment}')
    if teacher_policy not in ('rollout','oracle'):
        raise ValueError(f'unsupported teacher policy: {teacher_policy}')
    if teacher_policy=='oracle' and environment not in ('ttt','nim','lineworld'):
        raise ValueError('oracle teacher is currently implemented only for ttt, nim, and lineworld')
    builder,defaults=builders[environment]
    state_count=defaults[0] if nstates is None else nstates
    rollout_count=defaults[1] if rollouts is None else rollouts
    return builder(seed=seed,nstates=state_count,rollouts=rollout_count,teacher_policy=teacher_policy)

class NoveltyMonitor:
    def __init__(self, probes, corr_threshold=.995, window=8):
        self.probes=probes;self.threshold=corr_threshold;self.window=window;self.struct=set();self.behaviors=[];self.hist=[];self.stagnant=0
    def observe(self,pop):
        new_s=0;new_b=0;fps=[]
        for t in pop:
            k=canon(t)
            if k not in self.struct:self.struct.add(k);new_s+=1
            v=ev(t,self.probes); sd=np.std(v)
            if sd<1e-8: continue
            v=(v-v.mean())/(sd+1e-8)
            if all(abs(corr(v,z))<self.threshold for z in self.behaviors):
                self.behaviors.append(v);new_b+=1
            fps.append(v)
        div=0.
        if len(fps)>1:
            ds=[]
            for i in range(min(len(fps),40)):
                for j in range(i):ds.append(1-abs(corr(fps[i],fps[j])))
            div=float(np.mean(ds)) if ds else 0.
        self.hist.append({'structurally_new':new_s,'behaviorally_new':new_b,'behavioral_diversity':div})
        self.stagnant=0 if new_b else self.stagnant+1
        return self.hist[-1]

    def reset_stagnation(self):
        self.stagnant=0


def population_objectives(pop,X,y,Xval,yval):
    """Maximize robust signed prediction, validation fit, low overfit gap, simplicity and novelty.

    Signed correlation matters: an inverted feature is not treated as equally useful merely
    because its magnitude is high. The GP can discover an explicit negation if needed.
    The robust objective is the worse of train/validation correlation, discouraging
    expressions that fit one split while failing on the other.
    """
    outputs=[ev(t,Xval) for t in pop]
    normalized=[]
    for v in outputs:
        sd=float(np.std(v))
        normalized.append((v-v.mean())/(sd+1e-8) if sd>=1e-8 else None)
    objectives=[]
    neighbor_count=min(5,max(0,len(pop)-1))
    for i,t in enumerate(pop):
        train_v=ev(t,X)
        validation_v=outputs[i]
        valid=(np.std(train_v)>=1e-8 and np.std(validation_v)>=1e-8
               and np.all(np.isfinite(train_v)) and np.all(np.isfinite(validation_v)))
        if not valid:
            objectives.append((-1.,-1.,-1.,-float(node_size(t)),-1.))
            continue
        train_corr=corr(train_v,y)
        validation_corr=corr(validation_v,yval)
        robust=min(train_corr,validation_corr)
        generalization_gap=abs(train_corr-validation_corr)
        distances=[]
        for j,other in enumerate(normalized):
            if i!=j and other is not None:
                distances.append(1.-abs(corr(normalized[i],other)))
        novelty=float(np.mean(sorted(distances)[:neighbor_count])) if distances and neighbor_count else 0.
        objectives.append((robust,validation_corr,-generalization_gap,-float(node_size(t)),novelty))
    return objectives


def nondominated_fronts(objectives):
    """Return NSGA-II fronts, with all objective values treated as maximization."""
    dominates=[[] for _ in objectives]
    dominated_count=[0 for _ in objectives]
    first=[]
    for i,a in enumerate(objectives):
        for j,b in enumerate(objectives):
            if i==j:
                continue
            if all(x>=y for x,y in zip(a,b)) and any(x>y for x,y in zip(a,b)):
                dominates[i].append(j)
            elif all(x>=y for x,y in zip(b,a)) and any(x>y for x,y in zip(b,a)):
                dominated_count[i]+=1
        if dominated_count[i]==0:
            first.append(i)
    fronts=[]
    current=first
    while current:
        fronts.append(current)
        following=[]
        for i in current:
            for j in dominates[i]:
                dominated_count[j]-=1
                if dominated_count[j]==0:
                    following.append(j)
        current=following
    return fronts


def crowding_distances(front,objectives):
    distances={i:0. for i in front}
    if len(front)<=2:
        return {i:float('inf') for i in front}
    for objective_index in range(len(objectives[0])):
        ordered=sorted(front,key=lambda i:objectives[i][objective_index])
        low=objectives[ordered[0]][objective_index]
        high=objectives[ordered[-1]][objective_index]
        distances[ordered[0]]=float('inf')
        distances[ordered[-1]]=float('inf')
        if high==low:
            continue
        for position in range(1,len(ordered)-1):
            i=ordered[position]
            if math.isfinite(distances[i]):
                before=objectives[ordered[position-1]][objective_index]
                after=objectives[ordered[position+1]][objective_index]
                distances[i]+=(after-before)/(high-low)
    return distances


def nsga_ranking(objectives):
    fronts=nondominated_fronts(objectives)
    ranks={i:rank for rank,front in enumerate(fronts) for i in front}
    crowding={}
    for front in fronts:
        crowding.update(crowding_distances(front,objectives))
    order=sorted(range(len(objectives)),key=lambda i:(ranks[i],-crowding[i]))
    return order,ranks,crowding


def update_elite_archive(archive,pop,X,y,Xval,yval,limit):
    candidates=[];seen=set()
    for tree in archive+pop:
        key=canon(tree)
        if key not in seen:
            candidates.append(tree);seen.add(key)
    if not candidates or limit<=0:
        return []
    candidate_objectives=population_objectives(candidates,X,y,Xval,yval)
    selected=[]
    for front in nondominated_fronts(candidate_objectives):
        distances=crowding_distances(front,candidate_objectives)
        ordered=sorted(front,key=lambda i:distances[i],reverse=True)
        selected.extend(ordered[:limit-len(selected)])
        if len(selected)>=limit:
            break
    return [candidates[i] for i in selected]


def restart_population(archive,size,rng,primitives,random_fraction=.65,
                       mutation_intensity=3,subtree_depth=3):
    if size<1:
        raise ValueError('restart population size must be positive')
    if not 0.0<=random_fraction<=1.0:
        raise ValueError('restart random fraction must be between 0 and 1')
    elites=[]
    seen=set()
    for tree in archive:
        key=canon(tree)
        if key not in seen:
            elites.append(tree);seen.add(key)
        if len(elites)>=size:
            break
    if not elites:
        raise ValueError('restart requires at least one archived elite')
    remaining=size-len(elites)
    random_count=min(remaining,max(1,int(round(remaining*random_fraction)))) if remaining else 0
    restarted=elites.copy()
    restarted.extend(randtree(rng,maxd=4,primitives=primitives) for _ in range(random_count))
    while len(restarted)<size:
        child=mutate(
            rng.choice(elites),rng,primitives,
            intensity=mutation_intensity,subtree_depth=subtree_depth,
        )
        if depth(child)>6:
            child=randtree(rng,maxd=4,primitives=primitives)
        restarted.append(child)
    return restarted


def evolve_population(pop,X,y,Xval,yval,monitor,rng,primitives,generations,
                      stagnation_window=8,max_unsuccessful_restarts=3,
                      restart_random_fraction=.65,restart_mutation_generations=3):
    if max_unsuccessful_restarts<1:
        raise ValueError('max unsuccessful restarts must be positive')
    archive=[];logs=[];restart_events=[]
    unsuccessful_restarts=0;successful_restarts=0
    restarts_triggered=0;pending_restart=None;exploration_left=0
    archive_limit=max(1,len(pop)//10)

    for generation in range(generations):
        objectives=population_objectives(pop,X,y,Xval,yval)
        archive=update_elite_archive(archive,pop,X,y,Xval,yval,archive_limit)
        _,ranks,crowding=nsga_ranking(objectives)
        novelty=monitor.observe(pop)
        record={
            'generation':generation,
            **novelty,
            'pareto_front_size':sum(rank==0 for rank in ranks.values()),
            'best_robust_signed_corr':float(max(item[0] for item in objectives)),
            'best_validation_signed_corr':float(max(item[1] for item in objectives)),
            'best_generalization_consistency':float(max(item[2] for item in objectives)),
            'best_behavioral_novelty':float(max(item[4] for item in objectives)),
            'smallest_tree_nodes':int(-max(item[3] for item in objectives)),
            'restart_triggered':False,
            'restart_outcome':None,
        }
        logs.append(record)

        if pending_restart is not None:
            pending_restart['new_behaviors']+=novelty['behaviorally_new']
            pending_restart['generations_remaining']-=1
            if pending_restart['new_behaviors']>0:
                successful_restarts+=1
                unsuccessful_restarts=0
                logs[pending_restart['log_index']]['restart_outcome']='successful'
                logs[pending_restart['log_index']]['restart_new_behaviors']=pending_restart['new_behaviors']
                pending_restart=None
            elif pending_restart['generations_remaining']==0:
                unsuccessful_restarts+=1
                logs[pending_restart['log_index']]['restart_outcome']='unsuccessful'
                logs[pending_restart['log_index']]['restart_new_behaviors']=0
                logs[pending_restart['log_index']]['consecutive_unsuccessful_restarts']=unsuccessful_restarts
                pending_restart=None
                if unsuccessful_restarts>=max_unsuccessful_restarts:
                    break

        if monitor.stagnant>=stagnation_window and pending_restart is None:
            restart_id=restarts_triggered+1
            preserved_count=min(len(archive),len(pop))
            remaining_count=len(pop)-preserved_count
            random_count=min(remaining_count,max(1,int(round(remaining_count*restart_random_fraction)))) if remaining_count else 0
            pop=restart_population(
                archive,len(pop),rng,primitives,
                random_fraction=restart_random_fraction,
            )
            monitor.reset_stagnation()
            exploration_left=restart_mutation_generations
            restarts_triggered+=1
            record['restart_triggered']=True
            record['restart_id']=restart_id
            record['preserved_elites']=preserved_count
            record['random_individuals_injected']=random_count
            record['mutated_individuals_injected']=remaining_count-random_count
            record['high_mutation_generations']=restart_mutation_generations
            record['restart_outcome']='pending'
            restart_events.append({
                'restart_id':restart_id,
                'generation':generation,
                'outcome':'pending',
                'new_behaviors':None,
            })
            pending_restart={
                'restart_id':restart_id,
                'log_index':len(logs)-1,
                'generations_remaining':stagnation_window,
                'new_behaviors':0,
            }
            continue

        parent_pool=list(range(len(pop)))
        def tournament_parent():
            choices=rng.sample(parent_pool,2)
            return min(choices,key=lambda i:(ranks[i],-crowding.get(i,0.)))
        offspring=[]
        high_mutation=exploration_left>0
        mutation_probability=.75 if high_mutation else .45
        while len(offspring)<len(pop):
            if rng.random()<1.-mutation_probability:
                offspring.append(crossover(pop[tournament_parent()],pop[tournament_parent()],rng))
            else:
                offspring.append(mutate(
                    pop[tournament_parent()],rng,primitives,
                    intensity=3 if high_mutation else 1,
                    subtree_depth=3 if high_mutation else 2,
                ))
            if depth(offspring[-1])>5:
                offspring[-1]=randtree(rng,maxd=3,primitives=primitives)
        combined=pop+offspring
        combined_objectives=population_objectives(combined,X,y,Xval,yval)
        next_indices=[]
        for front in nondominated_fronts(combined_objectives):
            distances=crowding_distances(front,combined_objectives)
            front=sorted(front,key=lambda i:distances[i],reverse=True)
            next_indices.extend(front[:len(pop)-len(next_indices)])
            if len(next_indices)>=len(pop):
                break
        pop=[combined[i] for i in next_indices]
        if high_mutation:
            exploration_left-=1

    for event in restart_events:
        outcome_record=next(
            log for log in logs if log.get('restart_id')==event['restart_id']
        )
        event['outcome']=outcome_record['restart_outcome']
        event['new_behaviors']=outcome_record.get('restart_new_behaviors')
    summary={
        'restarts_triggered':restarts_triggered,
        'successful_restarts':successful_restarts,
        'unsuccessful_restarts':sum(event['outcome']=='unsuccessful' for event in restart_events),
        'consecutive_unsuccessful_restarts':unsuccessful_restarts,
        'max_unsuccessful_restarts':max_unsuccessful_restarts,
        'stopped_after_unsuccessful_restarts':unsuccessful_restarts>=max_unsuccessful_restarts,
        'events':restart_events,
    }
    archive=update_elite_archive(archive,pop,X,y,Xval,yval,archive_limit)
    return pop,archive,logs,summary


def parse_seed_list(value):
    seeds=[int(part.strip()) for part in value.split(',') if part.strip()]
    if not seeds or len(set(seeds))!=len(seeds):
        raise ValueError('downstream seeds must be a non-empty, comma-separated list of unique integers')
    return seeds


def resolve_brain_path(path):
    brain=Path(path)
    if not brain.is_absolute():
        brain=Path(__file__).resolve().parent/brain
    windows_executable=brain.with_suffix('.exe')
    if os.name=='nt' and brain.suffix.lower()!='.exe' and windows_executable.exists():
        brain=windows_executable
    if not brain.exists() and brain.suffix.lower()!='.exe':
        if windows_executable.exists():
            brain=windows_executable
    if not brain.is_file():
        raise FileNotFoundError(f'NEAT+PPO executable not found: {brain}')
    with brain.open('rb') as executable_file:
        if os.name=='nt' and executable_file.read(4)==b'\x7fELF':
            raise RuntimeError(f'{brain} is a Linux ELF executable; build a Windows executable or run this GP pipeline under Linux/WSL')
    return brain


def validate_brain_contract(brain):
    result=subprocess.run([str(brain),'--primitive-contract'],capture_output=True,text=True,check=False)
    if result.returncode!=0 or result.stdout.strip()!=PRIMITIVE_CONTRACT:
        raise RuntimeError(
            f'NEAT+PPO executable must report primitive contract {PRIMITIVE_CONTRACT}; '
            f'rebuild src/brain from the current sources. stdout={result.stdout!r}; stderr={result.stderr!r}'
        )


def run_brain_trial(brain,target,expressions,budget,seed,population):
    with tempfile.TemporaryDirectory(prefix='gp_downstream_') as directory:
        work=Path(directory)
        expression_file=work/'concepts.txt'
        expression_file.write_text('\n'.join(expressions)+'\n',encoding='utf-8')
        result=subprocess.run(
            [str(brain),target,str(expression_file),str(budget),str(seed),str(population)],
            cwd=work,capture_output=True,text=True,check=True,
        )
    matches=re.findall(r'HELDOUT return ([+-]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][+-]?\d+)?)',result.stdout)
    if not matches:
        raise RuntimeError(f'NEAT+PPO run did not report held-out return:\n{result.stdout}\n{result.stderr}')
    return float(matches[-1])


def evaluate_downstream(candidates,target_slug,budget,seeds,population,brain,min_improvement=0.,min_win_fraction=.6):
    if budget<=0:
        raise ValueError('downstream evaluation requires a positive environment-step budget')
    if not math.isfinite(min_improvement) or min_improvement<0:
        raise ValueError('minimum downstream improvement must be a finite non-negative value')
    if not math.isfinite(min_win_fraction) or not 0.0<=min_win_fraction<=1.0:
        raise ValueError('minimum downstream win fraction must be between zero and one')
    executable=resolve_brain_path(brain)
    validate_brain_contract(executable)
    target_name=GAME_NAMES[target_slug]
    zero='sub(state_mean,state_mean)'
    baseline_expressions=['action_norm',zero,zero,zero,zero]
    baselines={seed:run_brain_trial(executable,target_slug,baseline_expressions,budget,seed,population) for seed in seeds}
    for item in candidates:
        expressions=['action_norm',item['expression'],zero,zero,zero]
        returns={seed:run_brain_trial(executable,target_slug,expressions,budget,seed,population) for seed in seeds}
        differences={str(seed):returns[seed]-baselines[seed] for seed in seeds}
        difference_values=list(differences.values())
        mean_difference=float(np.mean(difference_values))
        win_fraction=float(np.mean([value>0.0 for value in difference_values]))
        median_difference=float(np.median(difference_values))
        item['downstream_utility']={
            'target_game':target_name,
            'budget_steps':budget,
            'population':population,
            'seeds':seeds,
            'baseline_returns':{str(seed):baselines[seed] for seed in seeds},
            'candidate_returns':{str(seed):returns[seed] for seed in seeds},
            'paired_differences':differences,
            'mean_paired_difference':mean_difference,
            'minimum_improvement':min_improvement,
            'median_paired_difference':median_difference,
            'win_fraction':win_fraction,
            'minimum_win_fraction':min_win_fraction,
            'passed':mean_difference>min_improvement and median_difference>0.0 and win_fraction>=min_win_fraction,
        }
    return candidates


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--out',default='results')
    ap.add_argument('--seed',type=int,default=2026)
    ap.add_argument('--pop',type=int,default=100)
    ap.add_argument('--gens',type=int,default=30)
    ap.add_argument('--source',choices=ENVIRONMENTS,default='ttt')
    ap.add_argument('--target',choices=ENVIRONMENTS,help='Transfer target; defaults to a paired game different from the source.')
    ap.add_argument('--shortlist',type=int,default=4)
    ap.add_argument('--min-source-robust-corr',type=float,default=.02,help='Minimum signed train/validation correlation for a source candidate.')
    ap.add_argument('--nstates',type=int,default=0,help='Override per-split source/target state sample counts; 0 uses environment defaults.')
    ap.add_argument('--rollouts',type=int,default=0,help='Override rollouts per candidate action; 0 uses environment defaults.')
    ap.add_argument('--teacher-policy',choices=('rollout','oracle'),default='rollout',help='Use random-rollout labels or exact oracle labels where supported (ttt, nim, lineworld).')
    ap.add_argument('--stagnation-window',type=int,default=8)
    ap.add_argument('--max-unsuccessful-restarts',type=int,default=3)
    ap.add_argument('--restart-random-fraction',type=float,default=.65)
    ap.add_argument('--restart-mutation-generations',type=int,default=3)
    ap.add_argument('--downstream-budget',type=int,default=0,help='Enable fresh target NEAT+PPO admission evaluation; 0 leaves candidates pending.')
    ap.add_argument('--downstream-seeds',default='11,22,33,44,55,66,77,88,99,111')
    ap.add_argument('--downstream-pop',type=int,default=12)
    ap.add_argument('--downstream-min-improvement',type=float,default=0.)
    ap.add_argument('--same-game-control',action='store_true',help='Allow source==target: discovers concepts on the target game itself (target-trained GP control).')
    ap.add_argument('--downstream-min-win-fraction',type=float,default=.6,help='Minimum fraction of paired learner seeds that must improve for concept admission.')
    ap.add_argument('--brain',default='src/brain',help='Path to the compiled NEAT+PPO executable.')
    args=ap.parse_args()
    if (args.pop<2 or args.gens<1 or args.shortlist<1 or args.downstream_pop<1
            or args.stagnation_window<1 or args.max_unsuccessful_restarts<1 or args.nstates<0 or args.rollouts<0
            or args.restart_mutation_generations<0):
        ap.error('pop must be at least 2; gens, shortlist, downstream-pop, stagnation-window, and max-unsuccessful-restarts must be positive; restart-mutation-generations cannot be negative')
    if not 0.0<=args.restart_random_fraction<=1.0:
        ap.error('restart-random-fraction must be between 0 and 1')
    if not math.isfinite(args.min_source_robust_corr) or not -1.0<=args.min_source_robust_corr<=1.0:
        ap.error('min-source-robust-corr must be between -1 and 1')
    if args.downstream_budget<0:
        ap.error('downstream-budget cannot be negative')
    if not math.isfinite(args.downstream_min_improvement) or args.downstream_min_improvement<0:
        ap.error('downstream-min-improvement must be a finite non-negative value')
    if not math.isfinite(args.downstream_min_win_fraction) or not 0.0<=args.downstream_min_win_fraction<=1.0:
        ap.error('downstream-min-win-fraction must be between zero and one')
    seeds=parse_seed_list(args.downstream_seeds) if args.downstream_budget else []
    if args.downstream_budget:
        downstream_brain=resolve_brain_path(args.brain)
        validate_brain_contract(downstream_brain)
    else:
        downstream_brain=args.brain

    out=Path(args.out);out.mkdir(parents=True,exist_ok=True);rng=random.Random(args.seed)
    target_slug=args.target or DEFAULT_TARGETS[args.source]
    if target_slug==args.source and not args.same_game_control:
        ap.error('source and target must be different games for cross-game transfer')
    sample_states=args.nstates or None
    sample_rollouts=args.rollouts or None
    Xt,yt=make_dataset(args.source,args.seed+410,nstates=sample_states,rollouts=sample_rollouts,teacher_policy=args.teacher_policy)
    Xval,yval=make_dataset(args.source,args.seed+441,nstates=sample_states,rollouts=sample_rollouts,teacher_policy=args.teacher_policy)
    Xtest,ytest=make_dataset(args.source,args.seed+442,nstates=sample_states,rollouts=sample_rollouts,teacher_policy=args.teacher_policy)
    Xtarget,_=make_dataset(target_slug,args.seed+912,nstates=sample_states,rollouts=sample_rollouts,teacher_policy=args.teacher_policy if target_slug in ('ttt','nim','lineworld') else 'rollout')
    source_name,target_name=GAME_NAMES[args.source],GAME_NAMES[target_slug]
    source_primitives=available_primitives(args.source)
    target_primitives=set(available_primitives(target_slug))
    pop=[randtree(rng,primitives=source_primitives) for _ in range(args.pop)]
    monitor=NoveltyMonitor(Xval,window=args.stagnation_window)
    pop,archive,logs,restart_summary=evolve_population(
        pop,Xt,yt,Xval,yval,monitor,rng,source_primitives,args.gens,
        stagnation_window=args.stagnation_window,
        max_unsuccessful_restarts=args.max_unsuccessful_restarts,
        restart_random_fraction=args.restart_random_fraction,
        restart_mutation_generations=args.restart_mutation_generations,
    )
    final_population=[];seen_trees=set()
    for tree in archive+pop:
        key=canon(tree)
        if key not in seen_trees:
            final_population.append(tree);seen_trees.add(key)
    final_objectives=population_objectives(final_population,Xt,yt,Xval,yval)
    final_order,_,_=nsga_ranking(final_objectives)
    chosen=[];target_vecs=[]
    for i in final_order:
        tree=final_population[i]
        required=set(dependencies(tree))
        state_dependencies=required-{'action_norm','legal_norm'}
        if not state_dependencies:
            continue
        if final_objectives[i][0] < args.min_source_robust_corr:
            continue
        if not required.issubset(target_primitives):
            continue
        if depth(tree)>5 or np.std(ev(tree,Xt))<1e-8 or np.std(ev(tree,Xval))<1e-8:
            continue
        target_values=ev(tree,Xtarget)
        if np.std(target_values)<.01:
            continue
        if any(abs(corr(target_values,previous))>=.995 for previous in target_vecs):
            continue
        objective=final_objectives[i]
        item={
            'id':f'{args.source}_gp_{len(chosen)+1}',
            'expression':render(tree),
            'canonical':canon(tree),
            'source_game':source_name,
            'target_game':target_name,
            'primitive_contract':PRIMITIVE_CONTRACT,
            'primitive_dependencies':sorted(required),
            'compatible_games':[game for game in ENVIRONMENTS if required.issubset(available_primitives(game))],
            'source_robust_signed_corr':float(objective[0]),
            'source_validation_signed_corr':float(objective[1]),
            'source_generalization_consistency':float(objective[2]),
            'source_test_abs_corr':float(abs(corr(ev(tree,Xtest),ytest))),
            'behavioral_novelty':float(objective[4]),
            'target_output_std':float(np.std(target_values)),
            'complexity':node_size(tree),
            'source_data_protocol':DATASET_PROTOCOL if args.teacher_policy=='rollout' else 'exact_oracle_labels_v1',
            'teacher_policy':args.teacher_policy,
            'source_data_config':{'train_states':int(args.nstates or {'ttt':100,'cartpole':45,'connect_four':60,'nim':60,'lineworld':60}[args.source]),'rollouts_per_action':int(args.rollouts or {'ttt':5,'cartpole':2,'connect_four':2,'nim':4,'lineworld':4}[args.source])},
            'depth':depth(tree),
            'admission_status':'pending_downstream',
            'downstream_utility':None,
        }
        chosen.append(item);target_vecs.append(target_values)
        if len(chosen)>=args.shortlist:
            break

    if args.downstream_budget and chosen:
        evaluate_downstream(
            chosen,target_slug,args.downstream_budget,seeds,args.downstream_pop,
            downstream_brain,args.downstream_min_improvement,args.downstream_min_win_fraction,
        )
    if args.downstream_budget:
        for item in chosen:
            if item['downstream_utility'] is not None:
                item['admission_status']='admitted' if item['downstream_utility']['passed'] else 'rejected_downstream'
    admitted=[item for item in chosen if item['admission_status']=='admitted']
    candidate_expressions=[item['expression'] for item in chosen]
    admitted_expressions=[item['expression'] for item in admitted]
    (out/'downstream_shortlist.txt').write_text(
        '# Candidates passed source validation and target feature compatibility; downstream admission may still be pending.\n'
        +''.join(expression+'\n' for expression in candidate_expressions),encoding='utf-8')
    (out/'transferred_concepts.txt').write_text(
        f'# Downstream-admitted concepts; primitive contract {PRIMITIVE_CONTRACT}.\n'
        +''.join(expression+'\n' for expression in admitted_expressions),encoding='utf-8')
    (out/'compatible_target_concepts.txt').write_text(
        '# Concepts admitted by fixed-budget fresh target-learner evaluation.\n'
        +''.join(expression+'\n' for expression in admitted_expressions),encoding='utf-8')
    (out/'concept_candidates.json').write_text(json.dumps({
        'version':2,
        'primitive_contract':PRIMITIVE_CONTRACT,
        'primitive_metadata':PRIMITIVE_METADATA,
        'target_game':target_name,
        'downstream_evaluation_enabled':bool(args.downstream_budget),
        'candidates':chosen,
    },indent=2),encoding='utf-8')
    (out/'concept_bank.json').write_text(json.dumps({
        'version':2,
        'primitive_contract':PRIMITIVE_CONTRACT,
        'primitive_metadata':PRIMITIVE_METADATA,
        'concepts':admitted,
    },indent=2),encoding='utf-8')
    (out/'novelty_history.json').write_text(json.dumps({
        'stopped_for_stagnation':restart_summary['stopped_after_unsuccessful_restarts'],
        'stagnation_window':monitor.window,
        'generations_run':len(logs),
        'restart_summary':restart_summary,
        'history':logs,
    },indent=2),encoding='utf-8')
    final_fronts=nondominated_fronts(final_objectives)
    summary={
        'generations_run':len(logs),
        'stopped_for_stagnation':restart_summary['stopped_after_unsuccessful_restarts'],
        'stopped_after_unsuccessful_restarts':restart_summary['stopped_after_unsuccessful_restarts'],
        'final_stagnant_generations':monitor.stagnant,
        'restart_summary':restart_summary,
        'unique_structures':len(monitor.struct),
        'unique_behaviors':len(monitor.behaviors),
        'shortlist_size':len(chosen),
        'concepts_admitted':len(admitted),
        'downstream_evaluation_enabled':bool(args.downstream_budget),
        'source_game':source_name,
        'target_game':target_name,
        'dataset_protocol':DATASET_PROTOCOL if args.teacher_policy=='rollout' else 'exact_oracle_labels_v1',
        'teacher_policy':args.teacher_policy,
        'primitive_contract':PRIMITIVE_CONTRACT,
        'fitness_version':'signed_robust_v1',
        'min_source_robust_corr':args.min_source_robust_corr,
        'downstream_min_win_fraction':args.downstream_min_win_fraction,
        'dataset_overrides':{'nstates':args.nstates,'rollouts':args.rollouts},
        'primitive_metadata':PRIMITIVE_METADATA,
        'available_source_primitives':source_primitives,
        'available_target_primitives':sorted(target_primitives),
        'final_pareto_front_size':len(final_fronts[0]) if final_fronts else 0,
        'concepts':admitted,
    }
    (out/'gp_summary.json').write_text(json.dumps(summary,indent=2),encoding='utf-8')
    print(json.dumps(summary,indent=2))

if __name__=='__main__':main()
