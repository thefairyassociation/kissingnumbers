"""dim-13 search in the global-linear-code framework.

Configuration = 312 D13 roots (+-2,+-2,0^11) + |S| * 2^dim(V) weight-8 vectors.
Searches over linear codes V <= F_2^13 and families of 8-supports, then solves
the coset system; every hit is Gram-checked in exact integer arithmetic and
written to disk.
"""
import sys, os, json, random, time
sys.path.insert(0, 'lower_bound/lib')
import numpy as np
from gsearch import pc, code_words, good_supports, kerdims_fast, build_adj, greedy_clique
from cosets import solve_cosets_ls, build_vectors

N = 13
DROOTS = 4 * N * (N - 1) // 2          # 312
RECORD = 1154
OUT = 'lower_bound/dim13/v2/configs'
LOG = 'lower_bound/logs/progress.log'

def log(msg):
    with open(LOG, 'a') as f:
        f.write(time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime()) + ' | 13 | globalcode | ' + msg + '\n')

def gram_ok(W):
    A = np.array(W, dtype=np.int64)
    G = A @ A.T
    np.fill_diagonal(G, -99)
    return int(G.max()), len(set(map(tuple, W))) == len(W), int((A * A).sum(1).min()), int((A * A).sum(1).max())

def try_family(sups, V, seed, shrink=True):
    """solve cosets, shrinking the family if needed; returns (sups, c) or None"""
    cur = list(sups)
    while len(cur) >= 4:
        c = solve_cosets_ls(cur, V, N, seed=seed, iters=250000)
        if c is not None:
            return cur, c
        if not shrink: return None
        cur.pop(random.Random(seed + len(cur)).randrange(len(cur)))
    return None

def run(seed, budget, dims=(5, 6)):
    rng = random.Random(seed)
    best = 0
    t0 = time.time()
    while time.time() - t0 < budget:
        d = rng.choice(dims)
        basis = [rng.randrange(1, 1 << N) for _ in range(d)]
        V = code_words(basis)
        if len(set(V)) != 1 << d: continue
        G = good_supports(V, N)
        if len(G) < 8: continue
        K = kerdims_fast(V, N)
        adj = build_adj(G, V, N, d, K)
        cl = greedy_clique(adj, rng, 25)
        if len(cl) * (1 << d) + DROOTS <= max(best, 900): continue
        sel = [G[i] for i in cl]
        r = try_family(sel, V, seed)
        if not r: continue
        sel, c = r
        W = build_vectors(sel, V, c, N)
        mx, distinct, n1, n2 = gram_ok(W)
        if mx <= 4 and distinct and n1 == 8 and n2 == 8:
            tot = len(W) + DROOTS
            if tot > best:
                best = tot
                log(f'd={d} supports={len(sel)} weight8={len(W)} total={tot} '
                    f'{"BEATS" if tot > RECORD else "below"}-record')
                print(f'seed{seed}: d={d} sup={len(sel)} w8={len(W)} TOTAL={tot}', flush=True)
                json.dump({'V': V, 'supports': sel, 'cosets': c, 'dim': d,
                           'weight8': len(W), 'total': tot},
                          open(f'{OUT}/gc_seed{seed}_{tot}.json', 'w'))
    return best

if __name__ == '__main__':
    seed = int(sys.argv[1]); budget = float(sys.argv[2])
    print('best total for seed', seed, ':', run(seed, budget))
