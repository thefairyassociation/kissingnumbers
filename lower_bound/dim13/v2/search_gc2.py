"""dim-13 global-linear-code search, growing support families under the
exact F_2 coset-feasibility constraint.

count = 4*C(13,2) D-roots + |F| * 2^dim(V) weight-8 vectors."""
import sys, os, json, random, time
sys.path.insert(0, 'lower_bound/lib')
import numpy as np
from gsearch import pc, code_words, good_supports, kerdims_fast
import cosets2
from cosets import build_vectors

N = 13; DROOTS = 312; RECORD = 1154
OUT = 'lower_bound/dim13/v2/configs'; LOG = 'lower_bound/logs/progress.log'
par = lambda x: pc(x) & 1

def log(msg):
    with open(LOG, 'a') as f:
        f.write(time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime()) + ' | 13 | gc2 | ' + msg + '\n')

def prep(V, n, d):
    """perp basis, functional coordinates, good supports, kernel dims"""
    B = []
    P = []
    for a in range(1 << n):
        if all(par(a & v) == 0 for v in V):
            if a: P.append(a)
            x = a
            for b in B: x = min(x, x ^ b)
            if x: B.append(x); B.sort(reverse=True)
    k = len(B)
    coord = {}
    for a in P:
        x = a; lam = 0
        for idx, b in enumerate(B):
            if x ^ b < x: x ^= b; lam |= 1 << idx
        coord[a] = lam
    return B, k, P, coord

def grow(G, V, n, d, K, P, coord, k, rng):
    """greedily grow a support family keeping the forced linear system consistent"""
    full = (1 << n) - 1
    order = list(G); rng.shuffle(order)
    F = []; piv = {}
    def addrow(pv, row, rhs):
        while row:
            h = row.bit_length() - 1
            if h in pv: pr, prhs = pv[h]; row ^= pr; rhs ^= prhs
            else: pv[h] = (row, rhs); return True
        return rhs == 0
    for S in order:
        rows = []; ok = True
        for idx, T in enumerate(F):
            X = S & T; t = pc(X)
            if t <= 4: continue
            if t >= 7: ok = False; break
            if d - K[X] >= t: ok = False; break
            A = [a for a in P if not (a & (full ^ X))]
            if not A: ok = False; break
            if len(A) == 1:
                lam = coord[A[0]]; row = 0
                i, j = idx, len(F)
                for b in range(k):
                    if lam >> b & 1: row ^= (1 << (i * k + b)) ^ (1 << (j * k + b))
                rows.append(row)
        if not ok: continue
        trial = dict(piv); good = True
        for row in rows:
            if not addrow(trial, row, 1): good = False; break
        if good: piv = trial; F.append(S)
    return F

def run(seed, budget, dims=(5, 6, 4)):
    rng = random.Random(seed); best = 0; t0 = time.time(); tried = 0
    while time.time() - t0 < budget:
        d = rng.choice(dims); tried += 1
        basis = [rng.randrange(1, 1 << N) for _ in range(d)]
        V = code_words(basis)
        if len(set(V)) != 1 << d: continue
        G = good_supports(V, N)
        if len(G) * (1 << d) + DROOTS < 900: continue
        K = kerdims_fast(V, N)
        B, k, P, coord = prep(V, N, d)
        for _rep in range(6):
            F = grow(G, V, N, d, K, P, coord, k, rng)
            if len(F) * (1 << d) + DROOTS <= best: continue
            c = cosets2.solve(F, V, N, seed=rng.randrange(10 ** 9), restarts=40, sweeps=3000)
            if c is None: continue
            W = build_vectors(F, V, c, N)
            A = np.array(W, dtype=np.int64); Gm = A @ A.T; np.fill_diagonal(Gm, -99)
            if Gm.max() > 4 or len(set(map(tuple, W))) != len(W): continue
            if (A * A).sum(1).min() != 8 or (A * A).sum(1).max() != 8: continue
            tot = len(W) + DROOTS
            if tot > best:
                best = tot
                print(f'seed{seed} d={d} sup={len(F)} w8={len(W)} TOTAL={tot}', flush=True)
                log(f'seed={seed} d={d} supports={len(F)} weight8={len(W)} total={tot} '
                    f'({"BEATS" if tot > RECORD else "below"} record 1154)')
                json.dump({'V': V, 'supports': F, 'cosets': c, 'dim_V': d,
                           'weight8': len(W), 'total': tot, 'n': N},
                          open(f'{OUT}/gc2_s{seed}_{tot}.json', 'w'))
    print(f'seed {seed}: tried {tried} codes, best total {best}')
    return best

if __name__ == '__main__':
    run(int(sys.argv[1]), float(sys.argv[2]))
