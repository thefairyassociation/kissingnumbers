"""dim 14, dim V = 6: 25 supports x 64 signs = 1600 weight-8 vectors, plus the
364 D-roots, is 1964 > 1932.

Sharper candidate generation than scanning all 3003 8-subsets.  For dim V = 6 the
restricted dual  D_S = {a in V^perp : supp(a) <= S}|_S  must have dimension
exactly 2 (so V|_S has dimension 6) and its three nonzero words must cover all 8
coordinates of S (no zero column <=> V|_S has minimum weight >= 2).  Each covered
coordinate lies in exactly two of the three words, so

    |w1| + |w2| + |w3| = 16   and   S = supp(w1) u supp(w2) u supp(w3).

So the usable supports are exactly the support-unions of the 2-dimensional
subspaces of V^perp whose union has size 8 — 10795 subspaces to walk instead of
3003 blind subsets, and it hands back the class partition of S for free.
"""
import sys, random, itertools, collections
sys.path.insert(0, 'kissing/lib')
import numpy as np
from gcode import Ctx, Elim, pair_rows, build, check, code_words, pc

n = 14; DROOTS = 364; RECORD = 1932
PCT = np.array([bin(x).count('1') for x in range(1 << n)], dtype=np.int8)
SUPS = np.array([m for m in range(1 << n) if bin(m).count('1') == 8], dtype=np.int32)

def perp_of(V):
    return [a for a in range(1 << n) if all(pc(a & v) % 2 == 0 for v in V)]

def supports_from_perp(P):
    """support-unions of 2-dim subspaces of V^perp that have size 8"""
    nz = [a for a in P if a]
    out = {}
    for i in range(len(nz)):
        a = nz[i]
        for j in range(i + 1, len(nz)):
            b = nz[j]; c = a ^ b
            if c < b: continue                 # each subspace once
            S = a | b
            if PCT[S] != 8: continue
            if pc(a) + pc(b) + pc(c) != 16: continue   # every column nonzero
            out.setdefault(S, (a, b))
    return out

def grow(ctx, cands, rng, order=None):
    """greedy family growth keeping the forced F_2 system consistent, taking the
    candidate that adds the FEWEST new forced equations at each step"""
    F = []; el = Elim()
    pool = list(cands)
    rng.shuffle(pool)
    while True:
        best = None
        for S in pool:
            rows = pair_rows(ctx, F, S, rng)
            if rows is None: continue
            tr = el.copy()
            if not all(tr.add(r, 1) for r in rows): continue
            if best is None or len(rows) < best[0]:
                best = (len(rows), S, tr)
                if len(rows) == 0: break
        if best is None: break
        _, S, tr = best
        el = tr; F.append(S); pool.remove(S)
    return F, el

def run(seed, budget, dimV=6):
    import time
    rng = random.Random(seed); t0 = time.time(); best = 0; tried = 0
    while time.time() - t0 < budget:
        tried += 1
        V = code_words([rng.randrange(1, 1 << n) for _ in range(dimV)])
        if len(set(V)) != 1 << dimV: continue
        P = perp_of(V)
        cands = supports_from_perp(P)
        if len(cands) < 20: continue
        ctx = Ctx(V, n, PCT, SUPS)
        good = set(ctx.good) & set(cands)
        if len(good) < 20: continue
        F, el = grow(ctx, good, rng)
        if len(F) * (1 << dimV) + DROOTS <= best: continue
        c = None
        for _ in range(300):
            y = el.sample(len(F) * ctx.k, rng)
            cand = [ctx.rep[sum(y[i * ctx.k + b] << b for b in range(ctx.k))]
                    for i in range(len(F))]
            W0 = build(F, V, cand, n)
            if check(W0, n)[0]: c = cand; break
        if c is None: continue
        W = build(F, V, c, n); tot = len(W) + DROOTS
        if tot > best:
            best = tot
            print(f"seed{seed}: |cands|={len(cands)} good={len(good)} supports={len(F)} "
                  f"weight8={len(W)} TOTAL={tot} {'*** BEATS 1932 ***' if tot > RECORD else ''}",
                  flush=True)
            if tot > RECORD:
                import json
                json.dump({'V': V, 'supports': F, 'cosets': c, 'dim_V': dimV,
                           'weight8': len(W), 'total': tot, 'n': n},
                          open(f'kissing/dim14/configs/dimv6_{tot}.json', 'w'))
    print(f"seed {seed}: {tried} codes, best total {best}", flush=True)

if __name__ == '__main__':
    run(int(sys.argv[1]), float(sys.argv[2]))
