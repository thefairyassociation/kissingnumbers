"""dim 14, Fano-product 49-design: enumerate the global codes V that make every
support good, then look for a 50th support.

Supports are S = P u Q with P,Q complements of Fano lines, so S^c = L u L'.
Goodness (|v & S| >= 2 for every nonzero codeword) is exactly

      def(v_A) + def(v_B) >= 2 ,   def(x) = wt(x) - max_{Fano line L} |x & L|,

where v = (v_A, v_B) splits over the two 7-sets.  So V\\{0} must lie in the
explicit set G of vectors with that property; we build 5-dimensional subspaces
inside G and test each for a 50th support."""
import sys, json, random, time, collections
sys.path.insert(0, 'lower_bound/lib')
import numpy as np
from gcode import Ctx, Elim, pair_rows, build, check

n = 14
PCT = np.array([bin(x).count('1') for x in range(1 << n)], dtype=np.int8)
SUPS = np.array([m for m in range(1 << n) if bin(m).count('1') == 8], dtype=np.int32)
lines = [(1 << ((0 + i) % 7)) | (1 << ((1 + i) % 7)) | (1 << ((3 + i) % 7)) for i in range(7)]
FULL7 = (1 << 7) - 1
deff = [bin(x).count('1') - max(bin(x & L).count('1') for L in lines) for x in range(1 << 7)]
design = [((FULL7 ^ lines[i]) | ((FULL7 ^ lines[j]) << 7)) for i in range(7) for j in range(7)]
G = [v for v in range(1, 1 << n) if deff[v & FULL7] + deff[v >> 7] >= 2]
print(f"allowed codeword set G: {len(G)} of {(1<<n)-1}")
Gset = set(G)

def random_subspace(d, rng, tries=300):
    for _ in range(tries):
        basis = []; words = [0]
        for _ in range(d):
            cand = [x for x in G if all((x ^ w) in Gset for w in words)]
            if not cand: break
            b = rng.choice(cand); basis.append(b); words = words + [w ^ b for w in words]
        if len(basis) == d and len(set(words)) == 1 << d: return words
    return None

rng = random.Random(int(sys.argv[1]) if len(sys.argv) > 1 else 0)
budget = float(sys.argv[2]) if len(sys.argv) > 2 else 600
t0 = time.time(); nV = 0; nCons = 0; bestadd = 0; wd = collections.Counter()
while time.time() - t0 < budget:
    V = random_subspace(5, rng)
    if V is None: continue
    nV += 1
    wd[tuple(sorted(collections.Counter(bin(v).count('1') for v in V).items()))] += 1
    ctx = Ctx(V, n, PCT, SUPS)
    el = Elim(); F = []; ok = True
    for S in design:
        rows = pair_rows(ctx, F, S, rng)
        if rows is None or not all(el.add(r, 1) for r in rows): ok = False; break
        F.append(S)
    if not ok: continue
    nCons += 1
    add = []
    for S in ctx.good:
        if S in design: continue
        rows = pair_rows(ctx, F, S, rng)
        if rows is None: continue
        tr = el.copy()
        if all(tr.add(r, 1) for r in rows): add.append(S)
    if len(add) > bestadd:
        bestadd = len(add)
        print(f"code #{nCons}: 49-design consistent; addable 50th supports = {len(add)}", flush=True)
    if add:
        F2 = F + [add[0]]
        el2 = el.copy()
        for r in pair_rows(ctx, F, add[0], rng): el2.add(r, 1)
        for _ in range(600):
            y = el2.sample(len(F2) * ctx.k, rng)
            c = [ctx.rep[sum(y[i * ctx.k + b] << b for b in range(ctx.k))] for i in range(len(F2))]
            W = build(F2, V, c, n)
            if check(W, n)[0]:
                tot = len(W) + 364
                print(f"*** {len(F2)} supports, {len(W)} weight-8 vectors, TOTAL {tot} ***", flush=True)
                json.dump({'V': V, 'supports': F2, 'cosets': c, 'weight8': len(W),
                           'total': tot, 'n': n},
                          open(f'lower_bound/dim14/configs/fano50_{tot}.json', 'w'))
                sys.exit(0)
print(f"subspaces built {nV}, consistent with the 49-design {nCons}, max addable {bestadd}")
print("weight distributions seen:", dict(list(wd.most_common(4))))
