"""dim 14: fix the Fano-product 49-support design and search over global codes V
   (dim 5, min weight 6) for one that admits a FIFTIETH support.

   49 supports x 32 signs = 1568 weight-8 vectors; a 50th gives 1600 and, with the
   364 D-roots, 1964 > 1932.  Ganzhinov's own V admits none."""
import sys, json, random, itertools, time
sys.path.insert(0, 'lower_bound/lib')
import numpy as np
from gcode import Ctx, Elim, pair_rows, build, check, code_words

n = 14
PCT = np.array([bin(x).count('1') for x in range(1 << n)], dtype=np.int8)
SUPS = np.array([m for m in range(1 << n) if bin(m).count('1') == 8], dtype=np.int32)

# the Fano plane on {0..6}: lines {0,1,3}+i mod 7; the design blocks are their complements
lines = [frozenset(((0 + i) % 7, (1 + i) % 7, (3 + i) % 7)) for i in range(7)]
blocksA = [frozenset(range(7)) - L for L in lines]
blocksB = [frozenset(x + 7 for x in (frozenset(range(7)) - L)) for L in lines]
design = [sum(1 << i for i in (P | Q)) for P in blocksA for Q in blocksB]
assert len(design) == 49 and all(bin(m).count('1') == 8 for m in design)
inter = collections = {}
import collections as _c
cnt = _c.Counter(bin(design[i] & design[j]).count('1')
                 for i in range(49) for j in range(i + 1, 49))
print("Fano-product design: 49 supports, pair intersections", dict(cnt))

def random_code_minwt(d, wmin, rng, tries=200):
    for _ in range(tries):
        basis = []; words = [0]; ok = True
        for _ in range(d):
            cand = [x for x in range(1, 1 << n)
                    if all(bin(x ^ w).count('1') >= wmin for w in words)]
            if not cand: ok = False; break
            b = rng.choice(cand); basis.append(b); words = words + [w ^ b for w in words]
        if ok and len(set(words)) == 1 << d: return words
    return None

def all_good(V, design):
    return all(all(bin(v & S).count('1') >= 2 for S in design) for v in V if v)

rng = random.Random(int(sys.argv[1]) if len(sys.argv) > 1 else 0)
budget = float(sys.argv[2]) if len(sys.argv) > 2 else 600
t0 = time.time(); nV = 0; nGood = 0; best = 0
while time.time() - t0 < budget:
    V = random_code_minwt(5, 6, rng)
    if V is None: continue
    nV += 1
    if not all_good(V, design): continue
    nGood += 1
    ctx = Ctx(V, n, PCT, SUPS)
    el = Elim(); F = []
    ok = True
    for S in design:
        rows = pair_rows(ctx, F, S, rng)
        if rows is None or not all(el.add(r, 1) for r in rows): ok = False; break
        F.append(S)
    if not ok: continue
    add = []
    for S in ctx.good:
        if S in design: continue
        rows = pair_rows(ctx, F, S, rng)
        if rows is None: continue
        tr = el.copy()
        if all(tr.add(r, 1) for r in rows): add.append(S)
    if len(add) > best:
        best = len(add)
        print(f"V #{nGood}: 49-design consistent, addable 50th supports: {len(add)}", flush=True)
    if add:
        F2 = F + [add[0]]
        rows = pair_rows(ctx, F, add[0], rng); el2 = el.copy()
        for r in rows: el2.add(r, 1)
        for _ in range(400):
            y = el2.sample(len(F2) * ctx.k, rng)
            c = [ctx.rep[sum(y[i * ctx.k + b] << b for b in range(ctx.k))] for i in range(len(F2))]
            W = build(F2, V, c, n)
            if check(W, n)[0]:
                tot = len(W) + 364
                print(f"*** {len(F2)} supports, {len(W)} weight-8, TOTAL {tot} ***", flush=True)
                json.dump({'V': V, 'supports': F2, 'cosets': c, 'weight8': len(W),
                           'total': tot, 'n': n}, open(f'lower_bound/dim14/configs/fano50_{tot}.json', 'w'))
                break
print(f"codes tried {nV}, good for the 49-design {nGood}, best #addable {best}")
