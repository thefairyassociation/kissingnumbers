"""dim 14, Fano-product 49-design: enumerate ALL global codes V of the right
shape and test each for a 50th support.

V must contain the all-ones word (that is exactly antipodality of the vector
set), so its 31 nonzero words come in complementary pairs; combined with
minimum weight 6 (Griesmer-optimal for [14,5]) every word other than 1^14 has
weight in [6,8].  Goodness for the design is the explicit condition
def(v_A) + def(v_B) >= 2 with def(x) = wt(x) - max_L |x & L| over Fano lines.
So V is a 5-dimensional subspace inside an explicit candidate set."""
import sys, json, random, time, collections
sys.path.insert(0, 'lower_bound/lib')
import numpy as np
from gcode import Ctx, Elim, pair_rows, build, check

n = 14; ONES = (1 << n) - 1
PCT = np.array([bin(x).count('1') for x in range(1 << n)], dtype=np.int8)
SUPS = np.array([m for m in range(1 << n) if bin(m).count('1') == 8], dtype=np.int32)
lines = [(1 << ((0 + i) % 7)) | (1 << ((1 + i) % 7)) | (1 << ((3 + i) % 7)) for i in range(7)]
F7 = (1 << 7) - 1
deff = [bin(x).count('1') - max(bin(x & L).count('1') for L in lines) for x in range(1 << 7)]
design = [((F7 ^ lines[i]) | ((F7 ^ lines[j]) << 7)) for i in range(7) for j in range(7)]
ing = lambda v: deff[v & F7] + deff[v >> 7] >= 2
C = [v for v in range(1, 1 << n) if 6 <= bin(v).count('1') <= 8 and ing(v)]
Cset = set(C) | {ONES}
print(f"candidate codewords: {len(C)} (plus the all-ones word)")

def build_V(rng, tries=4000):
    for _ in range(tries):
        words = [0, ONES]
        for _ in range(4):
            cand = [x for x in C if all((x ^ w) in Cset for w in words)]
            if not cand: break
            b = rng.choice(cand); words = words + [w ^ b for w in words]
        if len(words) == 32 and len(set(words)) == 32: return sorted(words)
    return None

rng = random.Random(int(sys.argv[1]) if len(sys.argv) > 1 else 0)
budget = float(sys.argv[2]) if len(sys.argv) > 2 else 600
t0 = time.time(); nV = 0; nCons = 0; bestadd = 0; seen = set()
while time.time() - t0 < budget:
    V = build_V(rng)
    if V is None: continue
    key = tuple(V)
    if key in seen: continue
    seen.add(key); nV += 1
    ctx = Ctx(V, n, PCT, SUPS)
    el = Elim(); F = []; ok = True
    for S in design:
        rows = pair_rows(ctx, F, S, None)
        if rows is None or not all(el.add(r, 1) for r in rows): ok = False; break
        F.append(S)
    if not ok: continue
    nCons += 1
    add = [S for S in ctx.good if S not in design
           and (lambda r: r is not None and all(el.copy().add(x, 1) for x in [])
                )(pair_rows(ctx, F, S, None))]
    add = []
    for S in ctx.good:
        if S in design: continue
        rows = pair_rows(ctx, F, S, None)
        if rows is None: continue
        tr = el.copy()
        if all(tr.add(r, 1) for r in rows): add.append(S)
    if len(add) > bestadd:
        bestadd = len(add)
        print(f"V #{nCons}: design consistent; addable 50th supports = {len(add)}", flush=True)
    for S50 in add[:5]:
        F2 = F + [S50]; el2 = el.copy()
        for r in pair_rows(ctx, F, S50, None): el2.add(r, 1)
        for _ in range(800):
            y = el2.sample(len(F2) * ctx.k, rng)
            c = [ctx.rep[sum(y[i * ctx.k + b] << b for b in range(ctx.k))] for i in range(len(F2))]
            W = build(F2, V, c, n)
            if check(W, n)[0]:
                tot = len(W) + 364
                print(f"*** {len(F2)} supports, {len(W)} weight-8, TOTAL {tot} ***", flush=True)
                json.dump({'V': V, 'supports': F2, 'cosets': c, 'weight8': len(W),
                           'total': tot, 'n': n},
                          open(f'lower_bound/dim14/configs/fano50_{tot}.json', 'w'))
                sys.exit(0)
print(f"distinct V built {nV}; consistent with the 49-design {nCons}; max addable {bestadd}")
