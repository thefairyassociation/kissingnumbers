"""Can Ganzhinov's dim-14 configuration be extended inside the global-code
framework?  Uses his own V (dim 5) and his 49 supports, then tries to add more,
and also re-grows families from scratch with the same V."""
import sys, json, random, time
sys.path.insert(0, 'lower_bound/lib')
import numpy as np
from gcode import Ctx, Elim, pair_rows, grow, solve_signs, build, check, code_words, pc

n = 14
PCT = np.array([bin(x).count('1') for x in range(1 << n)], dtype=np.int8)
SUPS = np.array([m for m in range(1 << n) if bin(m).count('1') == 8], dtype=np.int32)

d = json.load(open('lower_bound/lib/g14_struct.json'))
sups = [sum(1 << i for i in s) for s in d['supports']]
def span(g):
    B = []
    for x in g:
        for b in B: x = min(x, x ^ b)
        if x: B.append(x); B.sort(reverse=True)
    return B
def inspan(B, v):
    for b in B:
        if v ^ b < v: v ^= b
    return v == 0
def bits(m): return [b for b in range(n) if m >> b & 1]
lift = [[sum(1 << bits(S)[k] for k in range(8) if w >> k & 1) for w in C]
        for S, C in zip(sups, d['codes'])]
U = [span([w ^ L[0] for w in L]) for L in lift]
V = [x for x in range(1 << n) if all(inspan(U[i], x & sups[i]) for i in range(49))]
ctx = Ctx(V, n, PCT, SUPS)
print("Ganzhinov V: dim", ctx.d, " good supports:", len(ctx.good), " k =", ctx.k)

# rebuild the elimination state for his 49 supports, then try to extend
el = Elim(); F = []
for S in sups:
    rows = pair_rows(ctx, F, S)
    assert rows is not None, "his own support rejected"
    for r in rows: assert el.add(r, 1), "his own system inconsistent"
    F.append(S)
print("his 49 supports reproduce a consistent system; forced equations:", len(el.piv))

added = []
for S in ctx.good:
    if S in F: continue
    rows = pair_rows(ctx, F, S)
    if rows is None: continue
    tr = el.copy(); ok = all(tr.add(r, 1) for r in rows)
    if ok: added.append(S)
print("supports addable to his 49 (structurally + linearly):", len(added))

# how large can grow() get with this V, from scratch?
rng = random.Random(0); best = 0
t0 = time.time()
while time.time() - t0 < 60:
    F2, el2 = grow(ctx, rng)
    if len(F2) > best:
        best = len(F2)
        print(f"  grow from scratch: {best} supports -> weight8 {best*32}, total {best*32+364}", flush=True)
print("best m with his V:", best)
