"""Why can no 50th support join Ganzhinov's 49?  Classify every 8-subset of
[14] by the first condition it fails:
  (a) goodness  - some nonzero codeword of V meets S in <= 1 coordinate
  (b) t >= 7 with an existing support
  (c) cnt[X] = 0 for some t in {5,6} pair (V|X is everything, cosets can't be disjoint)
  (d) the forced F_2 system becomes inconsistent
"""
import sys, json, collections
sys.path.insert(0, 'lower_bound/lib')
import numpy as np
from gcode import Ctx, Elim, code_words, pc

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
bits = lambda m: [b for b in range(n) if m >> b & 1]
lift = [[sum(1 << bits(S)[k] for k in range(8) if w >> k & 1) for w in C]
        for S, C in zip(sups, d['codes'])]
U = [span([w ^ L[0] for w in L]) for L in lift]
V = [x for x in range(1 << n) if all(inspan(U[i], x & sups[i]) for i in range(49))]
ctx = Ctx(V, n, PCT, SUPS)
good = set(ctx.good)
el = Elim(); F = []
for S in sups:
    for i, T in enumerate(F):
        X = S & T; t = PCT[X]
        if t <= 4: continue
        if ctx.cnt[X] == 1:
            lam = int(ctx.lamx[X]); row = 0; j = len(F)
            for b in range(ctx.k):
                if lam >> b & 1: row ^= (1 << (i * ctx.k + b)) ^ (1 << (j * ctx.k + b))
            assert el.add(row, 1)
    F.append(S)

reason = collections.Counter()
for S in SUPS.tolist():
    if S in sups: reason['already in the family'] += 1; continue
    if S not in good: reason['(a) not good for V'] += 1; continue
    bad = None
    for T in sups:
        t = PCT[S & T]
        if t >= 7: bad = '(b) meets a support in >= 7'; break
        if t in (5, 6) and ctx.cnt[S & T] < 1: bad = '(c) V|X is everything'; break
    if bad: reason[bad] += 1; continue
    rows = []
    for i, T in enumerate(sups):
        X = S & T; t = PCT[X]
        if t <= 4: continue
        if ctx.cnt[X] == 1:
            lam = int(ctx.lamx[X]); row = 0; j = 49
            for b in range(ctx.k):
                if lam >> b & 1: row ^= (1 << (i * ctx.k + b)) ^ (1 << (j * ctx.k + b))
            rows.append(row)
    tr = el.copy()
    reason['(d) F_2 system inconsistent' if not all(tr.add(r, 1) for r in rows)
           else 'ADDABLE'] += 1
print(f"candidates: {len(SUPS)}   good for V: {len(good)}")
for k, v in reason.most_common(): print(f"  {k}: {v}")
# how do the (7,1)-split supports fare?
A = set(range(7))
sp = [S for S in SUPS.tolist() if len(set(bits(S)) & A) in (7, 1)]
print(f"\n(7,1)-split supports: {len(sp)}, of which good for V: {sum(1 for S in sp if S in good)}")
