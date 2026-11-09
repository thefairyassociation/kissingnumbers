"""Why does the dim-14, dim V = 6 family stop growing?

For each candidate support rejected after a family is grown, record whether it
failed structurally (meets a member in >= 7, or a t in {5,6} pair where V|_X is
everything) or because the forced F_2 system became inconsistent.  Also report
how many DISTINCT functionals the constrained pairs use: functionals are the
scarce resource, since three pairs sharing one give three equations summing to
0 = 1."""
import sys, random, collections
sys.path.insert(0, 'kissing/lib'); sys.path.insert(0, 'kissing/dim14')
import numpy as np
from gcode import Ctx, Elim, pair_rows, code_words, pc
from dimv6 import perp_of, supports_from_perp, grow, PCT, SUPS, n

rng = random.Random(int(sys.argv[1]) if len(sys.argv) > 1 else 1)
bestrec = None
for trial in range(int(sys.argv[2]) if len(sys.argv) > 2 else 300):
    V = code_words([rng.randrange(1, 1 << n) for _ in range(6)])
    if len(set(V)) != 64: continue
    P = perp_of(V); cands = supports_from_perp(P)
    if len(cands) < 40: continue
    ctx = Ctx(V, n, PCT, SUPS)
    good = sorted(set(ctx.good) & set(cands))
    if len(good) < 40: continue
    F, el = grow(ctx, good, rng)
    if bestrec is None or len(F) > bestrec[0]:
        bestrec = (len(F), V, ctx, good, F, el, P)
m, V, ctx, good, F, el, P = bestrec
print(f"best family: {len(F)} supports out of {len(good)} good candidates "
      f"(need 25 for 1964)")
wd = collections.Counter(pc(a) for a in P if a)
print(f"V^perp weight distribution: {dict(sorted(wd.items()))}")
reason = collections.Counter()
for S in good:
    if S in F: continue
    bad = None
    for T in F:
        t = PCT[S & T]
        if t >= 7: bad = 'meets a member in >= 7'; break
        if t in (5, 6) and ctx.cnt[S & T] < 1: bad = 'V|_X is all of F_2^X'; break
    if bad: reason[bad] += 1; continue
    rows = pair_rows(ctx, F, S, rng)
    tr = el.copy()
    reason['F_2 system inconsistent' if not all(tr.add(r, 1) for r in rows)
           else 'ADDABLE (greedy just missed it)'] += 1
print("why the other candidates cannot join:")
for k, v in reason.most_common(): print(f"   {k}: {v}")
lam = collections.Counter()
for i in range(len(F)):
    for j in range(i + 1, len(F)):
        X = F[i] & F[j]; t = PCT[X]
        if t in (5, 6): lam[int(ctx.lamx[X])] += 1
print(f"constrained pairs inside the family: {sum(lam.values())}, "
      f"using {len(lam)} distinct functionals")
print(f"   most reused functional appears on {max(lam.values()) if lam else 0} pairs "
      f"(3 on a triangle is already a contradiction)")
print(f"   unknowns available: {len(F)} supports x k={ctx.k} = {len(F)*ctx.k}")
