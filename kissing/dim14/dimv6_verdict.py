"""Is dim V = 6 in dimension 14 actually reachable?

The diagnostic says the family stops for a STRUCTURAL reason, not an F_2 one: a
pair at t in {5,6} needs some nonzero a in V^perp supported inside X = S & T, and
with dim V = 6 (so dim V^perp = 8) the low-weight words of V^perp simply do not
cover enough small sets.

The asymmetry that matters: a pair at intersection t is automatically fine
whenever dim V < t.  At dim V = 5, t=6 pairs are free and only t=5 needs a
witness -- which is why Ganzhinov's 49 x 32 works.  At dim V = 6, t=6 pairs need
a witness too.  This gives dim V = 6 its best possible shot by selecting codes
whose dual has the most low-weight words, and reports the cap.
"""
import sys, random, collections
sys.path.insert(0, 'kissing/lib'); sys.path.insert(0, 'kissing/dim14')
import numpy as np
from gcode import Ctx, Elim, pair_rows, code_words, pc
from dimv6 import perp_of, supports_from_perp, grow, PCT, SUPS, n

def lowweight(P, cap=6):
    return sum(1 for a in P if a and pc(a) <= cap)

rng = random.Random(int(sys.argv[1]) if len(sys.argv) > 1 else 0)
budget = float(sys.argv[2]) if len(sys.argv) > 2 else 300
import time
t0 = time.time()
rows = []
best = (0,)
while time.time() - t0 < budget:
    V = code_words([rng.randrange(1, 1 << n) for _ in range(6)])
    if len(set(V)) != 64: continue
    P = perp_of(V)
    lw = lowweight(P)
    cands = supports_from_perp(P)
    if len(cands) < 30: continue
    ctx = Ctx(V, n, PCT, SUPS)
    good = sorted(set(ctx.good) & set(cands))
    if len(good) < 30: continue
    F, el = grow(ctx, good, rng)
    rows.append((lw, len(good), len(F)))
    if len(F) > best[0]: best = (len(F), lw, len(good))
rows.sort()
print(f"{len(rows)} codes sampled")
print(f"best family: {best[0]} supports  (dual low-weight words {best[1]}, "
      f"{best[2]} good candidates)  -> {best[0]*64+364} total, need 25 supports for 1964")
# does having a richer dual help?
if rows:
    lo = [r for r in rows if r[0] <= np.median([x[0] for x in rows])]
    hi = [r for r in rows if r[0] >  np.median([x[0] for x in rows])]
    f = lambda g: (np.mean([x[0] for x in g]), np.mean([x[2] for x in g]))
    print(f"  dual-poor half: mean low-weight words {f(lo)[0]:.0f} -> mean family {f(lo)[1]:.1f}")
    print(f"  dual-rich half: mean low-weight words {f(hi)[0]:.0f} -> mean family {f(hi)[1]:.1f}")
    c = np.corrcoef([x[0] for x in rows], [x[2] for x in rows])[0,1]
    print(f"  correlation(low-weight words in V^perp, family size) = {c:.3f}")
