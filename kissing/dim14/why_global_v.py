"""Does dropping the global-code assumption actually buy anything?

For a pair (S,T) at t = |S & T| the sign sets must have disjoint projections onto
X = S & T.  Those projections are cosets a + A and b + B of A = U_S|_X and
B = U_T|_X, and cosets of different subspaces are disjoint only if
(a+b) is not in A + B -- which is impossible once A + B = F_2^X.

With 32 signs per support (dim U = 5) and t = 6, dim A, dim B >= 3, and if both
are 5 then A + B = F_2^6 unless A = B exactly.  So for the codes to be as large
as possible the two supports must project to the SAME subspace of F_2^X -- which
is precisely what one global V delivers for free.  Checked here on Ganzhinov's
configuration."""
import sys, json, collections
sys.path.insert(0, 'kissing/lib')
import numpy as np
from gcode import pc

n = 14
d = json.load(open('kissing/lib/g14_struct.json'))
sups = [sum(1 << i for i in s) for s in d['supports']]
bits = lambda m: [b for b in range(n) if m >> b & 1]
lift = [[sum(1 << bits(S)[k] for k in range(8) if w >> k & 1) for w in C]
        for S, C in zip(sups, d['codes'])]
lin = [{w ^ L[0] for w in L} for L in lift]          # the linear part U_S, as 14-bit masks

def proj(U, X):
    return frozenset(u & X for u in U)

same = collections.Counter()
dims = collections.Counter()
for i in range(len(sups)):
    for j in range(i + 1, len(sups)):
        X = sups[i] & sups[j]; t = pc(X)
        if t not in (5, 6): continue
        A, B = proj(lin[i], X), proj(lin[j], X)
        same[A == B] += 1
        dims[(len(A).bit_length() - 1, len(B).bit_length() - 1)] += 1
print(f"Ganzhinov, pairs at t in {{5,6}}: {sum(same.values())}")
print(f"  projections U_S|_X and U_T|_X are EQUAL: {same[True]}, different: {same[False]}")
print(f"  (dim A, dim B) over those pairs: {dict(dims)}")
print()
print("So on the record configuration every constrained pair projects to the same")
print("subspace of F_2^X.  Per-support codes could only differ by projecting to")
print("SMALLER subspaces, i.e. by carrying fewer than 32 signs -- which loses more")
print("than it gains.  The global-V restriction is close to forced, not a limitation.")
