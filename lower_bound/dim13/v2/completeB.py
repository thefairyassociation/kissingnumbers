"""Max 1/3-code in R^12 containing the 144 ZE99 diamonds and compatible
(cos <= 1/sqrt3) with the 816 tetrads.  ZE99 completes it with the 24 axials
+-e_i, giving 168 and dim-13 total 2 + 816 + 2*168 = 1154.
Any completion of size >= 169 gives 1156 and beats the record."""
import os, tempfile
# Intermediate arrays are shared between layers.py, completeB.py and greedyB.py.
SCRATCH = os.environ.get("KISS_SCRATCH", os.path.join(tempfile.gettempdir(), "kissing-scratch"))
os.makedirs(SCRATCH, exist_ok=True)
import sys, math, numpy as np
from scipy.optimize import linprog
S3 = 1 / math.sqrt(3)
A = np.load(os.path.join(SCRATCH, 'A12.npy'))     # 816 tetrads
B0 = np.load(os.path.join(SCRATCH, 'B12.npy'))    # 168 = 24 axials + 144 diamonds
dia = B0[np.abs(B0).max(1) < 0.9]                    # the 144 diamonds
print("tetrads", len(A), " diamonds", len(dia))

def feasible_point(M, rhs, c, n=12):
    r = linprog(-c, A_ub=M, b_ub=rhs, bounds=[(-3, 3)] * n, method='highs')
    return r.x if r.success else None

def complete(seed, greedy_maxnorm=False, rounds=40):
    rng = np.random.default_rng(seed)
    B = [b for b in dia]
    stall = 0
    while stall < rounds:
        M = np.vstack([A, np.array(B)])
        rhs = np.concatenate([np.full(len(A), S3), np.full(len(B), 1 / 3)])
        c = rng.standard_normal(12); c /= np.linalg.norm(c)
        u = None
        for _ in range(50):                          # push towards larger norm
            x = feasible_point(M, rhs, c)
            if x is None: break
            nx = np.linalg.norm(x)
            if nx < 1e-9: break
            cn = x / nx
            if cn @ c > 1 - 1e-13: u = x; break
            c = cn
            u = x
        if u is not None and np.linalg.norm(u) >= 1 - 1e-9:
            B.append(u / np.linalg.norm(u)); stall = 0
        else:
            stall += 1
    return np.array(B)

best = 0
for s in range(int(sys.argv[1]) if len(sys.argv) > 1 else 6):
    B = complete(s * 131 + 7)
    Gb = B @ B.T; np.fill_diagonal(Gb, -9)
    cross = (A @ B.T).max()
    tot = 2 + len(A) + 2 * len(B)
    ok = Gb.max() <= 1 / 3 + 1e-9 and cross <= S3 + 1e-9
    print(f"seed {s}: |B| = {len(B)}  maxcos(B) {Gb.max():.9f}  cross {cross:.9f} valid={ok}"
          f"  -> dim-13 total {tot} {'*** BEATS 1154 ***' if tot > 1154 and ok else ''}", flush=True)
    if len(B) > best:
        best = len(B)
        np.save(os.path.join(SCRATCH, 'Bbest.npy'), B)
print("best |B| =", best, " (ZE99 uses 168; >=169 wins)")
