"""Given a 1/2-code A in R^12 (the height-0 layer), greedily build the largest
1/3-code B with cross condition cos(A,B) <= 1/sqrt(3).
Total dim-13 count = 2 + |A| + 2|B|;  beat 1154 means |A| + 2|B| >= 1153."""
import os, tempfile
# Intermediate arrays are shared between layers.py, completeB.py and greedyB.py.
SCRATCH = os.environ.get("KISS_SCRATCH", os.path.join(tempfile.gettempdir(), "kissing-scratch"))
os.makedirs(SCRATCH, exist_ok=True)
import sys, math, numpy as np
from scipy.optimize import linprog

S3 = 1 / math.sqrt(3)

def farthest_point(M, rhs, n=12, starts=60, seed=0):
    """max ||b|| over {b : M b <= rhs}; returns (norm, b)"""
    rng = np.random.default_rng(seed); best = 0.0; bb = None
    bounds = [(-3, 3)] * n
    for _ in range(starts):
        c = rng.standard_normal(n); c /= np.linalg.norm(c)
        r = None
        for _ in range(50):
            r = linprog(-c, A_ub=M, b_ub=rhs, bounds=bounds, method='highs')
            if not r.success: break
            u = r.x; nu = np.linalg.norm(u)
            if nu < 1e-9: break
            cn = u / nu
            if cn @ c > 1 - 1e-14: c = cn; break
            c = cn
        if r is not None and r.success:
            nu = np.linalg.norm(r.x)
            if nu > best: best, bb = nu, r.x.copy()
    return best, bb

def grow_B(A, seed=0, starts=40, maxB=400, B0=None):
    B = [] if B0 is None else [b for b in B0]
    while len(B) < maxB:
        M = np.vstack([A] + ([np.array(B)] if B else []))
        rhs = np.concatenate([np.full(len(A), S3), np.full(len(B), 1 / 3)])
        nb, b = farthest_point(M, rhs, A.shape[1], starts=starts, seed=seed + len(B))
        if nb < 1 - 1e-9: break
        B.append(b / np.linalg.norm(b))
    return np.array(B) if B else np.zeros((0, A.shape[1]))

if __name__ == '__main__':
    import json
    tag = sys.argv[1]
    if tag == 'tetrads':
        A = np.load(os.path.join(SCRATCH, 'A12.npy'))
    elif tag == 'tetrads840':
        A0 = np.load(os.path.join(SCRATCH, 'A12.npy'))
        A = np.vstack([A0, np.eye(12), -np.eye(12)])
    elif tag == 'clebsch':
        d = json.load(open('lower_bound/dim12/configs/clebsch840.json'))
        import sympy
        A = np.array([[float(sympy.sympify(c)) for c in v] for v in d['vectors']])
        A = A / np.linalg.norm(A, axis=1, keepdims=True)
    G = A @ A.T; np.fill_diagonal(G, -9)
    print(f"A = {tag}: {len(A)} points, max cos {G.max():.9f}")
    for seed in range(int(sys.argv[2]) if len(sys.argv) > 2 else 3):
        B = grow_B(A, seed=seed * 977, starts=int(sys.argv[3]) if len(sys.argv) > 3 else 30)
        if len(B):
            GB = B @ B.T; np.fill_diagonal(GB, -9)
            cross = (A @ B.T).max()
        else:
            GB = np.array([[0]]); cross = 0
        tot = 2 + len(A) + 2 * len(B)
        print(f"  seed {seed}: |B| = {len(B)}  maxcos(B) {GB.max():.6f}  cross {cross:.6f} "
              f"-> dim-13 total {tot}  {'BEATS 1154' if tot > 1154 else ''}", flush=True)
        np.save(os.path.join(SCRATCH, f'B_{tag}_{seed}.npy'), B)
