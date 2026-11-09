"""Hole search: given a configuration, is there a unit vector u with
<v_i, u> <= 1/2 for all i?

P = {u : <v_i,u> <= 1/2} is a convex polytope containing 0, so such a u exists
iff some point of P has norm >= 1 (scale down along the segment from 0).
Maximise ||u|| over P by the fixed-point iteration u <- argmax_{P} <u,.>,
which strictly increases the norm, from many random starts.
"""
import numpy as np
from scipy.optimize import linprog

def max_norm_over_P(Vn, half=0.5, starts=400, iters=40, seed=0, tol=1e-12):
    """Vn: (N,n) unit rows.  Returns (best_norm, best_u)."""
    rng = np.random.default_rng(seed)
    n = Vn.shape[1]
    b = np.full(Vn.shape[0], half)
    bounds = [(-2, 2)] * n
    best = 0.0; bu = None
    for s in range(starts):
        c = rng.standard_normal(n); c /= np.linalg.norm(c)
        for _ in range(iters):
            r = linprog(-c, A_ub=Vn, b_ub=b, bounds=bounds, method='highs')
            if not r.success: break
            u = r.x; nu = np.linalg.norm(u)
            if nu < 1e-9: break
            cn = u / nu
            if np.dot(cn, c) > 1 - 1e-14: c = cn; break
            c = cn
        else:
            pass
        if r.success:
            nu = np.linalg.norm(r.x)
            if nu > best: best, bu = nu, r.x.copy()
    return best, bu

def holes_after_removal(Vn, drop_idx, half=0.5, starts=120, seed=0):
    keep = np.delete(Vn, drop_idx, axis=0)
    return max_norm_over_P(keep, half, starts=starts, seed=seed)
