"""Delsarte linear-programming upper bound for spherical codes.

M(n, s) <= min f(1)/f_0 over f = sum_k f_k G_k with f_k >= 0 for k >= 1, f_0 > 0,
and f(t) <= 0 for all t in [-1, s], where G_k are the Gegenbauer polynomials for
dimension n normalised by G_k(1) = 1.  Discretising [-1, s] turns this into an LP.

Used here to bound the layered dimension-13 framework
    N(13) <= 2 + M(12, 1/2) + 2 * M(12, 1/3)
for configurations that contain an antipodal pair of poles.
"""
import numpy as np
from scipy.optimize import linprog


def gegenbauer(n, kmax, t):
    """G_k^{(n)}(t) for k = 0..kmax, normalised so that G_k(1) = 1."""
    G = [np.ones_like(t), t.copy()]
    for k in range(1, kmax):
        # (k + n - 2) G_{k+1} = (2k + n - 2) t G_k - k G_{k-1}
        G.append(((2 * k + n - 2) * t * G[k] - k * G[k - 1]) / (k + n - 2))
    return np.array(G[: kmax + 1])


def lp_bound(n, s, deg=24, grid=4000):
    """Returns (bound, coefficients) or (None, message)."""
    t = np.linspace(-1, s, grid)
    G = gegenbauer(n, deg, t)
    # variables f_1..f_deg >= 0, with f_0 = 1 fixed; minimise f(1) = 1 + sum f_k
    # subject to 1 + sum_k f_k G_k(t) <= 0 on the grid
    r = linprog(np.ones(deg), A_ub=G[1:].T, b_ub=-np.ones(grid),
                bounds=[(0, None)] * deg, method='highs')
    if not r.success:
        return None, r.message
    return 1 + r.fun, r.x


if __name__ == '__main__':
    cases = [(8, 0.5, "K(8) = 240 exactly"), (12, 0.5, "K(12) >= 841"),
             (13, 0.5, "K(13) >= 1154"), (14, 0.5, "K(14) >= 1932"),
             (11, 1 / 3, "K(11,1/3)"), (12, 1 / 3, "K(12,1/3) >= 168"),
             (13, 1 / 3, "K(13,1/3)")]
    best = {}
    for (n, s, name) in cases:
        vals = []
        for deg in (12, 16, 20, 24, 30):
            v, _ = lp_bound(n, s, deg=deg)
            if v is not None:
                vals.append(v)
        b = min(vals)
        best[(n, s)] = b
        print(f"n={n:2d}  s={s:.5f}  LP bound {b:11.3f}   ({name})")
    print()
    print("layered dim-13 ceiling  2 + M(12,1/2) + 2*M(12,1/3) <= "
          f"{2 + best[(12,0.5)] + 2*best[(12,1/3)]:.1f}")
