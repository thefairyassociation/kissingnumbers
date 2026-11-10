#!/usr/bin/env python3
"""Bachoc-Vallentin three-point semidefinite bound for the kissing number.

NUMERICAL, NOT A PROOF.  This solves the semidefinite program in floating
point and prints the optimum.  The true optimum is a rigorous upper bound,
but a number printed by an interior-point solver is not.  make_certificate.py
turns a solution into exact rational data, and verify_certificate.py checks
that data in exact arithmetic; only what that verifier accepts is a proof.

The program (Bachoc & Vallentin, J. Amer. Math. Soc. 21 (2008), Theorem 4.2).
With P_k^n the Gegenbauer polynomials of R^n, P_k^n(1) = 1, put

    Q_k(u,v,t) = ((1-u^2)(1-v^2))^(k/2) P_k^(n-1)( (t-uv)/sqrt((1-u^2)(1-v^2)) )
    Y_k(u,v,t)_ij = u^i v^j Q_k(u,v,t),   0 <= i,j <= d-k
    S_k = average of Y_k over the six permutations of (u,v,t).

Then every spherical code with inner products in [-1, s] has size at most

    min  1 + sum_{k=1}^d a_k + b11 + <F_0, S_0(1,1,1)>
    s.t. a_k >= 0,  [[b11,b12],[b12,b22]] psd,  F_k psd  (k = 0..d)
    (i)  sum a_k P_k^n(u) + 2 b12 + b22 + 3 sum_k <F_k, S_k(u,u,1)> <= -1   on [-1, s]
    (ii) b22 + sum_k <F_k, S_k(u,v,t)> <= 0                                  on D

    D = { -1 <= u,v,t <= s,  1 + 2uvt - u^2 - v^2 - t^2 >= 0 }.

Setting F = 0 and b = 0 gives back the Delsarte LP, so the optimum is never
worse than the LP bound.  The two polynomial inequalities are imposed as
sum-of-squares identities with the multipliers (u+1)(s-u), ... and the Gram
determinant.

usage:  python3 upper_bound/sdp/three_point.py --n 4 --degree 7 [--s 0.5] [--solver CLARABEL]
"""
from __future__ import annotations

import argparse
import itertools
import json
import sys
import time

import numpy as np
import scipy.sparse as sp


# ----- small dense polynomial algebra in (u, v, t): arrays indexed [i, j, l] -----

def pmul(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    out = np.zeros(tuple(x + y - 1 for x, y in zip(a.shape, b.shape)))
    for idx in np.argwhere(a != 0):
        i, j, l = idx
        out[i:i + b.shape[0], j:j + b.shape[1], l:l + b.shape[2]] += a[i, j, l] * b
    return out


def ppow(a: np.ndarray, m: int) -> np.ndarray:
    out = np.ones((1, 1, 1))
    for _ in range(m):
        out = pmul(out, a)
    return out


def padd(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    shape = tuple(max(x, y) for x, y in zip(a.shape, b.shape))
    out = np.zeros(shape)
    out[:a.shape[0], :a.shape[1], :a.shape[2]] += a
    out[:b.shape[0], :b.shape[1], :b.shape[2]] += b
    return out


def mono(i: int, j: int, l: int) -> np.ndarray:
    out = np.zeros((i + 1, j + 1, l + 1))
    out[i, j, l] = 1.0
    return out


def gegenbauer_monomial(n: int, d: int) -> list[np.ndarray]:
    """Monomial coefficients (lowest first) of P_0..P_d for R^n, P_k(1) = 1."""
    P = [np.array([1.0]), np.array([0.0, 1.0])]
    for k in range(2, d + 1):
        nxt = np.zeros(k + 1)
        nxt[1:] += (2 * k + n - 4) * P[k - 1]
        nxt[:k - 1] -= (k - 1) * P[k - 2]
        P.append(nxt / (k + n - 3))
    return P[:d + 1]


def Q_poly(n: int, k: int) -> np.ndarray:
    """Q_k^{n-1}(u,v,t) as a polynomial."""
    c = gegenbauer_monomial(n - 1, k)[k] if k >= 1 else np.array([1.0])
    one_u = np.zeros((3, 1, 1)); one_u[0, 0, 0] = 1; one_u[2, 0, 0] = -1     # 1 - u^2
    one_v = np.zeros((1, 3, 1)); one_v[0, 0, 0] = 1; one_v[0, 2, 0] = -1     # 1 - v^2
    w = pmul(one_u, one_v)
    tuv = np.zeros((2, 2, 2)); tuv[0, 0, 1] = 1; tuv[1, 1, 0] = -1           # t - uv
    out = np.zeros((1, 1, 1))
    for m in range(k % 2, k + 1, 2):
        if c[m] != 0:
            out = padd(out, c[m] * pmul(ppow(tuv, m), ppow(w, (k - m) // 2)))
    return out


def symmetrise(a: np.ndarray) -> np.ndarray:
    M = max(a.shape)
    full = np.zeros((M, M, M))
    full[:a.shape[0], :a.shape[1], :a.shape[2]] = a
    return sum(np.transpose(full, perm) for perm in itertools.permutations(range(3))) / 6.0


def exps_upto(deg: int) -> list[tuple[int, int, int]]:
    return [(i, j, l) for i in range(deg + 1) for j in range(deg + 1 - i) for l in range(deg + 1 - i - j)]


def build(n: int, d: int, s: float):
    """Returns the data of the SDP as sparse linear maps on vec(F_k) etc."""
    D3 = 2 * d                                   # total degree of the 3-variable identity
    mons3 = exps_upto(D3)
    index3 = {e: r for r, e in enumerate(mons3)}
    S = []                                       # S[k][i][j]: symmetric 3-D coefficient arrays
    for k in range(d + 1):
        Qk = Q_poly(n, k)
        m = d - k + 1
        S.append([[symmetrise(pmul(mono(i, j, 0), Qk)) for j in range(m)] for i in range(m)])

    def coeff_rows(arr):
        nz = np.argwhere(np.abs(arr) > 0)
        return [(index3[tuple(e)], arr[tuple(e)]) for e in nz]

    # (ii): coefficient vector of sum_k <F_k, S_k> as A3[k] @ vec(F_k)
    A3 = []
    for k in range(d + 1):
        m = d - k + 1
        rows, cols, vals = [], [], []
        for i in range(m):
            for j in range(m):
                for r, v in coeff_rows(S[k][i][j]):
                    rows.append(r); cols.append(i * m + j); vals.append(v)
        A3.append(sp.csr_matrix((vals, (rows, cols)), shape=(len(mons3), m * m)))

    # (i): univariate coefficients (degree <= 2d) of <F_k, S_k(u,u,1)>
    A1 = []
    for k in range(d + 1):
        m = d - k + 1
        mat = np.zeros((2 * d + 1, m * m))
        for i in range(m):
            for j in range(m):
                arr = S[k][i][j]
                for e in np.argwhere(np.abs(arr) > 0):
                    deg = e[0] + e[1]            # v -> u, t -> 1
                    mat[deg, i * m + j] += arr[tuple(e)]
        A1.append(sp.csr_matrix(mat))
    Pn = gegenbauer_monomial(n, d)

    # SOS side, three variables: multipliers g_l and monomial bases
    g_box = lambda ax: _box(ax, s)
    gram = np.zeros((3, 3, 3))
    gram[0, 0, 0] = 1; gram[1, 1, 1] = 2; gram[2, 0, 0] = -1; gram[0, 2, 0] = -1; gram[0, 0, 2] = -1
    mults3 = [np.ones((1, 1, 1)), g_box(0), g_box(1), g_box(2), gram]
    B3, sizes3 = [], []
    for g in mults3:
        gdeg = max(int(e.sum()) for e in np.argwhere(g != 0))
        basis = exps_upto((D3 - gdeg) // 2)
        nb = len(basis)
        rows, cols, vals = [], [], []
        gnz = [(tuple(e), g[tuple(e)]) for e in np.argwhere(g != 0)]
        for a_, ea in enumerate(basis):
            for b_, eb in enumerate(basis):
                for eg, vg in gnz:
                    e = (ea[0] + eb[0] + eg[0], ea[1] + eb[1] + eg[1], ea[2] + eb[2] + eg[2])
                    rows.append(index3[e]); cols.append(a_ * nb + b_); vals.append(vg)
        B3.append(sp.csr_matrix((vals, (rows, cols)), shape=(len(mons3), nb * nb)))
        sizes3.append(nb)

    # SOS side, one variable: 1 and (u+1)(s-u)
    g1 = [np.array([1.0]), np.array([s, s - 1.0, -1.0])]
    B1, sizes1 = [], []
    for g in g1:
        nb = (2 * d - (len(g) - 1)) // 2 + 1
        mat = np.zeros((2 * d + 1, nb * nb))
        for a_ in range(nb):
            for b_ in range(nb):
                for eg, vg in enumerate(g):
                    mat[a_ + b_ + eg, a_ * nb + b_] += vg
        B1.append(sp.csr_matrix(mat)); sizes1.append(nb)
    return dict(A3=A3, A1=A1, Pn=Pn, B3=B3, sizes3=sizes3, B1=B1, sizes1=sizes1,
                const3=index3[(0, 0, 0)], nmon3=len(mons3))


def _box(axis: int, s: float) -> np.ndarray:
    """(x+1)(s-x) in the given variable."""
    shape = [1, 1, 1]; shape[axis] = 3
    g = np.zeros(shape)
    idx = [0, 0, 0]
    for p, c in enumerate([s, s - 1.0, -1.0]):
        idx[axis] = p
        g[tuple(idx)] = c
    return g


def solve(n: int, d: int, s: float = 0.5, solver: str = "CLARABEL", verbose: bool = False) -> dict:
    import cvxpy as cp
    t0 = time.time()
    data = build(n, d, s)
    a = cp.Variable(d, nonneg=True)
    b = cp.Variable((2, 2), symmetric=True)
    F = [cp.Variable((d - k + 1, d - k + 1), PSD=True) for k in range(d + 1)]
    Q3 = [cp.Variable((m, m), PSD=True) for m in data["sizes3"]]
    Q1 = [cp.Variable((m, m), PSD=True) for m in data["sizes1"]]
    cons = [b >> 0]

    # (ii)  -(b22 + sum <F_k,S_k>) = sum_l g_l * sos_l
    lhs3 = sum(data["A3"][k] @ cp.vec(F[k], order="C") for k in range(d + 1))
    e0 = np.zeros(data["nmon3"]); e0[data["const3"]] = 1.0
    rhs3 = sum(data["B3"][l] @ cp.vec(Q3[l], order="C") for l in range(len(Q3)))
    cons.append(-(lhs3 + b[1, 1] * e0) == rhs3)

    # (i)  -1 - p(u) = sos + (u+1)(s-u) sos
    Pmat = np.zeros((2 * d + 1, d))
    for k in range(1, d + 1):
        Pmat[:k + 1, k - 1] = data["Pn"][k]
    u0 = np.zeros(2 * d + 1); u0[0] = 1.0
    p = Pmat @ a + (2 * b[0, 1] + b[1, 1]) * u0 + 3 * sum(data["A1"][k] @ cp.vec(F[k], order="C")
                                                         for k in range(d + 1))
    rhs1 = sum(data["B1"][l] @ cp.vec(Q1[l], order="C") for l in range(2))
    cons.append(-u0 - p == rhs1)

    obj = 1 + cp.sum(a) + b[0, 0] + cp.sum(F[0])
    prob = cp.Problem(cp.Minimize(obj), cons)
    prob.solve(solver=solver, verbose=verbose)
    return {"n": n, "degree": d, "s": s, "status": prob.status, "value": float(prob.value),
            "solver": solver, "seconds": round(time.time() - t0, 1),
            "sos_block_sizes": data["sizes3"], "rigorous": False}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--n", type=int, required=True)
    ap.add_argument("--degree", type=int, default=6)
    ap.add_argument("--s", type=float, default=0.5)
    ap.add_argument("--solver", default="CLARABEL")
    ap.add_argument("--verbose", action="store_true")
    a = ap.parse_args()
    r = solve(a.n, a.degree, a.s, a.solver, a.verbose)
    print(json.dumps(r))
    return 0


if __name__ == "__main__":
    sys.exit(main())
