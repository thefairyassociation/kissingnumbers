#!/usr/bin/env python3
"""Exact verifier for three-point (Bachoc-Vallentin) semidefinite certificates.

Standard library only: integers and fractions.  Nothing here is floating
point, and this file shares no code with the program that produces the
certificates.

WHAT IS PROVED
==============
Let C be a set of N unit vectors in R^n whose pairwise inner products lie in
[-1, s].  Notation: P_k^m is the Gegenbauer polynomial of R^m with
P_k^m(1) = 1,

    Q_k(u,v,t)    = ((1-u^2)(1-v^2))^(k/2) P_k^(n-1)((t-uv)/sqrt((1-u^2)(1-v^2))),
    Y_k(u,v,t)_ij = u^i v^j Q_k(u,v,t)          (0 <= i,j <= d-k),
    S_k           = the average of Y_k over the six permutations of (u,v,t).

Two classical positivity facts are used and NOT re-proved here:

 (A) [Bachoc & Vallentin, J. Amer. Math. Soc. 21 (2008), Cor. 3.5]
     sum over (x,y,z) in C^3 of S_k(x.y, x.z, y.z) is positive semidefinite;
 (B) [Schoenberg 1942]  sum over (x,y) in C^2 of P_k^n(x.y) >= 0.

Given numbers a_k >= 0 (k = 1..d) and positive semidefinite matrices
B = [[b11,b12],[b12,b22]] and F_0, ..., F_d, pair (A) with F_k, (B) with a_k,
and note that N (1, N-1) B (1, N-1)^T >= 0.  Sorting the triples by how many
of x, y, z coincide gives

  0 <= N [ sum_k a_k + b11 + sum_k <F_k, S_k(1,1,1)> ]
       + sum_{x != y}        [ sum_k a_k P_k^n(u) + 2 b12 + b22 + 3 sum_k <F_k, S_k(u,u,1)> ]
       + sum_{x,y,z distinct} [ b22 + sum_k <F_k, S_k(u,v,t)> ].

So if
  (i)  the second bracket is <= -1 for every u in [-1, s], and
  (ii) the third bracket is <= 0 on
       D = { -1 <= u,v,t <= s,  1 + 2uvt - u^2 - v^2 - t^2 >= 0 },
then N - 1 <= sum_k a_k + b11 + sum_k <F_k, S_k(1,1,1)>.

A certificate supplies a_k, B, F_k together with sum-of-squares proofs of (i)
and (ii):

  -1 - [second bracket](u)  =  m1' Q1_0 m1  +  (u+1)(s-u) m1' Q1_1 m1
  -[third bracket](u,v,t)   =  m3' Q3_0 m3  +  sum_{l=1..3} (x_l+1)(s-x_l) m3' Q3_l m3
                               + (1 + 2uvt - u^2 - v^2 - t^2) m3' Q3_4 m3

with monomial vectors m and positive definite matrices Q.  Every multiplier is
nonnegative on the relevant set, so the identities imply (i) and (ii).

WHAT THIS PROGRAM CHECKS, ALL IN EXACT ARITHMETIC
=================================================
  1. a_k >= 0;
  2. B, every F_k and every Q matrix is symmetric positive definite
     (all leading principal minors positive, by fraction-free elimination);
  3. both polynomial identities hold coefficient by coefficient;
  4. the bound  floor(1 + sum a_k + b11 + sum_k <F_k, S_k(1,1,1)>).

usage:  python3 upper_bound/sdp/verify_certificate.py CERT.json [...]
"""
from __future__ import annotations

import itertools
import json
import sys
from fractions import Fraction
from pathlib import Path

Poly = dict  # {(i, j, l): Fraction} for u^i v^j t^l


# ---------------------------------------------------------------- polynomials
def padd(a: Poly, b: Poly, cb: Fraction = Fraction(1)) -> Poly:
    out = dict(a)
    for e, c in b.items():
        v = out.get(e, 0) + cb * c
        if v:
            out[e] = v
        else:
            out.pop(e, None)
    return out


def pmul(a: Poly, b: Poly) -> Poly:
    out: Poly = {}
    for (i1, j1, l1), c1 in a.items():
        for (i2, j2, l2), c2 in b.items():
            e = (i1 + i2, j1 + j2, l1 + l2)
            out[e] = out.get(e, 0) + c1 * c2
    return {e: c for e, c in out.items() if c}


def ppow(a: Poly, m: int) -> Poly:
    out: Poly = {(0, 0, 0): Fraction(1)}
    for _ in range(m):
        out = pmul(out, a)
    return out


def gegenbauer(m: int, d: int) -> list[list[Fraction]]:
    """Monomial coefficients (lowest first) of P_0..P_d for R^m, P_k(1) = 1."""
    P = [[Fraction(1)], [Fraction(0), Fraction(1)]]
    for k in range(2, d + 1):
        nxt = [Fraction(0)] * (k + 1)
        for i, v in enumerate(P[k - 1]):
            nxt[i + 1] += Fraction(2 * k + m - 4, k + m - 3) * v
        for i, v in enumerate(P[k - 2]):
            nxt[i] -= Fraction(k - 1, k + m - 3) * v
        P.append(nxt)
    return P[: d + 1]


def Q_poly(n: int, k: int) -> Poly:
    c = gegenbauer(n - 1, max(k, 1))[k]
    w = pmul({(0, 0, 0): Fraction(1), (2, 0, 0): Fraction(-1)},
             {(0, 0, 0): Fraction(1), (0, 2, 0): Fraction(-1)})          # (1-u^2)(1-v^2)
    tuv = {(0, 0, 1): Fraction(1), (1, 1, 0): Fraction(-1)}              # t - uv
    out: Poly = {}
    for m in range(k % 2, k + 1, 2):
        if c[m]:
            out = padd(out, pmul(ppow(tuv, m), ppow(w, (k - m) // 2)), c[m])
    return out


def symmetrise(p: Poly) -> Poly:
    out: Poly = {}
    for perm in itertools.permutations(range(3)):
        for e, c in p.items():
            f = (e[perm[0]], e[perm[1]], e[perm[2]])
            out[f] = out.get(f, 0) + c / 6
    return {e: c for e, c in out.items() if c}


def S_matrices(n: int, d: int) -> list[list[list[Poly]]]:
    S = []
    for k in range(d + 1):
        Qk = Q_poly(n, k)
        m = d - k + 1
        S.append([[symmetrise(pmul({(i, j, 0): Fraction(1)}, Qk)) for j in range(m)] for i in range(m)])
    return S


def exps_upto(deg: int) -> list[tuple[int, int, int]]:
    return [(i, j, l) for i in range(deg + 1) for j in range(deg + 1 - i) for l in range(deg + 1 - i - j)]


def multipliers3(s: Fraction) -> list[Poly]:
    def box(axis: int) -> Poly:
        out = {}
        for p, c in enumerate([s, s - 1, Fraction(-1)]):          # (x+1)(s-x) = s + (s-1)x - x^2
            e = [0, 0, 0]
            e[axis] = p
            if c:
                out[tuple(e)] = Fraction(c)
        return out
    gram = {(0, 0, 0): Fraction(1), (1, 1, 1): Fraction(2), (2, 0, 0): Fraction(-1),
            (0, 2, 0): Fraction(-1), (0, 0, 2): Fraction(-1)}
    return [{(0, 0, 0): Fraction(1)}, box(0), box(1), box(2), gram]


def degree(p: Poly) -> int:
    return max(sum(e) for e in p)


# ------------------------------------------------------------------- matrices
def load_matrix(obj: dict) -> tuple[list[list[int]], int]:
    rows, den = [[int(x) for x in r] for r in obj["rows"]], int(obj["den"])
    if den <= 0 or any(len(r) != len(rows) for r in rows):
        raise ValueError("malformed matrix")
    for i in range(len(rows)):
        for j in range(i):
            if rows[i][j] != rows[j][i]:
                raise ValueError("matrix is not symmetric")
    return rows, den


def is_positive_definite(M: list[list[int]]) -> bool:
    """Sylvester's criterion by fraction-free (Bareiss) elimination on an integer matrix."""
    A = [row[:] for row in M]
    n = len(A)
    prev = 1
    for k in range(n):
        if A[k][k] <= 0:          # A[k][k] is the (k+1)-th leading principal minor
            return False
        for i in range(k + 1, n):
            for j in range(k + 1, n):
                A[i][j] = (A[i][j] * A[k][k] - A[i][k] * A[k][j]) // prev
        prev = A[k][k]
    return True


def quad_form(M: list[list[int]], den: int, basis: list[tuple[int, int, int]]) -> Poly:
    """m' (M/den) m as a polynomial."""
    out: dict = {}
    for a, ea in enumerate(basis):
        for b, eb in enumerate(basis):
            if M[a][b]:
                e = (ea[0] + eb[0], ea[1] + eb[1], ea[2] + eb[2])
                out[e] = out.get(e, 0) + M[a][b]
    return {e: Fraction(c, den) for e, c in out.items() if c}


# ------------------------------------------------------------------ verifier
def verify(cert: dict) -> dict:
    n, d, s = int(cert["n"]), int(cert["d"]), Fraction(cert["s"])
    if n < 3 or d < 1 or not (-1 < s < 1):
        raise ValueError("need n >= 3, d >= 1, -1 < s < 1")
    a = [Fraction(x) for x in cert["a"]]
    if len(a) != d or any(x < 0 for x in a):
        raise ValueError("a_k must be d nonnegative numbers")
    B, Bden = load_matrix(cert["B"])
    if len(B) != 2:
        raise ValueError("B must be 2 x 2")
    F = [load_matrix(m) for m in cert["F"]]
    if len(F) != d + 1 or any(len(F[k][0]) != d - k + 1 for k in range(d + 1)):
        raise ValueError("F_k must have size d-k+1")
    Q3 = [load_matrix(m) for m in cert["Q3"]]
    Q1 = [load_matrix(m) for m in cert["Q1"]]
    for name, (M, _) in [("B", (B, Bden))] + [(f"F_{k}", F[k]) for k in range(d + 1)] + \
            [(f"Q3_{l}", Q3[l]) for l in range(len(Q3))] + [(f"Q1_{l}", Q1[l]) for l in range(len(Q1))]:
        if not is_positive_definite(M):
            raise ValueError(f"{name} is not positive definite")
    b11, b12, b22 = (Fraction(B[0][0], Bden), Fraction(B[0][1], Bden), Fraction(B[1][1], Bden))

    S = S_matrices(n, d)
    # third bracket:  b22 + sum_k <F_k, S_k>
    third: Poly = {(0, 0, 0): b22} if b22 else {}
    for k in range(d + 1):
        M, den = F[k]
        for i in range(d - k + 1):
            for j in range(d - k + 1):
                if M[i][j]:
                    third = padd(third, S[k][i][j], Fraction(M[i][j], den))
    D3 = 2 * d
    if third and degree(third) > D3:
        raise ValueError("unexpected degree")
    mults = multipliers3(s)
    if len(Q3) != len(mults):
        raise ValueError("need five Q3 matrices")
    rhs: Poly = {}
    for g, (M, den) in zip(mults, Q3):
        basis = exps_upto((D3 - degree(g)) // 2)
        if len(M) != len(basis):
            raise ValueError("Q3 block has the wrong size")
        rhs = padd(rhs, pmul(g, quad_form(M, den, basis)))
    if padd(rhs, third):                       # rhs must equal -third
        raise ValueError("identity (ii) does not hold")

    # second bracket, a polynomial in u alone:  v -> u, t -> 1
    second = [Fraction(0)] * (2 * d + 1)
    second[0] += 2 * b12 + b22
    Pn = gegenbauer(n, d)
    for k in range(1, d + 1):
        for i, c in enumerate(Pn[k]):
            second[i] += a[k - 1] * c
    for k in range(d + 1):
        M, den = F[k]
        for i in range(d - k + 1):
            for j in range(d - k + 1):
                if M[i][j]:
                    for e, c in S[k][i][j].items():
                        second[e[0] + e[1]] += 3 * Fraction(M[i][j], den) * c
    g1 = [[Fraction(1)], [s, s - 1, Fraction(-1)]]
    if len(Q1) != 2:
        raise ValueError("need two Q1 matrices")
    rhs1 = [Fraction(0)] * (2 * d + 1)
    for g, (M, den) in zip(g1, Q1):
        nb = (2 * d - (len(g) - 1)) // 2 + 1
        if len(M) != nb:
            raise ValueError("Q1 block has the wrong size")
        for p in range(nb):
            for q in range(nb):
                for eg, cg in enumerate(g):
                    rhs1[p + q + eg] += cg * Fraction(M[p][q], den)
    lhs1 = [-c for c in second]
    lhs1[0] -= 1
    if lhs1 != rhs1:
        raise ValueError("identity (i) does not hold")

    bound = 1 + sum(a) + b11
    for k in range(d + 1):
        M, den = F[k]
        for i in range(d - k + 1):
            for j in range(d - k + 1):
                bound += Fraction(M[i][j], den) * sum(S[k][i][j].values())    # value at (1,1,1)
    return {"n": n, "s": str(s), "degree": d, "bound_float": float(bound),
            "bound": bound.numerator // bound.denominator,
            "psd_block_sizes": [len(m[0]) for m in Q3] + [len(m[0]) for m in Q1]}


def main(argv: list[str]) -> int:
    if not argv:
        print(__doc__)
        return 2
    ok = True
    for name in argv:
        try:
            r = verify(json.loads(Path(name).read_text()))
            print(f"{name}: VERIFIED  n={r['n']} s={r['s']} degree={r['degree']}  "
                  f"|C| <= {r['bound_float']:.6f}, hence <= {r['bound']}")
        except (ValueError, AssertionError, KeyError) as exc:
            ok = False
            print(f"{name}: REJECTED  {exc}")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
