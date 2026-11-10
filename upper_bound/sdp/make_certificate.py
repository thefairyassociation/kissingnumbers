#!/usr/bin/env python3
"""Turn a numerical three-point SDP solution into an exact certificate.

The output is a JSON file of integers that verify_certificate.py checks in
exact arithmetic.  Nothing computed in floating point here is trusted: this
program only has to *find* numbers that the verifier accepts.

Steps.
  1. Solve the SDP of three_point.py, but instead of minimising the bound,
     fix it at --target and maximise the smallest eigenvalue of every
     positive semidefinite block.  That gives a solution in the interior.
  2. Round a_k, B, F_k and all sum-of-squares Gram matrices except the two
     unweighted ones (Q3_0 and Q1_0) to rationals with denominator --den.
  3. Compute, in exact rational arithmetic, the two polynomials that the
     unweighted Gram matrices must represent, and move the (tiny) difference
     between them and the rounded Q3_0, Q1_0 into those matrices.  The two
     identities then hold exactly.
  4. Run the exact verifier.  It fails if the eigenvalue margin from step 1
     did not survive steps 2-3; raise --target (more room) in that case.

usage:
    python3 upper_bound/sdp/make_certificate.py --n 3 --degree 5 --target 12.95 \
        --out upper_bound/certificates/sdp/three_point_n03.json
"""
from __future__ import annotations

import argparse
import json
import sys
from fractions import Fraction
from math import gcd
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import three_point as tp            # noqa: E402  (numerical model)
import verify_certificate as vc     # noqa: E402  (exact polynomials and the verifier)


def interior_solution(n: int, d: int, s: float, target: float, solver: str, verbose: bool):
    import cvxpy as cp
    data = tp.build(n, d, s)
    t = cp.Variable()
    a = cp.Variable(d, nonneg=True)

    def shifted(m):
        Z = cp.Variable((m, m), PSD=True)
        return Z + t * np.eye(m)

    b = shifted(2)
    F = [shifted(d - k + 1) for k in range(d + 1)]
    Q3 = [shifted(m) for m in data["sizes3"]]
    Q1 = [shifted(m) for m in data["sizes1"]]
    lhs3 = sum(data["A3"][k] @ cp.vec(F[k], order="C") for k in range(d + 1))
    e0 = np.zeros(data["nmon3"]); e0[data["const3"]] = 1.0
    rhs3 = sum(data["B3"][l] @ cp.vec(Q3[l], order="C") for l in range(len(Q3)))
    Pmat = np.zeros((2 * d + 1, d))
    for k in range(1, d + 1):
        Pmat[:k + 1, k - 1] = data["Pn"][k]
    u0 = np.zeros(2 * d + 1); u0[0] = 1.0
    p = Pmat @ a + (2 * b[0, 1] + b[1, 1]) * u0 + 3 * sum(data["A1"][k] @ cp.vec(F[k], order="C")
                                                         for k in range(d + 1))
    rhs1 = sum(data["B1"][l] @ cp.vec(Q1[l], order="C") for l in range(2))
    cons = [-(lhs3 + b[1, 1] * e0) == rhs3, -u0 - p == rhs1,
            1 + cp.sum(a) + b[0, 0] + cp.sum(F[0]) <= target]
    prob = cp.Problem(cp.Maximize(t), cons)
    prob.solve(solver=solver, verbose=verbose)
    if t.value is None or t.value <= 0:
        raise RuntimeError(f"no interior solution at target {target} (status {prob.status}, t={t.value})")
    val = lambda e: np.array(e.value, dtype=float)
    return dict(t=float(t.value), a=val(a), b=val(b), F=[val(x) for x in F],
                Q3=[val(x) for x in Q3], Q1=[val(x) for x in Q1], status=prob.status)


def round_sym(M: np.ndarray, den: int) -> list[list[int]]:
    M = (M + M.T) / 2
    R = [[int(round(float(x) * den)) for x in row] for row in M]
    for i in range(len(R)):
        for j in range(i):
            R[i][j] = R[j][i]
    return R


def lcm(a: int, b: int) -> int:
    return a * b // gcd(a, b)


def absorb(target: dict, basis: list, M: list[list[int]], den: int):
    """Smallest change to M/den (as a Gram matrix on `basis`) that makes m'Mm equal `target` exactly.

    `target` maps exponents to Fractions.  Returns (integer matrix, denominator).
    """
    index = {e: i for i, e in enumerate(basis)}
    current: dict = {}
    for a_, ea in enumerate(basis):
        for b_, eb in enumerate(basis):
            if M[a_][b_]:
                e = tuple(x + y for x, y in zip(ea, eb))
                current[e] = current.get(e, 0) + Fraction(M[a_][b_], den)
    delta = {e: target.get(e, 0) - current.get(e, 0) for e in set(target) | set(current)}
    delta = {e: c for e, c in delta.items() if c}
    Mq = [[Fraction(x, den) for x in row] for row in M]
    for e, c in delta.items():
        best = None
        for a_, ea in enumerate(basis):
            eb = tuple(x - y for x, y in zip(e, ea))
            if min(eb) < 0 or eb not in index:
                continue
            score = abs(sum(ea) - sum(eb)) * 10 + (0 if ea == eb else 1)
            if best is None or score < best[0]:
                best = (score, a_, index[eb])
        if best is None:
            raise RuntimeError(f"monomial {e} is not a product of two basis monomials")
        _, i, j = best
        if i == j:
            Mq[i][i] += c
        else:
            Mq[i][j] += c / 2
            Mq[j][i] += c / 2
    D = 1
    for row in Mq:
        for x in row:
            D = lcm(D, x.denominator)
    return [[int(x * D) for x in row] for row in Mq], D


def make(n: int, d: int, s: Fraction, target: float, den: int, solver: str, verbose: bool) -> dict:
    sol = interior_solution(n, d, float(s), target, solver, verbose)
    print(f"interior solution: status {sol['status']}, eigenvalue margin t = {sol['t']:.3e}", flush=True)
    a = [Fraction(max(0, int(round(float(x) * den))), den) for x in sol["a"]]
    B = round_sym(sol["b"], den)
    F = [round_sym(M, den) for M in sol["F"]]
    Q3 = [round_sym(M, den) for M in sol["Q3"]]
    Q1 = [round_sym(M, den) for M in sol["Q1"]]
    b11, b12, b22 = (Fraction(B[0][0], den), Fraction(B[0][1], den), Fraction(B[1][1], den))

    S = vc.S_matrices(n, d)
    third: dict = {(0, 0, 0): b22} if b22 else {}
    for k in range(d + 1):
        for i in range(d - k + 1):
            for j in range(d - k + 1):
                if F[k][i][j]:
                    third = vc.padd(third, S[k][i][j], Fraction(F[k][i][j], den))
    mults = vc.multipliers3(s)
    D3 = 2 * d
    need = {e: -c for e, c in third.items()}                 # what the SOS side must equal
    for g, M in list(zip(mults, Q3))[1:]:
        basis = vc.exps_upto((D3 - vc.degree(g)) // 2)
        need = vc.padd(need, vc.pmul(g, vc.quad_form(M, den, basis)), Fraction(-1))
    Q3_0, den3 = absorb(need, vc.exps_upto(d), Q3[0], den)

    second = [Fraction(0)] * (2 * d + 1)
    second[0] += 2 * b12 + b22
    Pn = vc.gegenbauer(n, d)
    for k in range(1, d + 1):
        for i, c in enumerate(Pn[k]):
            second[i] += a[k - 1] * c
    for k in range(d + 1):
        for i in range(d - k + 1):
            for j in range(d - k + 1):
                if F[k][i][j]:
                    for e, c in S[k][i][j].items():
                        second[e[0] + e[1]] += 3 * Fraction(F[k][i][j], den) * c
    need1 = [-c for c in second]
    need1[0] -= 1
    g = [s, s - 1, Fraction(-1)]
    nb1 = len(Q1[1])
    for p in range(nb1):
        for q in range(nb1):
            for eg, cg in enumerate(g):
                need1[p + q + eg] -= cg * Fraction(Q1[1][p][q], den)
    Q1_0, den1 = absorb({(i,): c for i, c in enumerate(need1) if c}, [(i,) for i in range(d + 1)], Q1[0], den)

    as_obj = lambda M, D=den: {"den": D, "rows": M}
    cert = {
        "comment": "three-point (Bachoc-Vallentin) certificate; check with upper_bound/sdp/verify_certificate.py",
        "n": n, "d": d, "s": str(s),
        "a": [str(x) for x in a], "B": as_obj(B), "F": [as_obj(M) for M in F],
        "Q3": [as_obj(Q3_0, den3)] + [as_obj(M) for M in Q3[1:]],
        "Q1": [as_obj(Q1_0, den1), as_obj(Q1[1])],
    }
    result = vc.verify(cert)          # raises ValueError if the margin did not survive rounding
    cert["bound"] = result["bound"]
    cert["bound_float"] = result["bound_float"]
    return cert


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--n", type=int, required=True)
    ap.add_argument("--degree", type=int, required=True)
    ap.add_argument("--target", type=float, required=True, help="value the bound is held at (above the SDP optimum)")
    ap.add_argument("--s", default="1/2")
    ap.add_argument("--den", type=int, default=10 ** 12)
    ap.add_argument("--solver", default="CLARABEL")
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--verbose", action="store_true")
    a = ap.parse_args()
    cert = make(a.n, a.degree, Fraction(a.s), a.target, a.den, a.solver, a.verbose)
    a.out.parent.mkdir(parents=True, exist_ok=True)
    a.out.write_text(json.dumps(cert, separators=(",", ":")) + "\n")
    print(f"VERIFIED and written: n={a.n} degree={a.degree}  |C| <= {cert['bound_float']:.6f}, hence <= {cert['bound']}"
          f"   ({a.out}, {a.out.stat().st_size // 1024} KiB)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
