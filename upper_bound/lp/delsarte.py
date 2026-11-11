#!/usr/bin/env python3
"""Produce exact Delsarte LP certificates for the kissing number.

The floating-point part of this file is only a *search* for good coefficients.
What it writes is a list of rational numbers, and the claim attached to them
is checked by verify_certificate.py in exact arithmetic (and re-checked by the
tests).  Nothing computed in floating point here is part of the proof.

Method.  Minimise 1 + sum_{k>=1} f_k subject to f_k >= 0 and
1 + sum f_k P_k(t) <= 0 on [-1, s], with the interval replaced by a point set
that is refined at the violating local maxima until the solution stabilises.
The coefficients are then rounded to rationals and f_0 is lowered by a small
exact amount until F is strictly negative on the whole interval -- which the
verifier proves with a Sturm sequence.

usage:
    python3 upper_bound/lp/delsarte.py --n 8 [--degree 30] [--s 1/2] [--out CERT.json]
    python3 upper_bound/lp/delsarte.py --all        # dimensions 3..24 -> certificates/lp/
"""
from __future__ import annotations

import argparse
import json
import sys
from fractions import Fraction
from pathlib import Path

import numpy as np
from scipy.optimize import linprog

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import verify_certificate as vc  # noqa: E402

CERT_DIR = HERE.parent / "certificates" / "lp"


def gegenbauer_values(n: int, d: int, t: np.ndarray) -> np.ndarray:
    """Rows k = 0..d of P_k(t), by the three-term recurrence (stable on [-1, 1])."""
    P = np.empty((d + 1, t.size))
    P[0] = 1.0
    if d >= 1:
        P[1] = t
    for k in range(2, d + 1):
        P[k] = ((2 * k + n - 4) * t * P[k - 1] - (k - 1) * P[k - 2]) / (k + n - 3)
    return P


def solve_lp(n: int, d: int, s: float, rounds: int = 12) -> np.ndarray:
    """Float coefficients f_1..f_d (f_0 = 1) of a near-optimal Delsarte polynomial."""
    # Chebyshev-distributed points on [-1, s], plus both end points
    m = 40 * d
    x = np.cos(np.pi * (np.arange(m) + 0.5) / m)
    pts = np.unique(np.concatenate([(s - 1) / 2 + (s + 1) / 2 * x, [-1.0, s]]))
    fine = np.linspace(-1.0, s, 200001)
    Pf = gegenbauer_values(n, d, fine)
    f = None
    for _ in range(rounds):
        P = gegenbauer_values(n, d, pts)
        res = linprog(np.ones(d), A_ub=P[1:].T, b_ub=-np.ones(pts.size), bounds=[(0, None)] * d,
                      method="highs")
        if res.status != 0:
            raise RuntimeError(f"LP failed for n={n}, d={d}: {res.message}")
        f = res.x
        F = 1.0 + f @ Pf[1:]
        # local maxima of F on the fine grid that violate F <= 0
        inner = (F[1:-1] >= F[:-2]) & (F[1:-1] >= F[2:]) & (F[1:-1] > 1e-11)
        new = fine[1:-1][inner]
        if new.size == 0:
            break
        pts = np.unique(np.concatenate([pts, new]))
    return f


def certify(n: int, d: int, s: Fraction, digits: int = 14) -> dict:
    f = solve_lp(n, d, float(s))
    scale = 10 ** digits
    fr = [Fraction(1)] + [Fraction(max(0, round(v * scale)), scale) for v in f]
    # Lower f_0 by an exact amount eps until the verifier accepts.  Subtracting a
    # constant from F is the same as subtracting it from f_0, since P_0 = 1.
    eps = Fraction(1, 10 ** 10)
    for _ in range(40):
        cert = {"n": n, "s": str(s), "f": [str(fr[0] - eps)] + [str(c) for c in fr[1:]]}
        try:
            result = vc.verify(cert)
        except ValueError:
            eps *= 4
            continue
        cert["bound"] = result["bound"]
        cert["bound_float"] = result["bound_float"]
        cert["comment"] = ("Delsarte LP certificate: F = sum f_k P_k with f_0 > 0, f_k >= 0 and F < 0 on [-1, s]; "
                           "check with upper_bound/lp/verify_certificate.py")
        return cert
    raise RuntimeError(f"could not make the rounded polynomial strictly negative (n={n}, d={d})")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--n", type=int)
    ap.add_argument("--degree", type=int, default=30)
    ap.add_argument("--s", default="1/2")
    ap.add_argument("--out", type=Path)
    ap.add_argument("--all", action="store_true", help="dimensions 3..24, written to certificates/lp/")
    a = ap.parse_args()
    s = Fraction(a.s)
    dims = range(3, 25) if a.all else [a.n]
    if not a.all and a.n is None:
        ap.error("give --n or --all")
    for n in dims:
        cert = certify(n, a.degree, s)
        out = a.out if (a.out and not a.all) else CERT_DIR / f"delsarte_n{n:02d}.json"
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(cert, indent=1) + "\n")
        print(f"n={n:2d}  degree={a.degree}  bound {cert['bound_float']:.6f} -> {cert['bound']}   {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
