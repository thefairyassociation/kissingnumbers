#!/usr/bin/env python3
"""Exact certificate for a floating-point kissing configuration.

A float configuration whose maximum cosine is *strictly* below 1/2 is a proof
once it is pinned down exactly: round every coordinate to an integer at scale
2^B, and the kissing condition

    cos(x, y) <= 1/2   <=>   <x,y> <= 0   or   4 <x,y>^2 <= |x|^2 |y|^2

becomes a statement about integers, checked here with Python's exact big
integers.  No algebraic structure needs to be identified: rounding moves each
cosine by about sqrt(n) 2^-B, so any margin far above that survives it.  (A
configuration that is *tight* -- max cosine exactly 1/2 -- cannot be certified
this way and still needs exact algebraic coordinates; see verify_exact.py.)

The certificate also proves an explicit margin: with --margin d it checks
4 <x,y>^2 <= (1-2d)^2 |x|^2 |y|^2, i.e. every cosine <= 1/2 - d.  By default
it searches for the largest d of the form k * 10^-e that it can prove.

usage:
    python3 kissing/lib/certify_float.py COORDS.txt [--bits 40]
        [--margin D] [--json-out CERT.json] [--record R]

The JSON written by --json-out holds the integer vectors themselves, in the
same {"dimension", "count", "field": "Q", "vectors"} shape verify_exact.py
reads (vectors are not normalised, so there is no scale2: use this script, or
any exact checker of the angle condition, to re-verify).
"""
from __future__ import annotations

import argparse
import json
import sys
from fractions import Fraction

import numpy as np


def load_float(path: str) -> np.ndarray:
    X = np.loadtxt(path, comments="#", ndmin=2)
    if not np.all(np.isfinite(X)):
        raise ValueError(f"{path}: non-finite coordinates")
    return X


def to_integer(X: np.ndarray, bits: int) -> list[list[int]]:
    U = X / np.linalg.norm(X, axis=1, keepdims=True)
    scale = float(2 ** bits)
    return [[int(round(c * scale)) for c in row] for row in U]


def exact_pairs(V: list[list[int]]):
    """Yield (i, j, dot, |x_i|^2, |x_j|^2) for every pair with dot > 0."""
    norms = [sum(c * c for c in v) for v in V]
    N = len(V)
    # numpy object arrays keep Python ints (exact) but vectorise the loops.
    A = np.array(V, dtype=object)
    for i in range(N - 1):
        d = A[i + 1:] @ A[i]
        for off in np.nonzero(d > 0)[0]:
            j = i + 1 + int(off)
            yield i, j, int(d[off]), norms[i], norms[j]


def certify(V: list[list[int]], margin: Fraction = Fraction(0)):
    """Return (ok, worst) where worst is the pair with the largest exact
    4<x,y>^2 / (|x|^2|y|^2) (as a Fraction), proving every cosine <= 1/2 - margin
    iff ok."""
    lim = (1 - 2 * margin) ** 2  # need 4 d^2 <= lim * nx * ny
    ok = True
    worst = (Fraction(-1), None)
    norms_nonzero = all(any(c for c in v) for v in V)
    if not norms_nonzero:
        return False, worst
    seen = set()
    for i, j, d, nx, ny in exact_pairs(V):
        q = Fraction(4 * d * d, nx * ny)
        if q > worst[0]:
            worst = (q, (i, j))
        if q > lim:
            ok = False
    for v in V:  # distinctness (a repeated vector has cosine 1)
        t = tuple(v)
        if t in seen:
            return False, (Fraction(4), None)
        seen.add(t)
    return ok, worst


def proven_margin(worst_q: Fraction) -> Fraction:
    """Largest d = k*10^-e (2 significant digits) with (1-2d)^2 >= worst_q."""
    best = Fraction(0)
    for e in range(2, 18):
        for k in range(99, 0, -1):
            d = Fraction(k, 10 ** e)
            if (1 - 2 * d) ** 2 >= worst_q:
                return d
    return best


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("coords")
    ap.add_argument("--bits", type=int, default=40)
    ap.add_argument("--margin", type=str, default=None,
                    help="prove every cosine <= 1/2 - MARGIN (a rational, e.g. 1e-8 or 1/100000000)")
    ap.add_argument("--json-out")
    ap.add_argument("--record", type=int, default=None,
                    help="also report whether the count beats this record")
    a = ap.parse_args()

    X = load_float(a.coords)
    N, n = X.shape
    V = to_integer(X, a.bits)
    margin = Fraction(0)
    if a.margin is not None:
        margin = Fraction(a.margin) if "/" in a.margin else Fraction(a.margin)
    ok, (wq, wpair) = certify(V, margin)
    # The float view of the worst cosine, for orientation only.
    worst_cos = float(wq) ** 0.5 / 2 if wq >= 0 else float("nan")
    print(f"{a.coords}: N={N} n={n} integer scale 2^{a.bits}")
    print(f"  worst pair {wpair}: exact 4<x,y>^2/(|x|^2|y|^2) = {float(wq):.17g}"
          f"  (cosine ~ {worst_cos:.15f})")
    if not ok:
        print(f"  NOT CERTIFIED: some cosine exceeds 1/2 - {margin}")
        return 1
    d = proven_margin(wq) if a.margin is None else margin
    print(f"  CERTIFIED (exact integer arithmetic): all {N*(N-1)//2} pairs have cosine <= 1/2"
          + (f" - {d} ({float(d):.2e})" if d > 0 else ""))
    if a.record is not None:
        print(f"  count {N} vs record {a.record}: {'BEATS' if N > a.record else 'does not beat'} it")
    if a.json_out:
        with open(a.json_out, "w") as f:
            json.dump({"dimension": n, "count": N, "field": "Q",
                       "integer_scale_bits": a.bits,
                       "certified": "every pair: <x,y> <= 0 or 4<x,y>^2 <= (1-2d)^2 |x|^2 |y|^2",
                       "margin_d": str(d),
                       "source": a.coords,
                       "vectors": [[str(c) for c in v] for v in V]}, f)
        print(f"  wrote {a.json_out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
