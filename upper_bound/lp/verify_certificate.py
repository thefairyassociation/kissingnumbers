#!/usr/bin/env python3
"""Exact verifier for Delsarte linear-programming certificates.

Standard library only (fractions): nothing here is floating point, and it
shares no code with the program that produced the certificates.

The theorem (Delsarte-Goethals-Seidel 1977, Kabatiansky-Levenshtein 1978).
Let P_k be the Gegenbauer polynomials of R^n normalised by P_k(1) = 1.  If

    F(t) = sum_{k=0}^{d} f_k P_k(t),   f_0 > 0,   f_k >= 0 for k >= 1,
    F(t) <= 0  for all t in [-1, s],

then every set C of unit vectors in R^n with pairwise inner products <= s has

    |C| <= F(1) / f_0.

Proof: the matrices (P_k(x.y))_{x,y in C} are positive semidefinite, so
sum_{x,y} F(x.y) >= f_0 |C|^2; and the off-diagonal terms are <= 0, so the
same sum is <= |C| F(1).  With s = 1/2 this bounds the kissing number.

Two certificate forms are accepted, both JSON with rationals written "p/q".

{"n", "s", "f": [f_0, ..., f_d]} -- F given by its Gegenbauer coefficients:

  1. check f_0 > 0 and f_k >= 0;
  2. expand F exactly, using the rational three-term recurrence
        (k+n-3) P_k = (2k+n-4) t P_{k-1} - (k-1) P_{k-2};
  3. prove F < 0 on the whole closed interval [-1, s]: F(-1) < 0, F(s) < 0,
     and a Sturm sequence shows F has no root in (-1, s];
  4. return floor(F(1)/f_0).

{"n", "s", "roots": [[r, m], ...]} -- F = prod (t - r)^m with rational roots:

  1. F <= 0 on [-1, s] because F is negative at the midpoint of every gap
     between consecutive roots / end points inside the interval (its sign
     cannot change inside a gap);
  2. expand F in the Gegenbauer basis exactly and check f_0 > 0, f_k >= 0;
  3. return floor(F(1)/f_0).

The second form is for the two sharp cases, n = 8 and n = 24, where the
optimal polynomial vanishes at the inner products of the E8 and Leech
configurations and so cannot be strictly negative.

usage:  python3 upper_bound/lp/verify_certificate.py CERT.json [CERT.json ...]
"""
from __future__ import annotations

import json
import sys
from fractions import Fraction
from math import gcd
from pathlib import Path


def gegenbauer(n: int, d: int) -> list[list[Fraction]]:
    """P_0..P_d for R^n as coefficient lists (lowest degree first), P_k(1) = 1."""
    P = [[Fraction(1)], [Fraction(0), Fraction(1)]]
    for k in range(2, d + 1):
        a, b, c = Fraction(2 * k + n - 4, k + n - 3), Fraction(k - 1, k + n - 3), P[k - 1]
        nxt = [Fraction(0)] * (k + 1)
        for i, v in enumerate(c):
            nxt[i + 1] += a * v
        for i, v in enumerate(P[k - 2]):
            nxt[i] -= b * v
        P.append(nxt)
    return P[: d + 1]


def evaluate(p: list, t: Fraction) -> Fraction:
    acc = Fraction(0)
    for c in reversed(p):
        acc = acc * t + c
    return acc


def _primitive(p: list[int]) -> list[int]:
    g = 0
    for c in p:
        g = gcd(g, abs(c))
    return [c // g for c in p] if g > 1 else p


def _to_int(p: list[Fraction]) -> list[int]:
    den = 1
    for c in p:
        den = den * c.denominator // gcd(den, c.denominator)
    return _primitive([int(c * den) for c in p])


def _neg_rem(a: list[int], b: list[int]) -> list[int]:
    """-(a mod b) up to a POSITIVE integer factor (so signs are those of a Sturm chain)."""
    a = a[:]
    lb = b[-1]
    while len(a) >= len(b):
        if a[-1] != 0:
            # multiply a by |lb| (positive) and cancel the leading term
            la, shift = a[-1], len(a) - len(b)
            a = [c * abs(lb) for c in a]
            q = (la * abs(lb)) // lb
            for i, c in enumerate(b):
                a[i + shift] -= q * c
        a.pop()
    while a and a[-1] == 0:
        a.pop()
    return _primitive([-c for c in a]) if a else []


def sturm_chain(p: list[Fraction]) -> list[list[int]]:
    p0 = _to_int(p)
    p1 = _primitive([i * c for i, c in enumerate(p0)][1:])
    chain = [p0, p1]
    while True:
        r = _neg_rem(chain[-2], chain[-1])
        if not r:
            break
        chain.append(r)
    return chain


def _variations(chain: list[list[int]], t: Fraction) -> int:
    signs = []
    for q in chain:
        v = evaluate([Fraction(c) for c in q], t)
        if v != 0:
            signs.append(v > 0)
    return sum(1 for a, b in zip(signs, signs[1:]) if a != b)


def roots_in_half_open(p: list[Fraction], a: Fraction, b: Fraction) -> int:
    """Number of distinct real roots of p in (a, b]; requires p(a) != 0 != p(b)."""
    assert evaluate(p, a) != 0 and evaluate(p, b) != 0
    chain = sturm_chain(p)
    return _variations(chain, a) - _variations(chain, b)


def to_gegenbauer(F: list[Fraction], n: int) -> list[Fraction]:
    """Coefficients f_k with F = sum f_k P_k (exact; top-down elimination)."""
    d = len(F) - 1
    P = gegenbauer(n, d)
    rest = F[:]
    f = [Fraction(0)] * (d + 1)
    for k in range(d, -1, -1):
        f[k] = rest[k] / P[k][k]
        for i, c in enumerate(P[k]):
            rest[i] -= f[k] * c
    assert all(c == 0 for c in rest)
    return f


def verify_roots(cert: dict) -> dict:
    n = int(cert["n"])
    s = Fraction(cert["s"])
    roots = [(Fraction(r), int(m)) for r, m in cert["roots"]]
    if n < 3 or not (-1 < s < 1) or any(m < 1 for _, m in roots):
        raise ValueError("need n >= 3, -1 < s < 1 and positive multiplicities")
    F = [Fraction(1)]
    for r, m in roots:
        for _ in range(m):
            nxt = [Fraction(0)] * (len(F) + 1)
            for i, c in enumerate(F):
                nxt[i + 1] += c
                nxt[i] -= r * c
            F = nxt
    cuts = sorted({Fraction(-1), s} | {r for r, _ in roots if -1 < r < s})
    for a, b in zip(cuts, cuts[1:]):
        if not evaluate(F, (a + b) / 2) < 0:
            raise ValueError(f"F is not negative between {a} and {b}")
    f = to_gegenbauer(F, n)
    if not f[0] > 0:
        raise ValueError("f_0 must be positive")
    if any(c < 0 for c in f[1:]):
        raise ValueError("a Gegenbauer coefficient is negative")
    bound = evaluate(F, Fraction(1)) / f[0]
    return {"n": n, "s": str(s), "degree": len(F) - 1, "bound_exact": str(bound),
            "bound_float": float(bound), "bound": bound.numerator // bound.denominator}


def verify(cert: dict) -> dict:
    if "roots" in cert:
        return verify_roots(cert)
    n = int(cert["n"])
    s = Fraction(cert["s"])
    f = [Fraction(x) for x in cert["f"]]
    d = len(f) - 1
    if n < 3 or not (-1 < s < 1):
        raise ValueError("need n >= 3 and -1 < s < 1")
    if not f[0] > 0:
        raise ValueError("f_0 must be positive")
    if any(c < 0 for c in f[1:]):
        raise ValueError("f_k must be nonnegative")
    P = gegenbauer(n, d)
    for k, pk in enumerate(P):
        assert evaluate(pk, Fraction(1)) == 1, k
    F = [Fraction(0)] * (d + 1)
    for fk, pk in zip(f, P):
        for i, c in enumerate(pk):
            F[i] += fk * c
    while len(F) > 1 and F[-1] == 0:
        F.pop()
    lo, hi = Fraction(-1), s
    if not evaluate(F, lo) < 0:
        raise ValueError("F(-1) is not negative")
    if not evaluate(F, hi) < 0:
        raise ValueError("F(s) is not negative")
    nroots = roots_in_half_open(F, lo, hi)
    if nroots != 0:
        raise ValueError(f"F has {nroots} root(s) in (-1, s]: it is not negative on the interval")
    bound = evaluate(F, Fraction(1)) / f[0]
    return {"n": n, "s": str(s), "degree": d, "bound_exact": str(bound),
            "bound_float": float(bound), "bound": bound.numerator // bound.denominator}


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
