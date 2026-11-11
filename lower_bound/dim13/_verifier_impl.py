#!/usr/bin/env python3
"""Exact kissing-number verifier.

A configuration is a list of n-dimensional vectors with coordinates in a
number field (Q, Q(sqrt(d)), or a product of quadratic fields) represented
either as:

* strings understood by sympy (e.g. "2", "-2*sqrt(3)", "(1 + sqrt(5))/2"), or
* integer / Rational tuples, optionally with a shared squared-norm.

The kissing condition for unit vectors is max_{i!=j} <vi,vj> <= 1/2.
Equal-norm vectors may be stored un-normalized: we check
2 <vi,vj> <= ||vi||^2 with all norms equal, which is equivalent.

Never uses floating-point for the certificate.
"""

from __future__ import annotations

import argparse
import json
import sys
from fractions import Fraction
from typing import Any, Iterable, Sequence

import sympy as sp


HALF = sp.Rational(1, 2)


def parse_coord(x: Any) -> sp.Expr:
    if isinstance(x, sp.Expr):
        return sp.simplify(sp.nsimplify(x, rational=True))
    if isinstance(x, (int, sp.Integer)):
        return sp.Integer(x)
    if isinstance(x, Fraction):
        return sp.Rational(x.numerator, x.denominator)
    if isinstance(x, float):
        raise ValueError("float coordinates are not allowed in the exact verifier")
    if isinstance(x, str):
        expr = sp.sympify(x, evaluate=True)
        return sp.simplify(expr)
    if isinstance(x, dict) and "num" in x and "den" in x:
        return sp.Rational(x["num"], x["den"])
    raise TypeError(f"unsupported coordinate type {type(x)!r}: {x!r}")


def parse_vector(v: Sequence[Any]) -> tuple[sp.Expr, ...]:
    return tuple(parse_coord(c) for c in v)


def inner(u: Sequence[sp.Expr], v: Sequence[sp.Expr]) -> sp.Expr:
    if len(u) != len(v):
        raise ValueError("dimension mismatch")
    s = sp.Integer(0)
    for a, b in zip(u, v):
        s += a * b
    return sp.expand(s)


def norm2(v: Sequence[sp.Expr]) -> sp.Expr:
    return inner(v, v)


def vectors_equal(u: Sequence[sp.Expr], v: Sequence[sp.Expr]) -> bool:
    return all(sp.expand(a - b) == 0 for a, b in zip(u, v))


def leq_half_unit_inner(ip: sp.Expr) -> bool:
    """Return True iff ip <= 1/2 exactly."""
    diff = sp.simplify(sp.expand(HALF - ip))
    if diff == 0:
        return True
    # Positive iff 1/2 - ip >= 0.
    try:
        return bool(sp.ask(sp.Q.nonnegative(diff)))
    except Exception:
        pass
    # Fall back to isolating radicals by comparing to zero via squared
    # minimal polynomial sign when the expression is real algebraic.
    num = sp.N(diff, 80)
    if num.is_number and abs(float(num)) < 1e-40:
        # Ambiguous numerically — force algebraic comparison.
        return sp.simplify(diff) == 0
    # Exact: rewrite as a real algebraic and test the sign of a primitive element.
    rdiff = sp.simplify(diff)
    if rdiff.is_rational:
        return rdiff >= 0
    # Use exact comparison of algebraic number to 0.
    try:
        alg = sp.AlgebraicNumber(rdiff)
        # AlgebraicNumber comparison to 0 via isolating interval.
        return alg >= 0
    except (ValueError, NotImplementedError, TypeError):
        # Last resort: evaluate all conjugates / use evalf with huge precision
        # only as a sanity check, then require the simplified difference to
        # have a proven nonnegative minimal polynomial evaluation.
        val = sp.simplify(rdiff.evalf(200))
        if val.is_number and val > 1e-50:
            # Not a proof. Refuse.
            raise ValueError(
                f"could not prove {ip} <= 1/2 exactly (diff={rdiff})"
            )
        if val.is_number and val < -1e-50:
            return False
        raise ValueError(f"could not decide sign of {rdiff}")


def verify_unit_vectors(
    vectors: Sequence[Sequence[Any]],
    *,
    dim: int | None = None,
    max_inner: Any = HALF,
) -> dict[str, Any]:
    """Verify a spherical code of unit vectors with max inner product <= max_inner."""
    pts = [parse_vector(v) for v in vectors]
    n = len(pts)
    if n == 0:
        raise ValueError("empty configuration")
    d = len(pts[0])
    if dim is not None and d != dim:
        raise ValueError(f"expected dimension {dim}, got {d}")
    if any(len(v) != d for v in pts):
        raise ValueError("ragged vectors")

    bound = parse_coord(max_inner)
    diag_ok = True
    off_ok = True
    max_off = None
    n_tight = 0
    n_equal = 0
    distinct = True

    for i, v in enumerate(pts):
        n2 = sp.simplify(sp.expand(norm2(v) - 1))
        if n2 != 0:
            diag_ok = False
            return {
                "ok": False,
                "reason": f"vector {i} has norm^2 = {sp.simplify(norm2(v))} != 1",
                "count": n,
                "dim": d,
            }

    seen: dict[tuple, int] = {}
    for i, v in enumerate(pts):
        key = tuple(sp.expand(c) for c in v)
        if key in seen:
            distinct = False
            n_equal += 1
        else:
            seen[key] = i

    worst = None
    worst_pair = None
    for i in range(n):
        for j in range(i + 1, n):
            ip = sp.simplify(sp.expand(inner(pts[i], pts[j])))
            gap = sp.simplify(sp.expand(bound - ip))
            if gap == 0:
                n_tight += 1
            elif not _nonneg(gap):
                off_ok = False
                return {
                    "ok": False,
                    "reason": f"inner product vectors {i},{j} = {ip} > {bound}",
                    "count": n,
                    "dim": d,
                    "violating_inner": str(ip),
                }
            if worst is None or _strict_gt(ip, worst):
                worst = ip
                worst_pair = (i, j)

    ok = diag_ok and off_ok and distinct
    return {
        "ok": ok,
        "count": n,
        "dim": d,
        "distinct": distinct,
        "n_duplicate_pairs": n_equal,
        "n_tight_pairs": n_tight,
        "max_offdiag": str(worst) if worst is not None else None,
        "max_offdiag_pair": worst_pair,
        "bound": str(bound),
        "all_unit": diag_ok,
        "all_offdiag_leq_bound": off_ok,
    }


def _nonneg(expr: sp.Expr) -> bool:
    expr = sp.simplify(sp.expand(expr))
    if expr == 0:
        return True
    if expr.is_rational:
        return expr >= 0
    try:
        if bool(sp.ask(sp.Q.nonnegative(expr))):
            return True
        if bool(sp.ask(sp.Q.negative(expr))):
            return False
    except Exception:
        pass
    try:
        return sp.AlgebraicNumber(expr) >= 0
    except Exception:
        pass
    # Quadratic a + b*sqrt(k): decide by casework.
    decided = _nonneg_quadratic(expr)
    if decided is not None:
        return decided
    raise ValueError(f"cannot decide nonnegativity of {expr} in exact arithmetic")


def _strict_gt(a: sp.Expr, b: sp.Expr) -> bool:
    return not _nonneg(sp.simplify(sp.expand(b - a)))


def _nonneg_quadratic(expr: sp.Expr) -> bool | None:
    """Decide a + b*sqrt(k) >= 0 for squarefree k > 0, a,b rational."""
    expr = sp.expand(expr)
    sqrts = list(expr.atoms(sp.Pow, sp.sqrt))
    radicals = []
    for s in expr.atoms(sp.Pow):
        if s.exp == sp.Rational(1, 2):
            radicals.append(s)
    radicals += [z for z in expr.atoms(sp.sqrt)]
    radicals = list({sp.powsimp(r) for r in radicals})
    if len(radicals) != 1:
        return None
    r = radicals[0]
    k = sp.simplify(r**2)
    if not k.is_rational or k <= 0:
        return None
    a = sp.simplify(expr.subs(r, 0))
    b = sp.simplify((expr - a) / r)
    if not a.is_rational or not b.is_rational:
        return None
    # a + b sqrt(k) >= 0
    if b == 0:
        return a >= 0
    if b > 0:
        if a >= 0:
            return True
        return a * a <= b * b * k
    # b < 0
    if a < 0:
        return False
    return a * a >= b * b * k


def verify_equal_norm(
    vectors: Sequence[Sequence[Any]],
    *,
    dim: int | None = None,
) -> dict[str, Any]:
    """Equal-norm kissing: 2 <vi,vj> <= ||vi||^2 for i!=j, all norms equal, distinct.

    Equivalent to unit inner products <= 1/2 after scaling.
    """
    pts = [parse_vector(v) for v in vectors]
    n = len(pts)
    d = len(pts[0])
    if dim is not None and d != dim:
        raise ValueError(f"expected dimension {dim}, got {d}")
    n2s = [sp.simplify(sp.expand(norm2(v))) for v in pts]
    n0 = n2s[0]
    if n0 == 0:
        return {"ok": False, "reason": "zero vector", "count": n, "dim": d}
    for i, n2 in enumerate(n2s):
        if sp.simplify(n2 - n0) != 0:
            return {
                "ok": False,
                "reason": f"unequal norms: ||v0||^2={n0}, ||v{i}||^2={n2}",
                "count": n,
                "dim": d,
            }

    seen: dict[tuple, int] = {}
    for i, v in enumerate(pts):
        key = tuple(sp.expand(c) for c in v)
        if key in seen:
            return {
                "ok": False,
                "reason": f"duplicate vectors {seen[key]} and {i}",
                "count": n,
                "dim": d,
            }
        seen[key] = i

    n_tight = 0
    worst = None
    worst_pair = None
    half_n0 = sp.simplify(n0 / 2)
    for i in range(n):
        for j in range(i + 1, n):
            ip = sp.simplify(sp.expand(inner(pts[i], pts[j])))
            gap = sp.simplify(sp.expand(half_n0 - ip))
            if gap == 0:
                n_tight += 1
            elif not _nonneg(gap):
                return {
                    "ok": False,
                    "reason": (
                        f"inner product vectors {i},{j} = {ip} > half-norm {half_n0}"
                    ),
                    "count": n,
                    "dim": d,
                    "violating_inner": str(ip),
                    "norm2": str(n0),
                }
            if worst is None or _strict_gt(ip, worst):
                worst = ip
                worst_pair = (i, j)

    unit_max = sp.simplify(worst / n0) if worst is not None else None
    return {
        "ok": True,
        "count": n,
        "dim": d,
        "distinct": True,
        "n_tight_pairs": n_tight,
        "norm2": str(n0),
        "max_offdiag_unnormalized": str(worst),
        "max_offdiag_unit": str(unit_max),
        "max_offdiag_pair": worst_pair,
        "bound_unnormalized": str(half_n0),
        "all_unit_after_scale": True,
        "all_offdiag_leq_bound": True,
    }


def to_unit_strings(vectors: Sequence[Sequence[Any]]) -> list[list[str]]:
    pts = [parse_vector(v) for v in vectors]
    n0 = sp.simplify(norm2(pts[0]))
    scale = sp.sqrt(n0)
    out = []
    for v in pts:
        out.append([str(sp.simplify(c / scale)) for c in v])
    return out


def load_config(path: str) -> dict[str, Any]:
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def verify_config_file(path: str) -> dict[str, Any]:
    cfg = load_config(path)
    vecs = cfg["vectors"]
    dim = cfg.get("dimension")
    if cfg.get("unit"):
        result = verify_unit_vectors(vecs, dim=dim)
    else:
        result = verify_equal_norm(vecs, dim=dim)
    result["path"] = path
    result["method"] = cfg.get("method")
    return result


def main(argv: Iterable[str] | None = None) -> int:
    p = argparse.ArgumentParser(description="Exact kissing-number verifier")
    p.add_argument("config", help="JSON configuration with exact coordinates")
    args = p.parse_args(list(argv) if argv is not None else None)
    result = verify_config_file(args.config)
    print(json.dumps(result, indent=2, default=str))
    if not result.get("ok"):
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
