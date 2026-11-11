#!/usr/bin/env python3
"""Exact-arithmetic kissing-number verifier for dimension 14.

Accepts integer (or rational) coordinates. Success requires:
  - N distinct unit vectors in R^14 (after normalization),
  - Gram diagonal identically 1,
  - off-diagonal inner products <= 1/2,
  - N strictly larger than the live record (default 1932),
all checked in exact arithmetic (Python int / fractions, or sympy).
"""

from __future__ import annotations

import argparse
import json
import sys
from fractions import Fraction
from pathlib import Path

import numpy as np


LIVE_RECORD = 1932
LIVE_RECORD_SOURCE = "https://cohn.mit.edu/kissing-numbers"
LIVE_UPPER = 3174


def _as_int_matrix(pts: np.ndarray) -> np.ndarray:
    arr = np.asarray(pts)
    if np.issubdtype(arr.dtype, np.integer):
        return arr.astype(np.int64, copy=False)
    # Try exact integers stored as floats.
    rounded = np.rint(arr)
    if np.allclose(arr, rounded, atol=0, rtol=0) or np.max(np.abs(arr - rounded)) < 1e-12:
        return rounded.astype(np.int64)
    raise TypeError("verifier expects an integer coordinate matrix for the Z-model")


def verify_integer_equal_norm(
    pts: np.ndarray,
    *,
    max_cosine: Fraction = Fraction(1, 2),
    live_record: int = LIVE_RECORD,
) -> dict:
    """Verify equal-norm integer vectors: <x,y> / <x,x> <= max_cosine for x != y."""
    a = _as_int_matrix(pts)
    n, d = a.shape
    if d != 14:
        raise ValueError(f"expected 14 columns, got {d}")

    # Distinctness.
    uniq = np.unique(a, axis=0)
    if uniq.shape[0] != n:
        return {
            "ok": False,
            "reason": f"duplicate vectors: {n} rows, {uniq.shape[0]} unique",
            "n": int(uniq.shape[0]),
            "dim": 14,
        }

    grams = a @ a.T  # int64; entries are small (norm^2 = 8)
    diags = np.diag(grams)
    if np.any(diags <= 0):
        return {"ok": False, "reason": "non-positive squared norms", "n": n, "dim": 14}
    if np.any(diags != diags[0]):
        # Unequal norms: check 2 <x,y> * denom vs norms (cross-shell).
        return verify_integer_mixed_norm(a, max_cosine=max_cosine, live_record=live_record)

    norm2 = int(diags[0])
    # <x,y> <= max_cosine * norm2  for x != y.
    # max_cosine = p/q => q * <x,y> <= p * norm2
    p, q = max_cosine.numerator, max_cosine.denominator
    # cosine <= p/q  iff  q * <x,y> <= p * norm2  (norm2>0).
    thresh = p * norm2
    off = grams.copy()
    np.fill_diagonal(off, -10**9)
    worst = int(off.max())
    viol_i, viol_j = np.where(q * off > thresh)
    n_viol = int(viol_i.size)
    ok_geom = n_viol == 0
    beats = n > live_record and ok_geom
    return {
        "ok": ok_geom,
        "beats_record": beats,
        "n": n,
        "dim": 14,
        "norm2": norm2,
        "max_offdiag_inner": worst,
        "max_cosine_exact": str(Fraction(worst, norm2)),
        "max_allowed_inner": thresh if q == 1 else str(Fraction(thresh, q)),
        "violations": n_viol,
        "live_record": live_record,
        "diagonal_ones_after_normalization": True,
        "distinct": True,
        "arithmetic": "exact integer Gram (Python/numpy int64)",
    }


def verify_integer_mixed_norm(
    a: np.ndarray,
    *,
    max_cosine: Fraction = Fraction(1, 2),
    live_record: int = LIVE_RECORD,
) -> dict:
    """Scale-free exact test: (2 <x,y>)^2 <= <x,x><y,y>  AND  <x,y> <= 0 is ok;
    more generally 2 q <x,y> <= p * 2 sqrt(nx ny) is messy, so use:

        q^2 * <x,y>^2  <= p^2 * nx * ny   when <x,y> > 0
        always ok when <x,y> <= 0
    which is equivalent to cosine <= p/q for positive inner products.
    """
    n = a.shape[0]
    grams = a @ a.T
    norms = np.diag(grams).astype(np.int64)
    p, q = max_cosine.numerator, max_cosine.denominator
    viol = 0
    worst_num = 0
    worst_den = 1
    for i in range(n):
        for j in range(i + 1, n):
            ip = int(grams[i, j])
            if ip <= 0:
                continue
            # q^2 ip^2 <= p^2 ni nj
            if q * q * ip * ip > p * p * int(norms[i]) * int(norms[j]):
                viol += 1
            # track max cosine^2 = ip^2 / (ni nj)
            if ip * ip * worst_den > worst_num * int(norms[i]) * int(norms[j]):
                worst_num = ip * ip
                worst_den = int(norms[i]) * int(norms[j])
    ok_geom = viol == 0
    return {
        "ok": ok_geom,
        "beats_record": n > live_record and ok_geom,
        "n": n,
        "dim": 14,
        "violations": viol,
        "live_record": live_record,
        "max_cosine_squared_exact": str(Fraction(worst_num, worst_den)) if worst_den else None,
        "arithmetic": "exact integer mixed-norm Gram",
        "distinct": True,
    }


def verify_sympy_rational(pts, max_cosine=None):
    """Optional sympy path for Q-coordinates (used if integer path is not applicable)."""
    import sympy as sp

    if max_cosine is None:
        max_cosine = sp.Rational(1, 2)
    M = sp.Matrix(pts)
    n, d = M.shape
    if d != 14:
        raise ValueError("expected 14 columns")
    # Distinct
    rows = [tuple(M.row(i)) for i in range(n)]
    if len(set(rows)) != n:
        return {"ok": False, "reason": "duplicates", "n": n}
    G = sp.simplify(M * M.T)
    viol = 0
    for i in range(n):
        if G[i, i] != 1:
            # try normalize
            pass
    # Normalize rows
    norms = [sp.sqrt(sp.simplify(G[i, i])) for i in range(n)]
    for i in range(n):
        if norms[i] == 0:
            return {"ok": False, "reason": "zero vector", "n": n}
    worst = None
    for i in range(n):
        for j in range(i + 1, n):
            c = sp.simplify(G[i, j] / (norms[i] * norms[j]))
            if c > max_cosine:
                viol += 1
            if worst is None or c > worst:
                worst = c
    ok = viol == 0
    return {
        "ok": ok,
        "beats_record": n > LIVE_RECORD and ok,
        "n": n,
        "dim": 14,
        "violations": viol,
        "max_offdiag_cosine": str(worst),
        "arithmetic": "sympy exact",
        "live_record": LIVE_RECORD,
    }


def load_points(path: Path) -> np.ndarray:
    if path.suffix == ".npy":
        return np.load(path)
    if path.suffix == ".json":
        data = json.loads(path.read_text())
        if isinstance(data, dict) and "points" in data:
            return np.array(data["points"], dtype=np.int64)
        return np.array(data, dtype=np.int64)
    # whitespace / comma separated
    rows = []
    for line in path.read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        line = line.replace(",", " ")
        rows.append([int(x) for x in line.split()])
    return np.array(rows, dtype=np.int64)


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("points", nargs="?", help="path to integer points (.npy/.json/.txt)")
    p.add_argument("--ganzhinov", action="store_true", help="verify constructed Ganzhinov 1932")
    p.add_argument("--live-record", type=int, default=LIVE_RECORD)
    args = p.parse_args(argv)

    if args.ganzhinov:
        sys.path.insert(0, str(Path(__file__).resolve().parent))
        from constructions.ganzhinov import ganzhinov_1932

        pts = ganzhinov_1932()
    elif args.points:
        pts = load_points(Path(args.points))
    else:
        p.error("provide a points file or --ganzhinov")

    result = verify_integer_equal_norm(pts, live_record=args.live_record)
    print(json.dumps(result, indent=2))
    print(f"LIVE RECORD used: {args.live_record} from {LIVE_RECORD_SOURCE} (upper {LIVE_UPPER})")
    if result.get("ok"):
        print(f"GEOM OK: n={result['n']} distinct unit vectors in R^14, off-diagonal inner products <= 1/2.")
        if result.get("beats_record"):
            print("SUCCESS: strictly beats the live lower bound (exact arithmetic).")
        else:
            print("Does not beat the live record (reproducing the record is baseline, not progress).")
        return 0
    print("FAIL: configuration is not a valid kissing arrangement.")
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
