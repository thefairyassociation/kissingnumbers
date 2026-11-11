#!/usr/bin/env python3
"""Exact kissing-arrangement verifier (sympy / Q(sqrt(2)) / rational shells).

A configuration of unit vectors in R^n is a kissing arrangement iff every
off-diagonal Gram entry is <= 1/2. Floats are never used as a certificate.

Usage:
  python3 verify.py path/to/config.json
  python3 verify.py --builtin-840
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import sympy as sp


def parse_entry(s) -> sp.Expr:
    if isinstance(s, (int, float)):
        raise ValueError(f"refusing float/bare-numeric coordinate {s!r}; use exact strings")
    if not isinstance(s, str):
        raise ValueError(f"coordinate must be a string, got {type(s)}")
    expr = sp.sympify(s, evaluate=True)
    if expr.free_symbols:
        raise ValueError(f"unexpected free symbols in {s!r}: {expr.free_symbols}")
    return sp.simplify(expr)


def gram_stats(vectors: list[list[sp.Expr]], ip_max: sp.Expr) -> dict:
    n = len(vectors)
    dim = len(vectors[0])
    half = sp.Rational(1, 2)
    max_off = None
    max_pair = None
    for i, v in enumerate(vectors):
        if len(v) != dim:
            return {"ok": False, "reason": f"vector {i} has length {len(v)} != {dim}"}
        nrm = sp.simplify(sum(c * c for c in v))
        if nrm != 1:
            return {
                "ok": False,
                "reason": f"vector {i} has squared norm {nrm} != 1",
                "count": n,
                "dim": dim,
            }
    for i in range(n):
        vi = vectors[i]
        for j in range(i + 1, n):
            ip = sp.simplify(sum(a * b for a, b in zip(vi, vectors[j])))
            if ip > half:
                return {
                    "ok": False,
                    "reason": f"inner product ({i},{j}) = {ip} > 1/2",
                    "count": n,
                    "dim": dim,
                    "bad_ip": str(ip),
                }
            if max_off is None or ip > max_off:
                max_off = ip
                max_pair = (i, j)
    return {
        "ok": True,
        "count": n,
        "dim": dim,
        "max_offdiag": str(max_off),
        "max_offdiag_pair": list(max_pair) if max_pair else None,
        "ip_max_allowed": str(ip_max),
    }


def load_config(path: Path) -> dict:
    data = json.loads(path.read_text())
    vecs = data["vectors"]
    parsed = [[parse_entry(c) for c in row] for row in vecs]
    return {**data, "_parsed": parsed}


def verify_config_dict(data: dict) -> dict:
    parsed = data["_parsed"]
    t0 = time.time()
    stats = gram_stats(parsed, sp.Rational(1, 2))
    stats["seconds"] = round(time.time() - t0, 4)
    stats["method"] = data.get("method")
    stats["claimed_count"] = data.get("count")
    if stats.get("ok") and stats["count"] != data.get("count"):
        stats["ok"] = False
        stats["reason"] = f"claimed count {data.get('count')} != actual {stats['count']}"
    return stats


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("config", nargs="?", type=Path)
    ap.add_argument("--builtin-840", action="store_true")
    ap.add_argument("--qsqrt2", action="store_true", help="use fast Q(sqrt(2)) verifier")
    args = ap.parse_args()

    if args.builtin_840:
        sys.path.insert(0, str(Path(__file__).resolve().parent))
        from constructions.clebsch840 import construction_840, verify_qsqrt2, vectors_as_strings

        vecs = construction_840()
        t0 = time.time()
        result = verify_qsqrt2(vecs)
        result["seconds"] = round(time.time() - t0, 4)
        result["method"] = "clebsch_48system_K6_bridges"
        print(json.dumps(result, indent=2, default=str))
        return 0 if result.get("ok") else 1

    if args.config is None:
        ap.error("config path or --builtin-840 required")

    data = load_config(args.config)
    if args.qsqrt2:
        sys.path.insert(0, str(Path(__file__).resolve().parent))
        from constructions.clebsch840 import verify_qsqrt2

        # parse strings back through sympy then into floats is forbidden;
        # for generic JSON use sympy path below unless triples are stored.
        result = verify_config_dict(data)
    else:
        result = verify_config_dict(data)
    print(json.dumps(result, indent=2, default=str))
    return 0 if result.get("ok") else 1


if __name__ == "__main__":
    raise SystemExit(main())
