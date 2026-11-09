#!/usr/bin/env python3
"""E8 even-weight spinors (128) + optional 8 orts in R^8, plus a 24-cell in R^4,
plus 2+2 half-bridges. All coordinates in Q(sqrt(2))."""

from __future__ import annotations

import itertools
import json
import sys
import time
from pathlib import Path

import numpy as np

_DIM12 = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_DIM12))
from constructions.clebsch840 import (
    cmp_le_half,
    inner_qsqrt2,
    is_one,
    triple_to_sympy_str,
)

ROOT = Path(__file__).resolve().parent.parent
Z = (0, 0, 1)


def e8_spinors() -> list[list[tuple[int, int, int]]]:
    """128 vectors (√2/4)(±1^8) with even number of minuses, padded by 4 zeros."""
    vecs = []
    for bits in range(1 << 8):
        signs = [1 if ((bits >> i) & 1) == 0 else -1 for i in range(8)]
        if signs.count(-1) % 2:
            continue
        v = [Z] * 12
        for i, s in enumerate(signs):
            v[i] = (0, s, 4)  # s * sqrt(2) / 4
        vecs.append(v)
    assert len(vecs) == 128
    return vecs


def orts8() -> list[list[tuple[int, int, int]]]:
    vecs = []
    for i in range(8):
        for s in (1, -1):
            v = [Z] * 12
            v[i] = (s, 0, 1)
            vecs.append(v)
    return vecs


def cell24_last4() -> list[list[tuple[int, int, int]]]:
    vecs = []
    for i in range(4):
        for s in (1, -1):
            v = [Z] * 12
            v[8 + i] = (s, 0, 1)
            vecs.append(v)
    for signs in itertools.product((-1, 1), repeat=4):
        v = [Z] * 12
        for i, s in enumerate(signs):
            v[8 + i] = (s, 0, 2)
        vecs.append(v)
    return vecs


def d8_roots() -> list[list[tuple[int, int, int]]]:
    """112 D8 unit roots (1/√2)(±e_i ± e_j) in first 8 coords."""
    vecs = []
    for i, j in itertools.combinations(range(8), 2):
        for s1, s2 in itertools.product((-1, 1), repeat=2):
            v = [Z] * 12
            v[i] = (0, s1, 2)  # s1 * sqrt(2)/2 = s1/sqrt(2)
            v[j] = (0, s2, 2)
            vecs.append(v)
    assert len(vecs) == 112
    return vecs


def bridges() -> list[list[tuple[int, int, int]]]:
    vecs = []
    for e1 in itertools.combinations(range(8), 2):
        for e2 in itertools.combinations(range(4), 2):
            for signs in itertools.product((-1, 1), repeat=4):
                v = [Z] * 12
                v[e1[0]] = (signs[0], 0, 2)
                v[e1[1]] = (signs[1], 0, 2)
                v[8 + e2[0]] = (signs[2], 0, 2)
                v[8 + e2[1]] = (signs[3], 0, 2)
                vecs.append(v)
    return vecs


def to_float(vecs):
    s2 = 2.0**0.5
    return np.array([[(a + b * s2) / d for a, b, d in v] for v in vecs], np.float64)


def greedy(core, cands):
    X = to_float(core)
    kept = list(core)
    for v in cands:
        fv = to_float([v])[0]
        mx = float(np.max(X @ fv))
        if mx > 0.5 + 1e-10 or mx > 1 - 1e-8:
            continue
        ok = True
        ipv = inner_qsqrt2(v, v)
        if not is_one(ipv):
            continue
        for u in kept:
            if not cmp_le_half(inner_qsqrt2(u, v)):
                ok = False
                break
        if ok:
            kept.append(v)
            X = np.vstack([X, fv])
    return kept


def verify(vecs):
    n = len(vecs)
    for i in range(n):
        if not is_one(inner_qsqrt2(vecs[i], vecs[i])):
            return {"ok": False, "i": i, "count": n}
        for j in range(i + 1, n):
            if not cmp_le_half(inner_qsqrt2(vecs[i], vecs[j])):
                return {"ok": False, "pair": (i, j), "count": n}
    return {"ok": True, "count": n}


def main():
    t0 = time.time()
    spin = e8_spinors()
    print("spinors", len(spin), "self", verify(spin), flush=True)
    c24 = cell24_last4()
    print("24-cell", len(c24), flush=True)
    core = spin + c24
    print("core", verify(core), flush=True)

    br = bridges()
    print("bridge cands", len(br), flush=True)
    kept = greedy(core, br)
    print("core+bridges", len(kept), verify(kept), flush=True)

    kept2 = greedy(kept, orts8())
    print("plus orts8", len(kept2), verify(kept2), flush=True)

    kept3 = greedy(kept2, d8_roots())
    print("plus D8", len(kept3), verify(kept3), flush=True)

    report = {
        "spinors_plus_24cell": 152,
        "plus_bridges": len(kept),
        "plus_orts": len(kept2),
        "plus_D8": len(kept3),
        "seconds": round(time.time() - t0, 2),
    }
    print(report)
    (ROOT / "configs" / "e8_spinor_24cell.json").write_text(json.dumps(report, indent=2) + "\n")
    with (ROOT / "progress.log").open("a") as f:
        f.write(
            f"{time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime())} "
            f"E8_SPINOR_24CELL {json.dumps(report)}\n"
        )


if __name__ == "__main__":
    main()
