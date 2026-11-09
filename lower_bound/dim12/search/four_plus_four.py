#!/usr/bin/env python3
"""Greedy 4+4+4 (three D4 blocks + 2+2 bridges) and related exact searches."""

from __future__ import annotations

import itertools
import json
import sys
import time
from math import gcd
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

ROOT = Path(__file__).resolve().parent.parent
SQRT2 = 2**0.5


def d4_roots(offset: int) -> list[np.ndarray]:
    """24 D4 unit roots in a 4-dimensional coordinate block."""
    vecs = []
    for i, j in itertools.combinations(range(4), 2):
        for s1, s2 in itertools.product((-1.0, 1.0), repeat=2):
            v = np.zeros(12)
            v[offset + i] = s1 / SQRT2
            v[offset + j] = s2 / SQRT2
            vecs.append(v)
    return vecs


def bridges_2_2(off_a: int, off_b: int, same_color_only: bool) -> list[np.ndarray]:
    """±1/2 on two coords in block a and two in block b."""
    K4 = [
        ((0, 1), (2, 3)),
        ((0, 2), (1, 3)),
        ((0, 3), (1, 2)),
    ]
    vecs = []
    if same_color_only:
        pairs = []
        for factor in K4:
            for e1 in factor:
                for e2 in factor:
                    pairs.append((e1, e2))
    else:
        edges = list(itertools.combinations(range(4), 2))
        pairs = [(e1, e2) for e1 in edges for e2 in edges]
    for e1, e2 in pairs:
        for signs in itertools.product((-1.0, 1.0), repeat=4):
            v = np.zeros(12)
            v[off_a + e1[0]] = signs[0] * 0.5
            v[off_a + e1[1]] = signs[1] * 0.5
            v[off_b + e2[0]] = signs[2] * 0.5
            v[off_b + e2[1]] = signs[3] * 0.5
            vecs.append(v)
    return vecs


def max_ip_with(X: np.ndarray, v: np.ndarray) -> float:
    if len(X) == 0:
        return -1.0
    return float(np.max(X @ v))


def greedy_add(core: list[np.ndarray], cands: list[np.ndarray], tol: float = 1e-12) -> np.ndarray:
    X = np.stack(core)
    added = 0
    for v in cands:
        if abs(np.linalg.norm(v) - 1) > 1e-9:
            continue
        if max_ip_with(X, v) <= 0.5 + tol and float(np.max(X @ v)) < 1 - 1e-8:
            X = np.vstack([X, v])
            added += 1
    return X


def exact_d4_and_half_ok(X: np.ndarray) -> bool:
    """Exact check: every vector is D4-root type (Q(sqrt(2))) or 4-sparse halves (Q).

    Inner products: 4 <vi,vj>^2 * nrm_i_unnorm * ... easier via sympy-free
    comparison in Q(sqrt(2)).
    Represent coords as (a, b, d) with a+b√2 over d, d|4.
    """
    triples = []
    for v in X:
        row = []
        ok = True
        for x in v:
            found = None
            for a, b, d in (
                (0, 0, 1),
                (1, 0, 1),
                (-1, 0, 1),
                (1, 0, 2),
                (-1, 0, 2),
                (0, 1, 2),
                (0, -1, 2),
            ):
                val = (a + b * SQRT2) / d
                if abs(x - val) < 1e-10:
                    found = (a, b, d)
                    break
            if found is None:
                ok = False
                break
            row.append(found)
        if not ok:
            return False
        triples.append(row)
    # exact Gram
    n = len(triples)
    for i in range(n):
        for j in range(i, n):
            p = q = 0
            for (a1, b1, d1), (a2, b2, d2) in zip(triples[i], triples[j]):
                s1 = 4 // d1
                s2 = 4 // d2
                A1, B1 = a1 * s1, b1 * s1
                A2, B2 = a2 * s2, b2 * s2
                p += A1 * A2 + 2 * B1 * B2
                q += A1 * B2 + A2 * B1
            # ip = (p + q√2)/16
            if i == j:
                if not (q == 0 and p == 16):
                    return False
            else:
                # (p + q√2)/16 <= 1/2 iff p - 8 + q√2 <= 0
                s = p - 8
                t = q
                if t == 0:
                    if s > 0:
                        return False
                elif t > 0:
                    if s > 0:
                        return False
                    if 2 * t * t > s * s:
                        return False
                else:
                    if s > 0 and 2 * t * t < s * s:
                        return False
    return True


def main() -> None:
    t0 = time.time()
    core = d4_roots(0) + d4_roots(4) + d4_roots(8)
    print("D4*3", len(core), flush=True)

    # same-color bridges
    sc = bridges_2_2(0, 4, True) + bridges_2_2(0, 8, True) + bridges_2_2(4, 8, True)
    Xsc = greedy_add(core, sc)
    print("same-color greedy", len(Xsc), "maxIP", float(((Xsc @ Xsc.T) - np.eye(len(Xsc))).max()), flush=True)

    # all 2+2 bridges
    allb = bridges_2_2(0, 4, False) + bridges_2_2(0, 8, False) + bridges_2_2(4, 8, False)
    Xall = greedy_add(core, allb)
    print("all 2+2 greedy", len(Xall), "maxIP", float(((Xall @ Xall.T) - np.eye(len(Xall))).max()), flush=True)

    # also try adding coordinate ±e_i
    orts = []
    for i in range(12):
        for s in (-1.0, 1.0):
            v = np.zeros(12)
            v[i] = s
            orts.append(v)
    Xort = greedy_add(list(Xall), orts)
    print("plus orts", len(Xort), flush=True)

    # 3-in-one-block + 1 in another: mixed 3+1 halves
    mixed = []
    for oa, ob in ((0, 4), (0, 8), (4, 0), (4, 8), (8, 0), (8, 4)):
        for trip in itertools.combinations(range(4), 3):
            for k in range(4):
                for signs in itertools.product((-1.0, 1.0), repeat=4):
                    v = np.zeros(12)
                    v[oa + trip[0]] = signs[0] * 0.5
                    v[oa + trip[1]] = signs[1] * 0.5
                    v[oa + trip[2]] = signs[2] * 0.5
                    v[ob + k] = signs[3] * 0.5
                    mixed.append(v)
    Xm = greedy_add(list(Xort), mixed)
    print("plus 3+1", len(Xm), flush=True)

    report = {
        "D4x3": 72,
        "same_color_total": int(len(Xsc)),
        "all_2plus2_total": int(len(Xall)),
        "plus_orts": int(len(Xort)),
        "plus_3plus1": int(len(Xm)),
        "seconds": round(time.time() - t0, 2),
    }
    # exact-check the largest
    for name, X in (
        ("same_color", Xsc),
        ("all_2plus2", Xall),
        ("plus_orts", Xort),
        ("plus_3plus1", Xm),
    ):
        ok = exact_d4_and_half_ok(X)
        report[f"{name}_exact"] = ok
        print(name, "exact", ok, "count", len(X), flush=True)

    path = ROOT / "configs" / "four_plus_four_greedy.json"
    path.write_text(json.dumps(report, indent=2) + "\n")
    print(report)
    with (ROOT / "progress.log").open("a") as f:
        f.write(
            f"{time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime())} "
            f"FOUR_PLUS_FOUR {json.dumps(report)}\n"
        )


if __name__ == "__main__":
    main()
