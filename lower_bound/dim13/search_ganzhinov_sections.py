#!/usr/bin/env python3
"""13-dimensional sections of Ganzhinov's 1932-point code in R^14.

For a vector n in R^14, the set {v in C : <v,n>=0} lies in n^perp ≅ R^13
and inherits the kissing condition. We scan:
  - n ranging over C itself (code equators)
  - n ranging over small integer vectors (coordinate and pairing hyperplanes)
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "constructions"))
sys.path.insert(0, str(ROOT))

from ganzhinov14 import ganzhinov_1932  # noqa: E402
from integer_kiss import verify_integer_equal_norm  # noqa: E402
from io_config import log_progress, save_config, maybe_update_best  # noqa: E402


def equator(pts: np.ndarray, n: np.ndarray) -> np.ndarray:
    ip = pts.astype(np.int32) @ n.astype(np.int32)
    return pts[ip == 0]


def span_rank(X: np.ndarray) -> int:
    if len(X) == 0:
        return 0
    return int(np.linalg.matrix_rank(X.astype(np.float64), tol=1e-8))


def save_if_ok(pts13: np.ndarray, method: str, notes: str, *, save: bool = False) -> dict:
    r = verify_integer_equal_norm(pts13)
    rnk = span_rank(pts13)
    r["span_rank"] = rnk
    status = "pass" if r["ok"] and rnk == 13 else ("fail" if not r["ok"] else "pass-lower-rank")
    log_progress(
        method,
        13,
        int(r["count"]),
        status,
        notes + f" rank={rnk} max_ip={r.get('max_offdiag_unnormalized')}",
    )
    if save and r["ok"] and rnk == 13:
        vecs = [[str(int(x)) for x in row] for row in pts13]
        path = save_config(
            dimension=13,
            count=int(r["count"]),
            vectors=vecs,
            max_off_diagonal=str(r["max_offdiag_unit"]),
            method=method,
            unit=False,
            extra={"norm2": r["norm2"], "notes": notes, "span_rank": rnk},
            filename=f"ganzhinov_section_{r['count']}_{method.split('-')[1]}.json",
            verified=True,
            verifier=r,
        )
        maybe_update_best(path)
    return r


def main() -> None:
    pts = ganzhinov_1932()
    P = pts.astype(np.int32)
    G = P @ P.T
    print("Ganzhinov 1932 built", pts.shape, "max_ip", int(G.max()), "diag", int(np.diag(G).min()), int(np.diag(G).max()))
    # self-verify 14D
    from integer_kiss import verify_integer_equal_norm as v
    r14 = v(pts)
    print("14D self-check", r14)
    log_progress("ganzhinov-1932-selfcheck", 14, 1932, "pass" if r14["ok"] else "fail", json.dumps(r14))

    n0s = (G == 0).sum(axis=1)
    print("equator sizes vs code vectors: min", int(n0s.min()), "max", int(n0s.max()), "mean", float(n0s.mean()))
    best_i = int(np.argmax(n0s))
    eq = pts[G[best_i] == 0]
    save_if_ok(eq, "ganzhinov-code-equator", f"largest equator of a code vector, idx={best_i}", save=True)

    uniq, cnt = np.unique(n0s, return_counts=True)
    print("equator size histogram", dict(zip(uniq.tolist(), cnt.tolist())))

    best_eq = None
    best_meta = None
    for k in range(14):
        n = np.zeros(14, dtype=np.int32)
        n[k] = 1
        eqk = equator(pts, n)
        if best_eq is None or len(eqk) > len(best_eq):
            best_eq, best_meta = eqk, ("coord", k, n)
    save_if_ok(best_eq, "ganzhinov-coord-hyperplane", f"best x_i=0 is i={best_meta[1]} |eq|={len(best_eq)}", save=True)
    print("best coordinate hyperplane", len(best_eq), best_meta[1])

    best_eq = None
    best_meta = None
    for i in range(14):
        for j in range(i + 1, 14):
            for s in (1, -1):
                n = np.zeros(14, dtype=np.int32)
                n[i] = 1
                n[j] = s
                eqk = equator(pts, n)
                if best_eq is None or len(eqk) > len(best_eq):
                    best_eq, best_meta = eqk, (i, j, s)
    save_if_ok(
        best_eq,
        "ganzhinov-pair-hyperplane",
        f"best x_{best_meta[0]}{'+' if best_meta[2]==1 else '-'}x_{best_meta[1]}=0 |eq|={len(best_eq)}",
        save=True,
    )
    print("best pairing hyperplane", len(best_eq), best_meta)

    best_eq = None
    best_n = None
    tried = 0
    for a in range(14):
        for b in range(a + 1, 14):
            for c in range(b + 1, 14):
                for sa, sb, sc in ((1, 1, 1), (1, 1, -1), (1, -1, 1), (1, -1, -1)):
                    n = np.zeros(14, dtype=np.int32)
                    n[a], n[b], n[c] = sa, sb, sc
                    eqk = equator(pts, n)
                    tried += 1
                    if span_rank(eqk) != 13:
                        continue
                    if best_eq is None or len(eqk) > len(best_eq):
                        best_eq, best_n = eqk, n.copy()
    if best_eq is not None:
        print("best wt-3 normal", len(best_eq), best_n.tolist())
        save_if_ok(best_eq, "ganzhinov-wt3-hyperplane", f"n={best_n.tolist()}", save=True)
    print("wt3 tried", tried)


if __name__ == "__main__":
    main()
