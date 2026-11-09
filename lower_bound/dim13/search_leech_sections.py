#!/usr/bin/env python3
"""13-dimensional coordinate and pairing sections of Leech minima."""

from __future__ import annotations

import sys
from itertools import combinations
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "constructions"))
sys.path.insert(0, str(ROOT))

from leech import all_leech_minima, golay24_codewords  # noqa: E402
from integer_kiss import verify_integer_equal_norm  # noqa: E402
from io_config import log_progress, save_config, maybe_update_best  # noqa: E402


def span_rank(X: np.ndarray) -> int:
    if len(X) == 0:
        return 0
    return int(np.linalg.matrix_rank(X.astype(np.float64), tol=1e-8))


def save_section(pts: np.ndarray, method: str, notes: str) -> dict:
    r = verify_integer_equal_norm(pts)
    r["span_rank"] = span_rank(pts)
    status = "pass" if r["ok"] and r["span_rank"] == 13 else ("fail" if not r["ok"] else "pass-lower-rank")
    log_progress(method, 13, int(r["count"]), status, notes + f" rank={r['span_rank']}")
    if r["ok"] and r["span_rank"] == 13 and r["count"] >= 500:
        vecs = [[str(int(x)) for x in row] for row in pts]
        path = save_config(
            dimension=13,
            count=int(r["count"]),
            vectors=vecs,
            max_off_diagonal=str(r["max_offdiag_unit"]),
            method=method,
            unit=False,
            extra={"norm2": r["norm2"], "notes": notes, "span_rank": r["span_rank"]},
            filename=f"leech_section_{r['count']}.json",
            verified=True,
            verifier={k: (int(v) if isinstance(v, (np.integer,)) else v) for k, v in r.items()},
        )
        maybe_update_best(path)
    return r


def coordinate_section(M: np.ndarray, keep: list[int]) -> np.ndarray:
    rest = [i for i in range(24) if i not in keep]
    mask = np.all(M[:, rest] == 0, axis=1)
    return M[mask][:, keep]


def main() -> None:
    print("building Leech minima...")
    M = all_leech_minima()
    print("Leech minima", M.shape, "norm2 unique", np.unique(np.sum(M.astype(np.int32) ** 2, axis=1)))
    log_progress("leech-minima-build", 24, int(M.shape[0]), "pass", f"shape={M.shape}")

    # All 13-subsets of coordinates is C(24,13)=2.7e6, too many.
    # Scan: all 13-sets containing 0..k for structured blocks, plus random sample,
    # plus complements of Steiner 11-sets, plus first 13, last 13, etc.
    C = golay24_codewords()
    wt = C.sum(axis=1)
    octads = C[wt == 8]
    print("octads", len(octads))

    candidates = []
    candidates.append(list(range(13)))
    candidates.append(list(range(11, 24)))
    candidates.append(list(range(0, 24, 2))[:13] if False else sorted(list(range(0, 24, 2)) + [1])[:13])
    # 13-sets that contain a given octad (octad + 5 extra)
    rng = np.random.default_rng(13)
    for oc in octads[:: max(1, len(octads) // 40)]:
        pos = np.flatnonzero(oc).tolist()
        extra = [i for i in range(24) if i not in pos]
        # all C(16,5) is 4368 * 40 = too many; take a few
        for _ in range(8):
            add = rng.choice(extra, size=5, replace=False).tolist()
            candidates.append(sorted(pos + add))

    # complements of 11-sets that are Golay supports of weight 11? Golay has no wt 11.
    # random 13-sets
    for _ in range(200):
        candidates.append(sorted(rng.choice(24, size=13, replace=False).tolist()))

    best = None
    best_keep = None
    seen = set()
    for keep in candidates:
        key = tuple(keep)
        if key in seen:
            continue
        seen.add(key)
        sec = coordinate_section(M, keep)
        if best is None or len(sec) > best:
            best = len(sec)
            best_keep = keep
            best_sec = sec
            print("new best coord section", best, "keep", keep[:8], "...")
    print("BEST coordinate 13-section", best, "keep", best_keep)
    save_section(best_sec, "leech-coord-13-section", f"keep={best_keep}")

    # Pairing constraints: 11 pairings among 22 coords, leftover 2 coords
    # (x_{2i} = x_{2i+1}) for i=0..10, keep 13 free... that's 11 constraints -> 13D
    # Count minima with x0=x1, x2=x3, ..., x20=x21 (x22,x23 free) — 11 constraints, 13 dof.
    def pairing_section(pairs: list[tuple[int, int]], signs: list[int]) -> np.ndarray:
        # constraints x_a = s x_b for each pair; remaining coords free
        used = {i for p in pairs for i in p}
        free = [i for i in range(24) if i not in used]
        # 11 pairs => 22 used, 2 free, dim = 11+2 = 13
        mask = np.ones(len(M), dtype=bool)
        for (a, b), s in zip(pairs, signs):
            mask &= M[:, a] == s * M[:, b]
        sub = M[mask]
        # map to R^13: one coord per pair (the a-value) plus the free coords
        cols = [a for a, _ in pairs] + free
        return sub[:, cols]

    # Standard pairing  (0,1),(2,3),...,(20,21)
    pairs0 = [(2 * i, 2 * i + 1) for i in range(11)]
    for signs in (
        [1] * 11,
        [-1] + [1] * 10,
        [1, -1] * 5 + [1],
        [-1] * 11,
    ):
        sec = pairing_section(pairs0, signs)
        print("pairing", signs[:4], "...", "n", len(sec), "rank", span_rank(sec))
        save_section(sec, "leech-pairing-13", f"pairs={pairs0} signs={signs}")

    # Random pairings
    for t in range(30):
        perm = rng.permutation(24)
        pairs = [(int(perm[2 * i]), int(perm[2 * i + 1])) for i in range(11)]
        signs = [int(rng.choice([-1, 1])) for _ in range(11)]
        sec = pairing_section(pairs, signs)
        if len(sec) >= (best or 0):
            print("random pairing n", len(sec), "rank", span_rank(sec))
            save_section(sec, "leech-random-pairing-13", f"pairs={pairs} signs={signs}")
            if len(sec) > (best or 0) and span_rank(sec) == 13:
                best = len(sec)


if __name__ == "__main__":
    main()
