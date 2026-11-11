#!/usr/bin/env python3
"""Try to add extra unit vectors inside one R^6 factor of the 840 construction.

Bridge constraints force |w_i| + |w_j| <= 1 for all coordinate pairs, hence
||w||_∞ <= 1/2 after also requiring |w · (±e_k)| <= 1/2.
"""

from __future__ import annotations

import itertools
import json
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from constructions.clebsch840 import sixty_block, triple_to_float

ROOT = Path(__file__).resolve().parent.parent
HALF = 0.5
TOL = 1e-9


def sixty_float() -> np.ndarray:
    return np.array([[triple_to_float(c) for c in v] for v in sixty_block()], dtype=np.float64)


def scan_pattern(B: np.ndarray, mag: np.ndarray, name: str) -> int:
    """mag is length-6 nonnegative; all signs and all placements? mag is a fixed placement."""
    nz = np.where(mag > 0)[0]
    k = len(nz)
    hits = 0
    n = 0
    signs = itertools.product((-1.0, 1.0), repeat=k)
    for s in signs:
        n += 1
        w = np.zeros(6)
        w[nz] = np.array(s) * mag[nz]
        if abs(np.linalg.norm(w) - 1) > 1e-8:
            continue
        if np.max(np.abs(w)) > HALF + TOL:
            continue
        mx = float(np.max(B @ w))
        if mx <= HALF + TOL and mx < 1 - 1e-8:
            hits += 1
    print(f"  {name}: n={n} hits={hits}")
    return hits


def main() -> None:
    t0 = time.time()
    B = sixty_float()
    print("6D block", B.shape, "maxIP", float(((B @ B.T) - np.eye(60)).max()))

    hits = {}
    # equal-weight in 6D
    for wt, val, name in (
        (4, 0.5, "6d_wt4_half"),
        (6, 6 ** -0.5, "6d_wt6_1/sqrt6"),
        (5, 5 ** -0.5, "6d_wt5"),
        (3, 3 ** -0.5, "6d_wt3"),
    ):
        h = 0
        for supp in itertools.combinations(range(6), wt):
            mag = np.zeros(6)
            mag[list(supp)] = val
            h += scan_pattern(B, mag, f"{name}_{supp}")
        hits[name] = h

    # mixed: two halves + four 1/√8
    h = 0
    a = 8 ** -0.5
    for p, q in itertools.combinations(range(6), 2):
        mag = np.full(6, a)
        mag[p] = mag[q] = 0.5
        h += scan_pattern(B, mag, f"two_half_four_1sqrt8_{p}{q}")
    hits["two_half_four_1sqrt8"] = h

    # random unit vectors in the cube
    rng = np.random.default_rng(0)
    rand_hits = 0
    tried = 0
    best = 1.0
    for _ in range(200_000):
        x = rng.normal(size=6)
        x /= np.linalg.norm(x)
        if np.max(np.abs(x)) > HALF + TOL:
            continue
        tried += 1
        mx = float(np.max(B @ x))
        if mx < best:
            best = mx
        if mx <= HALF + TOL:
            rand_hits += 1
    hits["random_in_cube"] = rand_hits
    print("random accepted (l_inf<=1/2)", tried, "hits", rand_hits, "best_mx", best)

    report = {
        "hits": hits,
        "random_tried_in_cube": tried,
        "random_best_maxIP": best,
        "seconds": round(time.time() - t0, 2),
        "note": "Zero hits means the 60-point 6D block cannot be enlarged inside the bridge cube.",
    }
    (ROOT / "configs" / "add_in_6d.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))
    with (ROOT / "progress.log").open("a") as f:
        f.write(
            f"{time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime())} "
            f"ADD_IN_6D {json.dumps(hits)} best_random={best:.6f}\n"
        )


if __name__ == "__main__":
    main()
