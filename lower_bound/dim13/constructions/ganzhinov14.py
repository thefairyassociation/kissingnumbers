"""Ganzhinov 1932-point kissing configuration in R^14.

Realification of C^7 as in arXiv:2207.08266 §5.5:

  Φ3 ∪ ((1+i)/√2) Φ2 ∪ ((1-i)/√2) Φ2 ∪ Φ4 ∪ i Φ4

with |Φ3|=1512, |Φ2|=126 (E7 roots), |Φ4|=84 (D7 roots).

Integer model (scale 2√2): vectors in Z^14 of squared length 8.
Unit inner product ≤ 1/2 iff integer inner product ≤ 4.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from e7_frames import d7_roots_half, e7_roots_half, orthogonal_frames  # noqa: E402


def _unique_rows(a: np.ndarray) -> np.ndarray:
    return np.unique(a.astype(np.int8, copy=False), axis=0)


def ganzhinov_1932() -> np.ndarray:
    roots = e7_roots_half()  # (126, 7) sq-norm 4
    frames = orthogonal_frames(roots)
    d7 = d7_roots_half()  # (84, 7) sq-norm 4

    parts: list[np.ndarray] = []

    # Φ3: (1/√2)(v + i w) for distinct axes v,w of one frame, with signs.
    # Integer: concat(s * axis_v, t * axis_w), s,t = ±1.
    phi3 = []
    for fr in frames:
        for i in range(7):
            for j in range(7):
                if i == j:
                    continue
                v = fr[i]
                w = fr[j]
                for s in (1, -1):
                    for t in (1, -1):
                        phi3.append(np.concatenate([s * v, t * w]))
    phi3 = _unique_rows(np.stack(phi3, axis=0))
    if phi3.shape[0] != 1512:
        raise RuntimeError(f"Φ3 size {phi3.shape[0]} != 1512")
    parts.append(phi3)

    # ((1+i)/√2) Φ2 and ((1-i)/√2) Φ2: concat(u, u) and concat(u, -u).
    phi2_phases = []
    for u in roots:
        phi2_phases.append(np.concatenate([u, u]))
        phi2_phases.append(np.concatenate([u, -u]))
    phi2_phases = _unique_rows(np.stack(phi2_phases, axis=0))
    if phi2_phases.shape[0] != 252:
        raise RuntimeError(f"Φ2 phases size {phi2_phases.shape[0]} != 252")
    parts.append(phi2_phases)

    # Φ4 in first R^7 and i Φ4 in second R^7.
    zeros = np.zeros((d7.shape[0], 7), dtype=np.int8)
    phi4 = _unique_rows(np.concatenate([d7, zeros], axis=1))
    iphi4 = _unique_rows(np.concatenate([zeros, d7], axis=1))
    if phi4.shape[0] != 84 or iphi4.shape[0] != 84:
        raise RuntimeError("Φ4 sizes wrong")
    parts.append(phi4)
    parts.append(iphi4)

    pts = _unique_rows(np.concatenate(parts, axis=0))
    if pts.shape[0] != 1932:
        raise RuntimeError(f"total size {pts.shape[0]} != 1932")
    sq = np.sum(pts.astype(np.int32) ** 2, axis=1)
    if not np.all(sq == 8):
        raise RuntimeError("not all squared lengths are 8")
    return pts


def d14_type22() -> np.ndarray:
    """All 364 vectors in Z^14 of type (±2,±2,0^12)."""
    pts = []
    for i in range(14):
        for j in range(i + 1, 14):
            for si in (2, -2):
                for sj in (2, -2):
                    v = np.zeros(14, dtype=np.int8)
                    v[i] = si
                    v[j] = sj
                    pts.append(v)
    arr = np.stack(pts, axis=0)
    assert arr.shape[0] == 364
    return arr


if __name__ == "__main__":
    pts = ganzhinov_1932()
    t22 = d14_type22()
    # How many type-(2,2) vs weight-8.
    w = np.sum(np.abs(pts) == 1, axis=1)
    n_w8 = int(np.sum(w == 8))
    n_22 = int(np.sum(np.sum(np.abs(pts) == 2, axis=1) == 2))
    print(f"Ganzhinov 1932: n={pts.shape[0]} type22={n_22} weight8={n_w8}")
    # Containment of all D14 type-22?
    tset = set(map(tuple, t22.tolist()))
    pset = set(map(tuple, pts.tolist()))
    missing = tset - pset
    print(f"D14 type-22 contained: {len(missing) == 0} (missing {len(missing)})")
