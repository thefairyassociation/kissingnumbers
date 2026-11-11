"""E7 roots in R^7 (norm 1) and their 9 cross-polytope frames.

Coordinates follow Ganzhinov, Highly symmetric lines, arXiv:2207.08266 / LAA 722 (2025):
B0 = all signed standard-basis vectors;
remaining roots are ±1/2 on the 4-set complements of the Fano plane 2-(7,3,1).
The 126 roots partition into 9 orthogonal frames of 14 vectors each.
"""

from __future__ import annotations

from itertools import product

import numpy as np

FANO_LINES = (
    (0, 1, 3),
    (0, 2, 6),
    (0, 4, 5),
    (1, 2, 4),
    (1, 5, 6),
    (2, 3, 5),
    (3, 4, 6),
)


def fano_complements() -> list[tuple[int, int, int, int]]:
    out = []
    universe = set(range(7))
    for line in FANO_LINES:
        comp = tuple(sorted(universe.difference(line)))
        assert len(comp) == 4
        out.append(comp)
    return out


def e7_roots_half() -> np.ndarray:
    """126 roots as integer vectors of squared length 4 (twice usual unit roots).

    B0: ±2 e_i  (squared length 4).
    4-support: ±1 on a Fano complement (squared length 4).
    Unit E7 roots are these vectors / 2.
    """
    roots = []
    for i in range(7):
        v = np.zeros(7, dtype=np.int8)
        v[i] = 2
        roots.append(v.copy())
        v[i] = -2
        roots.append(v.copy())
    for comp in fano_complements():
        for signs in product((-1, 1), repeat=4):
            v = np.zeros(7, dtype=np.int8)
            for s, j in zip(signs, comp):
                v[j] = s
            roots.append(v)
    arr = np.unique(np.stack(roots, axis=0), axis=0)
    if arr.shape[0] != 126:
        raise RuntimeError(f"expected 126 E7 roots, got {arr.shape[0]}")
    return arr


def _positive_rep(v: np.ndarray) -> np.ndarray:
    """Choose a hemisphere representative (first nonzero coordinate > 0)."""
    for x in v:
        if x != 0:
            return v if x > 0 else -v
    raise ValueError("zero vector")


def orthogonal_frames(roots: np.ndarray) -> list[np.ndarray]:
    """Partition the 126 roots into 9 frames; each frame is (7,7) positive axes.

    Each row of a frame is a positive-hemisphere root; the full cross-polytope
    is {±row}. Rows are pairwise orthogonal (integer inner product 0) and each
    has squared length 4.
    """
    pos = []
    seen = set()
    for v in roots:
        r = tuple(int(x) for x in _positive_rep(v))
        if r not in seen:
            seen.add(r)
            pos.append(np.array(r, dtype=np.int16))
    pos = np.stack(pos, axis=0)
    if pos.shape[0] != 63:
        raise RuntimeError(f"expected 63 positive roots, got {pos.shape[0]}")

    dots = pos @ pos.T
    unused = set(range(63))
    frames: list[np.ndarray] = []

    def clique_from(start: int, pool: set[int]) -> list[int] | None:
        clique = [start]
        candidates = [j for j in pool if j != start and dots[start, j] == 0]
        while len(clique) < 7:
            best = None
            best_deg = -1
            for j in candidates:
                deg = sum(1 for k in candidates if dots[j, k] == 0)
                if deg > best_deg:
                    best_deg = deg
                    best = j
            if best is None:
                return None
            clique.append(best)
            candidates = [j for j in candidates if j != best and dots[best, j] == 0]
        return clique

    # Prefer the coordinate frame if present.
    b0_idx = []
    for i, v in enumerate(pos):
        if int(np.count_nonzero(v)) == 1:
            b0_idx.append(i)
    if len(b0_idx) == 7:
        frames.append(pos[np.array(b0_idx)])
        unused.difference_update(b0_idx)

    while unused:
        order = sorted(unused, key=lambda i: -sum(1 for j in unused if i != j and dots[i, j] == 0))
        found = None
        for start in order:
            found = clique_from(start, unused)
            if found:
                break
        if not found:
            raise RuntimeError(f"failed to complete frames; unused={len(unused)}")
        frames.append(pos[np.array(found)])
        unused.difference_update(found)

    if len(frames) != 9:
        raise RuntimeError(f"expected 9 frames, got {len(frames)}")
    for fr in frames:
        g = fr @ fr.T
        if not np.array_equal(g, 4 * np.eye(7, dtype=g.dtype)):
            raise RuntimeError("frame is not orthogonal of squared length 4")
    return frames


def d7_roots_half() -> np.ndarray:
    """84 D7 roots as integer vectors of squared length 4: perms of (±2,±2,0^5)."""
    roots = []
    for i in range(7):
        for j in range(i + 1, 7):
            for si, sj in product((-2, 2), repeat=2):
                v = np.zeros(7, dtype=np.int8)
                v[i] = si
                v[j] = sj
                roots.append(v)
    arr = np.stack(roots, axis=0)
    if arr.shape[0] != 84:
        raise RuntimeError(f"expected 84 D7 roots, got {arr.shape[0]}")
    return arr
