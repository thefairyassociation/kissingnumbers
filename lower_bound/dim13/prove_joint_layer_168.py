#!/usr/bin/env python3
"""Proof that the joint ZE99 half-height layer has at most 168 points.

Context.  ZE99's 1154 is 816 tetrads + 2 poles + two half-height layers of 168
points.  search_ze99_joint_layer.py asks whether one of those layers can have
169 points when it is drawn from the finite family of all 4248 primitive
directions of magnitude <= 2 that are compatible with the 816 tetrads: 4096
sign vectors (+-1)^12, 128 signed triples and 24 signed axes.  A 169-point
layer would give 1155 > 1154.  The CP-SAT search for it ran out of time
(status UNKNOWN) both in CI and here.

This script settles the question: the maximum is exactly 168, attained only
with no triple at all.  It is a proof by a short case analysis, every
ingredient of which is checked below in exact integer arithmetic, plus four
classical values of A(n,4), the largest binary code of length n and minimum
distance 4:

    A(9,4) = 20,  A(10,4) = 40,  A(11,4) = 72,  A(12,4) = 144
    (Best, Brouwer, MacWilliams, Odlyzko, Sloane, IEEE Trans. Inform. Theory
     24 (1978) 81-93; https://www.win.tue.nl/~aeb/codes/binary-1.html).

The argument.  Write a layer as C + T + X: sign vectors, triples, axes.

 (1) Two distinct sign vectors are compatible iff their Hamming distance is
     >= 4, so C is a binary code of length 12 and minimum distance 4.
 (2) A sign vector and a triple conflict iff they agree on the triple's three
     coordinates.  An axis and a triple conflict iff the axis is one of the
     triple's three signed coordinates.  Axes are compatible with every sign
     vector and with each other.
 (3) There are 16 triple supports, 8 sign patterns on each, and every
     coordinate lies in exactly 4 supports; so any 13 supports cover all 12
     coordinates.
 (4) Suppose T uses k sign patterns on some support S.  Then C avoids k of
     the 8 subcubes "x restricted to S equals sigma".  Splitting the allowed
     patterns into faces of the 3-cube bounds |C| by 20 per vertex, 40 per
     edge and 72 per square face.  Let beta(k) be the largest such bound over
     all choices of k forbidden patterns (computed below):
         beta(1) = 132, beta(2) = 120, beta(3) = 100, beta(k) <= 20 (8 - k).
 (5) Let m >= 1 be the largest number of patterns used on one support, t = |T|
     and u the number of signed axes killed by T, so |X| <= 24 - u.  Then
     t <= m * (supports used), u >= 3, and u >= 12 once 13 supports are used.
     Hence  |C| + |T| + |X|  <=  beta(m) + 24 + max(t - u), and the script
     evaluates the right-hand side for m = 1..8: it never exceeds 165.
 (6) With T empty, |C| + |X| <= 144 + 24 = 168, which ZE99 attains.

usage:  python3 lower_bound/dim13/prove_joint_layer_168.py
"""
from __future__ import annotations

import itertools
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from search_ze99_joint_layer import enumerate_tetrad_compatible  # noqa: E402

A4 = {9: 20, 10: 40, 11: 72, 12: 144}   # A(n,4), classical


def compatible(q: np.ndarray, nq: np.ndarray, r: np.ndarray, nr: np.ndarray) -> np.ndarray:
    """Exact test cos(q,r) <= 1/3 for all pairs: dot <= 0 or 9 dot^2 <= |q|^2 |r|^2."""
    dot = q.astype(np.int64) @ r.astype(np.int64).T
    return (dot <= 0) | (9 * dot * dot <= nq[:, None].astype(np.int64) * nr[None, :].astype(np.int64))


def min_partition(allowed: frozenset[int]) -> int:
    """Cheapest partition of a set of vertices of the 3-cube into faces (20 / 40 / 72 / 144)."""
    faces = []
    for fixed in itertools.product([None, 0, 1], repeat=3):
        verts = frozenset(v for v in range(8) if all(f is None or (v >> i & 1) == f for i, f in enumerate(fixed)))
        dim = sum(f is None for f in fixed)
        faces.append((verts, A4[9 + dim]))
    best = {frozenset(): 0}

    def go(rest: frozenset[int]) -> int:
        if rest in best:
            return best[rest]
        v = min(rest)
        val = min(cost + go(rest - verts) for verts, cost in faces if v in verts and verts <= rest)
        best[rest] = val
        return val

    return go(allowed)


def main() -> int:
    q, n2, _ = enumerate_tetrad_compatible()
    assert q.shape == (4248, 12)
    C, T, X = q[n2 == 12], q[n2 == 3], q[n2 == 1]
    assert (len(C), len(T), len(X)) == (4096, 128, 24)
    nC, nT, nX = (np.full(len(a), v) for a, v in ((C, 12), (T, 3), (X, 1)))

    # (1) distinct sign vectors: compatible iff Hamming distance >= 4
    ham = (12 - C.astype(np.int64) @ C.astype(np.int64).T) // 2
    off = ~np.eye(len(C), dtype=bool)                # a vector is never paired with itself
    assert np.array_equal(compatible(C, nC, C, nC)[off], (ham >= 4)[off])
    # (2) triple vs sign vector: conflict iff they agree on all three coordinates
    agree_all = (C.astype(np.int64) @ T.astype(np.int64).T) == 3
    assert np.array_equal(compatible(C, nC, T, nT), ~agree_all)
    #     triple vs axis: conflict iff the axis is one of the triple's signed coordinates
    assert np.array_equal(compatible(T, nT, X, nX), (T.astype(np.int64) @ X.astype(np.int64).T) != 1)
    #     axes: compatible with every sign vector and with each other (distinct ones)
    assert compatible(C, nC, X, nX).all()
    assert (compatible(X, nX, X, nX) | np.eye(24, dtype=bool)).all()
    print("checked (1), (2): pairwise rules for all", 4096 * 4096 + 4096 * 128 + 128 * 24 + 4096 * 24 + 24 * 24,
          "ordered pairs, exact integers")

    # (3) support structure of the triples
    supports = sorted({tuple(np.flatnonzero(t)) for t in T})
    assert len(supports) == 16 and all(len(s) == 3 for s in supports)
    assert all(sum(tuple(np.flatnonzero(t)) == s for t in T) == 8 for s in supports)
    cover = [sum(i in s for s in supports) for i in range(12)]
    assert cover == [4] * 12
    avoid_max = max(16 - c for c in cover)          # most supports that can avoid one coordinate
    assert avoid_max == 12
    print("checked (3): 16 supports x 8 signs, every coordinate in exactly 4 supports")

    # (4) beta(k)
    beta = {}
    for k in range(0, 9):
        beta[k] = max(min_partition(frozenset(range(8)) - frozenset(K))
                      for K in itertools.combinations(range(8), k))
    assert beta[0] == A4[12]
    assert (beta[1], beta[2], beta[3]) == (132, 120, 100)
    assert all(beta[k] <= 20 * (8 - k) for k in range(2, 9))
    print("checked (4): beta =", [beta[k] for k in range(9)])

    # (5) the case analysis
    worst = 0
    for m in range(1, 9):
        slack = 0
        for used in range(1, 17):                   # number of supports carrying a triple
            t_max = m * used
            u_min = 12 if used > avoid_max else 3
            slack = max(slack, t_max - u_min)
        bound = beta[m] + 24 + slack
        worst = max(worst, bound)
        print(f"  some support carries {m} sign pattern(s): layer <= {beta[m]} + 24 + {slack} = {bound}")
    assert worst <= 165
    print("checked (5): any layer containing a triple has at most", worst, "points")
    print(f"(6) without triples: at most A(12,4) + 24 = {A4[12] + 24}")
    print("\nPROVED: the joint half-height layer has at most 168 points; no 169-point layer,")
    print("hence no 1155-point configuration, exists in this 4248-direction family.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
