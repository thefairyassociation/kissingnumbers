#!/usr/bin/env python3
"""Exact finite search helpers for replacing ZE99's 48 irrational vectors."""

from __future__ import annotations

import time
from collections import Counter
from pathlib import Path
from typing import Callable

import numpy as np

HERE = Path(__file__).resolve().parent


class SearchError(RuntimeError):
    pass


def constraint_rows(generate_ab: Callable[[], list]) -> tuple[np.ndarray, dict]:
    """Return the 816 signed tetrads and 144 distinct diamond sign rows."""
    vecs = generate_ab()
    if len(vecs) != 1154:
        raise SearchError(f"ZE99 generator returned {len(vecs)} vectors")

    tetrads = []
    for v in vecs[:816]:
        row = []
        for a, b in v[:12]:
            if b or a not in (-2, 0, 2):
                raise SearchError("unexpected tetrad coordinate")
            row.append(a // 2)
        tetrads.append(tuple(row))
    tetrads = list(dict.fromkeys(tetrads))

    diamonds = []
    for v in vecs[816:1104]:
        row = []
        for a, b in v[:12]:
            if b or a not in (-1, 1):
                raise SearchError("unexpected diamond coordinate")
            row.append(a)
        diamonds.append(tuple(row))
    diamonds = list(dict.fromkeys(diamonds))
    if (len(tetrads), len(diamonds)) != (816, 144):
        raise SearchError(
            f"constraint row counts {(len(tetrads), len(diamonds))} != (816,144)"
        )

    rows = np.asarray(tetrads + diamonds, dtype=np.int8)
    rowset = {tuple(row.tolist()) for row in rows}
    if any(tuple((-row).tolist()) not in rowset for row in rows):
        raise SearchError("constraint rows are not centrally symmetric")
    return rows, {
        "signed_tetrads": 816,
        "diamond_sign_rows": 144,
        "total": 960,
        "centrally_symmetric": True,
    }


def enumerate_candidates(rows: np.ndarray) -> tuple[np.ndarray, np.ndarray, dict]:
    """Exhaust q in {-1,0,1}^12 and apply exact fixed-base inequalities."""
    count = 3**12
    work = np.arange(count, dtype=np.int64)
    q = np.empty((count, 12), dtype=np.int8)
    for j in range(12):
        q[:, j] = (work % 3).astype(np.int8) - 1
        work //= 3
    k = np.count_nonzero(q, axis=1).astype(np.int16)
    nonzero = k > 0
    q, k = q[nonzero], k[nonzero]

    keep = np.zeros(len(q), dtype=bool)
    batch = 4096
    rows16 = rows.astype(np.int16)
    for start in range(0, len(q), batch):
        stop = min(start + batch, len(q))
        products = q[start:stop].astype(np.int16) @ rows16.T
        m = np.max(np.abs(products), axis=1).astype(np.int32)
        kk = k[start:stop].astype(np.int32)
        keep[start:stop] = 3 * m * m <= 4 * kk

    before = Counter(map(int, k.tolist()))
    q, k = q[keep], k[keep]
    after = Counter(map(int, k.tolist()))
    if len(q) != len({tuple(row.tolist()) for row in q}):
        raise SearchError("candidate enumeration produced duplicates")
    return q, k, {
        "tested": int(3**12 - 1),
        "feasible": int(len(q)),
        "support_distribution_before": dict(sorted(before.items())),
        "support_distribution_feasible": dict(sorted(after.items())),
    }


def compatibility_graph(
    q: np.ndarray, k: np.ndarray
) -> tuple[list[int], np.ndarray, dict]:
    """Build exact compatibility graph using 9(q.r)^2 <= |q||r|."""
    n = len(q)
    dots = q.astype(np.int16) @ q.astype(np.int16).T
    dots32 = dots.astype(np.int32)
    kk = k.astype(np.int32)
    compatible = (dots32 <= 0) | (9 * dots32 * dots32 <= kk[:, None] * kk[None, :])
    np.fill_diagonal(compatible, False)
    if not np.array_equal(compatible, compatible.T):
        raise SearchError("compatibility graph is not symmetric")

    degree = compatible.sum(axis=1)
    order = np.argsort(-degree, kind="stable").astype(np.int32)
    compatible = compatible[np.ix_(order, order)]
    adjacency = [
        int.from_bytes(np.packbits(row, bitorder="little").tobytes(), "little")
        for row in compatible
    ]
    edges = int(np.count_nonzero(np.triu(compatible, 1)))
    return adjacency, order, {
        "vertices": n,
        "edges": edges,
        "minimum_degree": int(degree.min()) if n else 0,
        "maximum_degree": int(degree.max()) if n else 0,
    }


def baseline_clique(q: np.ndarray, order: np.ndarray) -> list[int]:
    """Locate the known auxiliary code {+/- e_i} in reordered indices."""
    original = {tuple(row.tolist()): i for i, row in enumerate(q)}
    inverse = np.empty(len(order), dtype=np.int32)
    inverse[order] = np.arange(len(order), dtype=np.int32)
    out = []
    for axis in range(12):
        for sign in (-1, 1):
            row = [0] * 12
            row[axis] = sign
            idx = original.get(tuple(row))
            if idx is None:
                raise SearchError("known +/-e_i auxiliary vector was filtered out")
            out.append(int(inverse[idx]))
    return out


def exact_maximum_clique(
    adjacency: list[int],
    incumbent: list[int],
    time_limit: float,
    checkpoint: Callable[[list[int], int], None] | None = None,
) -> tuple[list[int], dict]:
    """Tomita-style branch-and-bound with exact Python-integer bitsets."""
    n = len(adjacency)
    best = list(incumbent)
    stack: list[int] = []
    nodes = 0
    deadline = time.monotonic() + time_limit
    timed_out = False

    def color_sort(vertices: int) -> tuple[list[int], list[int]]:
        ordered: list[int] = []
        colors: list[int] = []
        remaining = vertices
        color = 0
        while remaining:
            color += 1
            available = remaining
            while available:
                bit = available & -available
                v = bit.bit_length() - 1
                ordered.append(v)
                colors.append(color)
                remaining ^= bit
                available ^= bit
                available &= ~adjacency[v]
        return ordered, colors

    def expand(vertices: int) -> None:
        nonlocal best, nodes, timed_out
        nodes += 1
        if nodes & 4095 == 0 and time.monotonic() > deadline:
            timed_out = True
            return
        ordered, colors = color_sort(vertices)
        for pos in range(len(ordered) - 1, -1, -1):
            if timed_out or len(stack) + colors[pos] <= len(best):
                return
            v = ordered[pos]
            stack.append(v)
            candidates = vertices & adjacency[v]
            if candidates:
                expand(candidates)
            elif len(stack) > len(best):
                best = list(stack)
                if checkpoint:
                    checkpoint(best, nodes)
            stack.pop()
            vertices &= ~(1 << v)

    all_vertices = (1 << n) - 1
    if checkpoint:
        checkpoint(best, nodes)
    expand(all_vertices)
    return best, {
        "best": len(best),
        "nodes": nodes,
        "timed_out": timed_out,
        "proven_optimal": not timed_out,
        "algorithm": "exact branch-and-bound with greedy coloring and integer bitsets",
    }


def verify_selected(rows: np.ndarray, q: np.ndarray, k: np.ndarray) -> dict:
    products = q.astype(np.int16) @ rows.astype(np.int16).T
    m = np.max(np.abs(products), axis=1).astype(np.int32)
    kk = k.astype(np.int32)
    if np.any(3 * m * m > 4 * kk):
        raise SearchError("selected code violates a fixed-base inequality")
    tight_fixed = int(np.count_nonzero(3 * products.astype(np.int32) ** 2 == 4 * kk[:, None]))
    tight_pairs = 0
    for i in range(len(q)):
        dots = q[i].astype(np.int16) @ q[i + 1 :].astype(np.int16).T
        rhs = int(k[i]) * k[i + 1 :].astype(np.int32)
        positive = dots > 0
        if np.any(positive & (9 * dots.astype(np.int32) ** 2 > rhs)):
            raise SearchError("selected code contains an incompatible pair")
        tight_pairs += int(np.count_nonzero(positive & (9 * dots.astype(np.int32) ** 2 == rhs)))
    return {
        "count": int(len(q)),
        "fixed_base_inequalities_exact": True,
        "pairwise_inner_products_leq_one_third_exact": True,
        "tight_fixed_constraints": tight_fixed,
        "tight_auxiliary_pairs": tight_pairs,
    }
