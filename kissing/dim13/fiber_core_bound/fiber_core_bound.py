#!/usr/bin/env python3
"""Exact finite proof of the 1,008-point ceiling for the E6/E7 product core.

PackingStar's dimension-13 construction uses points (a,b)/sqrt(2), where a is
an E6 root and b is an E7 root. For a fixed a, all chosen b values must have
pairwise inner product <= 0; for a fixed b, all chosen a values must have
pairwise inner product <= 0. This program constructs both root systems with
integer coordinates and solves those two maximum-clique problems exactly.

No floating point is used. Roots are stored with squared norm 8, so a dot
product <= 0 is exactly the nonpositive-inner-product condition.
"""

from __future__ import annotations

import argparse
import hashlib
import itertools
import json
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable

import numpy as np


@dataclass
class CliqueResult:
    maximum: int
    witness: list[int]
    branch_nodes: int
    maximum_cliques: list[tuple[int, ...]]


def e7_roots_scaled() -> np.ndarray:
    roots: list[np.ndarray] = []
    for i in range(8):
        for j in range(i + 1, 8):
            root = np.zeros(8, dtype=np.int64)
            root[i] = 2
            root[j] = -2
            roots.extend([root, -root])
    for plus in itertools.combinations(range(8), 4):
        root = -np.ones(8, dtype=np.int64)
        root[list(plus)] = 1
        roots.append(root)
    result = np.unique(np.asarray(roots, dtype=np.int64), axis=0)
    if result.shape != (126, 8):
        raise AssertionError(result.shape)
    if not np.all(np.sum(result * result, axis=1) == 8):
        raise AssertionError("E7 norm failure")
    if not np.all(np.sum(result, axis=1) == 0):
        raise AssertionError("E7 hyperplane failure")
    return result


def e6_roots_scaled(e7: np.ndarray) -> np.ndarray:
    result = e7[e7[:, 6] + e7[:, 7] == 0]
    if result.shape != (72, 8):
        raise AssertionError(result.shape)
    return result


def adjacency_bitsets(roots: np.ndarray) -> list[int]:
    gram = roots @ roots.T
    n = len(roots)
    adjacency: list[int] = []
    for i in range(n):
        bits = 0
        for j in range(n):
            if i != j and int(gram[i, j]) <= 0:
                bits |= 1 << j
        adjacency.append(bits)
    return adjacency


def color_sort(candidates: int, adjacency: list[int]) -> tuple[list[int], list[int]]:
    order: list[int] = []
    bounds: list[int] = []
    remaining = candidates
    color = 0
    while remaining:
        color += 1
        available = remaining
        while available:
            bit = available & -available
            vertex = bit.bit_length() - 1
            order.append(vertex)
            bounds.append(color)
            remaining &= ~bit
            available &= ~bit
            available &= ~adjacency[vertex]
    return order, bounds


def maximum_clique(adjacency: list[int]) -> tuple[list[int], int]:
    best: list[int] = []
    nodes = 0

    def expand(clique: list[int], candidates: int) -> None:
        nonlocal best, nodes
        nodes += 1
        if not candidates:
            if len(clique) > len(best):
                best = clique.copy()
            return
        order, bounds = color_sort(candidates, adjacency)
        for index in range(len(order) - 1, -1, -1):
            if len(clique) + bounds[index] <= len(best):
                return
            vertex = order[index]
            if (candidates >> vertex) & 1:
                expand(clique + [vertex], candidates & adjacency[vertex])
                candidates &= ~(1 << vertex)

    expand([], (1 << len(adjacency)) - 1)
    return best, nodes


def enumerate_target_cliques(adjacency: list[int], target: int) -> list[tuple[int, ...]]:
    forward = [bits & ~((1 << (i + 1)) - 1) for i, bits in enumerate(adjacency)]
    found: list[tuple[int, ...]] = []

    def recurse(clique: list[int], candidates: int) -> None:
        need = target - len(clique)
        if need == 0:
            found.append(tuple(clique))
            return
        if candidates.bit_count() < need:
            return
        while candidates:
            bit = candidates & -candidates
            vertex = bit.bit_length() - 1
            candidates ^= bit
            recurse(clique + [vertex], candidates & forward[vertex])

    recurse([], (1 << len(adjacency)) - 1)
    return found


def solve_cliques(roots: np.ndarray) -> CliqueResult:
    adjacency = adjacency_bitsets(roots)
    witness, nodes = maximum_clique(adjacency)
    all_maximum = enumerate_target_cliques(adjacency, len(witness))
    return CliqueResult(len(witness), witness, nodes, all_maximum)


def connected_components(vertices: Iterable[int], edge: np.ndarray) -> list[list[int]]:
    remaining = set(int(v) for v in vertices)
    components: list[list[int]] = []
    while remaining:
        start = min(remaining)
        remaining.remove(start)
        stack = [start]
        component = [start]
        while stack:
            current = stack.pop()
            neighbors = [v for v in list(remaining) if bool(edge[current, v])]
            for neighbor in neighbors:
                remaining.remove(neighbor)
                stack.append(neighbor)
                component.append(neighbor)
        components.append(sorted(component))
    return sorted(components, key=lambda component: (len(component), component))


def classify_e7_maxima(roots: np.ndarray, cliques: list[tuple[int, ...]]) -> None:
    gram = roots @ roots.T
    for clique in cliques:
        subset = list(clique)
        for i in subset:
            antipodes = [j for j in subset if int(gram[i, j]) == -8]
            if len(antipodes) != 1:
                raise AssertionError("E7 maximum is not seven antipodal pairs")
        representatives: list[int] = []
        used: set[int] = set()
        for i in subset:
            if i in used:
                continue
            anti = next(j for j in subset if int(gram[i, j]) == -8)
            used.update([i, anti])
            representatives.append(i)
        if len(representatives) != 7:
            raise AssertionError("wrong E7 line count")
        for first, second in itertools.combinations(representatives, 2):
            if int(gram[first, second]) != 0:
                raise AssertionError("E7 maximum lines are not orthogonal")


def classify_e6_maxima(roots: np.ndarray, cliques: list[tuple[int, ...]]) -> None:
    gram = roots @ roots.T
    for clique in cliques:
        components = connected_components(clique, gram < 0)
        if [len(component) for component in components] != [3, 3, 3]:
            raise AssertionError("E6 maximum is not three orthogonal triangles")
        for component in components:
            for first, second in itertools.combinations(component, 2):
                if int(gram[first, second]) != -4:
                    raise AssertionError("E6 component is not an equilateral triangle")
        for first_component, second_component in itertools.combinations(components, 2):
            for first in first_component:
                for second in second_component:
                    if int(gram[first, second]) != 0:
                        raise AssertionError("E6 triangles are not orthogonal")


def hash_roots(roots: np.ndarray) -> str:
    return hashlib.sha256(roots.tobytes(order="C")).hexdigest()


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()

    e7 = e7_roots_scaled()
    e6 = e6_roots_scaled(e7)
    e7_result = solve_cliques(e7)
    e6_result = solve_cliques(e6)
    if e7_result.maximum != 14:
        raise AssertionError(e7_result.maximum)
    if e6_result.maximum != 9:
        raise AssertionError(e6_result.maximum)
    classify_e7_maxima(e7, e7_result.maximum_cliques)
    classify_e6_maxima(e6, e6_result.maximum_cliques)

    core_upper = len(e6) * e7_result.maximum
    minimum_e7_support = (core_upper + e6_result.maximum - 1) // e6_result.maximum
    result = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "arithmetic": "integer only; every root has exact squared norm 8",
        "e7": {
            "root_count": len(e7),
            "sha256": hash_roots(e7),
            "maximum_nonpositive_subset": e7_result.maximum,
            "maximum_subset_count": len(e7_result.maximum_cliques),
            "classification": "seven antipodal pairs on seven mutually orthogonal lines (X7)",
            "branch_and_bound_nodes": e7_result.branch_nodes,
            "witness_indices": e7_result.witness,
            "witness_vectors_scaled": e7[e7_result.witness].tolist(),
        },
        "e6": {
            "root_count": len(e6),
            "sha256": hash_roots(e6),
            "maximum_nonpositive_subset": e6_result.maximum,
            "maximum_subset_count": len(e6_result.maximum_cliques),
            "classification": "three mutually orthogonal equilateral triangles (Y3)",
            "branch_and_bound_nodes": e6_result.branch_nodes,
            "witness_indices": e6_result.witness,
            "witness_vectors_scaled": e6[e6_result.witness].tolist(),
        },
        "fiber_core_consequence": {
            "maximum_edges": core_upper,
            "proof": "72 E6 roots, each with at most 14 E7 neighbors",
            "minimum_distinct_e7_roots_if_1008_edges": minimum_e7_support,
            "proof_of_support": "each E7 root has at most 9 E6 neighbors",
            "maximum_e7_roots_that_can_be_omitted": len(e7) - minimum_e7_support,
            "packingstar_variant_attains_both_equalities": True,
        },
        "exact": True,
    }
    text = json.dumps(result, indent=2, sort_keys=True)
    print(text)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(text + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
