#!/usr/bin/env python3
"""Exact Weyl-orbit classification of maximum nonpositive E6 root subsets."""

from __future__ import annotations

import argparse
import hashlib
import itertools
import json
from collections import deque
from datetime import datetime, timezone
from pathlib import Path

import numpy as np


def e7_roots_scaled() -> np.ndarray:
    roots: list[np.ndarray] = []
    for i in range(8):
        for j in range(i + 1, 8):
            root = np.zeros(8, dtype=np.int64)
            root[i] = 2
            root[j] = -2
            roots.extend((root, -root))
    for plus in itertools.combinations(range(8), 4):
        root = -np.ones(8, dtype=np.int64)
        root[list(plus)] = 1
        roots.append(root)
    out = np.unique(np.asarray(roots, dtype=np.int64), axis=0)
    assert out.shape == (126, 8)
    assert np.all(np.sum(out * out, axis=1) == 8)
    assert np.all(np.sum(out, axis=1) == 0)
    return out


def e6_roots_scaled() -> np.ndarray:
    e7 = e7_roots_scaled()
    out = e7[e7[:, 6] + e7[:, 7] == 0]
    assert out.shape == (72, 8)
    assert np.all(np.sum(out * out, axis=1) == 8)
    return out


def adjacency_bitsets(roots: np.ndarray) -> list[int]:
    gram = roots @ roots.T
    adjacency: list[int] = []
    for i in range(len(roots)):
        bits = 0
        for j in range(len(roots)):
            if i != j and int(gram[i, j]) <= 0:
                bits |= 1 << j
        adjacency.append(bits)
    return adjacency


def enumerate_nine_cliques(adjacency: list[int]) -> list[tuple[int, ...]]:
    forward = [bits & ~((1 << (i + 1)) - 1) for i, bits in enumerate(adjacency)]
    found: list[tuple[int, ...]] = []

    def recurse(clique: list[int], candidates: int) -> None:
        need = 9 - len(clique)
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
    assert len(found) == 320
    return found


def reflection_permutations(roots: np.ndarray) -> list[tuple[int, ...]]:
    index = {tuple(row.tolist()): i for i, row in enumerate(roots)}
    permutations: set[tuple[int, ...]] = set()
    for alpha in roots:
        permutation: list[int] = []
        for root in roots:
            dot = int(root @ alpha)
            assert dot % 4 == 0
            reflected = root - (dot // 4) * alpha
            target = index.get(tuple(reflected.tolist()))
            assert target is not None
            permutation.append(target)
        assert sorted(permutation) == list(range(72))
        permutations.add(tuple(permutation))
    assert len(permutations) == 36
    return sorted(permutations)


def orbit(seed: tuple[int, ...], generators: list[tuple[int, ...]]) -> set[tuple[int, ...]]:
    seed = tuple(sorted(seed))
    seen = {seed}
    queue: deque[tuple[int, ...]] = deque([seed])
    while queue:
        current = queue.popleft()
        for permutation in generators:
            image = tuple(sorted(permutation[index] for index in current))
            if image not in seen:
                seen.add(image)
                queue.append(image)
    return seen


def classify_y3(roots: np.ndarray, clique: tuple[int, ...]) -> list[list[int]]:
    gram = roots @ roots.T
    remaining = set(clique)
    components: list[list[int]] = []
    while remaining:
        start = min(remaining)
        remaining.remove(start)
        component = [start]
        stack = [start]
        while stack:
            current = stack.pop()
            neighbors = [j for j in sorted(remaining) if int(gram[current, j]) < 0]
            for neighbor in neighbors:
                remaining.remove(neighbor)
                component.append(neighbor)
                stack.append(neighbor)
        components.append(sorted(component))
    components.sort()
    assert sorted(map(len, components)) == [3, 3, 3]
    for component in components:
        for i, j in itertools.combinations(component, 2):
            assert int(gram[i, j]) == -4
    for first, second in itertools.combinations(components, 2):
        for i in first:
            for j in second:
                assert int(gram[i, j]) == 0
    return components


def digest(items: list[tuple[int, ...]]) -> str:
    text = "\n".join(",".join(map(str, item)) for item in sorted(items)) + "\n"
    return hashlib.sha256(text.encode("ascii")).hexdigest()


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()

    roots = e6_roots_scaled()
    maxima = enumerate_nine_cliques(adjacency_bitsets(roots))
    generators = reflection_permutations(roots)
    reached = orbit(maxima[0], generators)
    assert reached == set(maxima)
    components = classify_y3(roots, maxima[0])

    result = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "arithmetic": "integer only; reflection x -> x - (x.alpha/4) alpha",
        "e6_root_count": 72,
        "maximum_nonpositive_subset_size": 9,
        "maximum_y3_count": len(maxima),
        "distinct_root_reflections": len(generators),
        "orbit_size_from_one_y3": len(reached),
        "weyl_transitive_on_y3": reached == set(maxima),
        "maximum_sets_sha256": digest(maxima),
        "witness_root_indices": list(maxima[0]),
        "witness_triangle_components": components,
        "witness_vectors_scaled": roots[list(maxima[0])].tolist(),
        "consequence": (
            "All 320 maximum nine-root Y3 subsets of E6 are isometric under exact "
            "root reflections. Changing a standalone maximum E6 fiber cannot create "
            "a new local geometry; any gain must come from globally different coupling."
        ),
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
