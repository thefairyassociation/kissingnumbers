#!/usr/bin/env python3
"""Enumerate primitive q/||q|| auxiliary directions with |q_i| <= 2."""

from __future__ import annotations

from collections import Counter
from math import gcd

import numpy as np

from ze99_aux_core import SearchError


def enumerate_bounded(rows: np.ndarray, bound: int = 2) -> tuple[np.ndarray, np.ndarray, dict]:
    if bound != 2:
        raise SearchError("the current exact enumerator is specialized to bound=2")

    tetrad_supports = np.unique(np.abs(rows[:816]), axis=0).astype(np.int16)
    diamonds = np.unique(rows[816:], axis=0).astype(np.int16)
    if (len(tetrad_supports), len(diamonds)) != (51, 144):
        raise SearchError(
            f"unexpected support/sign row counts {(len(tetrad_supports), len(diamonds))}"
        )

    total = 3**12
    work = np.arange(total, dtype=np.int64)
    magnitudes = np.empty((total, 12), dtype=np.int8)
    for j in range(12):
        magnitudes[:, j] = (work % 3).astype(np.int8)
        work //= 3
    norm2 = np.sum(magnitudes.astype(np.int16) ** 2, axis=1).astype(np.int16)

    nonzero = norm2 > 0
    primitive = np.any(magnitudes == 1, axis=1)
    keep = nonzero & primitive
    magnitude_total = int(np.count_nonzero(keep))
    magnitudes = magnitudes[keep]
    norm2 = norm2[keep]

    tetrad_keep = np.zeros(len(magnitudes), dtype=bool)
    batch = 4096
    for start in range(0, len(magnitudes), batch):
        stop = min(start + batch, len(magnitudes))
        sums = magnitudes[start:stop].astype(np.int16) @ tetrad_supports.T
        maximum = np.max(sums, axis=1).astype(np.int32)
        nn = norm2[start:stop].astype(np.int32)
        tetrad_keep[start:stop] = 3 * maximum * maximum <= 4 * nn
    magnitudes = magnitudes[tetrad_keep]
    norm2 = norm2[tetrad_keep]

    candidates: list[np.ndarray] = []
    candidate_norms: list[np.ndarray] = []
    signings_tested = 0
    signings_feasible = 0
    magnitude_supports = Counter()
    feasible_norms = Counter()

    for mag, nn in zip(magnitudes, norm2):
        positions = np.flatnonzero(mag)
        support = len(positions)
        magnitude_supports[support] += 1
        half = 1 << (support - 1)
        codes = np.arange(half, dtype=np.uint32)
        signed = np.zeros((half, 12), dtype=np.int8)
        signed[:, positions[0]] = mag[positions[0]]
        for bit, position in enumerate(positions[1:]):
            signs = np.where((codes >> bit) & 1, 1, -1).astype(np.int8)
            signed[:, position] = signs * mag[position]
        signings_tested += half
        products = signed.astype(np.int16) @ diamonds.T
        maximum = np.max(np.abs(products), axis=1).astype(np.int32)
        good = 3 * maximum * maximum <= 4 * int(nn)
        if not np.any(good):
            continue
        selected = signed[good]
        signings_feasible += 2 * len(selected)
        candidates.extend([selected, -selected])
        candidate_norms.extend(
            [
                np.full(len(selected), int(nn), dtype=np.int16),
                np.full(len(selected), int(nn), dtype=np.int16),
            ]
        )
        feasible_norms[int(nn)] += 2 * len(selected)

    if candidates:
        q = np.vstack(candidates)
        n = np.concatenate(candidate_norms)
    else:
        q = np.empty((0, 12), dtype=np.int8)
        n = np.empty(0, dtype=np.int16)

    keys = {(tuple(row.tolist()), int(nn)) for row, nn in zip(q, n)}
    if len(keys) != len(q):
        raise SearchError(f"bounded enumeration produced {len(q)-len(keys)} duplicates")
    for row in q:
        values = [abs(int(v)) for v in row if v]
        g = 0
        for value in values:
            g = gcd(g, value)
        if g != 1:
            raise SearchError("non-primitive direction survived")

    return q, n, {
        "bound": bound,
        "magnitude_patterns_total": magnitude_total,
        "magnitude_patterns_after_tetrads": int(len(magnitudes)),
        "magnitude_support_distribution_after_tetrads": dict(sorted(magnitude_supports.items())),
        "half_signings_tested_after_tetrads": signings_tested,
        "signed_candidates_feasible": signings_feasible,
        "feasible_norm_squared_distribution": dict(sorted(feasible_norms.items())),
    }
