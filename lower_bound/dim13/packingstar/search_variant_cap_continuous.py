#!/usr/bin/env python3
"""Structured cap search inside PackingStar's non-antipodal d=13 fiber model.

The exact 1,146-point variant has a 1,008-point E6/E7 fiber core, a 54-point
rank-6 cap, and an 84-point rank-7 cap. Holding the core and the rank-6 cap
fixed, a new d=13 record would require at least 93 compatible points in the
rank-7 cap.

This script explores the continuous feasible cap region

    P = {x in R^7 : <r,x> <= 1/sqrt(2) for every retained E7 root r}

beyond the polar vertices tested by analyze_fiber_variant.py. It samples
normalized points from edges, outer-vertex chords, cap-to-vertex chords, and
random convex combinations in P, then solves finite maximum-independent-set
problems in the resulting compatibility graph. All structural extraction of
the source configuration is exact on H=4G. The candidate search itself is
numerical and is never reported as an exact kissing result.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
import time
from collections import Counter, defaultdict
from datetime import datetime, timezone
from fractions import Fraction
from pathlib import Path
from typing import Any, Iterable

import numpy as np
from scipy.optimize import Bounds, LinearConstraint, milp
from scipy.sparse import coo_matrix
from scipy.spatial import ConvexHull, HalfspaceIntersection

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

from analyze_fiber import (  # noqa: E402
    GRAM_DEN,
    RationalMatrix,
    combined_subspace_coordinates,
    exact_scaled_gram,
    global_reduce,
    representatives,
    unique_row_classes,
)
from analyze_fiber_variant import nonzero_components  # noqa: E402

ROOT_THRESHOLD = 1 / math.sqrt(2)
PAIR_THRESHOLD = 0.5
CAP_RECORD = 84
CAP_TARGET = 93
FULL_FIXED_COUNT = 1008 + 54
LIVE_D13_RECORD = 1154


class SearchError(RuntimeError):
    pass


def utcnow() -> str:
    return datetime.now(timezone.utc).isoformat()


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def checkpoint(output_dir: Path, stage: str, payload: dict[str, Any]) -> None:
    data = {"generated_at": utcnow(), "stage": stage, **payload}
    write_json(output_dir / "checkpoint.json", data)
    print(json.dumps(data, indent=2, sort_keys=True), flush=True)


def exact_variant_rank7(path: Path) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    """Recover the 112 retained rank-7 roots and the 84-point cap.

    All decomposition/class assertions are made on the exact integer matrix H=4G.
    Only the final simultaneous Gram factorization uses floating point to choose an
    orthogonal coordinate basis for the numerical search.
    """
    H = exact_scaled_gram(path)
    antipodes = np.count_nonzero(H == -GRAM_DEN, axis=1)
    if set(antipodes.tolist()) != {0, 1}:
        raise SearchError(f"unexpected antipode counts: {sorted(set(antipodes.tolist()))}")
    core = np.flatnonzero(antipodes == 0).astype(int).tolist()
    residual = np.flatnonzero(antipodes == 1).astype(int).tolist()
    if (len(core), len(residual)) != (1008, 138):
        raise SearchError(f"unexpected core/residual sizes: {len(core)}, {len(residual)}")

    cap_components = nonzero_components(H, residual)
    if [len(component) for component in cap_components] != [54, 84]:
        raise SearchError(f"unexpected cap components: {[len(c) for c in cap_components]}")
    cap7 = cap_components[1]

    Hc = H[np.ix_(core, core)]
    H2 = Hc @ Hc
    component7 = global_reduce(336 * Hc - H2, 96)
    _, inverse7, multiplicities7 = unique_row_classes(component7.numerator)
    if len(multiplicities7) != 112 or set(multiplicities7.tolist()) != {9}:
        raise SearchError(
            f"rank-7 classes/multiplicity mismatch: {len(multiplicities7)}, "
            f"{set(multiplicities7.tolist())}"
        )
    reps7 = representatives(inverse7, 112)
    unique7 = RationalMatrix(
        component7.numerator[np.ix_(reps7, reps7)], component7.denominator
    ).reduced()

    cross_full = H[np.ix_(cap7, core)]
    cross7 = cross_full[:, reps7]
    if not np.array_equal(cross_full, cross7[:, inverse7]):
        raise SearchError("cap/core cross terms do not factor through 112 root classes")

    roots7, caps7, alignment = combined_subspace_coordinates(
        unique7,
        H[np.ix_(cap7, cap7)],
        cross7,
        7,
    )
    if roots7.shape != (112, 7) or caps7.shape != (84, 7):
        raise SearchError(f"unexpected coordinate shapes {roots7.shape}, {caps7.shape}")

    root_gram_error = float(
        np.max(
            np.abs(
                roots7 @ roots7.T
                - unique7.numerator.astype(float) / unique7.denominator
            )
        )
    )
    cap_gram_error = float(
        np.max(
            np.abs(
                caps7 @ caps7.T
                - H[np.ix_(cap7, cap7)].astype(float) / GRAM_DEN
            )
        )
    )
    root_cap_max = float(np.max(roots7 @ caps7.T))
    cap_pair = caps7 @ caps7.T
    np.fill_diagonal(cap_pair, -9.0)
    cap_pair_max = float(np.max(cap_pair))
    if root_gram_error > 3e-7 or cap_gram_error > 3e-7:
        raise SearchError(
            f"coordinate reconstruction errors {root_gram_error}, {cap_gram_error}"
        )
    if root_cap_max > ROOT_THRESHOLD + 3e-7 or cap_pair_max > 0.5 + 3e-7:
        raise SearchError(
            f"baseline cap invalid: root={root_cap_max}, pair={cap_pair_max}"
        )

    metadata = {
        "source": str(path),
        "exact_source_gram_denominator": GRAM_DEN,
        "exact_core_size": 1008,
        "exact_rank7_root_classes": 112,
        "exact_rank7_class_multiplicity": 9,
        "exact_cap7_size": 84,
        "coordinate_alignment": alignment,
        "root_gram_reconstruction_error": root_gram_error,
        "cap_gram_reconstruction_error": cap_gram_error,
        "baseline_root_cap_max": root_cap_max,
        "baseline_cap_pair_max": cap_pair_max,
    }
    return roots7, caps7, metadata


def row_key(row: np.ndarray, decimals: int = 9) -> tuple[int, ...]:
    scale = 10**decimals
    return tuple(np.rint(row * scale).astype(np.int64).tolist())


def normalize_feasible(
    raw: np.ndarray,
    roots: np.ndarray,
    *,
    norm_tolerance: float = 2e-9,
    root_tolerance: float = 3e-8,
) -> np.ndarray | None:
    norm = float(np.linalg.norm(raw))
    if norm < 1 - norm_tolerance:
        return None
    point = raw / norm
    if float(np.max(roots @ point)) > ROOT_THRESHOLD + root_tolerance:
        return None
    return point


def add_candidate(
    store: dict[tuple[int, ...], tuple[np.ndarray, set[str]]],
    raw: np.ndarray,
    roots: np.ndarray,
    label: str,
) -> None:
    point = normalize_feasible(raw, roots)
    if point is None:
        return
    for signed, suffix in ((point, "+"), (-point, "-")):
        if float(np.max(roots @ signed)) > ROOT_THRESHOLD + 3e-8:
            continue
        key = row_key(signed)
        if key in store:
            store[key][1].add(label + suffix)
        else:
            store[key] = (signed.copy(), {label + suffix})


def polar_vertices(roots: np.ndarray) -> np.ndarray:
    halfspaces = np.hstack(
        [roots, -np.full((len(roots), 1), ROOT_THRESHOLD, dtype=float)]
    )
    intersection = HalfspaceIntersection(
        halfspaces,
        np.zeros(roots.shape[1], dtype=float),
        qhull_options="Qx",
    )
    vertices = np.asarray(intersection.intersections, dtype=float)
    unique: dict[tuple[int, ...], np.ndarray] = {}
    for vertex in vertices:
        unique.setdefault(row_key(vertex), vertex)
    return np.asarray(list(unique.values()), dtype=float)


def hull_edges(vertices: np.ndarray) -> list[tuple[int, int]]:
    hull = ConvexHull(vertices, qhull_options="Qx")
    edges: set[tuple[int, int]] = set()
    for simplex in hull.simplices:
        indices = [int(value) for value in simplex]
        for i in range(len(indices)):
            for j in range(i + 1, len(indices)):
                a, b = sorted((indices[i], indices[j]))
                edges.add((a, b))
    return sorted(edges)


def generate_candidates(
    roots: np.ndarray,
    caps: np.ndarray,
    *,
    seed: int,
    random_samples: int,
) -> tuple[np.ndarray, list[list[str]], dict[str, Any]]:
    rng = np.random.default_rng(seed)
    vertices = polar_vertices(roots)
    norms = np.linalg.norm(vertices, axis=1)
    outer = np.flatnonzero(norms >= 1 - 2e-9).astype(int)
    edges = hull_edges(vertices)

    store: dict[tuple[int, ...], tuple[np.ndarray, set[str]]] = {}
    for cap in caps:
        add_candidate(store, cap, roots, "baseline_cap")
    for index in outer.tolist():
        add_candidate(store, vertices[index], roots, "polar_vertex")

    edge_fractions = np.linspace(0.0, 1.0, 33)[1:-1]
    for a, b in edges:
        if norms[a] < 1 - 2e-9 and norms[b] < 1 - 2e-9:
            continue
        va, vb = vertices[a], vertices[b]
        for t in edge_fractions:
            add_candidate(store, (1 - t) * va + t * vb, roots, "hull_edge")

    chord_fractions = np.linspace(0.0, 1.0, 17)[1:-1]
    seen_pairs: set[tuple[int, int]] = set()
    for a in outer.tolist():
        for b in range(len(vertices)):
            if a == b:
                continue
            pair = tuple(sorted((a, b)))
            if pair in seen_pairs:
                continue
            seen_pairs.add(pair)
            va, vb = vertices[pair[0]], vertices[pair[1]]
            for t in chord_fractions:
                add_candidate(store, (1 - t) * va + t * vb, roots, "outer_chord")

    for cap in caps:
        for index in outer.tolist():
            vertex = vertices[index]
            for t in chord_fractions:
                add_candidate(store, (1 - t) * cap + t * vertex, roots, "cap_chord")

    outer_list = outer.tolist()
    all_indices = np.arange(len(vertices))
    accepted_random = 0
    for _ in range(random_samples):
        width = int(rng.integers(2, min(9, len(vertices)) + 1))
        if rng.random() < 0.8:
            first = int(rng.choice(outer_list))
            others = rng.choice(all_indices, size=width - 1, replace=False).astype(int).tolist()
            indices = [first, *others]
        else:
            indices = rng.choice(all_indices, size=width, replace=False).astype(int).tolist()
        weights = rng.dirichlet(np.full(width, 0.28))
        raw = weights @ vertices[indices]
        before = len(store)
        add_candidate(store, raw, roots, "random_face")
        if len(store) > before:
            accepted_random += 1

    points = np.asarray([value[0] for value in store.values()], dtype=float)
    labels = [sorted(value[1]) for value in store.values()]
    root_max = np.max(roots @ points.T, axis=0)
    if float(np.max(root_max)) > ROOT_THRESHOLD + 4e-8:
        raise SearchError(f"generated candidate violates roots: {float(np.max(root_max))}")

    stats = {
        "polar_vertices": int(len(vertices)),
        "polar_norm_distribution": dict(
            Counter(f"{value:.9f}" for value in np.round(norms, 9)).most_common()
        ),
        "outer_vertices": int(len(outer)),
        "qhull_simplex_edges": int(len(edges)),
        "outer_vertex_pairs_sampled": int(len(seen_pairs)),
        "random_samples_requested": random_samples,
        "random_samples_adding_new_keys": accepted_random,
        "deduplicated_feasible_unit_candidates": int(len(points)),
        "source_label_counts": dict(
            Counter(label.rstrip("+-") for group in labels for label in group)
        ),
        "max_root_inner": float(np.max(root_max)),
    }
    return points, labels, stats


def conflict_mask_against_caps(point: np.ndarray, caps: np.ndarray) -> int:
    hits = np.flatnonzero(caps @ point > PAIR_THRESHOLD + 2e-8)
    mask = 0
    for index in hits.astype(int).tolist():
        mask |= 1 << index
    return mask


def bit_count(value: int) -> int:
    return int(value.bit_count())


def farthest_subset(
    indices: list[int], points: np.ndarray, limit: int, seed: int
) -> list[int]:
    if len(indices) <= limit:
        return indices
    rng = np.random.default_rng(seed)
    selected = [indices[int(rng.integers(0, len(indices)))]]
    remaining = np.array([i for i in indices if i != selected[0]], dtype=np.int64)
    minimum_distance = 1 - points[remaining] @ points[selected[0]]
    while len(selected) < limit and len(remaining):
        position = int(np.argmax(minimum_distance))
        chosen = int(remaining[position])
        selected.append(chosen)
        remaining = np.delete(remaining, position)
        minimum_distance = np.delete(minimum_distance, position)
        if len(remaining):
            minimum_distance = np.minimum(
                minimum_distance, 1 - points[remaining] @ points[chosen]
            )
    return selected


def reduce_candidate_pool(
    points: np.ndarray,
    labels: list[list[str]],
    caps: np.ndarray,
    *,
    seed: int,
    max_new: int,
    per_mask: int,
) -> tuple[np.ndarray, list[list[str]], dict[str, Any]]:
    baseline_keys = {row_key(row) for row in caps}
    new_indices = [i for i, row in enumerate(points) if row_key(row) not in baseline_keys]
    masks = {i: conflict_mask_against_caps(points[i], caps) for i in new_indices}
    groups: dict[int, list[int]] = defaultdict(list)
    for index in new_indices:
        groups[masks[index]].append(index)

    chosen: list[int] = []
    ordered_masks = sorted(groups, key=lambda mask: (bit_count(mask), -len(groups[mask]), mask))
    for position, mask in enumerate(ordered_masks):
        group = groups[mask]
        local_limit = per_mask
        if bit_count(mask) <= 2:
            local_limit = max(per_mask, 24)
        elif bit_count(mask) <= 4:
            local_limit = max(per_mask, 14)
        chosen.extend(farthest_subset(group, points, local_limit, seed + position))

    if len(chosen) > max_new:
        chosen = sorted(
            chosen,
            key=lambda i: (
                bit_count(masks[i]),
                -len(groups[masks[i]]),
                -len(labels[i]),
                i,
            ),
        )[:max_new]

    reduced_points = np.vstack([caps, points[chosen]])
    reduced_labels = [["baseline_cap"] for _ in range(len(caps))] + [labels[i] for i in chosen]
    selected_mask_counts = Counter(bit_count(masks[i]) for i in chosen)
    all_mask_counts = Counter(bit_count(mask) for mask in masks.values())
    stats = {
        "raw_nonbaseline_candidates": len(new_indices),
        "unique_old_cap_conflict_masks": len(groups),
        "all_old_conflict_count_distribution": dict(sorted(all_mask_counts.items())),
        "selected_new_candidates": len(chosen),
        "selected_old_conflict_count_distribution": dict(sorted(selected_mask_counts.items())),
        "combined_pool": int(len(reduced_points)),
        "max_new": max_new,
        "per_mask": per_mask,
    }
    return reduced_points, reduced_labels, stats


def conflict_matrix(points: np.ndarray) -> np.ndarray:
    gram = points @ points.T
    conflicts = gram > PAIR_THRESHOLD + 2e-8
    np.fill_diagonal(conflicts, True)
    return conflicts


def independent_check(indices: Iterable[int], conflicts: np.ndarray) -> bool:
    chosen = np.array(sorted(set(int(i) for i in indices)), dtype=np.int64)
    if len(chosen) == 0:
        return True
    sub = conflicts[np.ix_(chosen, chosen)].copy()
    np.fill_diagonal(sub, False)
    return not bool(np.any(sub))


def greedy_order(order: np.ndarray, conflicts: np.ndarray) -> list[int]:
    blocked = np.zeros(conflicts.shape[0], dtype=bool)
    chosen: list[int] = []
    for vertex in order.astype(int).tolist():
        if blocked[vertex]:
            continue
        chosen.append(vertex)
        blocked |= conflicts[vertex]
    return chosen


def randomized_greedy(
    conflicts: np.ndarray,
    *,
    seed: int,
    rounds: int,
    checkpoint_callback: Any | None = None,
) -> tuple[list[int], dict[str, Any]]:
    rng = np.random.default_rng(seed)
    n = conflicts.shape[0]
    static_degree = conflicts.sum(axis=1).astype(float)
    best = list(range(CAP_RECORD))
    if not independent_check(best, conflicts):
        raise SearchError("baseline cap is not independent in candidate graph")

    orders = [
        np.argsort(static_degree),
        np.concatenate([np.arange(CAP_RECORD), np.arange(CAP_RECORD, n)]),
    ]
    for order in orders:
        candidate = greedy_order(order, conflicts)
        if len(candidate) > len(best):
            best = candidate

    improvements: list[dict[str, Any]] = []
    completed = 0
    for iteration in range(rounds):
        completed = iteration + 1
        scale = float(rng.choice([0.05, 0.15, 0.35, 0.7, 1.5]))
        noise = rng.gumbel(size=n) * (np.std(static_degree) + 1) * scale
        score = static_degree + noise
        mode = iteration % 6
        if mode == 0:
            score[:CAP_RECORD] -= n
        elif mode == 1:
            score[CAP_RECORD:] -= 0.25 * n
        elif mode == 2:
            score[:CAP_RECORD] += 0.15 * n
        order = np.argsort(score)
        candidate = greedy_order(order, conflicts)
        if len(candidate) > len(best):
            best = candidate
            event = {
                "iteration": iteration,
                "size": len(best),
                "baseline_kept": int(sum(i < CAP_RECORD for i in best)),
            }
            improvements.append(event)
            print(f"GREEDY IMPROVEMENT {event}", flush=True)
            if checkpoint_callback is not None:
                checkpoint_callback(best, iteration)
        if len(best) >= CAP_TARGET:
            break
    return best, {
        "rounds_requested": rounds,
        "rounds_completed": completed,
        "improvements": improvements,
        "best": len(best),
    }


def greedy_independent_subset(vertices: list[int], conflicts: np.ndarray) -> list[int]:
    if not vertices:
        return []
    pool = np.array(vertices, dtype=np.int64)
    local_degree = conflicts[np.ix_(pool, pool)].sum(axis=1)
    return greedy_order(pool[np.argsort(local_degree)], conflicts)


def local_trade_improve(
    initial: list[int],
    conflicts: np.ndarray,
    *,
    max_remove: int = 4,
    passes: int = 30,
) -> tuple[list[int], list[dict[str, Any]]]:
    chosen = sorted(initial)
    events: list[dict[str, Any]] = []
    n = conflicts.shape[0]
    for pass_index in range(passes):
        selected = np.array(chosen, dtype=np.int64)
        selected_set = set(chosen)
        outside = [i for i in range(n) if i not in selected_set]
        groups: dict[tuple[int, ...], list[int]] = defaultdict(list)
        for vertex in outside:
            positions = np.flatnonzero(conflicts[vertex, selected]).astype(int)
            if 1 <= len(positions) <= max_remove:
                key = tuple(int(selected[p]) for p in positions.tolist())
                groups[key].append(vertex)

        best_delta = 0
        best_remove: tuple[int, ...] | None = None
        best_add: list[int] = []
        for remove, vertices in groups.items():
            add = greedy_independent_subset(vertices, conflicts)
            delta = len(add) - len(remove)
            if delta > best_delta:
                best_delta = delta
                best_remove = remove
                best_add = add
        if best_delta <= 0 or best_remove is None:
            break
        chosen = sorted((selected_set - set(best_remove)) | set(best_add))
        if not independent_check(chosen, conflicts):
            raise SearchError("local trade produced a conflicting set")
        event = {
            "pass": pass_index,
            "removed": list(best_remove),
            "added": best_add,
            "delta": best_delta,
            "size": len(chosen),
        }
        events.append(event)
        print(f"LOCAL TRADE {event}", flush=True)
        if len(chosen) >= CAP_TARGET:
            break
    return chosen, events


def milp_candidate_subpool(
    points: np.ndarray,
    conflicts: np.ndarray,
    *,
    max_vertices: int,
) -> list[int]:
    n = len(points)
    if n <= max_vertices:
        return list(range(n))
    old_conflict_count = conflicts[CAP_RECORD:, :CAP_RECORD].sum(axis=1)
    new = np.arange(CAP_RECORD, n)
    new_degree = conflicts[CAP_RECORD:, :].sum(axis=1)
    order = np.lexsort((new_degree, old_conflict_count))
    keep_new = new[order[: max_vertices - CAP_RECORD]].astype(int).tolist()
    return [*range(CAP_RECORD), *keep_new]


def solve_milp(
    conflicts: np.ndarray,
    vertices: list[int],
    *,
    time_limit: float,
) -> dict[str, Any]:
    local = conflicts[np.ix_(vertices, vertices)].copy()
    np.fill_diagonal(local, False)
    rows, cols = np.triu(local, k=1).nonzero()
    edge_count = int(len(rows))
    n = len(vertices)
    if edge_count > 1_500_000:
        return {
            "attempted": False,
            "reason": f"edge count {edge_count} exceeds safety limit",
            "vertices": n,
            "edges": edge_count,
        }
    data = np.ones(2 * edge_count, dtype=float)
    row_index = np.repeat(np.arange(edge_count, dtype=np.int64), 2)
    col_index = np.column_stack([rows, cols]).reshape(-1)
    matrix = coo_matrix((data, (row_index, col_index)), shape=(edge_count, n)).tocsr()
    constraint = LinearConstraint(
        matrix,
        lb=np.full(edge_count, -np.inf),
        ub=np.ones(edge_count),
    )
    start = time.monotonic()
    result = milp(
        c=-np.ones(n),
        integrality=np.ones(n),
        bounds=Bounds(np.zeros(n), np.ones(n)),
        constraints=constraint,
        options={
            "time_limit": float(time_limit),
            "mip_rel_gap": 0.0,
            "presolve": True,
        },
    )
    elapsed = time.monotonic() - start
    selected_local = (
        np.flatnonzero(np.asarray(result.x) > 0.5).astype(int).tolist()
        if result.x is not None
        else []
    )
    selected = [vertices[i] for i in selected_local]
    if selected and not independent_check(selected, conflicts):
        raise SearchError("MILP returned a conflicting set")
    return {
        "attempted": True,
        "status": int(result.status),
        "message": str(result.message),
        "success": bool(result.success),
        "vertices": n,
        "edges": edge_count,
        "elapsed_seconds": elapsed,
        "objective_count": len(selected),
        "selected_indices": selected,
        "mip_gap": None if getattr(result, "mip_gap", None) is None else float(result.mip_gap),
        "mip_node_count": None
        if getattr(result, "mip_node_count", None) is None
        else int(result.mip_node_count),
        "proven_optimal": bool(result.success and result.status == 0),
    }


def rational_level_diagnostic(matrix: np.ndarray, max_denominator: int = 512) -> dict[str, Any]:
    values = np.unique(np.round(matrix.reshape(-1), 10))
    best_denominator = None
    best_error = math.inf
    for denominator in range(1, max_denominator + 1):
        error = float(
            np.max(np.abs(values * denominator - np.rint(values * denominator)))
            / denominator
        )
        if error < best_error:
            best_error = error
            best_denominator = denominator
        if error < 2e-8:
            break
    fractions = Counter(
        str(Fraction(float(value)).limit_denominator(max_denominator)) for value in values
    )
    return {
        "unique_rounded_levels": int(len(values)),
        "best_common_denominator": best_denominator,
        "best_max_error": best_error,
        "sample_recognized_levels": sorted(fractions)[:100],
        "exact_certificate": False,
    }


def evaluate_selection(
    selected: list[int],
    points: np.ndarray,
    roots: np.ndarray,
    labels: list[list[str]],
) -> dict[str, Any]:
    selected = sorted(set(selected))
    chosen = points[selected]
    gram = chosen @ chosen.T
    off = gram.copy()
    np.fill_diagonal(off, -9.0)
    max_pair_flat = int(np.argmax(off))
    max_pair = tuple(map(int, np.unravel_index(max_pair_flat, off.shape)))
    root_cross = roots @ chosen.T
    source_counts = Counter(
        label.rstrip("+-")
        for index in selected
        for label in labels[index]
    )
    return {
        "count": len(selected),
        "full_dimension13_count_if_used": FULL_FIXED_COUNT + len(selected),
        "beats_live_dimension13_record_numerically": FULL_FIXED_COUNT + len(selected)
        > LIVE_D13_RECORD,
        "selected_indices": selected,
        "baseline_cap_points_kept": sum(index < CAP_RECORD for index in selected),
        "new_points_selected": sum(index >= CAP_RECORD for index in selected),
        "max_pair_inner": float(off[max_pair]),
        "max_pair_local_indices": list(max_pair),
        "max_root_inner": float(np.max(root_cross)),
        "max_norm_error": float(np.max(np.abs(np.sum(chosen * chosen, axis=1) - 1))),
        "source_label_counts": dict(source_counts),
        "cap_gram_rational_diagnostic": rational_level_diagnostic(gram),
        "root_cross_rational_diagnostic": rational_level_diagnostic(root_cross),
        "exact_verified": False,
        "note": "Numerical finite-pool candidate only; not an exact kissing configuration.",
    }


def report_text(result: dict[str, Any]) -> str:
    best = result["best_selection"]
    return "\n".join(
        [
            "# Continuous candidate search for the PackingStar variant rank-7 cap",
            "",
            f"Generated `{result['generated_at']}`.",
            "",
            "## Fixed exact framework",
            "",
            "The exact source decomposition is `1008 + 54 + 84 = 1146`. Holding the",
            "1,008-point fiber core and 54-point rank-6 cap fixed, beating the current",
            "dimension-13 record requires at least **93** rank-7 cap points.",
            "",
            "## Search result",
            "",
            f"- Best finite-pool cap: **{best['count']}**.",
            f"- Implied full count: **{best['full_dimension13_count_if_used']}**.",
            f"- Numerical max cap inner product: `{best['max_pair_inner']:.12g}`.",
            f"- Numerical max retained-root inner product: `{best['max_root_inner']:.12g}`.",
            f"- Exact certificate: **{best['exact_verified']}**.",
            "",
            "No count from this report is a new kissing-number lower bound unless an",
            "explicit exact certificate is subsequently produced.",
            "",
            "## Candidate generation",
            "",
            "```json",
            json.dumps(result["candidate_generation"], indent=2, sort_keys=True),
            "```",
            "",
            "## Pool reduction",
            "",
            "```json",
            json.dumps(result["pool_reduction"], indent=2, sort_keys=True),
            "```",
            "",
            "## Greedy/local search",
            "",
            "```json",
            json.dumps(result["greedy_search"], indent=2, sort_keys=True),
            "```",
            "",
            "## MILP",
            "",
            "```json",
            json.dumps(result["milp"], indent=2, sort_keys=True),
            "```",
            "",
            "## Best numerical selection",
            "",
            "```json",
            json.dumps(best, indent=2, sort_keys=True),
            "```",
            "",
        ]
    )


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--matrix", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--progress-log", required=True, type=Path)
    parser.add_argument("--seed", type=int, default=2026082301)
    parser.add_argument("--random-samples", type=int, default=80000)
    parser.add_argument("--max-new", type=int, default=2600)
    parser.add_argument("--per-mask", type=int, default=8)
    parser.add_argument("--greedy-rounds", type=int, default=2500)
    parser.add_argument("--milp-vertices", type=int, default=1200)
    parser.add_argument("--milp-seconds", type=float, default=420.0)
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    args.progress_log.parent.mkdir(parents=True, exist_ok=True)

    roots, caps, exact_metadata = exact_variant_rank7(args.matrix)
    checkpoint(args.output_dir, "exact_framework_recovered", exact_metadata)

    candidates, labels, generation = generate_candidates(
        roots,
        caps,
        seed=args.seed,
        random_samples=args.random_samples,
    )
    checkpoint(args.output_dir, "continuous_candidates_generated", generation)

    points, reduced_labels, reduction = reduce_candidate_pool(
        candidates,
        labels,
        caps,
        seed=args.seed + 1000,
        max_new=args.max_new,
        per_mask=args.per_mask,
    )
    checkpoint(args.output_dir, "candidate_pool_reduced", reduction)

    conflicts = conflict_matrix(points)
    edge_count = int(np.count_nonzero(np.triu(conflicts, k=1)))
    graph_stats = {
        "vertices": len(points),
        "conflict_edges": edge_count,
        "minimum_conflict_degree": int(conflicts.sum(axis=1).min() - 1),
        "maximum_conflict_degree": int(conflicts.sum(axis=1).max() - 1),
        "baseline_cap_independent": independent_check(range(CAP_RECORD), conflicts),
    }
    checkpoint(args.output_dir, "compatibility_graph_built", graph_stats)

    def save_best(selection: list[int], iteration: int) -> None:
        payload = evaluate_selection(selection, points, roots, reduced_labels)
        payload["greedy_iteration"] = iteration
        write_json(args.output_dir / "best_numerical.json", payload)

    greedy, greedy_stats = randomized_greedy(
        conflicts,
        seed=args.seed + 2000,
        rounds=args.greedy_rounds,
        checkpoint_callback=save_best,
    )
    improved, trade_events = local_trade_improve(greedy, conflicts)
    if len(improved) > len(greedy):
        greedy = improved
    greedy_stats["local_trade_events"] = trade_events
    greedy_stats["post_trade_best"] = len(greedy)
    save_best(greedy, args.greedy_rounds)
    checkpoint(
        args.output_dir,
        "greedy_and_local_search_complete",
        {"best": len(greedy), "statistics": greedy_stats},
    )

    milp_vertices = milp_candidate_subpool(
        points, conflicts, max_vertices=args.milp_vertices
    )
    milp_result = solve_milp(
        conflicts,
        milp_vertices,
        time_limit=args.milp_seconds,
    )
    milp_selection = milp_result.get("selected_indices", [])
    best = greedy
    if len(milp_selection) > len(best):
        best = milp_selection
    best, final_trades = local_trade_improve(best, conflicts)
    milp_result["post_milp_trade_events"] = final_trades
    milp_result["post_milp_best"] = len(best)

    best_evaluation = evaluate_selection(best, points, roots, reduced_labels)
    write_json(args.output_dir / "best_numerical.json", best_evaluation)
    np.save(args.output_dir / "candidate_pool.npy", points)
    np.save(args.output_dir / "best_cap_coordinates.npy", points[best])
    write_json(args.output_dir / "candidate_labels.json", reduced_labels)

    result = {
        "generated_at": utcnow(),
        "status": "completed_numerical_continuous_candidate_search",
        "dimension": 13,
        "live_record": LIVE_D13_RECORD,
        "fixed_exact_framework": exact_metadata,
        "cap_baseline": CAP_RECORD,
        "cap_target_for_record": CAP_TARGET,
        "candidate_generation": generation,
        "pool_reduction": reduction,
        "compatibility_graph": graph_stats,
        "greedy_search": greedy_stats,
        "milp": milp_result,
        "best_selection": best_evaluation,
        "exact_result": False,
    }
    write_json(args.output_dir / "analysis.json", result)
    (args.output_dir / "REPORT.md").write_text(report_text(result), encoding="utf-8")

    outcome = "pass" if best_evaluation["count"] > CAP_RECORD else "fail"
    with args.progress_log.open("a", encoding="utf-8") as log:
        log.write(
            f"{utcnow()} method='PackingStar variant continuous rank7 cap search' "
            f"dimension=13 count={best_evaluation['full_dimension13_count_if_used']} "
            f"result={outcome} exact=no cap_count={best_evaluation['count']} "
            f"target_cap={CAP_TARGET}\n"
        )
    print(json.dumps(result, indent=2, sort_keys=True), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
