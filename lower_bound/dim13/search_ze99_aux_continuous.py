#!/usr/bin/env python3
"""Continuous search for a 25th ZE99 auxiliary direction.

ZE99 consists of a fixed exact 1,106-vector base and two independent 24-point
auxiliary layers. A unit direction y in R^12 is admissible for either layer iff

    |r.y| <= 2/sqrt(3)   for all 960 exact constraint rows r,

and directions in the same layer must satisfy y.z <= 1/3. Replacing one
24-point layer by 25 admissible directions would give 1,155 vectors in R^13.

This program first samples vertices of the exact slab polytope by linear
programming, searches the finite compatibility graph, and then refines 25-point
codes directly on the sphere with a smooth-max Riemannian/L-BFGS objective.
All reported candidates are numerical until a separate exact reconstruction and
full Q(sqrt(3)) verifier pass.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
import time
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Iterable

import numpy as np
from scipy.optimize import Bounds, LinearConstraint, linprog, milp, minimize
from scipy.sparse import coo_matrix

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE / "constructions"))
from ze99 import generate_ab  # noqa: E402
from ze99_aux_core import constraint_rows, exact_maximum_clique  # noqa: E402

DIMENSION = 12
TARGET = 25
BASELINE = 24
FIXED_BASE = 1106
RECORD = 1154
PAIR_BOUND = 1.0 / 3.0
FIXED_BOUND = 2.0 / math.sqrt(3.0)


class SearchError(RuntimeError):
    pass


def utcnow() -> str:
    return datetime.now(timezone.utc).isoformat()


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def row_key(row: np.ndarray, decimals: int = 10) -> tuple[int, ...]:
    return tuple(np.rint(row * (10**decimals)).astype(np.int64).tolist())


def baseline_axes() -> np.ndarray:
    eye = np.eye(DIMENSION)
    return np.vstack([eye, -eye])


def metrics(points: np.ndarray, rows: np.ndarray) -> dict[str, float]:
    norms = np.linalg.norm(points, axis=1)
    normalized = points / norms[:, None]
    gram = normalized @ normalized.T
    np.fill_diagonal(gram, -np.inf)
    pair_max = float(np.max(gram))
    fixed_max = float(np.max(np.abs(normalized @ rows.T)))
    return {
        "pair_max": pair_max,
        "fixed_max": fixed_max,
        "pair_violation": pair_max - PAIR_BOUND,
        "fixed_violation": fixed_max - FIXED_BOUND,
        "max_violation": max(pair_max - PAIR_BOUND, fixed_max - FIXED_BOUND),
        "norm_max_error": float(np.max(np.abs(norms - 1.0))),
    }


def add_candidate(
    store: dict[tuple[int, ...], tuple[np.ndarray, set[str]]],
    raw: np.ndarray,
    rows: np.ndarray,
    label: str,
    *,
    tolerance: float = 3e-8,
) -> bool:
    norm = float(np.linalg.norm(raw))
    if norm < 1.0 - 3e-9:
        return False
    point = raw / norm
    if float(np.max(np.abs(rows @ point))) > FIXED_BOUND + tolerance:
        return False
    inserted = False
    for signed, suffix in ((point, "+"), (-point, "-")):
        key = row_key(signed)
        if key in store:
            store[key][1].add(label + suffix)
        else:
            store[key] = (signed.copy(), {label + suffix})
            inserted = True
    return inserted


def sample_slab_vertices(
    rows: np.ndarray,
    *,
    seed: int,
    random_directions: int,
    fixed_point_rounds: int,
    checkpoint: Callable[[dict[str, Any]], None] | None = None,
) -> tuple[np.ndarray, list[list[str]], dict[str, Any]]:
    rng = np.random.default_rng(seed)
    store: dict[tuple[int, ...], tuple[np.ndarray, set[str]]] = {}
    for axis in baseline_axes():
        add_candidate(store, axis, rows, "baseline_axis")

    directions: list[np.ndarray] = []
    directions.extend(np.eye(DIMENSION))
    directions.extend(-np.eye(DIMENSION))
    directions.extend(rows / np.linalg.norm(rows, axis=1)[:, None])
    directions.extend(rng.normal(size=(random_directions, DIMENSION)))

    solves = 0
    failures = 0
    norm_distribution: Counter[str] = Counter()
    new_from_lp = 0
    best_vertex_norm = -math.inf
    best_vertex: np.ndarray | None = None
    started = time.monotonic()

    for direction_index, raw_direction in enumerate(directions):
        direction = np.asarray(raw_direction, dtype=float)
        direction /= np.linalg.norm(direction)
        for _ in range(fixed_point_rounds):
            result = linprog(
                c=-direction,
                A_ub=rows,
                b_ub=np.full(len(rows), FIXED_BOUND),
                bounds=[(None, None)] * DIMENSION,
                method="highs",
            )
            if not result.success:
                failures += 1
                break
            solves += 1
            vertex = np.asarray(result.x, dtype=float)
            norm = float(np.linalg.norm(vertex))
            norm_distribution[f"{norm:.8f}"] += 1
            if norm > best_vertex_norm:
                best_vertex_norm = norm
                best_vertex = vertex.copy()
            before = len(store)
            add_candidate(store, vertex, rows, "slab_lp_vertex")
            if len(store) > before:
                new_from_lp += len(store) - before
            if norm < 1e-14:
                break
            new_direction = vertex / norm
            if float(np.linalg.norm(new_direction - direction)) < 1e-11:
                break
            direction = new_direction
        if checkpoint and direction_index and direction_index % 500 == 0:
            checkpoint(
                {
                    "stage": "slab_vertex_sampling",
                    "directions_completed": direction_index,
                    "directions_total": len(directions),
                    "lp_solves": solves,
                    "candidates": len(store),
                    "best_vertex_norm": best_vertex_norm,
                    "elapsed_seconds": time.monotonic() - started,
                }
            )

    points = np.asarray([item[0] for item in store.values()], dtype=float)
    labels = [sorted(item[1]) for item in store.values()]
    fixed_values = np.max(np.abs(points @ rows.T), axis=1)
    if float(np.max(fixed_values)) > FIXED_BOUND + 4e-8:
        raise SearchError("candidate pool contains a fixed-base violation")
    stats = {
        "directions": len(directions),
        "random_directions": random_directions,
        "fixed_point_rounds": fixed_point_rounds,
        "lp_solves": solves,
        "lp_failures": failures,
        "candidate_count": len(points),
        "new_signed_candidates_from_lp": new_from_lp,
        "best_vertex_norm": best_vertex_norm,
        "best_vertex": None if best_vertex is None else best_vertex.tolist(),
        "vertex_norm_distribution_most_common": norm_distribution.most_common(50),
        "candidate_fixed_max": float(np.max(fixed_values)),
        "source_label_counts": dict(
            Counter(label.rstrip("+-") for group in labels for label in group)
        ),
        "elapsed_seconds": time.monotonic() - started,
    }
    return points, labels, stats


def compatibility_graph(points: np.ndarray) -> tuple[np.ndarray, list[int], np.ndarray, dict[str, Any]]:
    gram = points @ points.T
    compatible = gram <= PAIR_BOUND + 2e-9
    np.fill_diagonal(compatible, False)
    degree = compatible.sum(axis=1)
    order = np.argsort(-degree, kind="stable").astype(np.int32)
    reordered = compatible[np.ix_(order, order)]
    adjacency = [
        int.from_bytes(np.packbits(row, bitorder="little").tobytes(), "little")
        for row in reordered
    ]
    return compatible, adjacency, order, {
        "vertices": len(points),
        "compatible_edges": int(np.count_nonzero(np.triu(compatible, k=1))),
        "minimum_compatible_degree": int(degree.min()) if len(degree) else 0,
        "maximum_compatible_degree": int(degree.max()) if len(degree) else 0,
    }


def locate_baseline(points: np.ndarray) -> list[int]:
    lookup = {row_key(row): index for index, row in enumerate(points)}
    indices = []
    for axis in baseline_axes():
        index = lookup.get(row_key(axis))
        if index is None:
            raise SearchError("baseline axis missing from candidate pool")
        indices.append(index)
    return indices


def independent(indices: Iterable[int], compatible: np.ndarray) -> bool:
    chosen = np.asarray(sorted(set(map(int, indices))), dtype=np.int64)
    if len(chosen) <= 1:
        return True
    sub = compatible[np.ix_(chosen, chosen)].copy()
    np.fill_diagonal(sub, True)
    return bool(np.all(sub))


def greedy_cliques(
    compatible: np.ndarray,
    baseline: list[int],
    *,
    seed: int,
    rounds: int,
    checkpoint: Callable[[list[int], int], None] | None = None,
) -> tuple[list[int], dict[str, Any]]:
    rng = np.random.default_rng(seed)
    n = len(compatible)
    best = baseline.copy()
    if not independent(best, compatible):
        raise SearchError("known 24-axis code is not compatible")
    improvements: list[dict[str, Any]] = []
    static_degree = compatible.sum(axis=1).astype(float)
    completed = 0

    for iteration in range(rounds):
        completed = iteration + 1
        available = np.ones(n, dtype=bool)
        chosen: list[int] = []
        while np.any(available):
            pool = np.flatnonzero(available)
            local_degree = compatible[np.ix_(pool, pool)].sum(axis=1).astype(float)
            if iteration == 0:
                position = int(np.argmax(local_degree))
            else:
                top = max(1, min(len(pool), 1 + len(pool) // 20))
                score = local_degree + 0.08 * static_degree[pool]
                top_positions = np.argpartition(score, -top)[-top:]
                weights = np.exp(
                    (score[top_positions] - np.max(score[top_positions]))
                    / max(1.0, np.std(score[top_positions]) + 1e-9)
                )
                weights /= weights.sum()
                position = int(rng.choice(top_positions, p=weights))
            vertex = int(pool[position])
            chosen.append(vertex)
            available &= compatible[vertex]
        if len(chosen) > len(best):
            best = chosen
            event = {"iteration": iteration, "size": len(best)}
            improvements.append(event)
            print(f"FINITE GREEDY IMPROVEMENT {event}", flush=True)
            if checkpoint:
                checkpoint(best, iteration)
        if len(best) >= TARGET:
            break
    return best, {
        "rounds_requested": rounds,
        "rounds_completed": completed,
        "best": len(best),
        "improvements": improvements,
    }


def finite_milp(
    compatible: np.ndarray,
    *,
    max_vertices: int,
    time_limit: float,
) -> dict[str, Any]:
    n = len(compatible)
    conflicts = ~compatible
    np.fill_diagonal(conflicts, False)
    rows, cols = np.triu(conflicts, k=1).nonzero()
    edge_count = len(rows)
    if n > max_vertices or edge_count > 1_500_000:
        return {
            "attempted": False,
            "reason": f"pool too large: vertices={n}, conflict_edges={edge_count}",
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
        options={"time_limit": time_limit, "mip_rel_gap": 0.0, "presolve": True},
    )
    selected = (
        np.flatnonzero(np.asarray(result.x) > 0.5).astype(int).tolist()
        if result.x is not None
        else []
    )
    if selected and not independent(selected, compatible):
        raise SearchError("MILP returned a non-clique")
    return {
        "attempted": True,
        "vertices": n,
        "conflict_edges": edge_count,
        "status": int(result.status),
        "message": str(result.message),
        "success": bool(result.success),
        "objective_count": len(selected),
        "selected_indices": selected,
        "elapsed_seconds": time.monotonic() - start,
        "mip_gap": None if getattr(result, "mip_gap", None) is None else float(result.mip_gap),
        "mip_node_count": None
        if getattr(result, "mip_node_count", None) is None
        else int(result.mip_node_count),
        "proven_optimal": bool(result.success and result.status == 0),
    }


def normalized_logmeanexp(values: np.ndarray, beta: float) -> tuple[float, np.ndarray]:
    maximum = float(np.max(values))
    exponentials = np.exp(beta * (values - maximum))
    total = float(np.sum(exponentials))
    weights = exponentials / total
    value = maximum + (math.log(total) - math.log(len(values))) / beta
    return value, weights


def smooth_objective(
    flat: np.ndarray,
    rows: np.ndarray,
    count: int,
    beta: float,
) -> tuple[float, np.ndarray]:
    raw = flat.reshape(count, DIMENSION)
    norms = np.linalg.norm(raw, axis=1)
    if np.any(norms < 1e-10):
        return 1e6, np.ones_like(flat) * 1e3
    points = raw / norms[:, None]

    first, second = np.triu_indices(count, k=1)
    pair_values = np.sum(points[first] * points[second], axis=1) - PAIR_BOUND
    fixed_products = points @ rows.T
    fixed_values = np.abs(fixed_products).reshape(-1) - FIXED_BOUND

    pair_loss, pair_weights = normalized_logmeanexp(pair_values, beta)
    fixed_loss, fixed_weights_flat = normalized_logmeanexp(fixed_values, beta)
    group_values = np.array([pair_loss, fixed_loss], dtype=float)
    group_max = float(np.max(group_values))
    group_exp = np.exp(beta * (group_values - group_max))
    group_weights = group_exp / group_exp.sum()
    loss = group_max + (math.log(float(group_exp.sum())) - math.log(2.0)) / beta

    gradient_points = np.zeros_like(points)
    weighted_second = pair_weights[:, None] * points[second]
    weighted_first = pair_weights[:, None] * points[first]
    np.add.at(gradient_points, first, group_weights[0] * weighted_second)
    np.add.at(gradient_points, second, group_weights[0] * weighted_first)

    fixed_weights = fixed_weights_flat.reshape(count, len(rows))
    fixed_signed = fixed_weights * np.sign(fixed_products)
    gradient_points += group_weights[1] * (fixed_signed @ rows)

    radial = np.sum(gradient_points * points, axis=1)
    gradient_raw = (gradient_points - radial[:, None] * points) / norms[:, None]
    return float(loss), gradient_raw.reshape(-1)


def refine_code(
    initial: np.ndarray,
    rows: np.ndarray,
    *,
    betas: list[float],
    maxiter: int,
) -> tuple[np.ndarray, dict[str, Any]]:
    points = np.asarray(initial, dtype=float).copy()
    points /= np.linalg.norm(points, axis=1)[:, None]
    stages: list[dict[str, Any]] = []
    evaluations = 0
    for beta in betas:
        def objective(flat: np.ndarray) -> tuple[float, np.ndarray]:
            nonlocal evaluations
            evaluations += 1
            return smooth_objective(flat, rows, len(points), beta)

        result = minimize(
            objective,
            points.reshape(-1),
            method="L-BFGS-B",
            jac=True,
            options={
                "maxiter": maxiter,
                "maxls": 50,
                "ftol": 1e-15,
                "gtol": 1e-10,
            },
        )
        points = np.asarray(result.x, dtype=float).reshape(len(points), DIMENSION)
        points /= np.linalg.norm(points, axis=1)[:, None]
        stage_metrics = metrics(points, rows)
        stages.append(
            {
                "beta": beta,
                "success": bool(result.success),
                "status": int(result.status),
                "message": str(result.message),
                "iterations": int(result.nit),
                "function_evaluations": int(result.nfev),
                "objective": float(result.fun),
                **stage_metrics,
            }
        )
        if stage_metrics["max_violation"] < -2e-9:
            break
    return points, {"stages": stages, "evaluations": evaluations, "final": metrics(points, rows)}


def random_initializations(
    pool: np.ndarray,
    finite_best: list[int],
    *,
    seed: int,
    restarts: int,
) -> list[tuple[str, np.ndarray]]:
    rng = np.random.default_rng(seed)
    axes = baseline_axes()
    initializations: list[tuple[str, np.ndarray]] = []

    if len(finite_best) >= TARGET:
        initializations.append(("finite_25", pool[finite_best[:TARGET]].copy()))
    else:
        chosen = pool[finite_best].copy() if finite_best else axes.copy()
        if len(chosen) < BASELINE:
            chosen = axes.copy()
        pair_to_chosen = pool @ chosen.T
        score = np.max(pair_to_chosen, axis=1)
        for index in np.argsort(score)[: min(16, len(pool))]:
            initial = np.vstack([chosen[:BASELINE], pool[int(index)]])
            initializations.append((f"finite_extra_{int(index)}", initial))

    for restart in range(restarts):
        amplitude = float(np.random.default_rng(seed + restart).choice([0.025, 0.06, 0.12, 0.22, 0.4]))
        perturbed = axes + amplitude * rng.normal(size=axes.shape)
        perturbed /= np.linalg.norm(perturbed, axis=1)[:, None]
        if len(pool):
            extra = pool[int(rng.integers(0, len(pool)))].copy()
        else:
            extra = rng.normal(size=DIMENSION)
            extra /= np.linalg.norm(extra)
        initializations.append((f"perturbed_axes_{restart}", np.vstack([perturbed, extra])))

    for restart in range(max(4, restarts // 3)):
        dense = rng.normal(size=(TARGET, DIMENSION))
        dense /= np.linalg.norm(dense, axis=1)[:, None]
        initializations.append((f"dense_random_{restart}", dense))
    return initializations


def selection_payload(
    indices: list[int],
    points: np.ndarray,
    labels: list[list[str]],
    rows: np.ndarray,
) -> dict[str, Any]:
    selected = points[indices]
    return {
        "count": len(indices),
        "indices": indices,
        "labels": [labels[index] for index in indices],
        "metrics": metrics(selected, rows),
        "coordinates": selected.tolist(),
        "exact": False,
    }


def report_text(result: dict[str, Any]) -> str:
    best = result["continuous_search"]["best"]
    return "\n".join(
        [
            "# Continuous ZE99 auxiliary-layer search",
            "",
            f"Generated `{result['generated_at']}`.",
            "",
            "The exact 1,106-vector ZE99 base is held fixed. One auxiliary sign layer",
            "currently has 24 directions; a feasible 25-direction layer would give",
            "`1106 + 25 + 24 = 1155` vectors in dimension 13.",
            "",
            "## Best numerical result",
            "",
            f"- Candidate count: **{best['count']}**.",
            f"- Pair maximum: `{best['metrics']['pair_max']:.12g}` (bound `1/3`).",
            f"- Fixed-base maximum: `{best['metrics']['fixed_max']:.12g}` (bound `2/sqrt(3)`).",
            f"- Maximum violation: `{best['metrics']['max_violation']:.12g}`.",
            f"- Numerical feasibility: **{best['numerically_feasible']}**.",
            "- Exact certificate: **false**.",
            "",
            "A numerical candidate is not a kissing-number result. Success requires exact",
            "reconstruction and the full pairwise Q(sqrt(3)) verifier.",
            "",
            "## Slab-polytope sampling",
            "",
            "```json",
            json.dumps(result["slab_sampling"], indent=2, sort_keys=True),
            "```",
            "",
            "## Finite compatibility search",
            "",
            "```json",
            json.dumps(result["finite_search"], indent=2, sort_keys=True),
            "```",
            "",
            "## Continuous optimization",
            "",
            "```json",
            json.dumps(result["continuous_search"], indent=2, sort_keys=True),
            "```",
            "",
        ]
    )


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--progress-log", required=True, type=Path)
    parser.add_argument("--seed", type=int, default=2026082302)
    parser.add_argument("--random-directions", type=int, default=2400)
    parser.add_argument("--fixed-point-rounds", type=int, default=4)
    parser.add_argument("--greedy-rounds", type=int, default=3000)
    parser.add_argument("--exact-seconds", type=float, default=240.0)
    parser.add_argument("--milp-vertices", type=int, default=1300)
    parser.add_argument("--milp-seconds", type=float, default=300.0)
    parser.add_argument("--restarts", type=int, default=14)
    parser.add_argument("--lbfgs-maxiter", type=int, default=350)
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    args.progress_log.parent.mkdir(parents=True, exist_ok=True)

    rows, row_info = constraint_rows(generate_ab)
    rows = rows.astype(float)
    if rows.shape != (960, DIMENSION):
        raise SearchError(f"unexpected constraint matrix shape {rows.shape}")

    def checkpoint(payload: dict[str, Any]) -> None:
        write_json(args.output_dir / "checkpoint.json", {"generated_at": utcnow(), **payload})
        print(json.dumps(payload, indent=2, sort_keys=True), flush=True)

    points, labels, sampling = sample_slab_vertices(
        rows,
        seed=args.seed,
        random_directions=args.random_directions,
        fixed_point_rounds=args.fixed_point_rounds,
        checkpoint=checkpoint,
    )
    checkpoint({"stage": "slab_sampling_complete", **sampling})
    np.save(args.output_dir / "slab_candidate_pool.npy", points)
    write_json(args.output_dir / "slab_candidate_labels.json", labels)

    compatible, adjacency, order, graph_info = compatibility_graph(points)
    baseline_original = locate_baseline(points)
    inverse = np.empty(len(order), dtype=np.int32)
    inverse[order] = np.arange(len(order), dtype=np.int32)
    baseline_reordered = [int(inverse[index]) for index in baseline_original]

    def finite_checkpoint(reordered_clique: list[int], iteration: int) -> None:
        original = order[np.asarray(reordered_clique, dtype=np.int32)].astype(int).tolist()
        payload = selection_payload(original, points, labels, rows)
        payload.update({"stage": "finite_search", "iteration": iteration})
        write_json(args.output_dir / "best_finite.json", payload)

    greedy_original, greedy_info = greedy_cliques(
        compatible,
        baseline_original,
        seed=args.seed + 1,
        rounds=args.greedy_rounds,
    )
    finite_best = greedy_original.copy()

    if len(points) <= 3500:
        exact_reordered, exact_info = exact_maximum_clique(
            adjacency,
            baseline_reordered,
            args.exact_seconds,
            lambda clique, nodes: finite_checkpoint(clique, nodes),
        )
        exact_original = order[np.asarray(exact_reordered, dtype=np.int32)].astype(int).tolist()
        if len(exact_original) > len(finite_best):
            finite_best = exact_original
    else:
        exact_info = {
            "attempted": False,
            "reason": f"candidate pool {len(points)} exceeds exact-search safety limit",
        }

    if len(points) <= args.milp_vertices:
        milp_info = finite_milp(
            compatible,
            max_vertices=args.milp_vertices,
            time_limit=args.milp_seconds,
        )
        milp_selected = milp_info.get("selected_indices", [])
        if len(milp_selected) > len(finite_best):
            finite_best = milp_selected
    else:
        milp_info = {
            "attempted": False,
            "reason": f"candidate pool {len(points)} exceeds {args.milp_vertices}",
        }

    finite_payload = selection_payload(finite_best, points, labels, rows)
    write_json(args.output_dir / "best_finite.json", finite_payload)
    finite_search = {
        "graph": graph_info,
        "greedy": greedy_info,
        "exact_bitset": exact_info,
        "milp": milp_info,
        "best": finite_payload,
    }
    checkpoint(
        {
            "stage": "finite_search_complete",
            "best": len(finite_best),
            "graph": graph_info,
        }
    )

    initializations = random_initializations(
        points,
        finite_best,
        seed=args.seed + 2,
        restarts=args.restarts,
    )
    best_points: np.ndarray | None = None
    best_metrics: dict[str, float] | None = None
    runs: list[dict[str, Any]] = []
    betas = [12.0, 35.0, 100.0, 280.0, 700.0]
    for run_index, (name, initial) in enumerate(initializations):
        if len(initial) != TARGET:
            continue
        refined, info = refine_code(
            initial,
            rows,
            betas=betas,
            maxiter=args.lbfgs_maxiter,
        )
        run_metrics = info["final"]
        runs.append(
            {
                "run": run_index,
                "name": name,
                "initial_metrics": metrics(initial, rows),
                **info,
            }
        )
        print(
            f"CONTINUOUS RUN {run_index} {name}: max_violation="
            f"{run_metrics['max_violation']:.12g}",
            flush=True,
        )
        if best_metrics is None or run_metrics["max_violation"] < best_metrics["max_violation"]:
            best_points = refined.copy()
            best_metrics = run_metrics.copy()
            write_json(
                args.output_dir / "best_numerical_25.json",
                {
                    "count": TARGET,
                    "metrics": best_metrics,
                    "coordinates": best_points.tolist(),
                    "run": run_index,
                    "name": name,
                    "numerically_feasible": best_metrics["max_violation"] <= 2e-9,
                    "exact": False,
                },
            )
            checkpoint(
                {
                    "stage": "continuous_improvement",
                    "run": run_index,
                    "name": name,
                    "metrics": best_metrics,
                }
            )
        if best_metrics is not None and best_metrics["max_violation"] < -2e-8:
            break

    if best_points is None or best_metrics is None:
        raise SearchError("continuous optimizer produced no 25-point run")
    best_continuous = {
        "count": TARGET,
        "metrics": best_metrics,
        "coordinates": best_points.tolist(),
        "numerically_feasible": best_metrics["max_violation"] <= 2e-9,
        "implied_full_count": FIXED_BASE + TARGET + BASELINE,
        "beats_record_numerically": (
            FIXED_BASE + TARGET + BASELINE > RECORD
            and best_metrics["max_violation"] <= 2e-9
        ),
        "exact": False,
    }
    np.save(args.output_dir / "best_numerical_25.npy", best_points)
    write_json(args.output_dir / "best_numerical_25.json", best_continuous)

    result = {
        "generated_at": utcnow(),
        "status": "completed",
        "dimension": 13,
        "record": RECORD,
        "fixed_base_count": FIXED_BASE,
        "other_auxiliary_layer_count": BASELINE,
        "target_layer_count": TARGET,
        "implied_target_full_count": FIXED_BASE + BASELINE + TARGET,
        "constraint_rows": row_info,
        "bounds": {"pair": "1/3", "fixed": "2/sqrt(3)"},
        "slab_sampling": sampling,
        "finite_search": finite_search,
        "continuous_search": {
            "betas": betas,
            "initializations": len(initializations),
            "runs": runs,
            "best": best_continuous,
        },
        "exact_result": False,
    }
    write_json(args.output_dir / "analysis.json", result)
    (args.output_dir / "REPORT.md").write_text(report_text(result), encoding="utf-8")

    outcome = "numerical-pass" if best_continuous["numerically_feasible"] else "fail"
    with args.progress_log.open("a", encoding="utf-8") as log:
        log.write(
            f"{utcnow()} method='ZE99 continuous auxiliary-layer search' dimension=13 "
            f"count={best_continuous['implied_full_count']} result={outcome} exact=no "
            f"max_violation={best_metrics['max_violation']:.17g}\n"
        )
    print(json.dumps(result, indent=2, sort_keys=True), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
