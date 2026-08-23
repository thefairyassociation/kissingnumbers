#!/usr/bin/env python3
"""Numerically scan one-axis-for-two-point trades in the ZE99 auxiliary code.

This is candidate generation only.  It samples vertices of the linear polytope
cut out by the fixed 1106-vector base and the 23 retained cross-polytope points.
A reported trade is not a result until reconstructed and verified exactly.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
from scipy.optimize import linprog

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE / "constructions"))
from ze99 import generate_ab  # noqa: E402
from ze99_aux_core import constraint_rows  # noqa: E402

FIXED_BOUND = 2 / math.sqrt(3)
PAIR_BOUND = 1 / 3


def write_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def retained_axis_constraints(removed_axis: int) -> tuple[np.ndarray, np.ndarray]:
    rows = []
    for axis in range(12):
        for sign in (-1, 1):
            if axis == removed_axis and sign == 1:
                continue
            row = np.zeros(12)
            row[axis] = sign
            rows.append(row)
    return np.asarray(rows), np.full(23, PAIR_BOUND)


def deduplicate(points: list[tuple[np.ndarray, np.ndarray]]) -> list[tuple[np.ndarray, np.ndarray]]:
    unique: dict[tuple[float, ...], tuple[np.ndarray, np.ndarray]] = {}
    for vertex, unit in points:
        key = tuple(np.round(unit, 9).tolist())
        old = unique.get(key)
        if old is None or np.linalg.norm(vertex) > np.linalg.norm(old[0]):
            unique[key] = (vertex, unit)
    return list(unique.values())


def scan_axis(
    fixed_rows: np.ndarray,
    removed_axis: int,
    samples: int,
    seed: int,
) -> dict:
    axis_rows, axis_bounds = retained_axis_constraints(removed_axis)
    A = np.vstack([fixed_rows, axis_rows])
    b = np.concatenate([np.full(len(fixed_rows), FIXED_BOUND), axis_bounds])
    rng = np.random.default_rng(seed)

    directions = [row.copy() for row in np.eye(12)]
    directions.extend([-row.copy() for row in np.eye(12)])
    directions.extend(rng.normal(size=(samples, 12)))
    biased = rng.normal(scale=0.35, size=(samples // 2, 12))
    biased[:, removed_axis] += 1.0
    directions.extend(biased)

    vertices: list[tuple[np.ndarray, np.ndarray]] = []
    solves = 0
    failures = 0
    for raw in directions:
        norm = float(np.linalg.norm(raw))
        if norm < 1e-14:
            continue
        direction = raw / norm
        result = linprog(
            -direction,
            A_ub=A,
            b_ub=b,
            bounds=[(None, None)] * 12,
            method="highs",
        )
        if not result.success:
            failures += 1
            continue
        solves += 1
        vertex = np.asarray(result.x, dtype=float)
        radius = float(np.linalg.norm(vertex))
        if radius < 1 - 2e-9:
            continue
        unit = vertex / radius
        if float(np.max(fixed_rows @ unit)) > FIXED_BOUND + 2e-8:
            continue
        if float(np.max(axis_rows @ unit)) > PAIR_BOUND + 2e-8:
            continue
        vertices.append((vertex, unit))

    vertices = deduplicate(vertices)
    if not vertices:
        return {
            "removed_axis": removed_axis,
            "lp_solves": solves,
            "lp_failures": failures,
            "candidate_directions": 0,
            "trade_found": False,
        }

    units = np.asarray([unit for _, unit in vertices])
    raw_vertices = np.asarray([vertex for vertex, _ in vertices])
    gram = units @ units.T
    np.fill_diagonal(gram, np.inf)
    pair_flat = int(np.argmin(gram))
    i, j = map(int, np.unravel_index(pair_flat, gram.shape))
    best_dot = float(gram[i, j])
    trade = best_dot <= PAIR_BOUND + 2e-8

    removed = np.zeros(12)
    removed[removed_axis] = 1
    retained = np.vstack(
        [
            sign * np.eye(12)[axis]
            for axis in range(12)
            for sign in (-1, 1)
            if not (axis == removed_axis and sign == 1)
        ]
    )
    y, z = units[i], units[j]
    diagnostics = {
        "removed_axis": removed_axis,
        "lp_solves": solves,
        "lp_failures": failures,
        "candidate_directions": len(vertices),
        "maximum_vertex_norm": float(np.max(np.linalg.norm(raw_vertices, axis=1))),
        "minimum_sampled_pair_inner_product": best_dot,
        "trade_found": trade,
        "candidate_y": y.tolist(),
        "candidate_z": z.tolist(),
        "vertex_y": raw_vertices[i].tolist(),
        "vertex_z": raw_vertices[j].tolist(),
        "checks": {
            "y_norm": float(np.linalg.norm(y)),
            "z_norm": float(np.linalg.norm(z)),
            "y_z": float(y @ z),
            "fixed_y_max": float(np.max(fixed_rows @ y)),
            "fixed_z_max": float(np.max(fixed_rows @ z)),
            "retained_y_max": float(np.max(retained @ y)),
            "retained_z_max": float(np.max(retained @ z)),
            "removed_y_inner": float(removed @ y),
            "removed_z_inner": float(removed @ z),
        },
        "active_y": np.flatnonzero(b - A @ raw_vertices[i] < 2e-7).astype(int).tolist(),
        "active_z": np.flatnonzero(b - A @ raw_vertices[j] < 2e-7).astype(int).tolist(),
    }
    return diagnostics


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--samples", type=int, default=384)
    parser.add_argument("--progress-log", type=Path)
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now(timezone.utc).isoformat()

    fixed_rows, row_info = constraint_rows(generate_ab)
    fixed_rows = fixed_rows.astype(float)
    results = []
    for axis in range(12):
        result = scan_axis(fixed_rows, axis, args.samples, 20260823 + axis)
        results.append(result)
        write_json(args.output_dir / "checkpoint.json", {"completed": results})

    trades = [item for item in results if item.get("trade_found")]
    best = min(
        results,
        key=lambda item: item.get("minimum_sampled_pair_inner_product", math.inf),
    )
    payload = {
        "generated_at": stamp,
        "status": "numerical_candidate_scan",
        "method": "random LP vertex sampling for one-axis-to-two-point trades",
        "dimension": 13,
        "samples_per_axis": args.samples,
        "constraint_rows": row_info,
        "axes_scanned": 12,
        "trade_candidates_found": len(trades),
        "best_sampled_axis": best.get("removed_axis"),
        "best_sampled_pair_inner_product": best.get("minimum_sampled_pair_inner_product"),
        "exact_result": False,
        "warning": "Floating-point candidate generation only; any trade must be reconstructed and verified exactly.",
        "results": results,
    }
    write_json(args.output_dir / "analysis.json", payload)
    (args.output_dir / "REPORT.md").write_text(
        "\n".join(
            [
                "# ZE99 one-for-two auxiliary trade scan",
                "",
                f"- Axes scanned: **12** (negative removals follow by central symmetry).",
                f"- LP samples per axis: **{args.samples}**.",
                f"- Numerical trade candidates: **{len(trades)}**.",
                f"- Best sampled pair inner product: **{payload['best_sampled_pair_inner_product']}**.",
                "- Status: **numerical only; not an exact kissing result**.",
                "",
                "```json",
                json.dumps(payload, indent=2, sort_keys=True),
                "```",
                "",
            ]
        ),
        encoding="utf-8",
    )
    if args.progress_log:
        args.progress_log.parent.mkdir(parents=True, exist_ok=True)
        with args.progress_log.open("a", encoding="utf-8") as log:
            log.write(
                f"{stamp} method='ZE99 one-for-two LP trade scan' dimension=13 "
                f"count={'1156?' if trades else 1154} result={'candidate' if trades else 'fail'} "
                "exact=false\n"
            )
    print(json.dumps(payload, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
