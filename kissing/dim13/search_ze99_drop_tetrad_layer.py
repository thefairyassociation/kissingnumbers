#!/usr/bin/env python3
"""Exact search for a 177-point half-height layer after deleting one ZE99 tetrad support.

Deleting one of the 51 four-coordinate tetrad supports removes 16 equatorial
vectors, leaving 800.  If the relaxed half-height feasible region contains a
177-point 1/3-code B, then using B at both heights gives

    800 + 2 poles + 2*177 = 1156,

which beats the dimension-13 lower bound 1154.

For a primitive integer direction q in Z^12, lift

    q -> (2*sqrt(3/||q||^2) q, +/-2).

All feasibility and pairwise comparisons are reduced to exact integer squared
inequalities.  CP-SAT is used only to choose a subset; any witness is rebuilt
and checked independently by the exact verifier in this file before output.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from collections import Counter
from datetime import datetime, timezone
from itertools import combinations
from pathlib import Path
from typing import Any, Iterable

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE / "constructions"))
from ze99 import ab_to_strings, generate_ab, verify_ab  # noqa: E402

DIMENSION = 13
RECORD = 1154
TARGET_LAYER = 177
TETRAD_VECTOR_COUNT = 816
DIAMOND_START = 816
DIAMOND_STOP = 1104
POLE_START = 1104
POLE_STOP = 1106
AUX_START = 1106


class SearchError(RuntimeError):
    pass


def utcnow() -> str:
    return datetime.now(timezone.utc).isoformat()


def write_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def support_mask_from_ab(vector: list[tuple[int, int]]) -> int:
    mask = 0
    for coordinate, (a, b) in enumerate(vector[:12]):
        if b or a not in (-2, 0, 2):
            raise SearchError("unexpected equatorial tetrad coordinate")
        if a:
            mask |= 1 << coordinate
    if mask.bit_count() != 4:
        raise SearchError("equatorial vector does not have support four")
    return mask


def tetrad_supports(original: list[list[tuple[int, int]]]) -> tuple[list[int], np.ndarray]:
    masks: list[int] = []
    for vector in original[:TETRAD_VECTOR_COUNT]:
        mask = support_mask_from_ab(vector)
        if mask not in masks:
            masks.append(mask)
    if len(masks) != 51:
        raise SearchError(f"expected 51 tetrad supports, found {len(masks)}")
    matrix = np.zeros((51, 12), dtype=np.int16)
    for row, mask in enumerate(masks):
        for coordinate in range(12):
            if (mask >> coordinate) & 1:
                matrix[row, coordinate] = 1
    return masks, matrix


def enumerate_candidates(
    support_matrix: np.ndarray, drop_index: int
) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    if not 0 <= drop_index < 51:
        raise SearchError("drop-index must be between 0 and 50")
    retained = np.delete(support_matrix, drop_index, axis=0)

    total = 3**12
    work = np.arange(total, dtype=np.int64)
    magnitudes = np.empty((total, 12), dtype=np.int8)
    for coordinate in range(12):
        magnitudes[:, coordinate] = (work % 3).astype(np.int8)
        work //= 3
    norm2 = np.sum(magnitudes.astype(np.int16) ** 2, axis=1).astype(np.int16)
    primitive = (norm2 > 0) & np.any(magnitudes == 1, axis=1)
    magnitudes = magnitudes[primitive]
    norm2 = norm2[primitive]

    keep = np.empty(len(magnitudes), dtype=bool)
    for start in range(0, len(magnitudes), 4096):
        batch = magnitudes[start : start + 4096].astype(np.int16)
        nn = norm2[start : start + len(batch)].astype(np.int32)
        maximum = np.max(batch @ retained.T, axis=1).astype(np.int32)
        keep[start : start + len(batch)] = 3 * maximum * maximum <= 4 * nn
    magnitudes = magnitudes[keep]
    norm2 = norm2[keep]

    signed_rows: list[np.ndarray] = []
    signed_norms: list[np.ndarray] = []
    magnitude_supports: Counter[int] = Counter()
    magnitude_norms: Counter[int] = Counter()
    for magnitude, nn in zip(magnitudes, norm2):
        positions = np.flatnonzero(magnitude)
        magnitude_supports[len(positions)] += 1
        magnitude_norms[int(nn)] += 1
        codes = np.arange(1 << len(positions), dtype=np.uint16)
        bits = (codes[:, None] >> np.arange(len(positions), dtype=np.uint16)) & 1
        signs = np.where(bits != 0, 1, -1).astype(np.int8)
        rows = np.zeros((len(codes), 12), dtype=np.int8)
        rows[:, positions] = signs * magnitude[positions]
        signed_rows.append(rows)
        signed_norms.append(np.full(len(rows), int(nn), dtype=np.int16))

    q = np.vstack(signed_rows)
    n = np.concatenate(signed_norms)
    keys = {tuple(row.tolist()) for row in q}
    if len(keys) != len(q):
        raise SearchError(f"candidate enumeration has {len(q)-len(keys)} duplicates")

    full_sign = np.all(np.abs(q) == 1, axis=1)
    if int(np.count_nonzero(full_sign)) != 4096:
        raise SearchError("the full binary sign cube was not preserved")

    retained16 = retained.astype(np.int16)
    for start in range(0, len(q), 512):
        batch = np.abs(q[start : start + 512]).astype(np.int16)
        nn = n[start : start + len(batch)].astype(np.int32)
        maximum = np.max(batch @ retained16.T, axis=1).astype(np.int32)
        if np.any(3 * maximum * maximum > 4 * nn):
            raise SearchError("signed candidate violates a retained tetrad constraint")

    return q, n, {
        "drop_index": drop_index,
        "retained_tetrad_supports": 50,
        "primitive_magnitude_patterns_tested": int(np.count_nonzero(primitive)),
        "magnitude_patterns_after_retained_tetrads": int(len(magnitudes)),
        "magnitude_support_distribution": dict(sorted(magnitude_supports.items())),
        "magnitude_norm_squared_distribution": dict(sorted(magnitude_norms.items())),
        "signed_direction_norm_squared_distribution": {
            str(int(value)): int(count)
            for value, count in zip(*np.unique(n, return_counts=True))
        },
        "candidate_count": int(len(q)),
        "binary_sign_directions": 4096,
        "nonbinary_directions": int(len(q) - 4096),
        "coordinate_bound": 2,
        "arithmetic": "integer squared inequalities; no floating point",
    }


def direction_compatible(q: np.ndarray, n: np.ndarray, i: int, j: int) -> bool:
    dot = int(q[i].astype(np.int32) @ q[j].astype(np.int32))
    return dot <= 0 or 9 * dot * dot <= int(n[i]) * int(n[j])


def sign_word(row: np.ndarray) -> int:
    if not np.all(np.abs(row) == 1):
        raise SearchError("not a binary sign direction")
    word = 0
    for coordinate, value in enumerate(row.tolist()):
        if value > 0:
            word |= 1 << coordinate
    return word


def edge_anticodes() -> list[tuple[int, ...]]:
    """All maximal diameter-three anticodes B1(x) union B1(y), d(x,y)=1."""
    anticodes: list[tuple[int, ...]] = []
    for x in range(1 << 12):
        for coordinate in range(12):
            if (x >> coordinate) & 1:
                continue
            y = x ^ (1 << coordinate)
            vertices = {x, y}
            for other in range(12):
                vertices.add(x ^ (1 << other))
                vertices.add(y ^ (1 << other))
            anticode = tuple(sorted(vertices))
            if len(anticode) != 24:
                raise SearchError("edge anticode does not have size 24")
            anticodes.append(anticode)
    if len(anticodes) != 24576 or len(set(anticodes)) != 24576:
        raise SearchError("edge anticode enumeration mismatch")
    return anticodes


def verify_anticode_cover(anticodes: list[tuple[int, ...]]) -> dict[str, int]:
    covered = np.zeros((1 << 12, 1 << 12), dtype=bool)
    for anticode in anticodes:
        array = np.asarray(anticode, dtype=np.int16)
        covered[np.ix_(array, array)] = True
    expected = 0
    missing = 0
    extraneous = 0
    for a in range(1 << 12):
        for b in range(a + 1, 1 << 12):
            close = (a ^ b).bit_count() < 4
            if close:
                expected += 1
                if not covered[a, b]:
                    missing += 1
            elif covered[a, b]:
                extraneous += 1
    if missing or extraneous:
        raise SearchError(
            f"invalid anticode cover: missing={missing}, extraneous={extraneous}"
        )
    return {
        "anticodes": len(anticodes),
        "vertices_per_anticode": 24,
        "forbidden_binary_pairs_covered": expected,
        "missing_forbidden_pairs": 0,
        "extraneous_compatible_pairs": 0,
    }


def baseline_directions(original: list[list[tuple[int, int]]]) -> list[np.ndarray]:
    directions: list[np.ndarray] = []
    for vector in original[DIAMOND_START:DIAMOND_STOP]:
        if vector[12] != (2, 0):
            continue
        row = np.asarray([a for a, b in vector[:12]], dtype=np.int8)
        if any(b != 0 for _, b in vector[:12]):
            raise SearchError("baseline diamond is not integral")
        directions.append(row)
    for vector in original[AUX_START:]:
        if vector[12] != (2, 0):
            continue
        row = np.zeros(12, dtype=np.int8)
        for coordinate, (a, b) in enumerate(vector[:12]):
            if a:
                raise SearchError("unexpected rational part in baseline axis")
            if b:
                if abs(b) != 2:
                    raise SearchError("unexpected irrational baseline coordinate")
                row[coordinate] = 1 if b > 0 else -1
        if int(np.count_nonzero(row)) != 1:
            raise SearchError("baseline auxiliary direction is not an axis")
        directions.append(row)
    if len(directions) != 168:
        raise SearchError(f"baseline half-height layer has {len(directions)} vectors")
    return directions


def solve_case(
    q: np.ndarray,
    n: np.ndarray,
    anticodes: list[tuple[int, ...]],
    original: list[list[tuple[int, int]]],
    case: str,
    time_limit: float,
    workers: int,
    seed: int,
) -> tuple[list[int] | None, dict[str, Any]]:
    try:
        from ortools.sat.python import cp_model
    except ImportError as exc:
        raise SearchError("OR-Tools is required for this exact finite search") from exc

    sign_indices = np.flatnonzero(np.all(np.abs(q) == 1, axis=1)).astype(int).tolist()
    nonsign_indices = np.flatnonzero(~np.all(np.abs(q) == 1, axis=1)).astype(int).tolist()
    if len(sign_indices) != 4096:
        raise SearchError("unexpected binary sign count")

    sign_original = [-1] * (1 << 12)
    for index in sign_indices:
        word = sign_word(q[index])
        if sign_original[word] >= 0:
            raise SearchError("duplicate binary sign word")
        sign_original[word] = index
    if any(index < 0 for index in sign_original):
        raise SearchError("binary sign cube is incomplete")

    model = cp_model.CpModel()
    sign_vars = [model.NewBoolVar(f"s_{word:03x}") for word in range(1 << 12)]
    nonsign_vars = [model.NewBoolVar(f"u_{local}") for local in range(len(nonsign_indices))]

    for anticode in anticodes:
        model.AddAtMostOne(sign_vars[word] for word in anticode)

    signs_q = q[np.asarray(sign_original, dtype=np.int32)].astype(np.int32)
    conditional_sign_terms = 0
    nonsign_forbidden_sizes: Counter[int] = Counter()
    for local, index in enumerate(nonsign_indices):
        dots = signs_q @ q[index].astype(np.int32)
        nn = int(n[index])
        incompatible = (dots > 0) & (9 * dots * dots > 12 * nn)
        words = np.flatnonzero(incompatible).astype(int).tolist()
        nonsign_forbidden_sizes[len(words)] += 1
        conditional_sign_terms += len(words)
        if words:
            model.Add(sum(sign_vars[word] for word in words) == 0).OnlyEnforceIf(
                nonsign_vars[local]
            )

    nonsign_conflicts = 0
    nq = q[nonsign_indices].astype(np.int32)
    nn = n[nonsign_indices].astype(np.int32)
    for start in range(0, len(nonsign_indices), 256):
        dots = nq[start : start + 256] @ nq.T
        left_norm = nn[start : start + len(dots), None]
        incompatible = (dots > 0) & (9 * dots * dots > left_norm * nn[None, :])
        for local_i in range(len(dots)):
            i = start + local_i
            for j in np.flatnonzero(incompatible[local_i, i + 1 :]).astype(int):
                jj = i + 1 + int(j)
                model.AddAtMostOne(nonsign_vars[i], nonsign_vars[jj])
                nonsign_conflicts += 1

    sign_count = sum(sign_vars)
    nonsign_count = sum(nonsign_vars)
    model.Add(sign_count <= 144)
    model.Add(sign_count + nonsign_count >= TARGET_LAYER)
    model.Add(nonsign_count >= TARGET_LAYER - 144)

    if case == "with_sign":
        model.Add(sign_vars[(1 << 12) - 1] == 1)
    elif case == "signless":
        model.Add(sign_count == 0)
    else:
        raise SearchError(f"unknown search case {case}")

    baseline = baseline_directions(original)
    anchor = next(row for row in baseline if np.all(np.abs(row) == 1))
    transformed = [row * anchor for row in baseline]
    candidate_lookup = {tuple(row.tolist()): i for i, row in enumerate(q)}
    hint_indices = {
        candidate_lookup[tuple(row.tolist())]
        for row in transformed
        if tuple(row.tolist()) in candidate_lookup
    }
    if case == "with_sign" and len(hint_indices) != 168:
        raise SearchError(f"known 168-code hint maps to {len(hint_indices)} candidates")

    for word, variable in enumerate(sign_vars):
        model.AddHint(variable, 1 if sign_original[word] in hint_indices else 0)
    for local, variable in enumerate(nonsign_vars):
        model.AddHint(variable, 1 if nonsign_indices[local] in hint_indices else 0)

    solver = cp_model.CpSolver()
    solver.parameters.max_time_in_seconds = float(time_limit)
    solver.parameters.num_search_workers = int(workers)
    solver.parameters.random_seed = int(seed)
    solver.parameters.log_search_progress = True
    solver.parameters.cp_model_presolve = True
    solver.parameters.symmetry_level = 2
    solver.parameters.linearization_level = 2
    solver.parameters.max_memory_in_mb = 7000
    status = solver.Solve(model)
    status_name = solver.StatusName(status)

    selected: list[int] | None = None
    if status in (cp_model.FEASIBLE, cp_model.OPTIMAL):
        selected = [
            sign_original[word]
            for word, variable in enumerate(sign_vars)
            if solver.Value(variable)
        ]
        selected.extend(
            nonsign_indices[local]
            for local, variable in enumerate(nonsign_vars)
            if solver.Value(variable)
        )
        if len(selected) < TARGET_LAYER:
            raise SearchError("CP-SAT returned a sub-target layer")
        if len(set(selected)) != len(selected):
            raise SearchError("CP-SAT returned duplicate candidate indices")
        for position, i in enumerate(selected):
            for j in selected[position + 1 :]:
                if not direction_compatible(q, n, i, j):
                    raise SearchError(f"CP-SAT selected incompatible pair {i},{j}")

    return selected, {
        "case": case,
        "status": status_name,
        "time_limit_seconds": time_limit,
        "wall_time_seconds": solver.WallTime(),
        "user_time_seconds": solver.UserTime(),
        "num_conflicts": solver.NumConflicts(),
        "num_branches": solver.NumBranches(),
        "response_stats": solver.ResponseStats(),
        "model": {
            "binary_sign_variables": len(sign_vars),
            "nonbinary_variables": len(nonsign_vars),
            "edge_anticode_constraints": len(anticodes),
            "conditional_sign_terms": conditional_sign_terms,
            "nonbinary_forbidden_sign_size_distribution": {
                str(key): int(value)
                for key, value in sorted(nonsign_forbidden_sizes.items())
            },
            "nonbinary_pair_conflicts": nonsign_conflicts,
            "target_layer": TARGET_LAYER,
            "known_binary_bound": "A_2(12,4)=144",
            "symmetry_break": (
                "all-plus binary sign selected" if case == "with_sign" else "no binary signs"
            ),
        },
        "solution": None
        if selected is None
        else {
            "layer_count": len(selected),
            "binary_signs": int(sum(np.all(np.abs(q[index]) == 1) for index in selected)),
            "nonbinary_directions": int(sum(not np.all(np.abs(q[index]) == 1) for index in selected)),
            "candidate_indices": selected,
        },
    }


def retained_equator(
    original: list[list[tuple[int, int]]], dropped_mask: int
) -> list[list[tuple[int, int]]]:
    kept = [
        vector
        for vector in original[:TETRAD_VECTOR_COUNT]
        if support_mask_from_ab(vector) != dropped_mask
    ]
    if len(kept) != 800:
        raise SearchError(f"retained equator has {len(kept)} vectors")
    return kept


def lifted_strings(row: np.ndarray, norm2: int, last: int) -> list[str]:
    import sympy as sp

    scale = 2 * sp.sqrt(sp.Rational(3, norm2))
    return [str(sp.simplify(int(value) * scale)) for value in row.tolist()] + [str(last)]


def exact_certificate(
    q: np.ndarray,
    n: np.ndarray,
    selected: list[int],
    support_matrix: np.ndarray,
    drop_index: int,
    original: list[list[tuple[int, int]]],
) -> tuple[list[list[str]], dict[str, Any]]:
    if len(selected) < TARGET_LAYER:
        raise SearchError("certificate layer is below target")
    selected_q = q[selected].astype(np.int32)
    selected_n = n[selected].astype(np.int32)
    if len({tuple(row.tolist()) for row in selected_q}) != len(selected_q):
        raise SearchError("certificate contains duplicate directions")
    if np.any(np.sum(selected_q * selected_q, axis=1) != selected_n):
        raise SearchError("certificate direction norm mismatch")

    retained = np.delete(support_matrix, drop_index, axis=0).astype(np.int32)
    magnitudes = np.abs(selected_q)
    maxima = np.max(magnitudes @ retained.T, axis=1)
    if np.any(3 * maxima * maxima > 4 * selected_n):
        raise SearchError("certificate violates a retained tetrad inequality")

    same_layer_pairs = 0
    opposite_layer_pairs = 0
    tight_same = 0
    tight_opposite = 0
    for i in range(len(selected_q)):
        dots = selected_q[i] @ selected_q[i + 1 :].T
        rhs = selected_n[i] * selected_n[i + 1 :]
        positive = dots > 0
        if np.any(positive & (9 * dots * dots > rhs)):
            raise SearchError("certificate half-layer has an incompatible pair")
        tight_same += int(np.count_nonzero(positive & (9 * dots * dots == rhs)))
        same_layer_pairs += len(dots)

        all_dots = selected_q[i] @ selected_q.T
        all_rhs = selected_n[i] * selected_n
        positive_all = all_dots > 0
        if np.any(positive_all & (all_dots * all_dots > all_rhs)):
            raise SearchError("opposite-height Cauchy check failed")
        tight_opposite += int(np.count_nonzero(positive_all & (all_dots * all_dots == all_rhs)))
        opposite_layer_pairs += len(all_dots)

    dropped_mask = 0
    for coordinate in np.flatnonzero(support_matrix[drop_index]).astype(int).tolist():
        dropped_mask |= 1 << coordinate
    equator = retained_equator(original, dropped_mask)
    poles = original[POLE_START:POLE_STOP]
    fixed_check = verify_ab(equator + poles)
    if not fixed_check.get("ok"):
        raise SearchError(f"retained integer fixed set failed exact verification: {fixed_check}")

    vectors: list[list[str]] = [ab_to_strings(vector) for vector in equator + poles]
    vectors.extend(
        lifted_strings(row, int(nn), 2) for row, nn in zip(selected_q, selected_n)
    )
    vectors.extend(
        lifted_strings(row, int(nn), -2) for row, nn in zip(selected_q, selected_n)
    )
    count = len(vectors)
    expected = 800 + 2 + 2 * len(selected)
    if count != expected or count <= RECORD:
        raise SearchError(f"bad full certificate count {count}, expected {expected}")
    if len({tuple(vector) for vector in vectors}) != len(vectors):
        raise SearchError("full coordinate list contains duplicates")

    total_pairs = count * (count - 1) // 2
    fixed_count = 802
    layer_count = len(selected)
    accounted = (
        fixed_count * (fixed_count - 1) // 2
        + 2 * fixed_count * layer_count
        + 2 * (layer_count * (layer_count - 1) // 2)
        + layer_count * layer_count
    )
    if accounted != total_pairs:
        raise SearchError("exact pair accounting mismatch")

    verification = {
        "ok": True,
        "dimension": DIMENSION,
        "count": count,
        "norm_squared": 16,
        "distinct": True,
        "all_off_diagonal_leq_8": True,
        "max_off_diagonal_unnormalized": "8",
        "max_off_diagonal_unit": "1/2",
        "arithmetic": "integers and exact squared radical inequalities; no floating point",
        "retained_fixed_verification": fixed_check,
        "retained_tetrad_supports": 50,
        "retained_equator_vectors": 800,
        "half_height_layer_count": layer_count,
        "half_height_direction_norm_squared_distribution": {
            str(int(value)): int(freq)
            for value, freq in zip(*np.unique(selected_n, return_counts=True))
        },
        "fixed_layer_pairs_checked": fixed_count * (fixed_count - 1) // 2,
        "fixed_to_half_height_pairs_checked_by_exact_squared_inequalities": 2 * fixed_count * layer_count,
        "same_height_pairs_checked_by_exact_squared_inequalities": 2 * same_layer_pairs,
        "opposite_height_pairs_checked_by_exact_squared_inequalities": opposite_layer_pairs,
        "total_unordered_pairs_checked": total_pairs,
        "tight_same_height_direction_pairs_per_layer": tight_same,
        "tight_opposite_height_pairs": tight_opposite,
    }
    return vectors, verification


def report_text(result: dict[str, Any]) -> str:
    lines = [
        "# Exact one-tetrad-deletion layer search",
        "",
        f"Generated `{result['generated_at']}`.",
        "",
        f"- Dropped support index: **{result['drop_index']}**.",
        f"- Dropped support: `{result['dropped_support']}`.",
        f"- Exact candidate directions: **{result['enumeration']['candidate_count']}**.",
        f"- Target half-height layer: **{TARGET_LAYER}**.",
        f"- Record witness found: **{result['beats_record']}**.",
        f"- Full dimension-13 count: **{result['candidate_count']}**.",
        "",
    ]
    for case in result["solver_cases"]:
        lines.append(f"- `{case['case']}` CP-SAT status: **{case['status']}**.")
    lines.append("")
    if result["beats_record"]:
        lines.extend(
            [
                "The emitted configuration passed the independent exact type-aware",
                "verifier, proving a dimension-13 lower bound above 1154.",
                "",
            ]
        )
    else:
        lines.extend(["No lower-bound improvement is claimed by this run.", ""])
    lines.extend(["```json", json.dumps(result, indent=2, sort_keys=True), "```", ""])
    return "\n".join(lines)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--drop-index", required=True, type=int)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--progress-log", type=Path)
    parser.add_argument("--time-limit", type=float, default=1350.0)
    parser.add_argument("--signless-seconds", type=float, default=120.0)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--seed", type=int, default=51)
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    stamp = utcnow()

    if args.progress_log:
        args.progress_log.parent.mkdir(parents=True, exist_ok=True)
        with args.progress_log.open("a", encoding="utf-8") as log:
            log.write(
                f"{stamp} method='drop one ZE99 tetrad support and search 177-layer' "
                f"drop_index={args.drop_index} dimension=13 count=? result=started\n"
            )

    original = generate_ab()
    baseline_check = verify_ab(original)
    if not baseline_check.get("ok"):
        raise SearchError("ZE99 baseline failed exact verification")
    masks, supports = tetrad_supports(original)
    q, n, enumeration = enumerate_candidates(supports, args.drop_index)
    anticodes = edge_anticodes()
    anticode_check = verify_anticode_cover(anticodes)

    selected, with_sign = solve_case(
        q, n, anticodes, original, "with_sign", args.time_limit, args.workers, args.seed
    )
    cases = [with_sign]
    if selected is None and args.signless_seconds > 0:
        signless_selected, signless = solve_case(
            q,
            n,
            anticodes,
            original,
            "signless",
            args.signless_seconds,
            args.workers,
            args.seed + 1000,
        )
        cases.append(signless)
        selected = signless_selected

    beats_record = selected is not None
    verification: dict[str, Any] | None = None
    config_name: str | None = None
    layer_payload: dict[str, Any] | None = None
    candidate_count = RECORD
    if selected is not None:
        vectors, verification = exact_certificate(q, n, selected, supports, args.drop_index, original)
        candidate_count = len(vectors)
        if candidate_count <= RECORD or not verification.get("ok"):
            raise SearchError("putative witness did not pass exact certification")
        config_name = f"best_exact_{candidate_count}.json"
        layer_payload = {
            "candidate_indices": selected,
            "directions_q": [q[index].astype(int).tolist() for index in selected],
            "direction_norm_squared": [int(n[index]) for index in selected],
        }
        write_json(
            args.output_dir / config_name,
            {
                "dimension": DIMENSION,
                "count": candidate_count,
                "norm_squared": 16,
                "unit_scaling": "divide every coordinate by 4",
                "coordinate_field": "real algebraic numbers",
                "vectors": vectors,
                "construction": {
                    "retained_equator": "800 ZE99 tetrad vectors after deleting one support",
                    "poles": 2,
                    "positive_half_height_layer": layer_payload,
                    "negative_half_height_layer": "same directions as positive layer",
                    "dropped_tetrad_support_index": args.drop_index,
                    "dropped_tetrad_support_mask": masks[args.drop_index],
                },
                "exact_verification": verification,
                "record_before": RECORD,
                "beats_record": True,
            },
        )
        write_json(
            args.output_dir / "best.json",
            {
                "dimension": DIMENSION,
                "count": candidate_count,
                "configuration": config_name,
                "record_before": RECORD,
                "beats_record": True,
                "exact": True,
                "max_off_diagonal_unit": "1/2",
            },
        )

    dropped_support = np.flatnonzero(supports[args.drop_index]).astype(int).tolist()
    result = {
        "generated_at": stamp,
        "status": "completed",
        "method": "delete one 16-vector ZE99 tetrad support and exactly search a mixed primitive bound-2 half-height layer",
        "dimension": DIMENSION,
        "record_before": RECORD,
        "target": RECORD + 1,
        "drop_index": args.drop_index,
        "dropped_support_mask": masks[args.drop_index],
        "dropped_support": dropped_support,
        "baseline_exact_verification": baseline_check,
        "enumeration": enumeration,
        "binary_distance_model": anticode_check,
        "solver_cases": cases,
        "beats_record": beats_record,
        "candidate_count": candidate_count,
        "configuration": config_name,
        "selected_layer": layer_payload,
        "exact_verification": verification,
    }
    write_json(args.output_dir / "analysis.json", result)
    (args.output_dir / "REPORT.md").write_text(report_text(result), encoding="utf-8")

    if args.progress_log:
        with args.progress_log.open("a", encoding="utf-8") as log:
            log.write(
                f"{utcnow()} method='drop one ZE99 tetrad support and search 177-layer' "
                f"drop_index={args.drop_index} dimension=13 count={candidate_count} "
                f"result={'pass' if beats_record else 'fail'} exact={bool(verification and verification.get('ok'))}\n"
            )
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
