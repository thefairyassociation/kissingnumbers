#!/usr/bin/env python3
"""Search an exact 169-point ZE99 half-height layer in a finite Q(sqrt(3)) family.

Keep ZE99's 816 tetrads and two poles. A vector at height +1/2 has the
form

    (2*sqrt(3/||q||^2) q, 2),

where q is a primitive integer direction in Z^12. Exhausting primitive
magnitudes <= 2 against the 51 tetrad supports leaves exactly:

* 4096 sign directions, ||q||^2 = 12;
* 128 signed triple directions, ||q||^2 = 3;
* 24 signed coordinate axes, ||q||^2 = 1.

The scale factors are respectively 1, 2, and 2*sqrt(3), so every coordinate
remains in Q(sqrt(3)). Same-height compatibility is exactly
q.r/(||q|| ||r||) <= 1/3.

A 169-point layer, combined with the untouched 168-point negative-height ZE99
layer, gives 816 + 2 + 169 + 168 = 1155 vectors. Any candidate witness is
checked by the repository's exact Q(sqrt(3)) verifier before it is emitted.
"""

from __future__ import annotations

import argparse
import json
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
sys.path.insert(0, str(HERE))
from ze99_aux_core import constraint_rows  # noqa: E402

DIM = 13
RECORD = 1154
TARGET_LAYER = 169
TETRADS = 816
DIAMONDS_START = 816
DIAMONDS_STOP = 1104
AXIALS_START = 1104
AUX_START = 1106
MASK12 = (1 << 12) - 1


class SearchError(RuntimeError):
    pass


def utcnow() -> str:
    return datetime.now(timezone.utc).isoformat()


def write_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def popcount(value: int) -> int:
    return int(value.bit_count())


def enumerate_tetrad_compatible() -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    rows, row_info = constraint_rows(generate_ab)
    supports = np.unique(np.abs(rows[:816]), axis=0).astype(np.int16)
    if supports.shape != (51, 12):
        raise SearchError(f"unexpected tetrad support shape {supports.shape}")

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

    maximum = np.empty(len(magnitudes), dtype=np.int16)
    for start in range(0, len(magnitudes), 4096):
        batch = magnitudes[start : start + 4096].astype(np.int16)
        maximum[start : start + len(batch)] = np.max(batch @ supports.T, axis=1)
    keep = 3 * maximum.astype(np.int32) ** 2 <= 4 * norm2.astype(np.int32)
    magnitudes = magnitudes[keep]
    norm2 = norm2[keep]

    candidates: list[np.ndarray] = []
    norms: list[np.ndarray] = []
    magnitude_supports: Counter[int] = Counter()
    magnitude_norms: Counter[int] = Counter()
    for magnitude, nn in zip(magnitudes, norm2):
        positions = np.flatnonzero(magnitude)
        magnitude_supports[len(positions)] += 1
        magnitude_norms[int(nn)] += 1
        codes = np.arange(1 << len(positions), dtype=np.uint16)
        bits = (codes[:, None] >> np.arange(len(positions), dtype=np.uint16)) & 1
        signs = np.where(bits != 0, 1, -1).astype(np.int8)
        signed = np.zeros((len(codes), 12), dtype=np.int8)
        signed[:, positions] = signs * magnitude[positions]
        candidates.append(signed)
        norms.append(np.full(len(signed), int(nn), dtype=np.int16))

    q = np.vstack(candidates)
    n = np.concatenate(norms)
    keys = {tuple(row.tolist()) for row in q}
    if len(keys) != len(q):
        raise SearchError(f"candidate directions contain {len(q)-len(keys)} duplicates")
    expected = {1: 24, 3: 128, 12: 4096}
    distribution = {int(k): int(v) for k, v in zip(*np.unique(n, return_counts=True))}
    if distribution != expected:
        raise SearchError(f"unexpected candidate norm distribution {distribution}")

    signed_tetrads = rows[:816].astype(np.int16)
    for start in range(0, len(q), 1024):
        qq = q[start : start + 1024].astype(np.int16)
        dots = qq @ signed_tetrads.T
        nn = n[start : start + len(qq)].astype(np.int32)
        if np.any(3 * dots.astype(np.int32) ** 2 > 4 * nn[:, None]):
            raise SearchError("enumerated direction violates a tetrad inequality")

    return q, n, {
        "constraint_rows": row_info,
        "primitive_magnitude_patterns_tested": int(np.count_nonzero(primitive)),
        "magnitude_patterns_after_tetrads": int(len(magnitudes)),
        "magnitude_support_distribution": dict(sorted(magnitude_supports.items())),
        "magnitude_norm_squared_distribution": dict(sorted(magnitude_norms.items())),
        "signed_direction_norm_squared_distribution": distribution,
        "candidate_count": int(len(q)),
        "arithmetic": "integer squared inequalities; no floating point",
    }


def direction_compatible(q: np.ndarray, n: np.ndarray, i: int, j: int) -> bool:
    dot = int(q[i].astype(np.int16) @ q[j].astype(np.int16))
    return dot <= 0 or 9 * dot * dot <= int(n[i]) * int(n[j])


def sign_code(row: np.ndarray) -> int:
    if not np.all(np.abs(row) == 1):
        raise SearchError("not a sign direction")
    value = 0
    for coordinate, entry in enumerate(row.tolist()):
        if entry > 0:
            value |= 1 << coordinate
    return value


def baseline_sign_code() -> list[int]:
    original = generate_ab()
    words: list[int] = []
    for vector in original[DIAMONDS_START:DIAMONDS_STOP]:
        if vector[12] != (2, 0):
            continue
        row = np.asarray([a for a, b in vector[:12]], dtype=np.int8)
        if any(b != 0 for _, b in vector[:12]):
            raise SearchError("baseline diamond is not integral")
        words.append(sign_code(row))
    if len(words) != 144 or len(set(words)) != 144:
        raise SearchError("could not recover the 144-word ZE99 binary code")
    for i, a in enumerate(words):
        for b in words[i + 1 :]:
            if popcount(a ^ b) < 4:
                raise SearchError("baseline binary code has distance below 4")
    return words


def edge_anticodes() -> list[tuple[int, ...]]:
    """All 24,576 maximal diameter-3 anticodes B1(x) union B1(y), d(x,y)=1."""
    out: list[tuple[int, ...]] = []
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
            if max(popcount(a ^ b) for a, b in combinations(anticode, 2)) > 3:
                raise SearchError("edge anticode has diameter greater than 3")
            out.append(anticode)
    if len(out) != 24576 or len(set(out)) != 24576:
        raise SearchError("edge anticode enumeration mismatch")
    return out


def verify_anticode_cover(anticodes: list[tuple[int, ...]]) -> dict[str, int]:
    """Prove every unordered binary pair at distance 1,2,3 is covered."""
    covered = np.zeros((1 << 12, 1 << 12), dtype=bool)
    for anticode in anticodes:
        array = np.asarray(anticode, dtype=np.int16)
        covered[np.ix_(array, array)] = True
    expected = 0
    missing = 0
    for a in range(1 << 12):
        for b in range(a + 1, 1 << 12):
            close = popcount(a ^ b) < 4
            if close:
                expected += 1
                if not covered[a, b]:
                    missing += 1
            elif covered[a, b]:
                raise SearchError("anticode cover contains a pair at distance at least 4")
    if missing:
        raise SearchError(f"anticode system misses {missing} forbidden pairs")
    return {
        "anticodes": len(anticodes),
        "vertices_per_anticode": 24,
        "forbidden_pairs_covered": expected,
        "missing_forbidden_pairs": 0,
        "extraneous_compatible_pairs": 0,
    }


def selected_layer_ab(q: np.ndarray, n: np.ndarray, selected: Iterable[int], last: int) -> list[list[tuple[int, int]]]:
    out: list[list[tuple[int, int]]] = []
    for index in selected:
        row = q[int(index)]
        nn = int(n[int(index)])
        vector: list[tuple[int, int]] = []
        if nn == 12:
            vector = [(int(value), 0) for value in row.tolist()]
        elif nn == 3:
            vector = [(2 * int(value), 0) for value in row.tolist()]
        elif nn == 1:
            vector = [(0, 2 * int(value)) for value in row.tolist()]
        else:
            raise SearchError(f"unsupported direction norm {nn}")
        vector.append((int(last), 0))
        out.append(vector)
    return out


def negative_baseline_layer(original: list[list[tuple[int, int]]]) -> list[list[tuple[int, int]]]:
    layer = [
        vector
        for vector in original[DIAMONDS_START:DIAMONDS_STOP]
        if vector[12] == (-2, 0)
    ]
    layer.extend(
        vector for vector in original[AUX_START:] if vector[12] == (-2, 0)
    )
    if len(layer) != 168:
        raise SearchError(f"negative baseline layer has {len(layer)} vectors")
    return layer


def exact_full_witness(q: np.ndarray, n: np.ndarray, selected: list[int]) -> tuple[list[list[tuple[int, int]]], dict[str, Any]]:
    original = generate_ab()
    vectors = list(original[:TETRADS])
    vectors.extend(original[AXIALS_START:AUX_START])
    vectors.extend(selected_layer_ab(q, n, selected, last=2))
    vectors.extend(negative_baseline_layer(original))
    expected = TETRADS + 2 + len(selected) + 168
    if len(vectors) != expected:
        raise SearchError(f"full witness count {len(vectors)} != {expected}")
    verification = verify_ab(vectors)
    if not verification.get("ok"):
        raise SearchError(f"full exact witness failed: {verification}")
    return vectors, verification


def solve_cp_sat(
    q: np.ndarray,
    n: np.ndarray,
    anticodes: list[tuple[int, ...]],
    time_limit: float,
    workers: int,
    seed: int,
) -> tuple[list[int] | None, dict[str, Any]]:
    try:
        from ortools.sat.python import cp_model
    except ImportError as exc:
        raise SearchError("OR-Tools is required for the joint-layer search") from exc

    sign_indices = np.flatnonzero(n == 12).astype(int).tolist()
    axis_indices = np.flatnonzero(n == 1).astype(int).tolist()
    triple_indices = np.flatnonzero(n == 3).astype(int).tolist()
    if (len(sign_indices), len(axis_indices), len(triple_indices)) != (4096, 24, 128):
        raise SearchError("candidate type counts changed")

    sign_original_by_word = [-1] * (1 << 12)
    for original_index in sign_indices:
        word = sign_code(q[original_index])
        if sign_original_by_word[word] != -1:
            raise SearchError("duplicate sign word")
        sign_original_by_word[word] = original_index
    if any(index < 0 for index in sign_original_by_word):
        raise SearchError("sign directions do not cover the 12-cube")

    axis_original_by_state: dict[tuple[int, int], int] = {}
    for original_index in axis_indices:
        positions = np.flatnonzero(q[original_index])
        if len(positions) != 1:
            raise SearchError("axis candidate has wrong support")
        coordinate = int(positions[0])
        sign = int(q[original_index, coordinate])
        axis_original_by_state[(coordinate, sign)] = original_index
    if len(axis_original_by_state) != 24:
        raise SearchError("axis state map is incomplete")

    model = cp_model.CpModel()
    sign_vars = [model.NewBoolVar(f"s_{word:03x}") for word in range(1 << 12)]
    triple_vars = [model.NewBoolVar(f"t_{local}") for local in range(len(triple_indices))]
    state_vars = {
        (coordinate, sign): model.NewBoolVar(
            f"u_{coordinate}_{'p' if sign > 0 else 'm'}"
        )
        for coordinate in range(12)
        for sign in (-1, 1)
    }

    for anticode in anticodes:
        model.AddAtMostOne(sign_vars[word] for word in anticode)

    triple_conflicts = 0
    triple_forbidden_sign_edges = 0
    state_incidence: dict[tuple[int, int], list[Any]] = {
        state: [] for state in state_vars
    }
    for local, original_index in enumerate(triple_indices):
        row = q[original_index]
        support = np.flatnonzero(row).astype(int).tolist()
        if len(support) != 3:
            raise SearchError("triple candidate has wrong support")
        forbidden: list[Any] = []
        for word in range(1 << 12):
            matches = True
            for coordinate in support:
                bit_positive = bool((word >> coordinate) & 1)
                if bit_positive != bool(row[coordinate] > 0):
                    matches = False
                    break
            if matches:
                forbidden.append(sign_vars[word])
        if len(forbidden) != 512:
            raise SearchError("triple direction does not forbid 512 sign words")
        model.Add(sum(forbidden) == 0).OnlyEnforceIf(triple_vars[local])
        triple_forbidden_sign_edges += len(forbidden)
        for coordinate in support:
            state = (coordinate, int(row[coordinate]))
            model.Add(triple_vars[local] <= state_vars[state])
            state_incidence[state].append(triple_vars[local])

    for state, variables in state_incidence.items():
        if len(variables) != 16:
            raise SearchError(f"oriented state {state} has incidence {len(variables)}")
        model.Add(state_vars[state] <= sum(variables))

    for a in range(len(triple_indices)):
        for b in range(a + 1, len(triple_indices)):
            if not direction_compatible(q, n, triple_indices[a], triple_indices[b]):
                model.AddAtMostOne(triple_vars[a], triple_vars[b])
                triple_conflicts += 1
    if triple_conflicts != 384:
        raise SearchError(f"unexpected triple conflict count {triple_conflicts}")

    sign_count = sum(sign_vars)
    triple_count = sum(triple_vars)
    used_state_count = sum(state_vars.values())
    layer_count = sign_count + triple_count + 24 - used_state_count

    model.Add(sign_count <= 144)
    model.Add(triple_count >= 1)
    model.Add(triple_count - used_state_count >= 1)
    model.Add(layer_count >= TARGET_LAYER)
    model.Add(sign_vars[MASK12] == 1)

    baseline = baseline_sign_code()
    anchor = baseline[0]
    translated = {MASK12 ^ (word ^ anchor) for word in baseline}
    if MASK12 not in translated or len(translated) != 144:
        raise SearchError("translated baseline hint is malformed")
    for word, variable in enumerate(sign_vars):
        model.AddHint(variable, 1 if word in translated else 0)
    for variable in triple_vars:
        model.AddHint(variable, 0)
    for variable in state_vars.values():
        model.AddHint(variable, 0)

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
    sign_selected: list[int] = []
    triple_selected: list[int] = []
    axes_selected: list[int] = []
    if status in (cp_model.FEASIBLE, cp_model.OPTIMAL):
        sign_selected = [
            sign_original_by_word[word]
            for word, variable in enumerate(sign_vars)
            if solver.Value(variable)
        ]
        triple_selected = [
            triple_indices[local]
            for local, variable in enumerate(triple_vars)
            if solver.Value(variable)
        ]
        used_states = {
            state for state, variable in state_vars.items() if solver.Value(variable)
        }
        axes_selected = [
            original_index
            for state, original_index in axis_original_by_state.items()
            if state not in used_states
        ]
        selected = sign_selected + triple_selected + axes_selected
        if len(selected) < TARGET_LAYER:
            raise SearchError("CP-SAT returned a solution below the target")
        if len(set(selected)) != len(selected):
            raise SearchError("CP-SAT selected duplicate candidate indices")
        for i, a in enumerate(selected):
            for b in selected[i + 1 :]:
                if not direction_compatible(q, n, a, b):
                    raise SearchError(f"CP-SAT selected incompatible directions {a},{b}")

    statistics = {
        "status": status_name,
        "wall_time_seconds": solver.WallTime(),
        "user_time_seconds": solver.UserTime(),
        "num_conflicts": solver.NumConflicts(),
        "num_branches": solver.NumBranches(),
        "best_objective_bound": solver.BestObjectiveBound(),
        "response_stats": solver.ResponseStats(),
        "time_limit_seconds": time_limit,
        "workers": workers,
        "random_seed": seed,
        "model": {
            "sign_variables": len(sign_vars),
            "triple_variables": len(triple_vars),
            "oriented_state_variables": len(state_vars),
            "edge_anticode_constraints": len(anticodes),
            "conditional_triple_sign_conflicts": triple_forbidden_sign_edges,
            "triple_triple_conflict_constraints": triple_conflicts,
            "target_layer_count": TARGET_LAYER,
            "symmetry_break": "all-plus sign word selected",
            "known_binary_bound": "A_2(12,4)=144",
        },
        "solution": None
        if selected is None
        else {
            "layer_count": len(selected),
            "sign_directions": len(sign_selected),
            "triple_directions": len(triple_selected),
            "axis_directions": len(axes_selected),
            "used_oriented_axis_states": 24 - len(axes_selected),
            "selected_candidate_indices": selected,
        },
    }
    return selected, statistics


def report_text(result: dict[str, Any]) -> str:
    solver = result["solver"]
    lines = [
        "# Exact joint ZE99 half-height layer search",
        "",
        f"Generated `{result['generated_at']}`.",
        "",
        "## Finite exact family",
        "",
        "- 4,096 sign directions of norm squared 12.",
        "- 128 signed triple directions of norm squared 3.",
        "- 24 signed coordinate axes of norm squared 1.",
        "- All coordinates after lifting lie in `Q(sqrt(3))`.",
        "- Same-height compatibility was encoded exactly; no floating point was used.",
        "",
        "## Search result",
        "",
        f"- CP-SAT status: **{solver['status']}**.",
        f"- Exact 169-point layer found: **{result['beats_record']}**.",
        f"- Full dimension-13 count: **{result['candidate_count']}**.",
        f"- Exact full verification: **{bool(result['exact_verification'] and result['exact_verification'].get('ok'))}**.",
        "",
    ]
    if result["beats_record"]:
        lines.extend(
            [
                "The emitted configuration proves `k(13) >= 1155` after full exact",
                "pairwise verification.",
                "",
            ]
        )
    elif solver["status"] == "INFEASIBLE":
        lines.extend(
            [
                "The 4,248-direction mixed finite family is exhausted: its maximum",
                "half-height layer has size 168, since ZE99 supplies 168 and CP-SAT",
                "proved that 169 is infeasible under the exact model.",
                "",
            ]
        )
    else:
        lines.extend(
            [
                "No record is claimed. The solver did not produce an exact 169-point",
                "layer within the configured resource limit.",
                "",
            ]
        )
    lines.extend(["```json", json.dumps(result, indent=2, sort_keys=True), "```", ""])
    return "\n".join(lines)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--progress-log", type=Path)
    parser.add_argument("--time-limit", type=float, default=1500.0)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--seed", type=int, default=51)
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    stamp = utcnow()

    if args.progress_log:
        args.progress_log.parent.mkdir(parents=True, exist_ok=True)
        with args.progress_log.open("a", encoding="utf-8") as log:
            log.write(
                f"{stamp} method='exact joint ZE99 half-height 4248-direction search' "
                "dimension=13 count=? result=started\n"
            )

    q, n, enumeration = enumerate_tetrad_compatible()
    anticodes = edge_anticodes()
    cover = verify_anticode_cover(anticodes)
    selected, solver = solve_cp_sat(
        q,
        n,
        anticodes,
        time_limit=args.time_limit,
        workers=args.workers,
        seed=args.seed,
    )

    beats_record = selected is not None and len(selected) >= TARGET_LAYER
    verification: dict[str, Any] | None = None
    config_name: str | None = None
    layer_payload: dict[str, Any] | None = None
    candidate_count = RECORD
    if beats_record and selected is not None:
        vectors, verification = exact_full_witness(q, n, selected)
        candidate_count = len(vectors)
        if candidate_count <= RECORD:
            raise SearchError("exact witness does not beat the record")
        config_name = f"best_exact_{candidate_count}.json"
        layer_payload = {
            "candidate_indices": selected,
            "directions_q": [q[index].astype(int).tolist() for index in selected],
            "direction_norm_squared": [int(n[index]) for index in selected],
        }
        write_json(
            args.output_dir / config_name,
            {
                "dimension": DIM,
                "count": candidate_count,
                "norm_squared": 16,
                "unit_scaling": "divide every coordinate by 4",
                "coordinate_field": "Q(sqrt(3))",
                "vectors": [ab_to_strings(vector) for vector in vectors],
                "new_positive_half_height_layer": layer_payload,
                "negative_half_height_layer": "unchanged 168-point ZE99 layer",
                "exact_verification": verification,
                "record_before": RECORD,
                "beats_record": True,
            },
        )
        write_json(
            args.output_dir / "best.json",
            {
                "dimension": DIM,
                "count": candidate_count,
                "configuration": config_name,
                "record_before": RECORD,
                "beats_record": True,
                "exact": True,
                "max_off_diagonal_unit": "1/2",
            },
        )

    result = {
        "generated_at": stamp,
        "status": "completed",
        "method": "exact joint search over all tetrad-compatible primitive q in {-2,-1,0,1,2}^12",
        "dimension": DIM,
        "record_before": RECORD,
        "target": RECORD + 1,
        "enumeration": enumeration,
        "binary_distance_model": cover,
        "solver": solver,
        "beats_record": beats_record,
        "candidate_count": candidate_count,
        "configuration": config_name,
        "selected_layer": layer_payload,
        "exact_verification": verification,
        "arithmetic": "integer inequalities and exact Q(sqrt(3)) verification; no floating point in feasibility or certification",
    }
    write_json(args.output_dir / "analysis.json", result)
    (args.output_dir / "REPORT.md").write_text(report_text(result), encoding="utf-8")

    if args.progress_log:
        with args.progress_log.open("a", encoding="utf-8") as log:
            log.write(
                f"{utcnow()} method='exact joint ZE99 half-height 4248-direction search' "
                f"dimension=13 count={candidate_count} "
                f"result={'pass' if beats_record else 'fail'} "
                f"solver_status={solver['status']} exact={bool(verification and verification.get('ok'))}\n"
            )
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
