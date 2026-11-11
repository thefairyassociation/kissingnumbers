#!/usr/bin/env python3
"""Exact fiber decoding of PackingStar's non-antipodal 1146-point variant.

This variant has a non-antipodal 1008-point core and an antipodal 138-point
residual.  The exact integer Gram matrix is decoded as

    1008 = 8 * 9 * 14,

using all 72 E6 roots and a 112-point rank-7 root subsystem.  The same 54- and
84-point orthogonal caps remain.  As in analyze_fiber.py, floating point is used
only for explicitly labelled hole and polar-vertex candidate diagnostics.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from collections import deque
from datetime import datetime, timezone
from fractions import Fraction
from pathlib import Path
from typing import Any

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from analyze_fiber import (  # noqa: E402
    GRAM_DEN,
    TARGET_RECORD,
    RationalMatrix,
    StructureError,
    combined_subspace_coordinates,
    connected_bipartite_components,
    exact_frame_identity,
    exact_gram_checks,
    exact_scaled_gram,
    fixed_point_hole_search,
    global_reduce,
    polar_vertex_pool,
    representatives,
    row_profile_groups,
    unique_row_classes,
)


def nonzero_components(H: np.ndarray, indices: list[int]) -> list[list[int]]:
    sub = H[np.ix_(indices, indices)]
    adjacency = sub != 0
    np.fill_diagonal(adjacency, False)
    seen = np.zeros(len(indices), dtype=bool)
    components: list[list[int]] = []
    for start in range(len(indices)):
        if seen[start]:
            continue
        queue: deque[int] = deque([start])
        seen[start] = True
        local: list[int] = []
        while queue:
            i = queue.popleft()
            local.append(indices[i])
            for j in np.flatnonzero(adjacency[i]):
                j = int(j)
                if not seen[j]:
                    seen[j] = True
                    queue.append(j)
        components.append(sorted(local))
    return sorted(components, key=len)


def repeated_unique_check(
    full: RationalMatrix,
    unique: RationalMatrix,
    inverse: np.ndarray,
    name: str,
) -> None:
    if full.denominator % unique.denominator != 0:
        raise StructureError(f"{name} denominator mismatch")
    rebuilt = unique.numerator[np.ix_(inverse, inverse)] * (
        full.denominator // unique.denominator
    )
    if not np.array_equal(full.numerator, rebuilt):
        raise StructureError(f"{name} repeated Gram reconstruction failed")


def analyze(path: Path, run_expensive: bool) -> dict[str, Any]:
    H = exact_scaled_gram(path)
    _, profiles = row_profile_groups(H)

    antipode_count = np.count_nonzero(H == -GRAM_DEN, axis=1)
    if set(antipode_count.tolist()) != {0, 1}:
        raise StructureError(
            f"unexpected antipode counts {sorted(set(antipode_count.tolist()))}"
        )
    residual = np.flatnonzero(antipode_count == 1).astype(int).tolist()
    core = np.flatnonzero(antipode_count == 0).astype(int).tolist()
    if (len(core), len(residual)) != (1008, 138):
        raise StructureError(
            f"expected non-antipodal core 1008 and antipodal residual 138, got "
            f"{len(core)} and {len(residual)}"
        )

    cap_components = nonzero_components(H, residual)
    if [len(c) for c in cap_components] != [54, 84]:
        raise StructureError(
            f"residual nonzero graph components have sizes {[len(c) for c in cap_components]}"
        )
    cap6, cap7 = cap_components
    if np.any(H[np.ix_(cap6, cap7)] != 0):
        raise StructureError("54- and 84-point residual caps are not orthogonal")

    Hc = H[np.ix_(core, core)]
    H2 = Hc @ Hc
    A = global_reduce(H2 - 288 * Hc, 96)
    B = global_reduce(336 * Hc - H2, 96)
    common = math.lcm(A.denominator, B.denominator, 2)
    if not np.array_equal(
        A.numerator * (common // A.denominator)
        + B.numerator * (common // B.denominator),
        Hc * (common // 2),
    ):
        raise StructureError("component Grams do not satisfy A+B=2G0")

    _, invA, countsA = unique_row_classes(A.numerator)
    _, invB, countsB = unique_row_classes(B.numerator)
    if len(countsA) != 72 or set(countsA.tolist()) != {14}:
        raise StructureError(
            f"E6 component classes={len(countsA)}, multiplicities={set(countsA.tolist())}"
        )
    if len(countsB) != 112 or set(countsB.tolist()) != {9}:
        raise StructureError(
            f"rank-7 component classes={len(countsB)}, multiplicities={set(countsB.tolist())}"
        )

    repsA = representatives(invA, 72)
    repsB = representatives(invB, 112)
    Au = RationalMatrix(A.numerator[np.ix_(repsA, repsA)], A.denominator).reduced()
    Bu = RationalMatrix(B.numerator[np.ix_(repsB, repsB)], B.denominator).reduced()
    repeated_unique_check(A, Au, invA, "E6")
    repeated_unique_check(B, Bu, invB, "rank-7 subset")

    exact_frame_identity(Au, 12, "E6 root")
    exact_frame_identity(Bu, 16, "112-point rank-7 subset")
    e6_info = exact_gram_checks(Au, "E6 root")
    e7subset_info = exact_gram_checks(Bu, "112-point rank-7 subset")
    if e6_info["rank"] != 6 or e7subset_info["rank"] != 7:
        raise StructureError(
            f"component ranks are {e6_info['rank']} and {e7subset_info['rank']}"
        )

    incidence = np.zeros((72, 112), dtype=np.int64)
    for a, b in zip(invA.tolist(), invB.tolist()):
        if incidence[a, b]:
            raise StructureError(f"duplicate component-root pair {(a, b)}")
        incidence[a, b] = 1
    comps = connected_bipartite_components(incidence)
    if len(comps) != 8:
        raise StructureError(f"fiber incidence has {len(comps)} components, expected 8")
    for comp in comps:
        if (comp["a_size"], comp["b_size"], comp["edges"]) != (9, 14, 126):
            raise StructureError(f"unexpected fiber component {comp}")
        if not comp["complete_bipartite"]:
            raise StructureError("fiber component is not complete bipartite")
        aidx, bidx = comp["a"], comp["b"]
        aoff = Au.numerator[np.ix_(aidx, aidx)].copy()
        boff = Bu.numerator[np.ix_(bidx, bidx)].copy()
        np.fill_diagonal(aoff, -10 * Au.denominator)
        np.fill_diagonal(boff, -10 * Bu.denominator)
        if int(aoff.max()) > 0 or int(boff.max()) > 0:
            raise StructureError("a variant fiber block is not nonpositive")

    if np.any(Au.numerator @ incidence @ Bu.numerator != 0):
        raise StructureError("the two component Grams are not exactly orthogonal")

    T6 = RationalMatrix(H[np.ix_(cap6, cap6)], GRAM_DEN).reduced()
    T7 = RationalMatrix(H[np.ix_(cap7, cap7)], GRAM_DEN).reduced()
    exact_frame_identity(T6, 9, "54-point cap")
    exact_frame_identity(T7, 12, "84-point cap")
    t6_info = exact_gram_checks(T6, "54-point cap")
    t7_info = exact_gram_checks(T7, "84-point cap")
    if t6_info["rank"] != 6 or t7_info["rank"] != 7:
        raise StructureError(f"cap ranks are {t6_info['rank']} and {t7_info['rank']}")

    C6full = H[np.ix_(cap6, core)]
    C7full = H[np.ix_(cap7, core)]
    C6 = C6full[:, repsA]
    C7 = C7full[:, repsB]
    if not np.array_equal(C6full, C6[:, invA]):
        raise StructureError("54-cap/core terms do not factor through E6 classes")
    if not np.array_equal(C7full, C7[:, invB]):
        raise StructureError("84-cap/core terms do not factor through the 112 classes")

    roots6, caps6, align6 = combined_subspace_coordinates(
        Au, H[np.ix_(cap6, cap6)], C6, 6
    )
    roots7, caps7, align7 = combined_subspace_coordinates(
        Bu, H[np.ix_(cap7, cap7)], C7, 7
    )

    result: dict[str, Any] = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "source": str(path),
        "status": "verified_exact_variant_structure",
        "dimension": 13,
        "count": 1146,
        "record": TARGET_RECORD,
        "beats_record": False,
        "row_profile_groups": profiles,
        "antipodal_decomposition": {
            "non_antipodal_core": len(core),
            "antipodal_residual": len(residual),
            "antipodal_pairs": len(residual) // 2,
            "cap_components": [len(cap6), len(cap7)],
            "cap_cross_orthogonal": True,
        },
        "decomposition": {
            "formula": "8*9*14 + 54 + 84 = 1146",
            "core": len(core),
            "cap_R6": len(cap6),
            "cap_R7": len(cap7),
        },
        "core_component_formula": {
            "E6_repeated": "G0*(G0-72I)/6",
            "rank7_repeated": "G0*(84I-G0)/6",
            "sum": "A+B=2G0",
            "orthogonal": True,
            "exact_denominators": {"A": A.denominator, "B": B.denominator},
        },
        "E6_roots": e6_info,
        "rank7_subset_112": e7subset_info,
        "cap54": t6_info,
        "cap84": t7_info,
        "fiber_components": comps,
        "fiber_summary": {
            "components": 8,
            "component_type": "K_9,14",
            "edges_per_component": 126,
            "total_core_points": int(incidence.sum()),
            "E6_class_multiplicity": [14],
            "rank7_class_multiplicity": [9],
        },
        "subspace_alignment_R6": align6,
        "subspace_alignment_R7": align7,
        "exact_cross_levels": {
            "cap54_to_E6_coefficients_times_sqrt2": {
                str(Fraction(int(v), GRAM_DEN)): int(c)
                for v, c in zip(*np.unique(C6, return_counts=True))
            },
            "cap84_to_rank7_subset_coefficients_times_sqrt2": {
                str(Fraction(int(v), GRAM_DEN)): int(c)
                for v, c in zip(*np.unique(C7, return_counts=True))
            },
        },
        "single_cap_extension_diagnostics": {
            "R6": fixed_point_hole_search(roots6, caps6, seed=861354),
            "R7": fixed_point_hole_search(roots7, caps7, seed=871384),
        },
    }
    if run_expensive:
        result["polar_vertex_candidate_search"] = {
            "R6": polar_vertex_pool(roots6, caps6, seed=860054),
            "R7": polar_vertex_pool(roots7, caps7, seed=870084),
        }
    else:
        result["polar_vertex_candidate_search"] = {"skipped": True}
    return result


def report_text(result: dict[str, Any]) -> str:
    return "\n".join(
        [
            "# Exact analysis of PackingStar's non-antipodal 1146 variant",
            "",
            f"Generated `{result['generated_at']}`.",
            "",
            "## Exact decomposition",
            "",
            f"`{result['decomposition']['formula']}`",
            "",
            "The integer Gram matrix has a 1,008-point core with no antipodal pairs",
            "and an antipodal 138-point residual. The residual splits exactly into",
            "orthogonal 54- and 84-point tight frames. The core consists of eight",
            "complete `K_9,14` fibers between all 72 E6 roots and a 112-point",
            "rank-7 tight root subsystem.",
            "",
            "## Single-point cap diagnostics",
            "",
            "```json",
            json.dumps(result["single_cap_extension_diagnostics"], indent=2),
            "```",
            "",
            "## Polar-vertex candidate family",
            "",
            "```json",
            json.dumps(result["polar_vertex_candidate_search"], indent=2),
            "```",
            "",
            "The exact configuration remains 1,146 points, nine short of the 1,155",
            "required to improve the current dimension-13 record.",
            "",
        ]
    )


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--matrix", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--expensive", action="store_true")
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    result = analyze(args.matrix, args.expensive)
    (args.output_dir / "variant_fiber_analysis.json").write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    (args.output_dir / "VARIANT_FIBER_REPORT.md").write_text(
        report_text(result), encoding="utf-8"
    )
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
