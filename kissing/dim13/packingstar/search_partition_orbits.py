#!/usr/bin/env python3
"""Identify and probe the 13 exact E6 partition orbits of PackingStar's d13 core.

The input is PackingStar's non-antipodal 1,146-point Gram matrix.  The script
recovers the exact E6/E7 component Grams, identifies the observed E6 Y3
partition orbit, constructs one exact 1,146-point configuration from every E6
partition orbit while keeping the E7 side and both caps fixed, verifies every
constructed Gram matrix in exact integer/rational arithmetic, and runs a
clearly labelled numerical single-point-hole diagnostic.
"""
from __future__ import annotations

import argparse
import hashlib
import itertools
import json
import math
import sys
from collections import Counter, deque
from datetime import datetime, timezone
from fractions import Fraction
from pathlib import Path
from typing import Any

import numpy as np
import sympy as sp
from scipy.optimize import linprog

sys.path.insert(0, str(Path(__file__).resolve().parent))
from analyze_fiber import (  # noqa: E402
    GRAM_DEN,
    RationalMatrix,
    connected_bipartite_components,
    exact_scaled_gram,
    global_reduce,
    representatives,
    unique_row_classes,
)

DIMENSION = 13
COUNT = 1146
RECORD = 1154


class SearchError(RuntimeError):
    pass


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


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


def derive_structure(path: Path) -> dict[str, Any]:
    H = exact_scaled_gram(path)
    if H.shape != (COUNT, COUNT):
        raise SearchError(f"expected {(COUNT, COUNT)}, got {H.shape}")
    antipodes = np.count_nonzero(H == -GRAM_DEN, axis=1)
    core = np.flatnonzero(antipodes == 0).astype(int).tolist()
    residual = np.flatnonzero(antipodes == 1).astype(int).tolist()
    if (len(core), len(residual)) != (1008, 138):
        raise SearchError("unexpected core/residual split")
    caps = nonzero_components(H, residual)
    if [len(c) for c in caps] != [54, 84]:
        raise SearchError(f"unexpected cap component sizes {[len(c) for c in caps]}")
    cap6, cap7 = caps

    Hc = H[np.ix_(core, core)]
    H2 = Hc @ Hc
    A = global_reduce(H2 - 288 * Hc, 96)
    B = global_reduce(336 * Hc - H2, 96)
    _, invA, countsA = unique_row_classes(A.numerator)
    _, invB, countsB = unique_row_classes(B.numerator)
    if len(countsA) != 72 or set(countsA.tolist()) != {14}:
        raise SearchError("E6 class reconstruction failed")
    if len(countsB) != 112 or set(countsB.tolist()) != {9}:
        raise SearchError("rank-7 class reconstruction failed")
    repsA = representatives(invA, 72)
    repsB = representatives(invB, 112)
    Au = RationalMatrix(A.numerator[np.ix_(repsA, repsA)], A.denominator).reduced()
    Bu = RationalMatrix(B.numerator[np.ix_(repsB, repsB)], B.denominator).reduced()
    if (Au.denominator, Bu.denominator) != (2, 2):
        raise SearchError(f"unexpected component denominators {(Au.denominator, Bu.denominator)}")

    incidence = np.zeros((72, 112), dtype=np.int64)
    for a, b in zip(invA.tolist(), invB.tolist()):
        if incidence[a, b]:
            raise SearchError(f"duplicate component pair {(a, b)}")
        incidence[a, b] = 1
    components = connected_bipartite_components(incidence)
    if len(components) != 8:
        raise SearchError(f"expected 8 fiber components, got {len(components)}")
    for component in components:
        if (component["a_size"], component["b_size"], component["edges"]) != (9, 14, 126):
            raise SearchError(f"bad component {component}")
        if not component["complete_bipartite"]:
            raise SearchError("fiber component is not complete bipartite")

    C6full = H[np.ix_(cap6, core)]
    C7full = H[np.ix_(cap7, core)]
    C6 = C6full[:, repsA]
    C7 = C7full[:, repsB]
    if not np.array_equal(C6full, C6[:, invA]):
        raise SearchError("cap6/core factorization failed")
    if not np.array_equal(C7full, C7[:, invB]):
        raise SearchError("cap7/core factorization failed")

    original_order = core + cap6 + cap7
    return {
        "H": H,
        "H_reordered": H[np.ix_(original_order, original_order)],
        "core": core,
        "cap6": cap6,
        "cap7": cap7,
        "Au": Au,
        "Bu": Bu,
        "T6": H[np.ix_(cap6, cap6)],
        "T7": H[np.ix_(cap7, cap7)],
        "C6": C6,
        "C7": C7,
        "invA": invA,
        "invB": invB,
        "components": components,
    }


def enumerate_cliques(adjacency: list[int], size: int) -> list[tuple[int, ...]]:
    forward = [bits & ~((1 << (i + 1)) - 1) for i, bits in enumerate(adjacency)]
    found: list[tuple[int, ...]] = []

    def recurse(clique: list[int], candidates: int) -> None:
        need = size - len(clique)
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


def exact_covers(universe: int, blocks: list[tuple[int, ...]]) -> list[tuple[int, ...]]:
    masks = [sum(1 << vertex for vertex in block) for block in blocks]
    containing: list[list[int]] = [[] for _ in range(universe)]
    for index, block in enumerate(blocks):
        for vertex in block:
            containing[vertex].append(index)
    full = (1 << universe) - 1
    covers: list[tuple[int, ...]] = []

    def recurse(used: int, selected: list[int]) -> None:
        if used == full:
            covers.append(tuple(sorted(selected)))
            return
        remaining = full ^ used
        options: list[int] | None = None
        while remaining:
            bit = remaining & -remaining
            vertex = bit.bit_length() - 1
            remaining ^= bit
            compatible = [i for i in containing[vertex] if not (masks[i] & used)]
            if not compatible:
                return
            if options is None or len(compatible) < len(options):
                options = compatible
            if len(options) == 1:
                break
        assert options is not None
        for index in options:
            recurse(used | masks[index], selected + [index])

    recurse(0, [])
    return sorted(covers)


def reflection_permutations_from_gram(N: np.ndarray, denominator: int) -> list[tuple[int, ...]]:
    scaled_rows = {tuple((row * denominator).tolist()): i for i, row in enumerate(N)}
    permutations: set[tuple[int, ...]] = set()
    for alpha in range(len(N)):
        permutation: list[int] = []
        for root in range(len(N)):
            reflected_row = N[root] * denominator - 2 * int(N[root, alpha]) * N[alpha]
            target = scaled_rows.get(tuple(reflected_row.tolist()))
            if target is None:
                raise SearchError(f"reflection lookup failed for alpha={alpha}, root={root}")
            permutation.append(target)
        if sorted(permutation) != list(range(len(N))):
            raise SearchError("reflection is not a permutation")
        permutations.add(tuple(permutation))
    if len(permutations) != 36:
        raise SearchError(f"expected 36 distinct E6 reflections, got {len(permutations)}")
    return sorted(permutations)


def induced_block_permutations(
    blocks: list[tuple[int, ...]], generators: list[tuple[int, ...]]
) -> list[tuple[int, ...]]:
    block_index = {tuple(sorted(block)): i for i, block in enumerate(blocks)}
    result = []
    for generator in generators:
        permutation = tuple(
            block_index[tuple(sorted(generator[vertex] for vertex in block))]
            for block in blocks
        )
        result.append(permutation)
    return result


def classify_orbits(
    covers: list[tuple[int, ...]], block_generators: list[tuple[int, ...]]
) -> list[list[tuple[int, ...]]]:
    cover_set = set(covers)
    unseen = set(covers)
    orbits: list[list[tuple[int, ...]]] = []
    while unseen:
        seed = min(unseen)
        seen = {seed}
        queue: deque[tuple[int, ...]] = deque([seed])
        while queue:
            current = queue.popleft()
            for permutation in block_generators:
                image = tuple(sorted(permutation[index] for index in current))
                if image not in cover_set:
                    raise SearchError("orbit left exact-cover set")
                if image not in seen:
                    seen.add(image)
                    queue.append(image)
        unseen -= seen
        orbits.append(sorted(seen))
    return sorted(orbits, key=lambda orbit: (len(orbit), orbit[0]))


def pair_profile(N: np.ndarray, blocks: list[tuple[int, ...]], cover: tuple[int, ...]) -> dict[str, int]:
    profiles: Counter[str] = Counter()
    chosen = [blocks[index] for index in cover]
    for first, second in itertools.combinations(chosen, 2):
        values, counts = np.unique(N[np.ix_(first, second)], return_counts=True)
        key = ",".join(f"{int(v)}:{int(c)}" for v, c in zip(values, counts))
        profiles[key] += 1
    return dict(sorted(profiles.items()))


def build_gram(
    Au: RationalMatrix,
    Bu: RationalMatrix,
    T6: np.ndarray,
    T7: np.ndarray,
    C6: np.ndarray,
    C7: np.ndarray,
    pairs: list[tuple[int, int]],
) -> np.ndarray:
    if len(pairs) != 1008 or len(set(pairs)) != 1008:
        raise SearchError("core pair list must contain 1008 distinct pairs")
    a = np.asarray([pair[0] for pair in pairs], dtype=np.int64)
    b = np.asarray([pair[1] for pair in pairs], dtype=np.int64)
    H = np.zeros((COUNT, COUNT), dtype=np.int64)
    H[:1008, :1008] = Au.numerator[np.ix_(a, a)] + Bu.numerator[np.ix_(b, b)]
    H[1008:1062, :1008] = C6[:, a]
    H[:1008, 1008:1062] = H[1008:1062, :1008].T
    H[1062:, :1008] = C7[:, b]
    H[:1008, 1062:] = H[1062:, :1008].T
    H[1008:1062, 1008:1062] = T6
    H[1062:, 1062:] = T7
    return H


def pairs_from_blocks(
    a_blocks: list[tuple[int, ...]], b_blocks: list[tuple[int, ...]]
) -> list[tuple[int, int]]:
    if len(a_blocks) != 8 or len(b_blocks) != 8:
        raise SearchError("expected eight blocks per side")
    pairs = [
        (a, b)
        for ablock, bblock in zip(sorted(a_blocks), sorted(b_blocks))
        for a in ablock
        for b in bblock
    ]
    return sorted(pairs)


def pivot_basis(H: np.ndarray, rank: int) -> list[int]:
    G = H.astype(float) / GRAM_DEN
    residual = np.diag(G).copy()
    factors = np.zeros((len(H), rank), dtype=float)
    selected: list[int] = []
    used = np.zeros(len(H), dtype=bool)
    for step in range(rank):
        candidates = residual.copy()
        candidates[used] = -np.inf
        pivot = int(np.argmax(candidates))
        if candidates[pivot] < 1e-9:
            raise SearchError(f"rank below {rank} at pivot {step}")
        selected.append(pivot)
        used[pivot] = True
        column = G[:, pivot].copy()
        if step:
            column -= factors[:, :step] @ factors[pivot, :step]
        factors[:, step] = column / math.sqrt(float(candidates[pivot]))
        residual -= factors[:, step] ** 2
        residual[np.abs(residual) < 1e-10] = 0.0
    return selected


def exact_verify(H: np.ndarray) -> tuple[dict[str, Any], np.ndarray]:
    if not np.array_equal(H, H.T):
        raise SearchError("Gram is not symmetric")
    if not np.all(np.diag(H) == GRAM_DEN):
        raise SearchError("Gram diagonal is not exactly one")
    off = H.copy()
    np.fill_diagonal(off, -10 * GRAM_DEN)
    maximum = int(off.max())
    if maximum > GRAM_DEN // 2:
        raise SearchError(f"off-diagonal violation {maximum}/{GRAM_DEN}")
    if np.unique(H, axis=0).shape[0] != COUNT:
        raise SearchError("duplicate vectors detected from duplicate Gram rows")

    basis = pivot_basis(H, DIMENSION)
    B = sp.Matrix(H[np.ix_(basis, basis)].tolist())
    leading = [int(B[:k, :k].det()) for k in range(1, DIMENSION + 1)]
    if min(leading) <= 0:
        raise SearchError(f"basis is not positive definite: {leading}")
    inverse = B.inv()
    inv_den = 1
    for value in inverse:
        inv_den = math.lcm(inv_den, int(value.q))
    inv_num = np.asarray(
        [[int(inverse[i, j] * inv_den) for j in range(DIMENSION)] for i in range(DIMENSION)],
        dtype=np.int64,
    )
    A = H[:, basis]
    max_a = int(np.max(np.abs(A)))
    max_m = int(np.max(np.abs(inv_num)))
    bound = DIMENSION * DIMENSION * max_a * max_m * max_a
    if bound >= 2**62:
        product = A.astype(object) @ inv_num.astype(object) @ A.T.astype(object)
        target = H.astype(object) * inv_den
        method = "Python arbitrary-precision integers"
    else:
        product = A @ inv_num @ A.T
        target = H * inv_den
        method = f"numpy int64 with proven absolute bound {bound} < 2^62"
    if not np.array_equal(product, target):
        where = np.argwhere(product != target)[0]
        raise SearchError(f"exact rank factorization failed at {tuple(map(int, where))}")

    B_float = H[np.ix_(basis, basis)].astype(float) / GRAM_DEN
    L = np.linalg.cholesky(B_float)
    X = (H[:, basis].astype(float) / GRAM_DEN) @ np.linalg.inv(L.T)
    norm_error = float(np.max(np.abs(np.sum(X * X, axis=1) - 1.0)))
    sample = X[:128] @ X[:128].T
    sample_error = float(np.max(np.abs(sample - H[:128, :128] / GRAM_DEN)))
    return {
        "count": COUNT,
        "dimension": DIMENSION,
        "exact": True,
        "diagonal_exactly_one": True,
        "all_off_diagonal_leq_half": True,
        "all_vectors_distinct": True,
        "rank": DIMENSION,
        "positive_semidefinite": True,
        "max_off_diagonal": str(Fraction(maximum, GRAM_DEN)),
        "basis_indices": basis,
        "leading_principal_minors": leading,
        "factorization_method": method,
        "coordinate_norm_error_numerical_crosscheck": norm_error,
        "sample_gram_error_numerical_crosscheck": sample_error,
        "beats_record": COUNT > RECORD,
    }, X


def numerical_hole_search(X: np.ndarray, seed: int) -> dict[str, Any]:
    rng = np.random.default_rng(seed)
    directions = [row.copy() for row in np.eye(DIMENSION)]
    directions += [row.copy() for row in -np.eye(DIMENSION)]
    directions += [row for row in rng.normal(size=(20, DIMENSION))]
    best_norm = -1.0
    best_point: np.ndarray | None = None
    successful = 0
    for raw in directions:
        direction = raw / np.linalg.norm(raw)
        previous: np.ndarray | None = None
        for _ in range(14):
            result = linprog(
                c=-direction,
                A_ub=X,
                b_ub=np.full(len(X), 0.5),
                bounds=[(None, None)] * DIMENSION,
                method="highs",
            )
            if not result.success:
                break
            successful += 1
            point = np.asarray(result.x, dtype=float)
            norm = float(np.linalg.norm(point))
            if norm > best_norm:
                best_norm = norm
                best_point = point.copy()
            if norm < 1e-14:
                break
            new_direction = point / norm
            if previous is not None and np.linalg.norm(new_direction - previous) < 1e-10:
                break
            previous = direction
            direction = new_direction
    if best_point is None:
        return {"success": False, "reason": "no LP solve succeeded", "exact": False}
    unit = best_point / np.linalg.norm(best_point)
    inner = X @ unit
    conflicts = np.flatnonzero(inner > 0.5 + 1e-10)
    return {
        "success": True,
        "exact": False,
        "method": "sampled fixed-point LP vertex search",
        "directions": len(directions),
        "successful_lp_solves": successful,
        "best_polytope_norm": best_norm,
        "single_point_addition_numerically_possible": bool(best_norm >= 1.0 - 1e-9),
        "normalized_candidate_max_inner": float(np.max(inner)),
        "normalized_candidate_conflicts": int(len(conflicts)),
        "note": "Numerical diagnostic only; any candidate requires exact reconstruction.",
    }


def frac_matrix(matrix: np.ndarray, denominator: int) -> dict[str, Any]:
    return {"denominator": denominator, "numerator": matrix.astype(int).tolist()}


def report_text(result: dict[str, Any]) -> str:
    lines = [
        "# PackingStar E6 partition-orbit search",
        "",
        f"Generated `{result['generated_at']}`.",
        "",
        "## Exact classification",
        "",
        f"The observed 1,146-point variant lies in E6 partition orbit **{result['observed_E6_orbit']['orbit_index']}** ",
        f"of size **{result['observed_E6_orbit']['orbit_size']}**. Exact enumeration again finds ",
        f"**{result['E6_partition_classification']['partition_count']}** partitions in ",
        f"**{result['E6_partition_classification']['orbit_count']}** Weyl orbits.",
        "",
        "## Exact variants and numerical hole diagnostics",
        "",
        "| variant | E6 orbit | orbit size | exact count | best sampled polytope norm | numerical hole |",
        "| --- | ---: | ---: | ---: | ---: | :---: |",
    ]
    for variant in result["variants"]:
        hole = variant["single_point_hole_diagnostic"]
        lines.append(
            f"| `{variant['name']}` | {variant['E6_orbit_index']} | {variant['E6_orbit_size']} | "
            f"{variant['exact_verification']['count']} | {hole.get('best_polytope_norm', float('nan')):.12f} | "
            f"{'yes' if hole.get('single_point_addition_numerically_possible') else 'no'} |"
        )
    lines += [
        "",
        "Every listed 1,146-point Gram matrix passed the exact integer/rational checks. The hole column is numerical only.",
        "None of these configurations improves the 1,154-point record; a record requires at least 1,155 exact vectors.",
        "",
    ]
    return "\n".join(lines)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--matrix", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    structure = derive_structure(args.matrix)
    Au: RationalMatrix = structure["Au"]
    Bu: RationalMatrix = structure["Bu"]
    adjacency = [
        sum(1 << j for j in range(72) if i != j and int(Au.numerator[i, j]) <= 0)
        for i in range(72)
    ]
    blocks = enumerate_cliques(adjacency, 9)
    if len(blocks) != 320:
        raise SearchError(f"expected 320 Y3 blocks, got {len(blocks)}")
    covers = exact_covers(72, blocks)
    if len(covers) != 17920:
        raise SearchError(f"expected 17920 Y3 partitions, got {len(covers)}")
    root_generators = reflection_permutations_from_gram(Au.numerator, Au.denominator)
    block_generators = induced_block_permutations(blocks, root_generators)
    orbits = classify_orbits(covers, block_generators)
    expected_sizes = [40, 360, 480, 480, 480, 960, 1440, 1440, 1440, 2160, 2880, 2880, 2880]
    if [len(orbit) for orbit in orbits] != expected_sizes:
        raise SearchError(f"unexpected orbit sizes {[len(orbit) for orbit in orbits]}")

    block_index = {tuple(sorted(block)): i for i, block in enumerate(blocks)}
    observed_a_blocks = [tuple(sorted(component["a"])) for component in structure["components"]]
    observed_b_blocks = [tuple(sorted(component["b"])) for component in structure["components"]]
    observed_cover = tuple(sorted(block_index[block] for block in observed_a_blocks))
    observed_orbit_index = next(i for i, orbit in enumerate(orbits) if observed_cover in set(orbit))
    observed_orbit = orbits[observed_orbit_index]

    original_pairs = list(zip(structure["invA"].astype(int).tolist(), structure["invB"].astype(int).tolist()))
    reconstructed_original = build_gram(
        Au, Bu, structure["T6"], structure["T7"], structure["C6"], structure["C7"], original_pairs
    )
    if not np.array_equal(reconstructed_original, structure["H_reordered"]):
        raise SearchError("exact reconstruction of the published variant failed")

    variants: list[dict[str, Any]] = []
    original_verification, original_X = exact_verify(reconstructed_original)
    variants.append(
        {
            "name": "published_variant_1",
            "E6_orbit_index": observed_orbit_index,
            "E6_orbit_size": len(observed_orbit),
            "E6_partition_block_indices": list(observed_cover),
            "E6_blocks": [list(block) for block in observed_a_blocks],
            "E7_blocks": [list(block) for block in observed_b_blocks],
            "pairing": "published incidence",
            "exact_verification": original_verification,
            "single_point_hole_diagnostic": numerical_hole_search(original_X, 2026082300),
        }
    )

    for orbit_index, orbit in enumerate(orbits):
        cover = orbit[0]
        a_blocks = [blocks[index] for index in cover]
        pairs = pairs_from_blocks(a_blocks, observed_b_blocks)
        H = build_gram(
            Au, Bu, structure["T6"], structure["T7"], structure["C6"], structure["C7"], pairs
        )
        verification, X = exact_verify(H)
        variants.append(
            {
                "name": f"canonical_E6_orbit_{orbit_index:02d}",
                "E6_orbit_index": orbit_index,
                "E6_orbit_size": len(orbit),
                "E6_partition_block_indices": list(cover),
                "E6_blocks": [list(block) for block in a_blocks],
                "E7_blocks": [list(block) for block in sorted(observed_b_blocks)],
                "pairing": "lexicographic block pairing with published E7 partition",
                "exact_verification": verification,
                "single_point_hole_diagnostic": numerical_hole_search(X, 2026082310 + orbit_index),
            }
        )

    result = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "source": str(args.matrix),
        "source_sha256": sha256_file(args.matrix),
        "status": "completed_exact_partition_orbit_probe",
        "dimension": DIMENSION,
        "record": RECORD,
        "record_target": RECORD + 1,
        "best_exact_count": COUNT,
        "beats_record": False,
        "observed_E6_orbit": {
            "orbit_index": observed_orbit_index,
            "orbit_size": len(observed_orbit),
            "partition_block_indices": list(observed_cover),
            "pair_profile": pair_profile(Au.numerator, blocks, observed_cover),
        },
        "E6_partition_classification": {
            "maximum_Y3_blocks": len(blocks),
            "partition_count": len(covers),
            "orbit_count": len(orbits),
            "orbit_sizes": expected_sizes,
            "orbit_witnesses": [list(orbit[0]) for orbit in orbits],
        },
        "shared_exact_components": {
            "E6_Gram": frac_matrix(Au.numerator, Au.denominator),
            "rank7_active_Gram": frac_matrix(Bu.numerator, Bu.denominator),
            "cap54_Gram": frac_matrix(structure["T6"], GRAM_DEN),
            "cap84_Gram": frac_matrix(structure["T7"], GRAM_DEN),
            "cap54_to_E6": frac_matrix(structure["C6"], GRAM_DEN),
            "cap84_to_rank7_active": frac_matrix(structure["C7"], GRAM_DEN),
            "full_Gram_formula": (
                "For each core edge (a,b), 4*<v_ab,v_cd> = N_E6[a,c]+N_E7[b,d]; "
                "cap/core numerators are the shared cross matrices; cap subspaces are orthogonal."
            ),
        },
        "variants": variants,
    }
    (args.output_dir / "partition_orbit_search.json").write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    (args.output_dir / "REPORT.md").write_text(report_text(result), encoding="utf-8")
    best = max(
        variants,
        key=lambda row: row["single_point_hole_diagnostic"].get("best_polytope_norm", -1.0),
    )
    (args.output_dir / "best.json").write_text(
        json.dumps(best, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    with (args.output_dir / "progress.log").open("w", encoding="utf-8") as log:
        stamp = result["generated_at"]
        for variant in variants:
            hole = variant["single_point_hole_diagnostic"]
            log.write(
                f"{stamp} method='exact E6 partition-orbit core variant' dimension=13 "
                f"count=1146 result=fail_record exact=true orbit={variant['E6_orbit_index']} "
                f"hole_norm={hole.get('best_polytope_norm', 'none')}\n"
            )
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
