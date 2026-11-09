#!/usr/bin/env python3
"""Exact audit and augmentation search for the ZE99 d=13 diamond shell.

A prior structural report claimed that 528 Steiner-style candidates give 264
independent zero-cost replacements of ZE99 diamond vectors, hence 2^264
distinct 1,154-point configurations.  This program reconstructs the reported
3,696-candidate generator and audits that interpretation before using it in a
record search.

It then asks the record-relevant question:

    Is there an additional norm-16 integer vector u such that every baseline
    conflict of u can be repaired by a genuine one-conflict replacement?

The fixed tetrad, axial, and irrational layers reduce the complete norm-16
integer search to the 8,192 vectors

    (s_0,...,s_11, t),  s_i in {+1,-1}, t in {+2,-2}.

All comparisons use integer arithmetic and exact Q(sqrt(3)) inequalities.  A
successful 1,155-vector witness is rechecked pairwise by the repository's exact
Q(sqrt(3)) verifier and emitted with actual algebraic coordinate strings.
"""

from __future__ import annotations

import argparse
import heapq
import itertools
import json
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np

HERE = Path(__file__).resolve().parent
import sys

sys.path.insert(0, str(HERE / "constructions"))
from ze99 import ab_to_strings, generate_ab, verify_ab  # noqa: E402

DIMENSION = 13
NORM2 = 16
BOUND = 8
RECORD = 1154

TETRAD_STOP = 816
DIAMOND_STOP = 1104
AXIAL_STOP = 1106
BASELINE_COUNT = 1154


class SearchError(RuntimeError):
    pass


def utcnow() -> str:
    return datetime.now(timezone.utc).isoformat()


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def integer_row(vector: list[tuple[int, int]]) -> np.ndarray:
    if any(b != 0 for _, b in vector):
        raise SearchError("expected an integer vector")
    return np.asarray([a for a, _ in vector], dtype=np.int16)


def ab_matrices(
    vectors: list[list[tuple[int, int]]],
) -> tuple[np.ndarray, np.ndarray]:
    a = np.asarray([[x for x, _ in vector] for vector in vectors], dtype=np.int64)
    b = np.asarray([[y for _, y in vector] for vector in vectors], dtype=np.int64)
    return a, b


def qsqrt3_leq(
    p: np.ndarray, q: np.ndarray, bound: int = BOUND
) -> np.ndarray:
    """Return the exact truth value of p + q*sqrt(3) <= bound elementwise."""

    p, q = np.broadcast_arrays(
        np.asarray(p, dtype=np.int64), np.asarray(q, dtype=np.int64)
    )
    out = np.zeros(p.shape, dtype=bool)

    zero = q == 0
    out[zero] = p[zero] <= bound

    positive = q > 0
    if np.any(positive):
        rhs = bound - p[positive]
        qq = q[positive]
        out[positive] = (rhs >= 0) & (rhs * rhs >= 3 * qq * qq)

    negative = q < 0
    if np.any(negative):
        pp = p[negative]
        qq = q[negative]
        easy = pp <= bound
        gap = pp - bound
        out[negative] = easy | (gap * gap <= 3 * qq * qq)

    return out


def diamond_shell() -> np.ndarray:
    codes = np.arange(1 << 12, dtype=np.uint16)[:, None]
    bits = (codes >> np.arange(12, dtype=np.uint16)[None, :]) & 1
    signs = np.where(bits, 1, -1).astype(np.int16)
    plus = np.hstack([signs, np.full((len(signs), 1), 2, dtype=np.int16)])
    minus = np.hstack([signs, np.full((len(signs), 1), -2, dtype=np.int16)])
    shell = np.vstack([plus, minus])
    if shell.shape != (8192, DIMENSION):
        raise SearchError(f"unexpected diamond-shell shape {shell.shape}")
    if not np.all(np.sum(shell.astype(np.int64) ** 2, axis=1) == NORM2):
        raise SearchError("diamond shell contains a non-norm-16 row")
    if len({tuple(row.tolist()) for row in shell}) != len(shell):
        raise SearchError("diamond shell contains duplicates")
    return shell


def raw_steiner_candidates() -> np.ndarray:
    """Reproduce the reported 924 blocks x 4 sign-flips generator.

    Choosing a six-subset B, a global sign s, and the last-coordinate sign
    emits s on B, -s on the complement, and +/-2 in coordinate 12.  The map is
    two-to-one because (B, s) and (B^c, -s) produce the same first 12 entries.
    """

    rows: list[np.ndarray] = []
    universe = set(range(12))
    for block_tuple in itertools.combinations(range(12), 6):
        block = set(block_tuple)
        complement = universe - block
        for sign in (1, -1):
            first = np.empty(12, dtype=np.int16)
            for index in block:
                first[index] = sign
            for index in complement:
                first[index] = -sign
            for last in (2, -2):
                rows.append(np.concatenate([first, np.asarray([last], dtype=np.int16)]))
    raw = np.vstack(rows)
    if raw.shape != (3696, DIMENSION):
        raise SearchError(f"unexpected raw Steiner shape {raw.shape}")
    return raw


def check_non_diamond_compatibility(
    candidates: np.ndarray,
    non_diamonds: list[list[tuple[int, int]]],
    *,
    batch_size: int = 512,
) -> None:
    a, b = ab_matrices(non_diamonds)
    for start in range(0, len(candidates), batch_size):
        stop = min(start + batch_size, len(candidates))
        batch = candidates[start:stop].astype(np.int64)
        p = batch @ a.T
        q = batch @ b.T
        good = qsqrt3_leq(p, q)
        if not np.all(good):
            local = np.argwhere(~good)[0]
            i = start + int(local[0])
            j = int(local[1])
            raise SearchError(
                f"candidate {i} conflicts with non-diamond {j}: "
                f"{int(p[local[0], j])}+{int(q[local[0], j])}*sqrt(3)"
            )


def derive_integer_shell_reduction(
    tetrads: np.ndarray,
) -> dict[str, Any]:
    """Exhaust all magnitude profiles not already in the diamond shell."""

    supports = np.unique((np.abs(tetrads[:, :12]) > 0).astype(np.int8), axis=0)
    if supports.shape != (51, 12):
        raise SearchError(f"unexpected tetrad-support shape {supports.shape}")
    if not np.all(np.sum(supports, axis=1) == 4):
        raise SearchError("a tetrad support does not have size four")

    # If |u_12|=0, the irrational layer forces |u_i|<=2.  The only
    # first-12-coordinate magnitude profiles with squared norm 16 are:
    #   4 twos; 3 twos + 4 ones; 2 twos + 8 ones.
    profiles: list[dict[str, Any]] = []
    surviving_patterns = 0
    for twos, ones in ((4, 0), (3, 4), (2, 8)):
        total = 0
        survivors = 0
        maximum_distribution: Counter[int] = Counter()
        first_survivor: list[int] | None = None
        for twos_pos in itertools.combinations(range(12), twos):
            remaining = [index for index in range(12) if index not in twos_pos]
            for ones_pos in itertools.combinations(remaining, ones):
                total += 1
                magnitude = np.zeros(12, dtype=np.int8)
                magnitude[list(twos_pos)] = 2
                magnitude[list(ones_pos)] = 1
                maximum = int(np.max(supports @ magnitude))
                maximum_distribution[maximum] += 1
                if maximum <= 4:
                    survivors += 1
                    if first_survivor is None:
                        first_survivor = magnitude.astype(int).tolist()
        surviving_patterns += survivors
        profiles.append(
            {
                "twos": twos,
                "ones": ones,
                "patterns": total,
                "patterns_satisfying_all_tetrad_inequalities": survivors,
                "maximum_tetrad_absolute_sum_distribution": {
                    str(key): value for key, value in sorted(maximum_distribution.items())
                },
                "first_survivor": first_survivor,
            }
        )

    if surviving_patterns != 0:
        raise SearchError(
            f"integer-shell reduction failed: {surviving_patterns} last-zero "
            "magnitude patterns survived"
        )

    return {
        "axial_constraint": "|u_12| <= 2",
        "irrational_constraints": "sqrt(3)*|u_i| + |u_12| <= 4 for i=0,...,11",
        "tetrad_constraints": "sum_{i in S}|u_i| <= 4 for each of 51 tetrad supports S",
        "last_coordinate_cases": {
            "abs(u_12)=2": (
                "all first 12 magnitudes equal 1; exactly the 8192-vector "
                "diamond shell"
            ),
            "abs(u_12)=1": (
                "impossible: irrational constraints force first-12 squared "
                "norm at most 12, but 15 is required"
            ),
            "u_12=0": (
                "all possible magnitude profiles were exhaustively rejected "
                "by the 51 tetrad inequalities"
            ),
        },
        "last_zero_profiles": profiles,
        "last_zero_surviving_magnitude_patterns": surviving_patterns,
        "integer_candidate_shell_size": 8192,
        "exact_family_exhaustive": True,
    }


def build_switch_catalog(
    shell: np.ndarray,
    diamonds: np.ndarray,
    non_diamonds: list[list[tuple[int, int]]],
) -> tuple[np.ndarray, np.ndarray, dict[int, list[int]], np.ndarray, dict[str, Any]]:
    """Audit the reported Steiner replacements and return genuine ones only."""

    dots = shell.astype(np.int16) @ diamonds.astype(np.int16).T
    conflicts = dots > BOUND
    conflict_counts = np.sum(conflicts, axis=1)

    baseline_keys = {tuple(row.tolist()): index for index, row in enumerate(diamonds)}
    is_baseline = np.asarray(
        [tuple(row.tolist()) in baseline_keys for row in shell], dtype=bool
    )

    # Genuine one-conflict replacements must be new vectors.  Baseline rows
    # have cset=1 solely because their inner product with themselves is 16.
    replacement_mask = (conflict_counts == 1) & ~is_baseline
    replacements = shell[replacement_mask].copy()
    replacement_groups = np.argmax(conflicts[replacement_mask], axis=1).astype(np.int16)

    if len({tuple(row.tolist()) for row in replacements}) != len(replacements):
        raise SearchError("switch catalogue contains duplicate replacement vectors")
    check_non_diamond_compatibility(replacements, non_diamonds)

    groups: dict[int, list[int]] = defaultdict(list)
    for index, group in enumerate(replacement_groups.astype(int).tolist()):
        groups[group].append(index)
    groups = dict(sorted(groups.items()))
    rigid = np.asarray(
        [index for index in range(len(diamonds)) if index not in groups],
        dtype=np.int16,
    )

    for replacement_index, group in enumerate(replacement_groups.astype(int)):
        bad = np.flatnonzero(replacements[replacement_index] @ diamonds.T > BOUND)
        if bad.tolist() != [group]:
            raise SearchError(
                f"replacement {replacement_index} has conflicts {bad.tolist()}, "
                f"expected only {group}"
            )

    if len(replacements):
        pair_dots = replacements.astype(np.int16) @ replacements.astype(np.int16).T
        pair_compatible = pair_dots <= BOUND
    else:
        pair_dots = np.empty((0, 0), dtype=np.int16)
        pair_compatible = np.empty((0, 0), dtype=bool)

    cross_conflicts = 0
    within_conflicts = 0
    for i in range(len(replacements)):
        for j in range(i + 1, len(replacements)):
            if pair_dots[i, j] <= BOUND:
                continue
            if replacement_groups[i] == replacement_groups[j]:
                within_conflicts += 1
            else:
                cross_conflicts += 1

    # Reproduce the exact counting behind the published 528/264 claim.
    raw = raw_steiner_candidates()
    check_non_diamond_compatibility(raw, non_diamonds)
    raw_dots = raw.astype(np.int16) @ diamonds.astype(np.int16).T
    raw_conflict_counts = np.sum(raw_dots > BOUND, axis=1)
    raw_cset1 = raw[raw_conflict_counts == 1]
    unique_raw = np.unique(raw, axis=0)
    unique_dots = unique_raw.astype(np.int16) @ diamonds.astype(np.int16).T
    unique_conflict_counts = np.sum(unique_dots > BOUND, axis=1)
    unique_cset1 = unique_raw[unique_conflict_counts == 1]

    unique_cset1_keys = {tuple(row.tolist()) for row in unique_cset1}
    cset1_baseline_indices = sorted(
        baseline_keys[key] for key in unique_cset1_keys if key in baseline_keys
    )
    nonbaseline_unique_cset1 = sorted(
        key for key in unique_cset1_keys if key not in baseline_keys
    )
    raw_multiplicity = Counter(tuple(row.tolist()) for row in raw_cset1)

    if len(raw) != 3696 or len(unique_raw) != 1848:
        raise SearchError("raw Steiner multiplicity audit failed")
    if len(raw_cset1) != 528 or len(unique_cset1) != 264:
        raise SearchError(
            "reported Steiner cset counts were not reproduced: "
            f"raw={len(raw_cset1)}, unique={len(unique_cset1)}"
        )
    if nonbaseline_unique_cset1:
        raise SearchError(
            f"unexpected genuine cset-1 candidates: {len(nonbaseline_unique_cset1)}"
        )
    if set(raw_multiplicity.values()) != {2}:
        raise SearchError(
            f"raw cset-1 multiplicities are not uniformly two: "
            f"{Counter(raw_multiplicity.values())}"
        )

    unrepresented_baseline = sorted(set(range(len(diamonds))) - set(cset1_baseline_indices))
    group_size_distribution = Counter(len(value) for value in groups.values())
    catalogue = {
        "shell_size": len(shell),
        "baseline_diamond_count": len(diamonds),
        "baseline_shell_rows": int(np.count_nonzero(is_baseline)),
        "nonbaseline_shell_rows": int(np.count_nonzero(~is_baseline)),
        "one_conflict_genuine_replacement_candidates": len(replacements),
        "genuine_touchable_directed_diamonds": len(groups),
        "genuine_rigid_directed_diamonds": len(rigid),
        "genuine_rigid_diamond_indices": rigid.astype(int).tolist(),
        "replacement_group_size_distribution": {
            str(key): value for key, value in sorted(group_size_distribution.items())
        },
        "replacement_pair_conflicts_within_same_group": within_conflicts,
        "replacement_pair_conflicts_across_groups": cross_conflicts,
        "conflict_count_distribution_over_shell": {
            str(key): value
            for key, value in sorted(Counter(conflict_counts.astype(int)).items())
        },
        "reported_steiner_generator_audit": {
            "raw_rows": len(raw),
            "unique_rows": len(unique_raw),
            "raw_cset1_rows": len(raw_cset1),
            "unique_cset1_rows": len(unique_cset1),
            "raw_cset1_multiplicity_distribution": {
                str(key): value
                for key, value in sorted(Counter(raw_multiplicity.values()).items())
            },
            "unique_cset1_rows_already_in_baseline": len(cset1_baseline_indices),
            "unique_cset1_rows_not_in_baseline": len(nonbaseline_unique_cset1),
            "baseline_diamond_indices_reproduced_by_generator": cset1_baseline_indices,
            "baseline_diamond_indices_absent_from_balanced_generator": unrepresented_baseline,
            "interpretation": (
                "The 528 raw cset=1 rows are two copies each of 264 existing "
                "ZE99 diamonds. Their sole conflict is self-inner-product 16. "
                "There are zero new cset=1 partners, so the claimed 2^264 "
                "independent switch family does not follow from this generator."
            ),
            "claimed_2_pow_264_switch_family_valid": False,
        },
    }
    return replacements, replacement_groups, groups, pair_compatible, catalogue


def choose_repairs(
    candidate: np.ndarray,
    affected: list[int],
    replacements: np.ndarray,
    groups: dict[int, list[int]],
    pair_compatible: np.ndarray,
) -> tuple[dict[int, int] | None, dict[str, Any]]:
    candidate_dots = replacements.astype(np.int16) @ candidate.astype(np.int16)
    domains: dict[int, list[int]] = {}
    empty: list[int] = []
    for group in affected:
        domain = [
            index for index in groups[group] if candidate_dots[index] <= BOUND
        ]
        domains[group] = domain
        if not domain:
            empty.append(group)
    if empty:
        return None, {
            "empty_repair_domains": empty,
            "domain_sizes": {str(group): len(domains[group]) for group in affected},
            "backtracking_nodes": 0,
        }

    nodes = 0

    def recurse(
        remaining: tuple[int, ...], selected: dict[int, int]
    ) -> dict[int, int] | None:
        nonlocal nodes
        nodes += 1
        if not remaining:
            return dict(selected)

        chosen_group = -1
        chosen_options: list[int] | None = None
        for group in remaining:
            options = [
                option
                for option in domains[group]
                if all(pair_compatible[option, other] for other in selected.values())
            ]
            if not options:
                return None
            if chosen_options is None or len(options) < len(chosen_options):
                chosen_group = group
                chosen_options = options
                if len(options) == 1:
                    break
        assert chosen_options is not None and chosen_group >= 0
        tail = tuple(group for group in remaining if group != chosen_group)
        for option in chosen_options:
            selected[chosen_group] = option
            answer = recurse(tail, selected)
            if answer is not None:
                return answer
            del selected[chosen_group]
        return None

    ordering = tuple(
        sorted(affected, key=lambda group: (len(domains[group]), group))
    )
    answer = recurse(ordering, {})
    return answer, {
        "empty_repair_domains": [],
        "domain_sizes": {str(group): len(domains[group]) for group in affected},
        "backtracking_nodes": nodes,
    }


def candidate_payload(
    candidate: np.ndarray,
    conflicts: list[int],
    rigid_hits: list[int],
    empty_domains: list[int],
    *,
    baseline_duplicate: bool,
) -> dict[str, Any]:
    return {
        "vector": candidate.astype(int).tolist(),
        "baseline_diamond_conflicts": conflicts,
        "conflict_count": len(conflicts),
        "rigid_conflicts": rigid_hits,
        "rigid_conflict_count": len(rigid_hits),
        "empty_repair_domains": empty_domains,
        "empty_repair_domain_count": len(empty_domains),
        "baseline_duplicate": baseline_duplicate,
    }


def exact_configuration(
    baseline: list[list[tuple[int, int]]],
    diamonds: np.ndarray,
    replacements: np.ndarray,
    candidate: np.ndarray,
    repairs: dict[int, int],
) -> tuple[list[list[tuple[int, int]]], dict[str, Any]]:
    selected = diamonds.copy()
    repair_rows: list[dict[str, Any]] = []
    for diamond_index, replacement_index in sorted(repairs.items()):
        original = selected[diamond_index].copy()
        replacement = replacements[replacement_index].copy()
        selected[diamond_index] = replacement
        repair_rows.append(
            {
                "diamond_index": diamond_index,
                "original": original.astype(int).tolist(),
                "replacement_index": replacement_index,
                "replacement": replacement.astype(int).tolist(),
                "original_replacement_inner_product": int(original @ replacement),
                "extra_replacement_inner_product": int(candidate @ replacement),
            }
        )

    selected_ab = [
        [(int(value), 0) for value in row.astype(int).tolist()] for row in selected
    ]
    extra_ab = [(int(value), 0) for value in candidate.astype(int).tolist()]
    vectors = (
        baseline[:TETRAD_STOP]
        + selected_ab
        + baseline[DIAMOND_STOP:]
        + [extra_ab]
    )
    if len(vectors) != RECORD + 1:
        raise SearchError(f"assembled {len(vectors)} vectors, expected {RECORD + 1}")

    verification = verify_ab(vectors)
    if not verification.get("ok"):
        raise SearchError(f"assembled candidate failed exact verification: {verification}")
    if verification.get("count") != RECORD + 1:
        raise SearchError(f"exact verifier returned unexpected count {verification}")
    return vectors, {
        "removed_and_replaced_diamonds": repair_rows,
        "extra_vector": candidate.astype(int).tolist(),
        "exact_verification": verification,
    }


def report_text(analysis: dict[str, Any]) -> str:
    lines = [
        "# Exact ZE99 Steiner-switch augmentation search",
        "",
        f"Generated `{analysis['generated_at']}`.",
        "",
        "## Result",
        "",
        f"- Current exact record: **{analysis['record']}**.",
        f"- Candidate count if successful: **{analysis['candidate_count']}**.",
        f"- Beats the record: **{analysis['beats_record']}**.",
        f"- Complete norm-16 integer family exhausted: **{analysis['integer_shell_reduction']['exact_family_exhaustive']}**.",
        f"- Integer candidates tested: **{analysis['search']['tested']}**.",
        "",
        "The fixed ZE99 tetrad, axial, and irrational layers reduce every compatible",
        "norm-16 integer vector to the 8,192-vector diamond shell. The program",
        "reproduces the reported Steiner generator, distinguishes raw duplicates from",
        "genuine replacements, then exhaustively tests whether any shell vector can",
        "be added after valid zero-cost repairs of all its diamond conflicts.",
        "",
        "A positive result is emitted only after the full 1,155-vector configuration",
        "passes the exact Q(sqrt(3)) verifier.",
        "",
        "## Switch catalogue",
        "",
        "```json",
        json.dumps(analysis["switch_catalogue"], indent=2, sort_keys=True),
        "```",
        "",
        "## Search summary",
        "",
        "```json",
        json.dumps(analysis["search"], indent=2, sort_keys=True),
        "```",
        "",
    ]
    if analysis.get("witness"):
        lines.extend(
            [
                "## Exact witness",
                "",
                "```json",
                json.dumps(analysis["witness"], indent=2, sort_keys=True),
                "```",
                "",
            ]
        )
    return "\n".join(lines)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--progress-log", type=Path)
    parser.add_argument("--near-misses", type=int, default=20)
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    stamp = utcnow()

    baseline = generate_ab()
    if len(baseline) != BASELINE_COUNT:
        raise SearchError(f"generated {len(baseline)} baseline vectors")
    baseline_verification = verify_ab(baseline)
    if not baseline_verification.get("ok"):
        raise SearchError(f"ZE99 baseline failed exact verification: {baseline_verification}")

    tetrads = np.vstack([integer_row(v) for v in baseline[:TETRAD_STOP]])
    diamonds = np.vstack(
        [integer_row(v) for v in baseline[TETRAD_STOP:DIAMOND_STOP]]
    )
    axials = np.vstack(
        [integer_row(v) for v in baseline[DIAMOND_STOP:AXIAL_STOP]]
    )
    irrationals = baseline[AXIAL_STOP:]
    if (tetrads.shape, diamonds.shape, axials.shape, len(irrationals)) != (
        (816, 13),
        (288, 13),
        (2, 13),
        48,
    ):
        raise SearchError("unexpected ZE99 layer sizes")

    non_diamonds = (
        baseline[:TETRAD_STOP] + baseline[DIAMOND_STOP:AXIAL_STOP] + irrationals
    )
    shell = diamond_shell()
    check_non_diamond_compatibility(shell, non_diamonds)
    reduction = derive_integer_shell_reduction(tetrads)

    (
        replacements,
        replacement_groups,
        groups,
        pair_compatible,
        catalogue,
    ) = build_switch_catalog(shell, diamonds, non_diamonds)
    touchable = set(groups)
    rigid = set(catalogue["genuine_rigid_diamond_indices"])
    baseline_diamond_keys = {tuple(row.tolist()) for row in diamonds}

    diamond_dots = shell.astype(np.int16) @ diamonds.astype(np.int16).T
    diamond_conflicts = diamond_dots > BOUND

    search_stats: dict[str, Any] = {
        "tested": 0,
        "nonbaseline_tested": 0,
        "baseline_duplicates": 0,
        "direct_extensions_without_repairs": 0,
        "rejected_by_rigid_diamond": 0,
        "rejected_by_empty_repair_domain": 0,
        "rejected_by_replacement_pair_conflicts": 0,
        "repairable_candidates": 0,
        "total_backtracking_nodes": 0,
        "candidate_conflict_count_distribution": {},
        "near_misses": [],
    }
    conflict_distribution: Counter[int] = Counter()
    near_heap: list[tuple[tuple[int, int, int, int], int, dict[str, Any]]] = []
    serial = 0
    witness_vectors: list[list[tuple[int, int]]] | None = None
    witness: dict[str, Any] | None = None

    for index, candidate in enumerate(shell):
        search_stats["tested"] += 1
        key = tuple(candidate.astype(int).tolist())
        duplicate = key in baseline_diamond_keys
        if duplicate:
            search_stats["baseline_duplicates"] += 1
        else:
            search_stats["nonbaseline_tested"] += 1

        conflicts = np.flatnonzero(diamond_conflicts[index]).astype(int).tolist()
        conflict_distribution[len(conflicts)] += 1
        rigid_hits = [group for group in conflicts if group in rigid]
        empty_domains: list[int] = []
        repairs: dict[int, int] | None = None
        diagnostics: dict[str, Any] = {
            "empty_repair_domains": [],
            "backtracking_nodes": 0,
        }

        if rigid_hits:
            search_stats["rejected_by_rigid_diamond"] += 1
        else:
            if not conflicts:
                search_stats["direct_extensions_without_repairs"] += 1
                repairs = {}
            else:
                if any(group not in touchable for group in conflicts):
                    raise SearchError("non-rigid conflict missing from switch catalogue")
                repairs, diagnostics = choose_repairs(
                    candidate,
                    conflicts,
                    replacements,
                    groups,
                    pair_compatible,
                )
                search_stats["total_backtracking_nodes"] += diagnostics[
                    "backtracking_nodes"
                ]
                empty_domains = diagnostics["empty_repair_domains"]
                if empty_domains:
                    search_stats["rejected_by_empty_repair_domain"] += 1
                elif repairs is None:
                    search_stats["rejected_by_replacement_pair_conflicts"] += 1

        payload = candidate_payload(
            candidate,
            conflicts,
            rigid_hits,
            empty_domains,
            baseline_duplicate=duplicate,
        )
        score = (
            len(rigid_hits),
            len(empty_domains),
            1 if repairs is None else 0,
            len(conflicts),
        )
        # Retain the lexicographically best near misses in a bounded max-heap.
        inverted = tuple(-value for value in score)
        item = (inverted, serial, payload)
        serial += 1
        if len(near_heap) < args.near_misses:
            heapq.heappush(near_heap, item)
        elif item > near_heap[0]:
            heapq.heapreplace(near_heap, item)

        if repairs is not None:
            try:
                vectors, exact = exact_configuration(
                    baseline, diamonds, replacements, candidate, repairs
                )
            except SearchError:
                # A duplicate or a hidden incompatibility is a hard correctness
                # error for the candidate, but not for the exhaustive scan.
                if duplicate:
                    continue
                raise
            search_stats["repairable_candidates"] += 1
            witness_vectors = vectors
            witness = {
                "shell_index": index,
                "baseline_conflicts": conflicts,
                "repairs": {
                    str(group): replacement
                    for group, replacement in sorted(repairs.items())
                },
                **exact,
            }
            break

    search_stats["candidate_conflict_count_distribution"] = {
        str(key): value for key, value in sorted(conflict_distribution.items())
    }
    near_misses = [
        item[2]
        for item in sorted(
            near_heap,
            key=lambda entry: tuple(-value for value in entry[0]),
        )
    ]
    search_stats["near_misses"] = near_misses

    success = witness_vectors is not None and witness is not None
    candidate_count = len(witness_vectors) if success else RECORD
    analysis: dict[str, Any] = {
        "generated_at": stamp,
        "status": "completed",
        "method": "exact audit of the reported ZE99 Steiner switches plus exhaustive integer-shell augmentation",
        "dimension": DIMENSION,
        "record": RECORD,
        "candidate_count": candidate_count,
        "beats_record": success,
        "baseline_exact_verification": baseline_verification,
        "integer_shell_reduction": reduction,
        "switch_catalogue": catalogue,
        "search": search_stats,
        "witness": witness,
        "arithmetic": (
            "integer dot products, exact squared comparisons in Q(sqrt(3)), "
            "and full exact Q(sqrt(3)) pairwise verification; no floating point"
        ),
    }
    write_json(args.output_dir / "analysis.json", analysis)
    (args.output_dir / "REPORT.md").write_text(report_text(analysis), encoding="utf-8")

    if success:
        assert witness_vectors is not None and witness is not None
        config_name = "best_exact_1155.json"
        configuration = {
            "dimension": DIMENSION,
            "count": len(witness_vectors),
            "norm_squared": NORM2,
            "unit_scaling": "divide every coordinate by 4",
            "coordinate_field": "Q(sqrt(3))",
            "vectors": [ab_to_strings(vector) for vector in witness_vectors],
            "verification": witness["exact_verification"],
            "record": RECORD,
            "beats_record": True,
            "construction": {
                "method": analysis["method"],
                "extra_vector": witness["extra_vector"],
                "removed_and_replaced_diamonds": witness[
                    "removed_and_replaced_diamonds"
                ],
            },
        }
        write_json(args.output_dir / config_name, configuration)
        write_json(
            args.output_dir / "best.json",
            {
                "dimension": DIMENSION,
                "count": len(witness_vectors),
                "configuration": f"ze99_steiner_switch_results/{config_name}",
                "max_off_diagonal_unit": "1/2",
                "record": RECORD,
                "beats_record": True,
                "exact": True,
            },
        )

    if args.progress_log:
        args.progress_log.parent.mkdir(parents=True, exist_ok=True)
        with args.progress_log.open("a", encoding="utf-8") as log:
            log.write(
                f"{utcnow()} method='ZE99 exact Steiner-switch audit and augmentation' "
                f"dimension=13 count={candidate_count} "
                f"result={'pass' if success else 'fail'} exact=true "
                "integer_family_exhausted=true\n"
            )

    print(json.dumps(analysis, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
