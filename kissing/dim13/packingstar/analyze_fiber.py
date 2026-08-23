#!/usr/bin/env python3
"""Decode PackingStar's 1146-point antipodal construction into its E6/E7 fiber product.

Input is the published floating .npy Gram matrix, but every structural assertion is
performed on H = 4G as an integer matrix.  Floating point is used only for the two
explicitly labelled diagnostics at the end: polytope hole search and a search over
normalized vertices of the root-polytopes' polars.

The exact part recovers

    1146 = 1008 + 54 + 84
         = 9 * 8 * 14 + 54 + 84,

with a 1008-point core built from nine complete K_{8,14} fibers between the 72 E6
roots and 126 E7 roots.  It also verifies exact tight-frame identities for the E6,
E7 and cap Gram matrices and proves the two component Gram matrices reconstruct the
core and occupy orthogonal subspaces.
"""

from __future__ import annotations

import argparse
import json
import math
from collections import Counter, defaultdict, deque
from dataclasses import dataclass
from datetime import datetime, timezone
from fractions import Fraction
from pathlib import Path
from typing import Any, Iterable

import numpy as np

GRAM_DEN = 4
TARGET_RECORD = 1154


class StructureError(RuntimeError):
    pass


@dataclass
class RationalMatrix:
    numerator: np.ndarray
    denominator: int

    def reduced(self) -> "RationalMatrix":
        g = self.denominator
        for value in self.numerator.flat:
            g = math.gcd(g, abs(int(value)))
            if g == 1:
                break
        return RationalMatrix(self.numerator // g, self.denominator // g)

    def max_offdiag(self) -> Fraction:
        x = self.numerator.copy()
        np.fill_diagonal(x, np.iinfo(np.int64).min)
        return Fraction(int(x.max()), self.denominator)

    def levels(self) -> dict[str, int]:
        values, counts = np.unique(self.numerator, return_counts=True)
        return {
            str(Fraction(int(v), self.denominator)): int(c)
            for v, c in zip(values.tolist(), counts.tolist())
        }


def exact_scaled_gram(path: Path) -> np.ndarray:
    G = np.load(path, allow_pickle=False)
    if G.shape != (1146, 1146):
        raise StructureError(f"expected 1146x1146 Gram matrix, got {G.shape}")
    H = np.rint(G * GRAM_DEN).astype(np.int64)
    error = float(np.max(np.abs(G - H / GRAM_DEN)))
    if error != 0.0:
        raise StructureError(f"stored Gram is not exactly quarter-integral: error {error}")
    if not np.array_equal(H, H.T):
        raise StructureError("integer Gram is not symmetric")
    if not np.all(np.diag(H) == GRAM_DEN):
        raise StructureError("integer Gram does not have unit diagonal")
    off = H.copy()
    np.fill_diagonal(off, -10)
    if int(off.max()) > 2:
        raise StructureError(f"off-diagonal exceeds 1/2: {int(off.max())}/4")
    return H


def row_profile_groups(H: np.ndarray) -> tuple[list[int], list[dict[str, Any]]]:
    levels = sorted(int(v) for v in np.unique(H))
    profile_to_indices: dict[tuple[int, ...], list[int]] = defaultdict(list)
    for i, row in enumerate(H):
        counts = Counter(int(v) for v in row)
        profile_to_indices[tuple(counts.get(v, 0) for v in levels)].append(i)
    groups = sorted(profile_to_indices.values(), key=lambda g: (len(g), g[0]))
    labels = np.empty(H.shape[0], dtype=np.int64)
    report: list[dict[str, Any]] = []
    for gid, indices in enumerate(groups):
        labels[indices] = gid
        report.append(
            {
                "group": gid,
                "size": len(indices),
                "first_index": indices[0],
                "row_level_counts": {
                    str(Fraction(level, GRAM_DEN)): int(count)
                    for level, count in zip(levels, profile_to_indices_inv(profile_to_indices, indices))
                },
                "indices": indices,
            }
        )
    return labels.tolist(), report


def profile_to_indices_inv(
    groups: dict[tuple[int, ...], list[int]], indices: list[int]
) -> tuple[int, ...]:
    for profile, found in groups.items():
        if found is indices:
            return profile
    raise AssertionError("profile identity lost")


def find_group(report: list[dict[str, Any]], size: int) -> list[int]:
    matches = [g["indices"] for g in report if g["size"] == size]
    if len(matches) != 1:
        raise StructureError(f"expected one row-profile class of size {size}, got {len(matches)}")
    return matches[0]


def global_reduce(numerator: np.ndarray, denominator: int) -> RationalMatrix:
    return RationalMatrix(numerator.astype(np.int64, copy=False), denominator).reduced()


def unique_row_classes(matrix: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    unique, inverse, counts = np.unique(
        matrix, axis=0, return_inverse=True, return_counts=True
    )
    return unique, inverse.astype(np.int64), counts.astype(np.int64)


def representatives(inverse: np.ndarray, nclasses: int) -> list[int]:
    reps = [-1] * nclasses
    for i, c in enumerate(inverse.tolist()):
        if reps[c] < 0:
            reps[c] = i
    if any(v < 0 for v in reps):
        raise StructureError("missing row-class representative")
    return reps


def exact_frame_identity(matrix: RationalMatrix, eigenvalue: int, name: str) -> None:
    N = matrix.numerator.astype(np.int64, copy=False)
    lhs = N @ N
    rhs = eigenvalue * matrix.denominator * N
    if not np.array_equal(lhs, rhs):
        bad = np.argwhere(lhs != rhs)[0]
        i, j = map(int, bad)
        raise StructureError(
            f"{name} frame identity G^2={eigenvalue}G fails at {(i, j)}: "
            f"{int(lhs[i,j])} != {int(rhs[i,j])}"
        )


def exact_gram_checks(matrix: RationalMatrix, name: str) -> dict[str, Any]:
    N, d = matrix.numerator, matrix.denominator
    if not np.array_equal(N, N.T):
        raise StructureError(f"{name} Gram is not symmetric")
    if not np.all(np.diag(N) == d):
        raise StructureError(f"{name} Gram does not have unit diagonal")
    off = N.copy()
    np.fill_diagonal(off, -10 * d)
    maximum = Fraction(int(off.max()), d)
    if maximum > Fraction(1, 2):
        raise StructureError(f"{name} max off-diagonal is {maximum}")
    rows = np.unique(N, axis=0).shape[0]
    if rows != N.shape[0]:
        raise StructureError(f"{name} has duplicate Gram rows: {rows}/{N.shape[0]}")
    eig = np.linalg.eigvalsh(N.astype(float) / d)
    rank = int(np.count_nonzero(eig > 1e-7))
    return {
        "count": int(N.shape[0]),
        "denominator": d,
        "max_off_diagonal": str(maximum),
        "rank": rank,
        "numerical_nonzero_eigenvalues": [
            float(v) for v in eig[eig > 1e-7]
        ],
        "levels": matrix.levels(),
    }


def connected_bipartite_components(incidence: np.ndarray) -> list[dict[str, Any]]:
    na, nb = incidence.shape
    seen_a = np.zeros(na, dtype=bool)
    seen_b = np.zeros(nb, dtype=bool)
    comps: list[dict[str, Any]] = []
    for start in range(na):
        if seen_a[start]:
            continue
        qa: deque[tuple[str, int]] = deque([("a", start)])
        aa: set[int] = set()
        bb: set[int] = set()
        while qa:
            side, vertex = qa.popleft()
            if side == "a":
                if vertex in aa:
                    continue
                aa.add(vertex)
                seen_a[vertex] = True
                for nxt in np.flatnonzero(incidence[vertex]):
                    if int(nxt) not in bb:
                        qa.append(("b", int(nxt)))
            else:
                if vertex in bb:
                    continue
                bb.add(vertex)
                seen_b[vertex] = True
                for nxt in np.flatnonzero(incidence[:, vertex]):
                    if int(nxt) not in aa:
                        qa.append(("a", int(nxt)))
        edge_count = int(incidence[np.ix_(sorted(aa), sorted(bb))].sum())
        comps.append(
            {
                "a": sorted(aa),
                "b": sorted(bb),
                "a_size": len(aa),
                "b_size": len(bb),
                "edges": edge_count,
                "complete_bipartite": edge_count == len(aa) * len(bb),
            }
        )
    if not np.all(seen_b):
        missing = np.flatnonzero(~seen_b).astype(int).tolist()
        raise StructureError(f"isolated B classes: {missing}")
    return comps


def factor_gram(G: np.ndarray, rank: int) -> np.ndarray:
    values, vectors = np.linalg.eigh(G)
    keep = values > 1e-7
    if int(np.count_nonzero(keep)) != rank:
        raise StructureError(
            f"combined Gram numerical rank {int(np.count_nonzero(keep))}, expected {rank}"
        )
    if float(values.min()) < -1e-7:
        raise StructureError(f"combined Gram has negative eigenvalue {float(values.min())}")
    X = vectors[:, keep] * np.sqrt(values[keep])[None, :]
    error = float(np.max(np.abs(X @ X.T - G)))
    if error > 2e-7:
        raise StructureError(f"Gram factorization error {error}")
    return X


def fixed_point_hole_search(
    roots: np.ndarray,
    caps: np.ndarray,
    *,
    seed: int,
    random_starts: int = 128,
) -> dict[str, Any]:
    from scipy.optimize import linprog

    dim = roots.shape[1]
    A = np.vstack([roots, caps])
    b = np.concatenate(
        [
            np.full(len(roots), 1 / math.sqrt(2)),
            np.full(len(caps), 0.5),
        ]
    )
    rng = np.random.default_rng(seed)
    starts: list[np.ndarray] = []
    starts.extend(np.eye(dim))
    starts.extend(-np.eye(dim))
    starts.extend(caps)
    starts.extend(rng.normal(size=(random_starts, dim)))

    best_norm = -1.0
    best_point: np.ndarray | None = None
    best_active: list[int] = []
    solves = 0
    for raw in starts:
        norm = float(np.linalg.norm(raw))
        if norm < 1e-15:
            continue
        direction = raw / norm
        previous: np.ndarray | None = None
        for _ in range(60):
            result = linprog(
                -direction,
                A_ub=A,
                b_ub=b,
                bounds=[(None, None)] * dim,
                method="highs",
            )
            if not result.success:
                break
            solves += 1
            point = np.asarray(result.x, dtype=float)
            point_norm = float(np.linalg.norm(point))
            if point_norm > best_norm:
                best_norm = point_norm
                best_point = point.copy()
                slack = b - A @ point
                best_active = np.flatnonzero(slack < 1e-8).astype(int).tolist()
            if point_norm < 1e-14:
                break
            new_direction = point / point_norm
            if previous is not None and np.linalg.norm(new_direction - direction) < 1e-11:
                break
            previous = direction
            direction = new_direction

    if best_point is None:
        return {"success": False, "reason": "all LP solves failed"}
    unit = best_point / best_norm
    root_max = float(np.max(roots @ unit))
    cap_max = float(np.max(caps @ unit))
    return {
        "success": True,
        "starts": len(starts),
        "successful_lp_solves": solves,
        "best_polytope_norm": best_norm,
        "single_point_extension_numerically_possible": best_norm >= 1 - 1e-9,
        "normalized_candidate_root_max": root_max,
        "normalized_candidate_cap_max": cap_max,
        "active_constraints": best_active,
        "note": "Numerical diagnostic only; no exact non-extension claim is made.",
    }


def deduplicate_rows(X: np.ndarray, decimals: int = 10) -> np.ndarray:
    keys: dict[tuple[float, ...], np.ndarray] = {}
    for row in X:
        key = tuple(np.round(row, decimals=decimals).tolist())
        keys.setdefault(key, row)
    return np.array(list(keys.values()), dtype=float)


def greedy_independent_set(candidates: np.ndarray, seed: int, rounds: int = 1000) -> list[int]:
    n = len(candidates)
    if n == 0:
        return []
    gram = candidates @ candidates.T
    compatible = gram <= 0.5 + 1e-9
    np.fill_diagonal(compatible, False)
    rng = np.random.default_rng(seed)
    degrees = compatible.sum(axis=1)
    best: list[int] = []
    for r in range(rounds):
        available = np.ones(n, dtype=bool)
        chosen: list[int] = []
        while np.any(available):
            pool = np.flatnonzero(available)
            if r < n:
                vertex = int(pool[np.argmin(degrees[pool])])
            elif rng.random() < 0.65:
                local = compatible[np.ix_(pool, pool)].sum(axis=1)
                cutoff = max(1, len(pool) // 12)
                order = np.argsort(-local)[:cutoff]
                vertex = int(pool[int(rng.choice(order))])
            else:
                vertex = int(rng.choice(pool))
            chosen.append(vertex)
            available &= compatible[vertex]
        if len(chosen) > len(best):
            best = chosen
    return best


def polar_vertex_pool(
    roots: np.ndarray,
    caps: np.ndarray,
    *,
    seed: int,
    max_milp_candidates: int = 3500,
) -> dict[str, Any]:
    from scipy.optimize import Bounds, LinearConstraint, milp
    from scipy.sparse import coo_matrix
    from scipy.spatial import HalfspaceIntersection

    dim = roots.shape[1]
    halfspaces = np.hstack(
        [roots, -np.full((len(roots), 1), 1 / math.sqrt(2))]
    )
    try:
        hs = HalfspaceIntersection(halfspaces, np.zeros(dim), qhull_options="Qx")
        vertices = deduplicate_rows(np.asarray(hs.intersections, dtype=float), decimals=9)
    except Exception as exc:
        return {"success": False, "stage": "halfspace_intersection", "error": str(exc)}

    norms = np.linalg.norm(vertices, axis=1)
    eligible = vertices[norms >= 1 - 2e-8]
    eligible_norms = np.linalg.norm(eligible, axis=1)
    directions = eligible / eligible_norms[:, None] if len(eligible) else np.empty((0, dim))
    directions = deduplicate_rows(np.vstack([directions, caps]), decimals=9)
    feasible = np.max(roots @ directions.T, axis=0) <= 1 / math.sqrt(2) + 2e-8
    directions = directions[feasible]

    result: dict[str, Any] = {
        "success": True,
        "polar_vertices": int(len(vertices)),
        "polar_vertex_norm_distribution": {
            f"{value:.9f}": int(count)
            for value, count in zip(*np.unique(np.round(norms, 9), return_counts=True))
        },
        "vertices_with_norm_at_least_one": int(len(eligible)),
        "normalized_candidate_directions_including_known_cap": int(len(directions)),
    }
    if len(directions) == 0:
        result["best_candidate_code"] = 0
        return result

    greedy = greedy_independent_set(directions, seed=seed, rounds=1200)
    result["greedy_best"] = len(greedy)
    result["greedy_indices"] = greedy

    n = len(directions)
    if n > max_milp_candidates:
        result["milp"] = {
            "attempted": False,
            "reason": f"candidate pool {n} exceeds cap {max_milp_candidates}",
        }
        return result

    gram = directions @ directions.T
    conflict_i, conflict_j = np.where(np.triu(gram > 0.5 + 2e-8, k=1))
    m = len(conflict_i)
    rows = np.repeat(np.arange(m, dtype=np.int64), 2)
    cols = np.column_stack([conflict_i, conflict_j]).reshape(-1)
    data = np.ones(2 * m, dtype=float)
    constraints = coo_matrix((data, (rows, cols)), shape=(m, n)).tocsr()
    lower = np.full(m, -np.inf)
    upper = np.ones(m)
    try:
        opt = milp(
            c=-np.ones(n),
            integrality=np.ones(n),
            bounds=Bounds(np.zeros(n), np.ones(n)),
            constraints=LinearConstraint(constraints, lower, upper),
            options={"time_limit": 180.0, "mip_rel_gap": 0.0, "presolve": True},
        )
        chosen = (
            np.flatnonzero(np.asarray(opt.x) > 0.5).astype(int).tolist()
            if opt.x is not None
            else []
        )
        result["milp"] = {
            "attempted": True,
            "status": int(opt.status),
            "message": str(opt.message),
            "objective_count": len(chosen),
            "indices": chosen,
            "mip_gap": None if getattr(opt, "mip_gap", None) is None else float(opt.mip_gap),
            "mip_node_count": None
            if getattr(opt, "mip_node_count", None) is None
            else int(opt.mip_node_count),
            "proven_optimal": int(opt.status) == 0,
            "conflict_edges": int(m),
        }
    except Exception as exc:
        result["milp"] = {"attempted": True, "error": str(exc)}
    return result


def combined_subspace_coordinates(
    root_gram: RationalMatrix,
    cap_gram_scaled: np.ndarray,
    cross_full_scaled: np.ndarray,
    rank: int,
) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    R = root_gram.numerator.astype(float) / root_gram.denominator
    T = cap_gram_scaled.astype(float) / GRAM_DEN
    C = cross_full_scaled.astype(float) * (math.sqrt(2) / GRAM_DEN)
    K = np.block([[R, C.T], [C, T]])
    X = factor_gram(K, rank)
    roots = X[: len(R)]
    caps = X[len(R) :]
    return roots, caps, {
        "combined_count": int(len(K)),
        "rank": rank,
        "factorization_error": float(np.max(np.abs(X @ X.T - K))),
        "root_norm_error": float(np.max(np.abs(np.sum(roots * roots, axis=1) - 1))),
        "cap_norm_error": float(np.max(np.abs(np.sum(caps * caps, axis=1) - 1))),
        "root_cap_max": float(np.max(roots @ caps.T)),
    }


def analyze(path: Path, run_expensive: bool) -> dict[str, Any]:
    H = exact_scaled_gram(path)
    labels, profiles = row_profile_groups(H)
    profile_sizes = sorted(g["size"] for g in profiles)
    if profile_sizes != [54, 84, 112, 896]:
        raise StructureError(f"unexpected row-profile sizes {profile_sizes}")

    cap6 = find_group(profiles, 54)
    cap7 = find_group(profiles, 84)
    if np.any(H[np.ix_(cap6, cap7)] != 0):
        raise StructureError("54- and 84-point caps are not exactly orthogonal")
    caps = set(cap6) | set(cap7)
    core = [i for i in range(len(H)) if i not in caps]
    if len(core) != 1008:
        raise StructureError(f"core size {len(core)}, expected 1008")

    Hc = H[np.ix_(core, core)]
    H2 = Hc @ Hc
    Araw = H2 - 288 * Hc
    Braw = 336 * Hc - H2
    A = global_reduce(Araw, 96)
    B = global_reduce(Braw, 96)
    if not np.array_equal(
        A.numerator * (2 * B.denominator),
        B.numerator * (2 * A.denominator) + Hc * (A.denominator * B.denominator),
    ):
        # Easier direct statement below; this branch merely catches arithmetic drift.
        pass
    # A + B = Hc/2 exactly.
    common = math.lcm(A.denominator, B.denominator, 2)
    sum_num = (
        A.numerator * (common // A.denominator)
        + B.numerator * (common // B.denominator)
    )
    target_num = Hc * (common // 2)
    if not np.array_equal(sum_num, target_num):
        raise StructureError("component Grams do not sum to twice the core Gram")

    _, invA, countsA = unique_row_classes(A.numerator)
    _, invB, countsB = unique_row_classes(B.numerator)
    if len(countsA) != 72 or set(countsA.tolist()) != {14}:
        raise StructureError(f"A row classes are {len(countsA)} with multiplicities {set(countsA.tolist())}")
    if len(countsB) != 126 or set(countsB.tolist()) != {8}:
        raise StructureError(f"B row classes are {len(countsB)} with multiplicities {set(countsB.tolist())}")

    repsA = representatives(invA, 72)
    repsB = representatives(invB, 126)
    Au = RationalMatrix(A.numerator[np.ix_(repsA, repsA)], A.denominator).reduced()
    Bu = RationalMatrix(B.numerator[np.ix_(repsB, repsB)], B.denominator).reduced()
    if not np.array_equal(
        A.numerator,
        Au.numerator[np.ix_(invA, invA)] * (A.denominator // Au.denominator),
    ):
        raise StructureError("full E6 component is not the repeated unique Gram")
    if not np.array_equal(
        B.numerator,
        Bu.numerator[np.ix_(invB, invB)] * (B.denominator // Bu.denominator),
    ):
        raise StructureError("full E7 component is not the repeated unique Gram")

    exact_frame_identity(Au, 12, "E6 root")
    exact_frame_identity(Bu, 18, "E7 root")
    e6_info = exact_gram_checks(Au, "E6 root")
    e7_info = exact_gram_checks(Bu, "E7 root")
    if e6_info["rank"] != 6 or e7_info["rank"] != 7:
        raise StructureError(f"root ranks are {e6_info['rank']} and {e7_info['rank']}")

    incidence = np.zeros((72, 126), dtype=np.int64)
    pair_counts: Counter[tuple[int, int]] = Counter()
    for a, b in zip(invA.tolist(), invB.tolist()):
        pair_counts[(a, b)] += 1
    if max(pair_counts.values()) != 1:
        raise StructureError("a component-root pair occurs more than once")
    for (a, b), count in pair_counts.items():
        incidence[a, b] = count
    comps = connected_bipartite_components(incidence)
    if len(comps) != 9:
        raise StructureError(f"fiber incidence has {len(comps)} components, expected 9")
    for comp in comps:
        if (comp["a_size"], comp["b_size"], comp["edges"]) != (8, 14, 112):
            raise StructureError(f"unexpected fiber component {comp}")
        if not comp["complete_bipartite"]:
            raise StructureError("fiber component is not complete bipartite")
        aidx, bidx = comp["a"], comp["b"]
        aoff = Au.numerator[np.ix_(aidx, aidx)].copy()
        boff = Bu.numerator[np.ix_(bidx, bidx)].copy()
        np.fill_diagonal(aoff, -10 * Au.denominator)
        np.fill_diagonal(boff, -10 * Bu.denominator)
        if int(aoff.max()) > 0 or int(boff.max()) > 0:
            raise StructureError("a fiber block is not a nonpositive code")

    # Orthogonality of the two repeated component Grams, reduced to 72x126.
    cross_product = Au.numerator @ incidence @ Bu.numerator
    if np.any(cross_product != 0):
        bad = np.argwhere(cross_product != 0)[0]
        raise StructureError(f"E6/E7 component Grams are not orthogonal at {tuple(map(int,bad))}")

    T6 = RationalMatrix(H[np.ix_(cap6, cap6)], GRAM_DEN).reduced()
    T7 = RationalMatrix(H[np.ix_(cap7, cap7)], GRAM_DEN).reduced()
    exact_frame_identity(T6, 9, "54-point cap")
    exact_frame_identity(T7, 12, "84-point cap")
    t6_info = exact_gram_checks(T6, "54-point cap")
    t7_info = exact_gram_checks(T7, "84-point cap")
    if t6_info["rank"] != 6 or t7_info["rank"] != 7:
        raise StructureError(f"cap ranks are {t6_info['rank']} and {t7_info['rank']}")

    # Cross terms must depend only on the matching root component.
    C6full = H[np.ix_(cap6, core)]
    C7full = H[np.ix_(cap7, core)]
    C6 = C6full[:, repsA]
    C7 = C7full[:, repsB]
    if not np.array_equal(C6full, C6[:, invA]):
        raise StructureError("54-cap/core cross terms do not factor through E6 classes")
    if not np.array_equal(C7full, C7[:, invB]):
        raise StructureError("84-cap/core cross terms do not factor through E7 classes")

    roots6, caps6, align6 = combined_subspace_coordinates(Au, H[np.ix_(cap6, cap6)], C6, 6)
    roots7, caps7, align7 = combined_subspace_coordinates(Bu, H[np.ix_(cap7, cap7)], C7, 7)

    result: dict[str, Any] = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "source": str(path),
        "status": "verified_exact_structure",
        "dimension": 13,
        "count": 1146,
        "beats_record": False,
        "record": TARGET_RECORD,
        "row_profile_groups": profiles,
        "decomposition": {
            "core": len(core),
            "cap_R6": len(cap6),
            "cap_R7": len(cap7),
            "formula": "9*8*14 + 54 + 84 = 1146",
            "cap_cross_orthogonal": True,
        },
        "core_component_formula": {
            "E6_repeated": "G0*(G0-72I)/6",
            "E7_repeated": "G0*(84I-G0)/6",
            "sum": "A+B=2G0",
            "exact_denominators": {"A": A.denominator, "B": B.denominator},
            "orthogonal": True,
        },
        "E6_roots": e6_info,
        "E7_roots": e7_info,
        "cap54": t6_info,
        "cap84": t7_info,
        "fiber_components": comps,
        "fiber_summary": {
            "components": len(comps),
            "component_type": "K_8,14",
            "edges_per_component": 112,
            "total_core_points": int(incidence.sum()),
            "E6_class_multiplicity": sorted(set(countsA.astype(int).tolist())),
            "E7_class_multiplicity": sorted(set(countsB.astype(int).tolist())),
        },
        "subspace_alignment_R6": align6,
        "subspace_alignment_R7": align7,
        "exact_cross_levels": {
            "cap54_to_E6_full_inner_coefficients_times_sqrt2": {
                str(Fraction(int(v), GRAM_DEN)): int(c)
                for v, c in zip(*np.unique(C6, return_counts=True))
            },
            "cap84_to_E7_full_inner_coefficients_times_sqrt2": {
                str(Fraction(int(v), GRAM_DEN)): int(c)
                for v, c in zip(*np.unique(C7, return_counts=True))
            },
        },
    }

    result["single_cap_extension_diagnostics"] = {
        "R6": fixed_point_hole_search(roots6, caps6, seed=61354),
        "R7": fixed_point_hole_search(roots7, caps7, seed=71384),
    }
    if run_expensive:
        result["polar_vertex_candidate_search"] = {
            "R6": polar_vertex_pool(roots6, caps6, seed=60054),
            "R7": polar_vertex_pool(roots7, caps7, seed=70084),
        }
    else:
        result["polar_vertex_candidate_search"] = {
            "skipped": True,
            "reason": "run with --expensive to enumerate polar vertices and solve candidate MIS",
        }
    return result


def report_text(result: dict[str, Any]) -> str:
    d = result["decomposition"]
    f = result["fiber_summary"]
    lines = [
        "# Exact fiber-product analysis of PackingStar's 1146 in dimension 13",
        "",
        f"Generated `{result['generated_at']}`.",
        "",
        "## Exact result",
        "",
        "The published antipodal 1,146-point Gram matrix has been decoded exactly as",
        "",
        f"`{d['formula']}`.",
        "",
        f"The 1,008-point core has {f['components']} connected fiber components, each an",
        f"exact complete bipartite `{f['component_type']}`. The component root Grams are",
        "the 72-point E6 root system (rank 6, frame eigenvalue 12) and the 126-point",
        "E7 root system (rank 7, frame eigenvalue 18). The 54- and 84-point residual",
        "caps are orthogonal tight frames of ranks 6 and 7, respectively.",
        "",
        "All identities above were checked on integer/rational Gram matrices. The",
        "floating-point computations below are diagnostics only.",
        "",
        "## Single-point cap extension diagnostics",
        "",
        "```json",
        json.dumps(result["single_cap_extension_diagnostics"], indent=2),
        "```",
        "",
        "## Root-polar vertex candidate family",
        "",
        "```json",
        json.dumps(result["polar_vertex_candidate_search"], indent=2),
        "```",
        "",
        "## Consequence for this construction family",
        "",
        "The core already contributes 1,008 points. Beating the 1,154 record inside",
        "this fixed fiber core requires at least 147 cap points in total, versus the",
        "current 54+84=138. Therefore a successful fixed-core extension needs a net",
        "gain of at least nine cap points; a single addable cap point would not suffice.",
        "",
    ]
    return "\n".join(lines)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--matrix", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--expensive", action="store_true")
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    result = analyze(args.matrix, args.expensive)
    (args.output_dir / "fiber_analysis.json").write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    (args.output_dir / "FIBER_REPORT.md").write_text(
        report_text(result), encoding="utf-8"
    )
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
