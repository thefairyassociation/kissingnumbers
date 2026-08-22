#!/usr/bin/env python3
"""Numerically inspect the public N=841, d=12 witness.

This is a read-only research diagnostic, not an exact certificate or a search
driver.  It uses the attributed coordinate fixture already used by
``test_faithful_841.py`` and writes JSON only when ``--json-out`` is supplied.
"""
from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
import sys

import tempfile

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from prepare_841_polish import decimal_gram_reconstruct  # noqa: E402


EXPECTED_MAX = 0.4999999377514321


def procrustes(a: np.ndarray, b: np.ndarray) -> tuple[np.ndarray, dict[str, float]]:
    """Globally rotate labelled unit rows of ``a`` to best fit ``b``."""
    u, _, vt = np.linalg.svd(a.T @ b, full_matrices=False)
    r = u @ vt
    aligned = a @ r
    error = np.linalg.norm(aligned - b, axis=1)
    return aligned, {
        "rms": float(np.sqrt(np.mean(error * error))),
        "median": float(np.median(error)),
        "max": float(error.max()),
    }


def components(adjacency: np.ndarray) -> dict[str, object]:
    n = len(adjacency)
    seen = np.zeros(n, dtype=bool)
    sizes: list[int] = []
    for start in range(n):
        if seen[start]:
            continue
        todo, seen[start], count = [start], True, 0
        while todo:
            i = todo.pop()
            count += 1
            nxt = np.flatnonzero(adjacency[i] & ~seen)
            seen[nxt] = True
            todo.extend(map(int, nxt))
        sizes.append(count)
    sizes.sort(reverse=True)
    return {"count": len(sizes), "largest": sizes[0], "sizes_over_1": [x for x in sizes if x > 1]}


def near_contact_stats(gram: np.ndarray, threshold: float) -> dict[str, object]:
    upper = np.triu(gram >= threshold, 1)
    degree = upper.sum(axis=0) + upper.sum(axis=1)
    adjacency_bool = upper | upper.T
    # Integer multiplication counts length-3 walks.  Boolean matmul would
    # collapse every nonzero count to True and undercount triangles.
    adjacency = adjacency_bool.astype(np.int64)
    return {
        "threshold": threshold,
        "edges": int(upper.sum()),
        "degree_min": int(degree.min()),
        "degree_mean": float(degree.mean()),
        "degree_max": int(degree.max()),
        "degree_histogram": {str(k): int(v) for k, v in sorted(Counter(map(int, degree)).items())},
        "components": components(adjacency_bool),
        "triangles": int(np.trace(adjacency @ adjacency @ adjacency) // 6),
    }


def antipodal_stats(gram: np.ndarray) -> dict[str, object]:
    work = gram.copy()
    np.fill_diagonal(work, np.inf)
    nearest = work.min(axis=1)
    partner = work.argmin(axis=1)
    mutual = [(i, int(partner[i])) for i in range(len(gram)) if i < partner[i] and partner[partner[i]] == i]
    return {
        "nearest_dot_quantiles": np.quantile(nearest, [0, .01, .1, .5, .9, .99, 1]).tolist(),
        "mutual_nearest_pairs": len(mutual),
        "rows_in_mutual_pairs": 2 * len(mutual),
        "rows_below_minus_0_999": int(np.count_nonzero(nearest < -0.999)),
        "rows_below_minus_0_99": int(np.count_nonzero(nearest < -0.99)),
        "unpaired_rows": np.flatnonzero(partner[partner] != np.arange(len(gram))).astype(int).tolist(),
    }


def o4_diagnostic(canonical: np.ndarray, solution: np.ndarray) -> dict[str, object]:
    """Test the natural labels against the local family from Theorem 2.

    This is deliberately a label-dependent least-squares diagnostic.  It does
    not decide membership after an unknown permutation or prove non-membership
    in another representation.
    """
    aligned, global_fit = procrustes(solution, canonical)
    blocks = {
        "left_48": (slice(0, 48), slice(1, 5)),
        "right_48": (slice(60, 108), slice(7, 11)),
    }
    fits: dict[str, object] = {}
    for name, (rows, cols) in blocks.items():
        u, _, vt = np.linalg.svd(canonical[rows, cols].T @ aligned[rows, cols], full_matrices=False)
        q = u @ vt
        residual = np.linalg.norm(canonical[rows, cols] @ q - aligned[rows, cols], axis=1)
        fits[name] = {
            "residual_rms": float(np.sqrt(np.mean(residual * residual))),
            "residual_max": float(residual.max()),
            "determinant": float(np.linalg.det(q)),
            "max_row_l1": float(np.abs(q).sum(axis=1).max()),
        }
    group_displacement = {}
    for name, rows in {
        "left_48": slice(0, 48), "left_orts": slice(48, 60),
        "right_48": slice(60, 108), "right_orts": slice(108, 120),
        "bridges": slice(120, 840),
    }.items():
        error = np.linalg.norm(aligned[rows] - canonical[rows], axis=1)
        group_displacement[name] = {"rms": float(np.sqrt(np.mean(error * error))), "max": float(error.max())}
    return {"labelled_global_fit": global_fit, "local_o4_fits": fits, "row_group_displacements": group_displacement}


def canonical_840(root: Path) -> np.ndarray:
    sys.path.insert(0, str(root / "kissing" / "dim12"))
    from constructions.clebsch840 import construction_840, vectors_as_float
    return vectors_as_float(construction_840())


def main() -> None:
    parser = argparse.ArgumentParser()
    root = Path(__file__).resolve().parents[2]
    parser.add_argument("--coordinates", type=Path, default=root / "kissing" / "lib" / "testdata" / "authors_841_coordinates.txt")
    parser.add_argument("--json-out", type=Path, help="optional path for complete numeric results")
    args = parser.parse_args()

    raw = np.loadtxt(args.coordinates, dtype=np.float64)
    if raw.shape != (841, 12) or not np.isfinite(raw).all():
        raise ValueError(f"expected finite (841, 12) coordinates, got {raw.shape}")
    norms = np.linalg.norm(raw, axis=1)
    if np.any(norms <= 1e-15):
        raise ValueError("zero-length input row")
    points = raw / norms[:, None]
    gram = points @ points.T
    upper = gram[np.triu_indices(841, 1)]
    max_ip = float(upper.max())
    where = np.argwhere(np.triu(gram, 1) == max_ip)[0].astype(int).tolist()
    eig = np.linalg.eigvalsh((gram + gram.T) / 2)
    # The authors' search-to-polish handoff: Gram serialized at %.10f,
    # symmetrized, truncated to rank 12, then row-normalized.  Run the real
    # text round-trip from prepare_841_polish.py rather than a second inline
    # copy of it, so this structural note and the polish input cannot drift.
    # (np.round(gram, 10) agrees with the %.10f round-trip on every entry of
    # this Gram, but it is not the same operation in general.)
    with tempfile.TemporaryDirectory(prefix="published-841-") as handoff:
        handoff_dir = Path(handoff)
        _, reconstructed_max = decimal_gram_reconstruct(
            points,
            handoff_dir / "gram_10dp.txt",
            handoff_dir / "reconstructed.txt",
        )
    canonical = canonical_840(root)
    canonical_gram = canonical @ canonical.T
    natural_solution, natural_fit = procrustes(points[:840], canonical)
    natural_delta = np.linalg.norm(natural_solution - canonical, axis=1)
    result: dict[str, object] = {
        "coordinate_verification": {
            "normalization_max_change": float(np.abs(norms - 1).max()),
            "max_offdiag": max_ip,
            "argmax_pair_0_based": where,
            "rank_abs_gt_1e-8": int(np.count_nonzero(np.abs(eig) > 1e-8)),
            "smallest_gram_eigenvalue": float(eig[0]),
            "positive_gram_eigenvalues_descending": eig[eig > 1e-8][::-1].tolist(),
            "authors_decimal_gram_reconstruction_max_offdiag": reconstructed_max,
        },
        "natural_labelled_840_comparison": {
            "global_orthogonal_procrustes": natural_fit,
            "per_row_displacement_rms": float(np.sqrt(np.mean(natural_delta * natural_delta))),
            "per_row_displacement_max": float(natural_delta.max()),
            "gram_frobenius_difference_per_840_row": float(np.linalg.norm(gram[:840, :840] - canonical_gram) / 840),
            "max_gram_entry_difference": float(np.max(np.abs(gram[:840, :840] - canonical_gram))),
        },
        "o4_family_diagnostic": o4_diagnostic(canonical, points[:840]),
        "near_contact": {str(t): near_contact_stats(gram, t) for t in (0.4999999, 0.499999, 0.49999, 0.4999, 0.499)},
        "near_antipodal": antipodal_stats(gram),
        "canonical_840_near_antipodal": antipodal_stats(canonical_gram),
    }
    if abs(max_ip - EXPECTED_MAX) > 5e-13:
        raise AssertionError(f"fixture max-IP changed: {max_ip:.17g}")
    if args.json_out:
        args.json_out.write_text(json.dumps(result, indent=2) + "\n")
    print(f"verified max-IP={max_ip:.16f} pair={where}; rank={result['coordinate_verification']['rank_abs_gt_1e-8']}; "
          f"decimal-Gram reconstruction={reconstructed_max:.16f}")
    print("natural-label Procrustes RMS=" f"{natural_fit['rms']:.7f}; "
          "O(4) residual RMS=" f"{result['o4_family_diagnostic']['local_o4_fits']['left_48']['residual_rms']:.7f}/"
          f"{result['o4_family_diagnostic']['local_o4_fits']['right_48']['residual_rms']:.7f}")
    print("near-antipodal mutual pairs=" f"{result['near_antipodal']['mutual_nearest_pairs']}")
    if args.json_out:
        print(f"wrote {args.json_out}")


if __name__ == "__main__":
    main()
