#!/usr/bin/env python3
"""Independently reconstruct and exactly verify PackingStar's dimension-13 Gram matrices.

The public PackingStar files are floating-point .npy serializations of rational Gram
matrices.  This program reconstructs a common exact denominator, proves the Gram
matrix is symmetric, positive semidefinite of rank 13, has diagonal 1, and has every
off-diagonal entry at most 1/2.  It then emits explicit exact coordinates in an
orthogonal basis: coordinate j is a rational coefficient times sqrt(r_j), where the
positive rational radicands r_j are stored once per file.

No floating-point calculation is used for the final certificate.  Floating point is
used only to discover a candidate denominator and a numerically independent basis;
every claimed property is subsequently checked over the integers/rationals.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import sys
from dataclasses import dataclass
from datetime import datetime, timezone
from fractions import Fraction
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import sympy as sp

TARGET_DIM = 13
LIVE_RECORD = 1154
HALF = Fraction(1, 2)


class VerificationError(RuntimeError):
    """Raised when an exact certificate cannot be established."""


@dataclass
class ExactGram:
    path: Path
    source_sha256: str
    count: int
    denominator: int
    matrix: np.ndarray
    reconstruction_max_error: float
    basis_indices: list[int]
    inverse_numerator: np.ndarray
    inverse_denominator: int
    leading_principal_minors: list[int]
    coordinate_radicands: list[Fraction]
    coordinate_coefficients: list[list[Fraction]]
    max_offdiag: Fraction
    max_offdiag_pair: tuple[int, int]
    tight_pairs: int
    antipodal_pairs: int
    unique_rows: int
    level_counts: dict[str, int]
    exact_factorization_method: str


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def lcm(values: Iterable[int]) -> int:
    out = 1
    for value in values:
        out = math.lcm(out, int(value))
    return out


def frac_json(x: Fraction) -> dict[str, int]:
    return {"num": x.numerator, "den": x.denominator}


def frac_text(x: Fraction) -> str:
    return str(x.numerator) if x.denominator == 1 else f"{x.numerator}/{x.denominator}"


def infer_common_denominator(
    matrix: np.ndarray,
    *,
    max_denominator: int = 8192,
    tolerance: float = 2e-10,
) -> tuple[int, float]:
    """Find the smallest q such that every matrix entry is within tolerance of Z/q."""
    if not np.issubdtype(matrix.dtype, np.number):
        raise VerificationError(f"unsupported non-numeric dtype {matrix.dtype}")
    if not np.all(np.isfinite(matrix)):
        raise VerificationError("matrix contains NaN or infinity")

    rounded = np.unique(np.round(matrix.reshape(-1), decimals=11))
    if rounded.size > 100_000:
        flat = matrix.reshape(-1)
        step = max(1, flat.size // 100_000)
        rounded = np.unique(np.round(flat[::step], decimals=11))

    best: tuple[int, float] | None = None
    for q in range(1, max_denominator + 1):
        err = float(np.max(np.abs(rounded * q - np.rint(rounded * q))) / q)
        if err <= tolerance:
            full_err = float(
                np.max(np.abs(matrix * q - np.rint(matrix * q))) / q
            )
            if full_err <= tolerance:
                best = (q, full_err)
                break
    if best is None:
        if rounded.size > 5000:
            raise VerificationError(
                f"no common denominator <= {max_denominator}; {rounded.size} levels remain"
            )
        fracs = [Fraction(float(v)).limit_denominator(max_denominator) for v in rounded]
        q = lcm(f.denominator for f in fracs)
        if q > 2_000_000:
            raise VerificationError(f"inferred common denominator is too large: {q}")
        full_err = float(np.max(np.abs(matrix * q - np.rint(matrix * q))) / q)
        if full_err > tolerance:
            raise VerificationError(
                f"rational reconstruction error {full_err:.3e} exceeds {tolerance:.3e}"
            )
        best = (q, full_err)
    return best


def pivoted_cholesky_basis(matrix: np.ndarray, rank: int) -> list[int]:
    """Choose a numerically independent principal basis; exact invertibility is checked later."""
    n = matrix.shape[0]
    residual = np.diag(matrix).astype(float).copy()
    factors = np.zeros((n, rank), dtype=float)
    chosen: list[int] = []
    used = np.zeros(n, dtype=bool)

    for k in range(rank):
        candidates = residual.copy()
        candidates[used] = -np.inf
        pivot = int(np.argmax(candidates))
        pivot_residual = float(candidates[pivot])
        if not math.isfinite(pivot_residual) or pivot_residual < 1e-9:
            raise VerificationError(
                f"numeric rank below {rank}: pivot {k} residual {pivot_residual}"
            )
        chosen.append(pivot)
        used[pivot] = True
        col = matrix[:, pivot].astype(float).copy()
        if k:
            col -= factors[:, :k] @ factors[pivot, :k]
        factors[:, k] = col / math.sqrt(pivot_residual)
        residual -= factors[:, k] ** 2
        residual[np.abs(residual) < 1e-10] = 0.0
    return chosen


def exact_ldl(matrix: sp.Matrix) -> tuple[sp.Matrix, list[sp.Rational]]:
    """Rational LDL^T decomposition of a positive-definite rational matrix."""
    n = matrix.rows
    L = sp.eye(n)
    d: list[sp.Rational] = []
    for j in range(n):
        value = sp.Rational(matrix[j, j])
        for k in range(j):
            value -= L[j, k] * L[j, k] * d[k]
        value = sp.cancel(value)
        if not bool(value > 0):
            raise VerificationError(f"LDL pivot {j} is not positive: {value}")
        d.append(sp.Rational(value))
        for i in range(j + 1, n):
            numer = sp.Rational(matrix[i, j])
            for k in range(j):
                numer -= L[i, k] * L[j, k] * d[k]
            L[i, j] = sp.cancel(numer / d[j])
    if sp.simplify(L * sp.diag(*d) * L.T - matrix) != sp.zeros(n):
        raise VerificationError("exact LDL reconstruction failed")
    return L, d


def matrix_product_exact(
    left: np.ndarray,
    middle: np.ndarray,
    right_t: np.ndarray,
) -> tuple[np.ndarray, str]:
    """Compute left @ middle @ right_t exactly, using int64 only when proved safe."""
    left_obj = left.astype(object, copy=False)
    middle_obj = middle.astype(object, copy=False)
    right_obj = right_t.astype(object, copy=False)

    r = left.shape[1]
    max_left = max((abs(int(v)) for v in left_obj.flat), default=0)
    max_middle = max((abs(int(v)) for v in middle_obj.flat), default=0)
    max_right = max((abs(int(v)) for v in right_obj.flat), default=0)
    first_bound = r * max_left * max_middle
    final_bound = r * first_bound * max_right
    if max(first_bound, final_bound) < 2**62:
        left64 = left_obj.astype(np.int64)
        middle64 = middle_obj.astype(np.int64)
        right64 = right_obj.astype(np.int64)
        first = left64 @ middle64
        return first @ right64, (
            f"numpy int64, proven bounds first={first_bound}, final={final_bound} < 2^62"
        )

    first_obj = left_obj @ middle_obj
    n_rows = left.shape[0]
    n_cols = right_t.shape[1]
    out = np.empty((n_rows, n_cols), dtype=object)
    block = 32
    for start in range(0, n_rows, block):
        stop = min(start + block, n_rows)
        out[start:stop] = first_obj[start:stop] @ right_obj
    return out, (
        "Python arbitrary-precision integers; int64 rejected because "
        f"first bound={first_bound}, final bound={final_bound}"
    )


def choose_exact_basis(H: np.ndarray, floating: np.ndarray, rank: int) -> list[int]:
    initial = pivoted_cholesky_basis(floating, rank)
    B = sp.Matrix(H[np.ix_(initial, initial)].tolist())
    if B.det() != 0:
        return initial

    indices: list[int] = []
    current_rank = 0
    for idx in range(H.shape[0]):
        trial = indices + [idx]
        sub = sp.Matrix(H[:, trial].tolist())
        rank_now = sub.rank()
        if rank_now > current_rank:
            indices.append(idx)
            current_rank = rank_now
            if current_rank == rank:
                break
    if len(indices) != rank:
        raise VerificationError(f"could not find {rank} independent columns")
    if sp.Matrix(H[np.ix_(indices, indices)].tolist()).det() == 0:
        raise VerificationError("independent columns did not yield invertible principal basis")
    return indices


def exact_verify_gram(path: Path, matrix: np.ndarray) -> ExactGram:
    if matrix.ndim != 2 or matrix.shape[0] != matrix.shape[1]:
        raise VerificationError(f"not a square matrix: shape={matrix.shape}")
    n = int(matrix.shape[0])
    if n < TARGET_DIM:
        raise VerificationError(f"count {n} is below dimension {TARGET_DIM}")

    q, reconstruction_error = infer_common_denominator(matrix)
    scaled = np.rint(matrix * q)
    if np.max(np.abs(scaled)) >= 2**62:
        raise VerificationError("scaled Gram entries exceed int64 safety margin")
    H = scaled.astype(np.int64)

    if not np.array_equal(H, H.T):
        bad = np.argwhere(H != H.T)[0]
        raise VerificationError(f"exact symmetry fails at {tuple(map(int, bad))}")
    if not np.all(np.diag(H) == q):
        i = int(np.flatnonzero(np.diag(H) != q)[0])
        raise VerificationError(f"diagonal entry {i} is {H[i, i]}/{q}, not 1")

    off = H.copy()
    np.fill_diagonal(off, np.iinfo(np.int64).min)
    max_flat = int(np.argmax(off))
    max_pair = tuple(map(int, np.unravel_index(max_flat, off.shape)))
    max_num = int(off[max_pair])
    max_offdiag = Fraction(max_num, q)
    if max_offdiag > HALF:
        raise VerificationError(
            f"max off-diagonal {max_offdiag} at {max_pair} exceeds 1/2"
        )
    tight_pairs = int(np.count_nonzero(np.triu(2 * H == q, k=1)))

    basis = choose_exact_basis(H, matrix, TARGET_DIM)
    B_int = sp.Matrix(H[np.ix_(basis, basis)].tolist())
    det_B = int(B_int.det())
    if det_B == 0:
        raise VerificationError("basis Gram determinant is zero")

    leading_minors: list[int] = []
    for k in range(1, TARGET_DIM + 1):
        minor = int(B_int[:k, :k].det())
        leading_minors.append(minor)
        if minor <= 0:
            raise VerificationError(
                f"basis Gram is not positive definite: leading minor {k}={minor}"
            )

    B_inv = B_int.inv()
    inv_den = lcm(term.q for term in B_inv)
    inv_values = [
        [int(term * inv_den) for term in B_inv.row(i)]
        for i in range(TARGET_DIM)
    ]
    inv_max = max(abs(v) for row in inv_values for v in row)
    inv_dtype = np.int64 if inv_max < 2**62 else object
    inv_num = np.array(inv_values, dtype=inv_dtype)

    A = H[:, basis]
    product, factorization_method = matrix_product_exact(A, inv_num, A.T)
    target = H.astype(object if product.dtype == object else np.int64) * inv_den
    if not np.array_equal(product, target):
        mismatch = np.argwhere(product != target)[0]
        i, j = map(int, mismatch)
        raise VerificationError(
            "rank-13 exact factorization failed at "
            f"({i},{j}): left={product[i,j]}, right={target[i,j]}"
        )

    unique_rows = int(np.unique(H, axis=0).shape[0])
    if unique_rows != n:
        raise VerificationError(f"Gram matrix has duplicate rows: {unique_rows} < {n}")

    antipodal_hits = np.count_nonzero(H == -q, axis=1)
    if np.any(antipodal_hits > 1):
        raise VerificationError("a vector has more than one exact antipode")
    antipodal_pairs = int(np.sum(antipodal_hits) // 2)

    values, counts = np.unique(H, return_counts=True)
    level_counts = {
        frac_text(Fraction(int(value), q)): int(count)
        for value, count in zip(values.tolist(), counts.tolist())
    }

    B_rat = B_int.applyfunc(lambda z: sp.Rational(z, q))
    L, d = exact_ldl(B_rat)
    AM = A.astype(object) @ inv_num.astype(object)
    coeffs: list[list[Fraction]] = []
    for i in range(n):
        row: list[Fraction] = []
        for j in range(TARGET_DIM):
            total = sp.Rational(0)
            for k in range(TARGET_DIM):
                total += sp.Rational(int(AM[i, k]), inv_den) * L[k, j]
            total = sp.cancel(total)
            row.append(Fraction(int(total.p), int(total.q)))
        coeffs.append(row)
    radicands = [Fraction(int(value.p), int(value.q)) for value in d]

    def coord_ip(i: int, j: int) -> Fraction:
        return sum(
            coeffs[i][k] * coeffs[j][k] * radicands[k]
            for k in range(TARGET_DIM)
        )

    for i in range(n):
        if coord_ip(i, i) != 1:
            raise VerificationError(f"coordinate norm check failed for vector {i}")
    deterministic_pairs = {(0, 1), (0, n - 1), max_pair}
    rng = np.random.default_rng(20260822)
    for _ in range(min(2000, n * 2)):
        i = int(rng.integers(0, n))
        j = int(rng.integers(0, n - 1))
        if j >= i:
            j += 1
        deterministic_pairs.add((min(i, j), max(i, j)))
    for i, j in deterministic_pairs:
        expected = Fraction(int(H[i, j]), q)
        if coord_ip(i, j) != expected:
            raise VerificationError(f"coordinate Gram check failed for pair {(i, j)}")

    return ExactGram(
        path=path,
        source_sha256=sha256_file(path),
        count=n,
        denominator=q,
        matrix=H,
        reconstruction_max_error=reconstruction_error,
        basis_indices=basis,
        inverse_numerator=inv_num,
        inverse_denominator=inv_den,
        leading_principal_minors=leading_minors,
        coordinate_radicands=radicands,
        coordinate_coefficients=coeffs,
        max_offdiag=max_offdiag,
        max_offdiag_pair=max_pair,
        tight_pairs=tight_pairs,
        antipodal_pairs=antipodal_pairs,
        unique_rows=unique_rows,
        level_counts=level_counts,
        exact_factorization_method=factorization_method,
    )


def numerical_hole_search(exact: ExactGram, starts: int = 64) -> dict[str, Any]:
    """Search for a single addable point; numerical output is not an exact certificate."""
    try:
        from scipy.optimize import linprog
    except Exception as exc:
        return {"available": False, "reason": f"scipy unavailable: {exc}"}

    n = exact.count
    r = TARGET_DIM
    B_int = sp.Matrix(exact.matrix[np.ix_(exact.basis_indices, exact.basis_indices)].tolist())
    B_rat = B_int.applyfunc(lambda z: sp.Rational(z, exact.denominator))
    L, d = exact_ldl(B_rat)
    Lf = np.array(L.tolist(), dtype=float)
    df = np.array([float(v) for v in d], dtype=float)
    A = exact.matrix[:, exact.basis_indices].astype(float)
    Binv = np.array(B_int.inv().tolist(), dtype=float)
    C = A @ Binv
    X = C @ Lf @ np.diag(np.sqrt(df))

    norm_error = float(np.max(np.abs(np.sum(X * X, axis=1) - 1.0)))
    sample_n = min(128, n)
    gram_sample = X[:sample_n] @ X[:sample_n].T
    target_sample = exact.matrix[:sample_n, :sample_n] / exact.denominator
    sample_error = float(np.max(np.abs(gram_sample - target_sample)))

    rng = np.random.default_rng(20260822)
    directions: list[np.ndarray] = []
    directions.extend(np.eye(r))
    directions.extend(-np.eye(r))
    directions.extend(rng.normal(size=(starts, r)))

    best_norm = -1.0
    best_point: np.ndarray | None = None
    best_iterations = 0
    successes = 0
    for raw in directions:
        direction = raw / np.linalg.norm(raw)
        last = None
        for iteration in range(40):
            result = linprog(
                c=-direction,
                A_ub=X,
                b_ub=np.full(n, 0.5),
                bounds=[(None, None)] * r,
                method="highs",
            )
            if not result.success:
                break
            successes += 1
            point = np.asarray(result.x, dtype=float)
            norm = float(np.linalg.norm(point))
            if norm > best_norm:
                best_norm = norm
                best_point = point.copy()
                best_iterations = iteration + 1
            if norm < 1e-15:
                break
            new_direction = point / norm
            if last is not None and np.linalg.norm(new_direction - last) < 1e-11:
                break
            last = direction
            direction = new_direction

    if best_point is None:
        return {
            "available": True,
            "success": False,
            "reason": "all LP solves failed",
            "coordinate_norm_error": norm_error,
            "sample_gram_error": sample_error,
        }

    unit = best_point / np.linalg.norm(best_point)
    inner = X @ unit
    conflicts = np.flatnonzero(inner > 0.5 + 1e-10)
    return {
        "available": True,
        "success": True,
        "starts": len(directions),
        "successful_lp_solves": successes,
        "best_polytope_norm": best_norm,
        "best_iterations": best_iterations,
        "single_point_addition_numerically_possible": bool(best_norm >= 1.0 - 1e-9),
        "normalized_candidate_max_inner": float(np.max(inner)),
        "normalized_candidate_conflicts": int(len(conflicts)),
        "normalized_candidate_conflict_indices": conflicts[:100].astype(int).tolist(),
        "coordinate_norm_error": norm_error,
        "sample_gram_error": sample_error,
        "note": "Numerical diagnostic only; it is not an exact kissing certificate.",
    }


def write_exact_config(exact: ExactGram, output: Path) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "dimension": TARGET_DIM,
        "count": exact.count,
        "unit": True,
        "coordinate_model": "x[i,j] = coefficient[i,j] * sqrt(radicand[j])",
        "coordinate_radicands": [frac_json(x) for x in exact.coordinate_radicands],
        "vectors": [
            [frac_json(value) for value in row]
            for row in exact.coordinate_coefficients
        ],
        "max_off_diagonal": frac_text(exact.max_offdiag),
        "max_off_diagonal_pair": list(exact.max_offdiag_pair),
        "exact_gram_denominator": exact.denominator,
        "basis_indices": exact.basis_indices,
        "source_file": exact.path.name,
        "source_sha256": exact.source_sha256,
        "method": (
            "Exact rational reconstruction of PackingStar Gram matrix; rank-13 "
            "factorization followed by rational LDL^T coordinates"
        ),
        "verification": {
            "diagonal_exactly_one": True,
            "all_off_diagonal_leq_half": True,
            "all_vectors_distinct": True,
            "positive_semidefinite": True,
            "rank": TARGET_DIM,
            "tight_pairs": exact.tight_pairs,
            "leading_principal_minors_of_scaled_basis": exact.leading_principal_minors,
            "factorization_method": exact.exact_factorization_method,
            "reconstruction_max_float_error": exact.reconstruction_max_error,
        },
        "beats_live_record": exact.count > LIVE_RECORD,
        "live_record": LIVE_RECORD,
    }
    output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def analyze_file(path: Path, out_dir: Path) -> dict[str, Any]:
    record: dict[str, Any] = {
        "source": str(path),
        "source_sha256": sha256_file(path),
        "status": "started",
    }
    try:
        matrix = np.load(path, allow_pickle=False)
        record["dtype"] = str(matrix.dtype)
        record["shape"] = list(matrix.shape)
        exact = exact_verify_gram(path, matrix)
        stem = path.stem.replace(" ", "_")
        config_path = out_dir / "configs" / f"{stem}_exact.json"
        write_exact_config(exact, config_path)
        hole = numerical_hole_search(exact)
        record.update(
            {
                "status": "verified_exactly",
                "dimension": TARGET_DIM,
                "count": exact.count,
                "exact_gram_denominator": exact.denominator,
                "reconstruction_max_float_error": exact.reconstruction_max_error,
                "rank": TARGET_DIM,
                "positive_semidefinite": True,
                "max_off_diagonal": frac_text(exact.max_offdiag),
                "max_off_diagonal_pair": list(exact.max_offdiag_pair),
                "tight_pairs": exact.tight_pairs,
                "antipodal_pairs": exact.antipodal_pairs,
                "unique_rows": exact.unique_rows,
                "basis_indices": exact.basis_indices,
                "leading_principal_minors": exact.leading_principal_minors,
                "factorization_method": exact.exact_factorization_method,
                "level_counts": exact.level_counts,
                "exact_config_path": str(config_path),
                "beats_live_record": exact.count > LIVE_RECORD,
                "live_record": LIVE_RECORD,
                "hole_search": hole,
            }
        )
    except Exception as exc:
        record.update(
            {
                "status": "failed",
                "error_type": type(exc).__name__,
                "error": str(exc),
            }
        )
    return record


def markdown_report(records: list[dict[str, Any]], source_dir: Path) -> str:
    verified = [r for r in records if r.get("status") == "verified_exactly"]
    lines = [
        "# PackingStar dimension-13 exact reconstruction",
        "",
        f"Generated {datetime.now(timezone.utc).isoformat()} from `{source_dir}`.",
        "",
        "This independently reconstructs exact rational Gram matrices from the public",
        "PackingStar `.npy` files. A successful row proves symmetry, unit diagonal,",
        "off-diagonal bound `<= 1/2`, positive semidefiniteness, rank 13, and distinctness",
        "using integer/rational arithmetic. It also emits explicit exact coordinates.",
        "",
        "| file | count | exact | max offdiag | rank | beats 1154 |",
        "| --- | ---: | :---: | ---: | ---: | :---: |",
    ]
    for r in records:
        lines.append(
            "| `{}` | {} | {} | {} | {} | {} |".format(
                Path(r["source"]).name,
                r.get("count", "—"),
                "yes" if r.get("status") == "verified_exactly" else "no",
                r.get("max_off_diagonal", "—"),
                r.get("rank", "—"),
                "yes" if r.get("beats_live_record") else "no",
            )
        )
    lines.extend(["", "## Results", ""])
    if not verified:
        lines.append("No matrix was exactly verified. See `analysis.json` for failures.")
    for r in verified:
        lines.extend(
            [
                f"### {Path(r['source']).name}",
                "",
                f"- Count: **{r['count']}**; current dimension-13 record: **{LIVE_RECORD}**.",
                f"- Exact max off-diagonal: **{r['max_off_diagonal']}**.",
                f"- Exact rank: **{r['rank']}**; positive semidefinite: **yes**.",
                f"- Tight unordered pairs: **{r['tight_pairs']}**.",
                f"- Exact antipodal pairs: **{r['antipodal_pairs']}**.",
                f"- Common Gram denominator: **{r['exact_gram_denominator']}**.",
                f"- Reconstruction error from stored floats: `{r['reconstruction_max_float_error']:.3e}`.",
                f"- Exact coordinate certificate: `{r['exact_config_path']}`.",
                "",
                "Raw exact-verifier result:",
                "",
                "```json",
                json.dumps(
                    {
                        key: r[key]
                        for key in (
                            "status",
                            "dimension",
                            "count",
                            "rank",
                            "positive_semidefinite",
                            "max_off_diagonal",
                            "max_off_diagonal_pair",
                            "tight_pairs",
                            "unique_rows",
                            "beats_live_record",
                            "live_record",
                            "factorization_method",
                        )
                    },
                    indent=2,
                ),
                "```",
                "",
                "Single-point hole diagnostic (numerical only):",
                "",
                "```json",
                json.dumps(r.get("hole_search", {}), indent=2),
                "```",
                "",
            ]
        )
    lines.extend(
        [
            "## Interpretation",
            "",
            "A verified 1,146-point rational configuration is a useful independent baseline,",
            "but it does **not** improve the live 1,154-point Zinoviev–Ericson record. Any",
            "record attempt starting from it needs a net gain of at least nine points.",
            "",
        ]
    )
    failures = [r for r in records if r.get("status") == "failed"]
    if failures:
        lines.extend(["## Failures", ""])
        for r in failures:
            lines.append(f"- `{Path(r['source']).name}`: {r.get('error_type')}: {r.get('error')}")
        lines.append("")
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-dir", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument(
        "--progress-log",
        type=Path,
        default=Path("kissing/dim13/progress.log"),
    )
    args = parser.parse_args(argv)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    candidates = sorted(
        path
        for path in args.input_dir.rglob("*.npy")
        if "13" in path.name and ("cos" in path.name.lower() or "gram" in path.name.lower())
    )
    if not candidates:
        candidates = sorted(args.input_dir.rglob("*.npy"))

    records: list[dict[str, Any]] = []
    for path in candidates:
        print(f"ANALYZE {path}", flush=True)
        record = analyze_file(path, args.output_dir)
        records.append(record)
        print(json.dumps(record, indent=2, default=str), flush=True)

    payload = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "live_record_dimension_13": LIVE_RECORD,
        "source_directory": str(args.input_dir),
        "records": records,
    }
    (args.output_dir / "analysis.json").write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    (args.output_dir / "REPORT.md").write_text(
        markdown_report(records, args.input_dir), encoding="utf-8"
    )

    verified = [r for r in records if r.get("status") == "verified_exactly"]
    best = max(verified, key=lambda r: int(r["count"]), default=None)
    (args.output_dir / "best.json").write_text(
        json.dumps(best or {"status": "no_exact_configuration"}, indent=2, sort_keys=True)
        + "\n",
        encoding="utf-8",
    )

    args.progress_log.parent.mkdir(parents=True, exist_ok=True)
    with args.progress_log.open("a", encoding="utf-8") as log:
        stamp = datetime.now(timezone.utc).isoformat()
        for r in records:
            method = "PackingStar rational Gram exact reconstruction"
            count = r.get("count", r.get("shape", ["?"])[0] if r.get("shape") else "?")
            outcome = "pass" if r.get("status") == "verified_exactly" else "fail"
            log.write(
                f"{stamp} method={method!r} dimension=13 count={count} result={outcome} "
                f"source={Path(r['source']).name}\n"
            )

    if not records:
        print("No .npy files found", file=sys.stderr)
        return 2
    return 0 if verified else 1


if __name__ == "__main__":
    raise SystemExit(main())
