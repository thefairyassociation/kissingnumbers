#!/usr/bin/env python3
"""Structurally valid O(4)-deformed 840-core seed generator.

This module is deliberately located beside ``seed841.py`` so its imports work
from a checkout of the repository, not from an agent-specific scratch tree.
It implements Theorem 2's near-identity O(4) family and audits each 840 core
before appending a uniform hypercube extra.  The emitted text format is the
one consumed by ``kissing/lib/riesz2``.

The default ``uniform-random`` extra mode is source-faithful.  The
``deterministic-best`` mode is retained only as an explicitly labelled legacy
diagnostic and must not be used to claim reproduction of the authors' search.
"""

from __future__ import annotations

import argparse
import hashlib
import itertools
import json
import math
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "kissing" / "dim12"))
from constructions.clebsch840 import construction_840, vectors_as_float  # noqa: E402

THEOREM_ROW_L1 = 3.0 / (2.0 * math.sqrt(2.0))
DEFAULT_SAFETY_ROW_L1 = 1.055
CORE_TOL = 2e-12



def _expm_skew(a: np.ndarray) -> np.ndarray:
    """Matrix exponential of a real skew-symmetric matrix, NumPy only.

    ``1j * a`` is Hermitian for real skew-symmetric ``a``, so ``eigh`` gives an
    exact unitary diagonalisation and ``exp`` is applied to the eigenvalues.
    This avoids a SciPy dependency in the one new module that needed it.
    """
    hermitian = 1j * a
    values, vectors = np.linalg.eigh(hermitian)
    exponential = (vectors * np.exp(-1j * values)) @ vectors.conj().T
    return np.real(exponential)


def expm(a: np.ndarray) -> np.ndarray:
    return _expm_skew(a)


def _clebsch48() -> np.ndarray:
    rows: list[np.ndarray] = []
    for eps in itertools.product((-1.0, 1.0), repeat=5):
        if np.prod(eps) != 1:
            continue
        e = np.asarray(eps, dtype=np.float64)
        x = np.r_[0.0, math.sqrt(2.0) * e[:4] / 3.0, e[4] / 3.0]
        b = np.r_[0.5, -math.sqrt(2.0) * e[:4] / 4.0, -0.5 * e[4]]
        nb = np.r_[-0.5, -math.sqrt(2.0) * e[:4] / 4.0, -0.5 * e[4]]
        rows.extend((x, b, nb))
    return np.asarray(rows, dtype=np.float64)


def _signed_orts() -> np.ndarray:
    rows = []
    for i in range(6):
        # Match constructions.clebsch840.signed_orts_6() ordering exactly.
        for sign in (1.0, -1.0):
            row = np.zeros(6, dtype=np.float64)
            row[i] = sign
            rows.append(row)
    return np.asarray(rows)


def _bridges() -> np.ndarray:
    factorization = [
        [(0, 1), (2, 3), (4, 5)],
        [(0, 2), (1, 4), (3, 5)],
        [(0, 3), (1, 5), (2, 4)],
        [(0, 4), (1, 3), (2, 5)],
        [(0, 5), (1, 2), (3, 4)],
    ]
    rows = []
    for matching in factorization:
        for a, b in matching:
            for c, d in matching:
                for signs in itertools.product((-1.0, 1.0), repeat=4):
                    row = np.zeros(12, dtype=np.float64)
                    row[[a, b, 6 + c, 6 + d]] = 0.5 * np.asarray(signs)
                    rows.append(row)
    return np.asarray(rows)


def canonical_core() -> np.ndarray:
    """Return the checked-in exact 840 construction as float64."""
    return vectors_as_float(construction_840())


def deformed_core(A_left: np.ndarray, A_right: np.ndarray) -> np.ndarray:
    """Apply Theorem 2's Q_A transforms to the two 48-systems."""
    c48 = _clebsch48()
    orts = _signed_orts()

    def block(A: np.ndarray) -> np.ndarray:
        q = np.eye(5)
        q[:4, :4] = A
        transformed = c48.copy()
        transformed[:, 1:] = transformed[:, 1:] @ q.T
        return np.vstack([transformed, orts])

    left = np.pad(block(A_left), ((0, 0), (0, 6)))
    right = np.pad(block(A_right), ((0, 0), (6, 0)))
    return np.vstack([left, right, _bridges()])


def _draw_o4(
    rng: np.random.Generator, scale: float, safety_row_l1: float
) -> tuple[np.ndarray, int]:
    """Draw a near-identity orthogonal 4x4 factor.

    ``expm`` of a skew-symmetric matrix always has determinant +1, so this
    samples SO(4) and never the reflection component of O(4).  That is the
    intended coverage: Theorem 2's family is a *near-identity* deformation and
    every reflection is at distance ~2 from the identity, well outside the
    row-L1 safety bound below.  The rejection loop also truncates the
    distribution, so draws are not Haar on SO(4); they only need to be a
    spread of admissible deformations, not a uniform sample.
    """
    for attempt in range(1, 100_001):
        skew = rng.normal(size=(4, 4))
        skew -= skew.T
        matrix = expm(scale * skew)
        if float(np.max(np.sum(np.abs(matrix), axis=1))) <= safety_row_l1:
            return matrix, attempt
    raise RuntimeError("O(4) rejection sampler exhausted")


def hypercube_vectors() -> np.ndarray:
    bits = np.arange(1 << 12, dtype=np.uint16)
    signs = np.empty((1 << 12, 12), dtype=np.float64)
    for k in range(12):
        signs[:, k] = np.where((bits >> k) & 1, 1.0, -1.0)
    return signs / math.sqrt(12.0)


def choose_extra(
    core: np.ndarray, rng: np.random.Generator, mode: str
) -> tuple[np.ndarray, dict[str, object]]:
    """Choose and record the extra point.

    ``uniform-random`` is the source-faithful mode.  ``deterministic-best``
    exists only for historical diagnostic comparisons.
    """
    all_signs = hypercube_vectors()
    if mode == "uniform-random":
        index = int(rng.integers(all_signs.shape[0]))
        extra = all_signs[index]
        label = "source-faithful uniform random hypercube"
    elif mode == "deterministic-best":
        scores = np.max(np.abs(all_signs @ core.T), axis=1)
        index = int(np.argmin(scores))
        extra = all_signs[index]
        label = "legacy diagnostic deterministic-best; not source-faithful"
    else:
        raise ValueError(f"unknown extra mode: {mode}")
    return extra, {
        "extra_mode": mode,
        "extra_mode_label": label,
        "extra_hypercube_index": index,
        "extra_max_abs_ip_vs_core": float(np.max(np.abs(core @ extra))),
    }


def _upper_values(X: np.ndarray) -> np.ndarray:
    gram = X @ X.T
    return np.ascontiguousarray(gram[np.triu_indices(X.shape[0], 1)])


def fingerprint(X: np.ndarray, tail_size: int = 4096) -> dict[str, object]:
    values = _upper_values(X)
    k = min(tail_size, values.size)
    tail = np.partition(values, -k)[-k:]
    tail.sort()
    return {
        "max_ip": float(values.max()),
        "tail_size": int(k),
        "tail_contacts_1e-10": int(np.count_nonzero(tail >= 0.5 - 1e-10)),
        "tail_contacts_1e-8": int(np.count_nonzero(tail >= 0.5 - 1e-8)),
        "tail_sha256_rounded_1e-10": hashlib.sha256(
            np.round(tail, 10).tobytes()
        ).hexdigest()[:24],
    }


def audit_core(core: np.ndarray, label: str) -> dict[str, object]:
    if core.shape != (840, 12):
        raise ValueError(f"{label}: expected (840, 12), got {core.shape}")
    if not np.isfinite(core).all():
        raise ValueError(f"{label}: non-finite core")
    norms = np.linalg.norm(core, axis=1)
    if np.any(norms <= 1e-15):
        raise ValueError(f"{label}: zero core row")
    fp = fingerprint(core)
    if fp["max_ip"] > 0.5 + CORE_TOL:
        raise ValueError(f"{label}: core max-IP={fp['max_ip']:.17g} exceeds tolerance")
    return {
        "label": label,
        "shape": [840, 12],
        "max_norm_error": float(np.max(np.abs(norms - 1.0))),
        "fingerprint": fp,
    }


def make_seed(
    arm: str,
    seed: int,
    scale: float = 0.006,
    safety_row_l1: float = DEFAULT_SAFETY_ROW_L1,
    extra_mode: str = "uniform-random",
) -> tuple[np.ndarray, dict[str, object]]:
    if safety_row_l1 > THEOREM_ROW_L1:
        raise ValueError("safety row-l1 limit exceeds theorem bound")
    if scale < 0:
        raise ValueError("scale must be nonnegative")
    rng = np.random.default_rng(seed)
    draws = {"left": 0, "right": 0}
    row_l1 = {"left": 1.0, "right": 1.0}
    if arm == "canonical":
        core = canonical_core()
    elif arm == "deformed":
        left, draws["left"] = _draw_o4(rng, scale, safety_row_l1)
        right, draws["right"] = _draw_o4(rng, scale, safety_row_l1)
        core = deformed_core(left, right)
        row_l1 = {
            "left": float(np.max(np.sum(np.abs(left), axis=1))),
            "right": float(np.max(np.sum(np.abs(right), axis=1))),
        }
    else:
        raise ValueError(f"unknown arm: {arm}")

    core_audit = audit_core(core, f"{arm}/seed{seed}")
    extra, extra_meta = choose_extra(core, rng, extra_mode)
    seed841 = np.vstack([core, extra])
    norms = np.linalg.norm(seed841, axis=1)
    if seed841.shape != (841, 12) or not np.isfinite(seed841).all():
        raise ValueError("invalid 841 seed")
    metadata: dict[str, object] = {
        "arm": arm,
        "seed": seed,
        "scale": scale,
        "theorem_row_l1_bound": THEOREM_ROW_L1,
        "safety_row_l1": safety_row_l1,
        "row_l1_max": row_l1,
        "draw_attempts": draws,
        "core_audit_before_extra": core_audit,
        "seed841_max_norm_error": float(np.max(np.abs(norms - 1.0))),
        "seed841_fingerprint": fingerprint(seed841),
        **extra_meta,
        "coordinate_format": "plain text, 841 rows x 12 float64 values; first 840 rows are audited core",
    }
    return seed841, metadata


def write_seed(path: Path, X: np.ndarray, metadata: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savetxt(path, X, fmt="%.17g")
    path.with_suffix(path.suffix + ".json").write_text(
        json.dumps({"seed_file": str(path), **metadata}, indent=2, sort_keys=True)
        + "\n"
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--outdir", type=Path, default=ROOT / "kissing" / "logs" / "o4-seeds")
    parser.add_argument("--arm", choices=["canonical", "deformed", "both"], default="both")
    parser.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2, 3])
    parser.add_argument("--scale", type=float, default=0.006)
    parser.add_argument("--safety-row-l1", type=float, default=DEFAULT_SAFETY_ROW_L1)
    parser.add_argument(
        "--extra-mode",
        choices=["uniform-random", "deterministic-best"],
        default="uniform-random",
        help="uniform-random is source-faithful; deterministic-best is legacy diagnostic only",
    )
    args = parser.parse_args()
    arms = ["canonical", "deformed"] if args.arm == "both" else [args.arm]
    records = []
    for arm in arms:
        for seed in args.seeds:
            X, metadata = make_seed(
                arm, seed, args.scale, args.safety_row_l1, args.extra_mode
            )
            path = args.outdir / f"seed841_{arm}_{seed}.txt"
            write_seed(path, X, metadata)
            records.append({"seed_file": str(path), **metadata})
            print(json.dumps(records[-1], sort_keys=True), flush=True)
    manifest = args.outdir / "manifest.json"
    manifest.write_text(json.dumps({"records": records}, indent=2, sort_keys=True) + "\n")
    print(f"wrote {manifest}")


if __name__ == "__main__":
    main()
