#!/usr/bin/env python3
"""Prepare an authors-style decimal-Gram input for the polish path.

The authors' search writes a candidate Gram matrix with ``%.10f`` precision.
Their ``polish_841.py`` then symmetrises that text matrix, keeps its leading
12 eigendirections, reconstructs coordinates, and normalises each row.  This
small standalone preparer makes that handoff explicit before invoking the
existing C polish-only path.  It never claims the reconstructed coordinates
are an exact kissing configuration.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np


def normalized_coordinates(path: Path, n_points: int = 841, dimension: int = 12) -> np.ndarray:
    x = np.loadtxt(path)
    if x.shape != (n_points, dimension):
        raise ValueError(f"expected {(n_points, dimension)}, got {x.shape}")
    if not np.isfinite(x).all():
        raise ValueError("input coordinates contain non-finite values")
    norms = np.linalg.norm(x, axis=1)
    if not np.isfinite(norms).all() or np.any(norms <= 1e-15):
        raise ValueError("input coordinates contain an invalid row norm")
    return x / norms[:, None]


def max_inner(x: np.ndarray) -> float:
    gram = x @ x.T
    return float(gram[np.triu_indices(x.shape[0], 1)].max())


def max_gram_entry(gram: np.ndarray) -> float:
    return float(gram[np.triu_indices(gram.shape[0], 1)].max())


def decimal_gram_reconstruct(
    x: np.ndarray,
    gram_path: Path,
    coordinates_path: Path,
    decimals: int = 10,
    dimension: int = 12,
) -> tuple[float, float]:
    gram = x @ x.T
    gram_path.parent.mkdir(parents=True, exist_ok=True)
    coordinates_path.parent.mkdir(parents=True, exist_ok=True)
    np.savetxt(gram_path, gram, fmt=f"%.{decimals}f")

    # Reload the text representation exactly as polish_841.py does.
    rounded = np.loadtxt(gram_path)
    if rounded.shape != gram.shape or not np.isfinite(rounded).all():
        raise ValueError("decimal Gram serialization failed")
    rounded = 0.5 * (rounded + rounded.T)

    eigenvalues, eigenvectors = np.linalg.eigh(rounded)
    order = np.argsort(eigenvalues)[::-1][:dimension]
    leading = np.maximum(eigenvalues[order], 0.0)
    reconstructed = eigenvectors[:, order] * np.sqrt(leading)[None, :]
    row_norms = np.linalg.norm(reconstructed, axis=1)
    if not np.isfinite(row_norms).all() or np.any(row_norms <= 1e-15):
        raise ValueError("decimal Gram reconstruction produced an invalid row")
    reconstructed /= row_norms[:, None]
    if not np.isfinite(reconstructed).all():
        raise ValueError("decimal Gram reconstruction contains non-finite values")

    np.savetxt(coordinates_path, reconstructed, fmt="%.17g")
    # Re-read the coordinates that will be handed to C, then independently
    # recompute its normalised Gram maximum.
    roundtrip = normalized_coordinates(coordinates_path, x.shape[0], dimension)
    return max_gram_entry(rounded), max_inner(roundtrip)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("coordinates", type=Path, help="841x12 coordinate file")
    parser.add_argument("--gram-out", type=Path, required=True)
    parser.add_argument("--coordinates-out", type=Path, required=True)
    parser.add_argument("--decimals", type=int, default=10)
    args = parser.parse_args()
    if args.decimals < 1:
        parser.error("--decimals must be positive")

    x = normalized_coordinates(args.coordinates)
    input_max = max_inner(x)
    gram_max, reconstructed_max = decimal_gram_reconstruct(
        x, args.gram_out, args.coordinates_out, args.decimals
    )
    print(f"input_normalized_max_ip={input_max:.17g}")
    print(f"decimal_gram_max_ip={gram_max:.17g}")
    print(f"reconstructed_normalized_max_ip={reconstructed_max:.17g}")
    print(f"wrote_gram={args.gram_out}")
    print(f"wrote_coordinates={args.coordinates_out}")


if __name__ == "__main__":
    main()
