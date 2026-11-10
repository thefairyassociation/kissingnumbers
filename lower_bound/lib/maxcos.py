"""The one place where "largest pairwise inner product of a coordinate file" is computed.

Every campaign verdict short of the exact certifier rests on this number, so
the search driver, the branch search and the collector all import it from
here instead of each keeping a copy.  NumPy only.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np


def max_cos(X: np.ndarray) -> float:
    """Largest off-diagonal inner product of the rows of X (rows assumed unit)."""
    G = X @ X.T
    np.fill_diagonal(G, -2)
    return float(G.max())


def indep_max(path: Path | str) -> float:
    """Normalise the rows of a coordinate file and return their largest inner product.

    Returns +inf for anything that cannot be a configuration (non-finite
    entries, a zero row, fewer than two rows), so that it can never be
    mistaken for a value below 1/2.
    """
    X = np.loadtxt(path, comments="#", ndmin=2)
    if X.shape[0] < 2 or not np.all(np.isfinite(X)):
        return float("inf")
    norms = np.linalg.norm(X, axis=1, keepdims=True)
    if not np.all(norms > 0):
        return float("inf")
    return max_cos(X / norms)
