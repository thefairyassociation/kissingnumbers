#!/usr/bin/env python3
"""Write a dim-12 seed: the classical 840 plus one extra unit vector.

Takhanov–Assylbekov–Yun seed the extra point as (±1)^12 / sqrt(12).  This
script picks the hypercube vertex with the smallest max inner product against
the 840 (equivalently, the least-bad start among those vectors).  --mode hole
puts the extra point in the deepest LP hole of the 840 instead.

usage: python3 kissing/lib/seed841.py [outfile] [--mode hypercube|hole|random]
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "kissing" / "dim12"))
from constructions.clebsch840 import construction_840, vectors_as_float  # noqa: E402


def best_hypercube_extra(X: np.ndarray) -> np.ndarray:
    """All 2^12 sign vectors, pick the one with smallest max |inner| vs X.

    Antipodes are equivalent on RP^11; we still scan all 4096 (cheap).
    """
    dim = X.shape[1]
    bits = np.arange(1 << dim)
    S = np.empty((1 << dim, dim), dtype=np.float64)
    for k in range(dim):
        S[:, k] = np.where((bits >> k) & 1, 1.0, -1.0)
    S /= np.sqrt(dim)
    ips = np.abs(S @ X.T).max(axis=1)
    return S[int(np.argmin(ips))]


def hole_extra(X: np.ndarray, seed: int = 0) -> np.ndarray:
    try:
        sys.path.insert(0, str(ROOT / "kissing" / "lib"))
        from holes import max_norm_over_P

        _, u = max_norm_over_P(X, starts=40, seed=seed)
        if u is not None:
            return u / np.linalg.norm(u)
    except Exception:
        pass
    rng = np.random.default_rng(seed)
    x = rng.standard_normal(X.shape[1])
    x /= np.linalg.norm(x)
    for _ in range(1200):
        ips = X @ x
        x = x - 0.04 * X[int(np.argmax(ips))]
        x /= np.linalg.norm(x)
    return x


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("outfile", nargs="?", default="kissing/logs/cl840_841.txt")
    ap.add_argument("--mode", choices=["hypercube", "hole", "random"], default="hypercube")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    X = vectors_as_float(construction_840())
    assert X.shape == (840, 12)
    G = X @ X.T
    np.fill_diagonal(G, -np.inf)
    print(f"840 max_offdiag={G.max():.12f}", flush=True)

    if args.mode == "hypercube":
        v = best_hypercube_extra(X)
    elif args.mode == "hole":
        v = hole_extra(X, seed=args.seed)
    else:
        rng = np.random.default_rng(args.seed)
        v = rng.standard_normal(12)
        v /= np.linalg.norm(v)

    mx = float(np.max(np.abs(X @ v)))
    print(f"extra mode={args.mode} max_inner_vs_840={mx:.12f}", flush=True)
    Y = np.vstack([X, v])
    path = Path(args.outfile)
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savetxt(path, Y, fmt="%.17g")
    print(f"wrote {Y.shape[0]} x {Y.shape[1]} -> {path}")


if __name__ == "__main__":
    main()
