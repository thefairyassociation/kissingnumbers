#!/usr/bin/env python3
"""Classify hypercube extras by their signed inner-product fingerprint.

The published 841-point search draws a random (±1)^12/sqrt(12) extra.  Absolute
compatibility with the classical 840 does not distinguish these vertices, but
their signed fingerprints do.  This utility writes one deterministic seed per
fingerprint class so a CPU multi-start can cover the genuinely different starts
without pretending that thousands of tied vertices are distinct.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "kissing" / "dim12"))
from constructions.clebsch840 import construction_840, vectors_as_float  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("outdir", type=Path)
    parser.add_argument("--decimals", type=int, default=12)
    args = parser.parse_args()

    base = vectors_as_float(construction_840())
    dim = base.shape[1]
    bits = np.arange(1 << dim, dtype=np.uint16)
    signs = np.where(((bits[:, None] >> np.arange(dim)) & 1) != 0, 1.0, -1.0)
    signs /= np.sqrt(dim)

    inner = np.round(signs @ base.T, args.decimals)
    values = np.unique(inner)
    counts = np.stack([(inner == value).sum(axis=1) for value in values], axis=1)
    fingerprints, inverse, sizes = np.unique(
        counts, axis=0, return_inverse=True, return_counts=True
    )

    args.outdir.mkdir(parents=True, exist_ok=True)
    print(f"classes={len(fingerprints)}")
    for class_id, size in enumerate(sizes):
        representative = int(np.flatnonzero(inverse == class_id)[0])
        seed = np.vstack([base, signs[representative]])
        output = args.outdir / f"cl840_841_hypercube_class{class_id}.txt"
        np.savetxt(output, seed, fmt="%.17g")
        print(
            f"class={class_id} size={int(size)} representative={representative} "
            f"min_inner={inner[representative].min():.12f} "
            f"max_inner={inner[representative].max():.12f} output={output}"
        )


if __name__ == "__main__":
    main()
