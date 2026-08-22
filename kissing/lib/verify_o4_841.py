#!/usr/bin/env python3
"""Independent verifier/fingerprint for O(4) 841 CPU pilot outputs.

This file intentionally does not import ``o4_breadth.py``.  It reads written
coordinates back from disk, normalizes rows only for geometric diagnostics,
and recomputes all upper-triangle Gram entries from scratch.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
from pathlib import Path

import numpy as np


EXTRA_RE = re.compile(r"extra_mode=([^ ]+).*extra_index=(\d+)")


def _fingerprint(values: np.ndarray) -> dict[str, object]:
    k = min(4096, values.size)
    tail = np.partition(values, -k)[-k:]
    tail.sort()
    return {
        "max_ip": float(values.max()),
        "tail_size": int(k),
        "tail_first8": [float(x) for x in tail[-8:]],
        "tail_last8": [float(x) for x in tail[:8]],
        "contacts_ge_half_minus_1e-10": int(np.count_nonzero(tail >= 0.5 - 1e-10)),
        "contacts_ge_half_minus_1e-8": int(np.count_nonzero(tail >= 0.5 - 1e-8)),
        "tail_histogram": [
            int(x)
            for x in np.histogram(
                tail,
                bins=np.array([0.49, 0.495, 0.499, 0.4999, 0.5, 0.5001, 0.501, 0.505, 0.51]),
            )[0]
        ],
        "basin_fingerprint_sha256_rounded_1e-10": hashlib.sha256(
            np.round(tail, 10).tobytes()
        ).hexdigest()[:24],
    }


def _upper(X: np.ndarray) -> np.ndarray:
    gram = X @ X.T
    return np.ascontiguousarray(gram[np.triu_indices(X.shape[0], 1)])


def verify(path: Path) -> dict[str, object]:
    X = np.loadtxt(path, dtype=np.float64)
    if X.shape != (841, 12):
        raise ValueError(f"{path}: expected (841, 12), got {X.shape}")
    if not np.isfinite(X).all():
        raise ValueError(f"{path}: non-finite coordinates")
    norms = np.linalg.norm(X, axis=1)
    if np.any(norms <= 1e-15):
        raise ValueError(f"{path}: zero coordinate row")
    Y = X / norms[:, None]
    core_values = _upper(Y[:840])
    full_values = _upper(Y)
    # Take the comment block, not a fixed number of lines: candidates now carry
    # a solver-provenance line as well as the faithful metadata line.
    header = "\n".join(
        line for line in path.read_text().splitlines() if line.startswith("#")
    )
    extra_match = EXTRA_RE.search(header)
    return {
        "file": str(path),
        "rows": 841,
        "dim": 12,
        "max_norm_error_before_normalize": float(np.max(np.abs(norms - 1.0))),
        "core": _fingerprint(core_values),
        "full": _fingerprint(full_values),
        "faithful_header_extra_mode": extra_match.group(1) if extra_match else None,
        "faithful_header_extra_index": int(extra_match.group(2)) if extra_match else None,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("files", nargs="+", type=Path)
    parser.add_argument("--json", type=Path)
    args = parser.parse_args()
    payload = {"results": [verify(path) for path in args.files]}
    print(json.dumps(payload, indent=2, sort_keys=True))
    if args.json:
        args.json.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")


if __name__ == "__main__":
    main()
