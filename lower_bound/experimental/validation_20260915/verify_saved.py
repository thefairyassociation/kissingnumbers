#!/usr/bin/env python3
"""Independently read back the saved calibration endpoints (NumPy only).

Recompute all floating-point pair products and give an exact rational
counterexample for each infeasible file. A counterexample disproves only
that particular configuration, never existence of a better configuration.
"""
from fractions import Fraction
import hashlib
import json
from pathlib import Path

import numpy as np


def verify(directory):
    record = json.loads((directory / "calibration.json").read_text())
    results = []
    for item in record["results"]:
        if item.get("returncode") != 0 or item["timed_out"]:
            raise ValueError("calibration did not finish successfully")
        path = directory / ("c.txt" if item["engine"] == "C" else "torch.txt")
        if hashlib.sha256(path.read_bytes()).hexdigest() != item["readback"]["sha256"]:
            raise ValueError(f"coordinate hash mismatch: {path}")
        rows = [line.split("#", 1)[0].split() for line in path.read_text().splitlines()]
        rows = [row for row in rows if row]
        x = np.array(rows, dtype=np.float64)
        if x.shape != (841, 12) or not np.isfinite(x).all():
            raise ValueError("invalid coordinate data")
        norms = np.linalg.norm(x, axis=1)
        if np.any(norms == 0) or not np.isfinite(norms).all():
            raise ValueError("invalid norms")
        x /= norms[:, None]
        gram = x @ x.T
        np.fill_diagonal(gram, -np.inf)
        pair = tuple(int(i) for i in np.unravel_index(gram.argmax(), gram.shape))
        maximum = float(gram[pair])
        if abs(maximum - item["readback"]["max_inner_product"]) > 1e-12:
            raise ValueError("reported maximum disagrees with independent readback")
        u, v = ([Fraction(s) for s in rows[i]] for i in pair)
        p = sum(a*b for a, b in zip(u, v))
        q = sum(a*a for a in u)
        r = sum(b*b for b in v)
        excess = 4*p*p - q*r
        violates = p > 0 and excess > 0
        if maximum > .5 and not violates:
            raise ValueError("floating-point failure was not confirmed exactly")
        if item["engine"] == "C":
            header = " ".join(item["header"])
            if "full_schedule=1" not in header or "search_updates=35000" not in header:
                raise ValueError("C full-schedule provenance missing")
        else:
            meta = item["metadata"]
            if meta["full_schedule"] is not True or meta["updates"] != 35000:
                raise ValueError("Torch full-schedule provenance missing")
        results.append(dict(engine=item["engine"], max_inner_product=maximum,
                            pair_zero_based=list(pair), exact_pair_violates_half=violates,
                            exact_positive_dot=str(p), exact_squared_excess=str(excess),
                            sha256=item["readback"]["sha256"]))
    return results


if __name__ == "__main__":
    print(json.dumps(verify(Path(__file__).with_name("results")), indent=2))
