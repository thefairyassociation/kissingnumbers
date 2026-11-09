#!/usr/bin/env python3
"""Estimate the largest extra unit vector that fits the frozen 840 core.

Repeated subgradient descent on t(x) = max_i <x, v_i> over the sphere.
If min t <= 1/2, an extra exact-ish point may exist without moving the core.
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from constructions.clebsch840 import construction_840, vectors_as_float

ROOT = Path(__file__).resolve().parent.parent


def minimize_max_ip(V: np.ndarray, n_restarts: int = 200, iters: int = 800, seed: int = 0):
    rng = np.random.default_rng(seed)
    best_t = 1.0
    best_x = None
    ts = []
    for r in range(n_restarts):
        x = rng.normal(size=12)
        x /= np.linalg.norm(x)
        lr = 0.08
        t = None
        for it in range(iters):
            ips = V @ x
            i = int(np.argmax(ips))
            t = float(ips[i])
            x = x - lr * V[i]
            x /= np.linalg.norm(x)
            if it == iters // 2:
                lr *= 0.3
        ts.append(t)
        if t < best_t:
            best_t = t
            best_x = x.copy()
        if (r + 1) % 40 == 0:
            print(f"  restart {r+1}/{n_restarts} best_t={best_t:.6f}", flush=True)
    return best_t, best_x, ts


def main() -> None:
    V = vectors_as_float(construction_840())
    t0 = time.time()
    best_t, best_x, ts = minimize_max_ip(V)
    report = {
        "core": 840,
        "best_max_ip_to_core": best_t,
        "room_for_unmoved_extra": bool(best_t <= 0.5 + 1e-12),
        "best_x_float": best_x.tolist() if best_x is not None else None,
        "t_min": float(min(ts)),
        "t_median": float(np.median(ts)),
        "seconds": round(time.time() - t0, 2),
        "note": "If best_max_ip_to_core > 1/2, the rigid 840 has no extra unit vector.",
    }
    path = ROOT / "configs" / "hole_search_840.json"
    path.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({k: v for k, v in report.items() if k != "best_x_float"}, indent=2))
    with (ROOT / "progress.log").open("a") as f:
        f.write(
            f"{time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime())} "
            f"HOLE_SEARCH_840 best_t={best_t:.8f} room={report['room_for_unmoved_extra']}\n"
        )


if __name__ == "__main__":
    main()
