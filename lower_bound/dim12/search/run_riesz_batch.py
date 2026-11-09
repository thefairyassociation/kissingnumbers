#!/usr/bin/env python3
"""Batch Riesz/hinge search for n=842,843,844 from the exact 840 core."""

from __future__ import annotations

import json
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from search.riesz import (
    adam_sphere,
    init_from_840_plus,
    max_offdiag,
    save_numeric,
    snap_to_rationals,
    HALF,
)

ROOT = Path(__file__).resolve().parent.parent


def one_run(kwargs):
    extra = kwargs["extra"]
    seed = kwargs["seed"]
    mode = kwargs["mode"]
    steps = kwargs["steps"]
    lr = kwargs["lr"]
    n = 840 + extra
    X = init_from_840_plus(extra, seed, mode)
    s_schedule = [4, 8, 16, 32, 64, 128, 256, 512, 1024, 2048]
    t0 = time.time()
    X = adam_sphere(
        X,
        steps=steps,
        s_schedule=s_schedule,
        lr=lr,
        seed=seed,
        report_every=max(steps // 10, 100),
    )
    mx = max_offdiag(X)
    elapsed = time.time() - t0
    rec = {
        "n": n,
        "extra": extra,
        "seed": seed,
        "mode": mode,
        "steps": steps,
        "maxIP_float": mx,
        "seconds": round(elapsed, 2),
        "below_half": bool(mx < HALF),
    }
    if mx < HALF:
        snapped, d, smx = snap_to_rationals(
            X, [2, 3, 4, 5, 6, 8, 12, 16, 24, 32, 48, 64, 96, 128, 192, 256, 384, 512]
        )
        rec["snap_denom"] = d
        rec["snap_maxIP"] = smx
        rec["snap_ok"] = bool(snapped is not None and smx <= HALF)
        out = ROOT / "configs" / f"riesz_{n}_seed{seed}_{mode}_SUCCESS.json"
        save_numeric(X, out, rec)
    else:
        rec["note"] = "float search, maxIP>=1/2, not a certificate"
    return rec, X, mx


def main():
    jobs = []
    for seed in range(4):
        jobs.append({"extra": 1, "seed": seed, "mode": "signs", "steps": 2200, "lr": 0.03})
        jobs.append({"extra": 1, "seed": seed, "mode": "hole", "steps": 2200, "lr": 0.03})
    for extra, nseed, steps in ((2, 6, 2800), (3, 4, 2400), (4, 3, 2000)):
        for mode in ("signs", "hole"):
            for seed in range(nseed):
                jobs.append(
                    {"extra": extra, "seed": seed, "mode": mode, "steps": steps, "lr": 0.025}
                )
    print(f"jobs={len(jobs)}", flush=True)
    results = []
    best_mx = 10.0
    best_rec = None
    # 4 workers
    with ProcessPoolExecutor(max_workers=4) as ex:
        futs = [ex.submit(one_run, j) for j in jobs]
        for fut in as_completed(futs):
            rec, X, mx = fut.result()
            results.append(rec)
            print("RESULT", rec, flush=True)
            if mx < best_mx:
                best_mx = mx
                best_rec = rec
                np.save(ROOT / "configs" / "riesz_best_X.npy", X)
    summary = {
        "n_jobs": len(jobs),
        "best_maxIP_float": best_mx,
        "best": best_rec,
        "n_below_half": sum(1 for r in results if r["below_half"]),
        "results": results,
    }
    (ROOT / "configs" / "riesz_batch_summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    with (ROOT / "progress.log").open("a") as f:
        f.write(
            f"{time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime())} "
            f"RIESZ_BATCH jobs={len(jobs)} best_maxIP={best_mx:.12f} "
            f"n_below_half={summary['n_below_half']}\n"
        )
    print("BEST", best_rec)


if __name__ == "__main__":
    main()
