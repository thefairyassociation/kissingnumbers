#!/usr/bin/env python3
"""One self-contained shard of a many-root search campaign.

This is the 2026-10-02 pipeline (docs/CALIBRATION_optimizer.md, "Multi-root
search") wrapped so that it can be run unattended for a fixed time budget, on
this machine or as one job of the `search campaign` GitHub Actions workflow:

  1. screen fresh seeds through the first two exponent stages with fastriesz
     and keep the ones whose core re-crystallises (many near-antipodal pairs
     after the first stage) -- about 6 s per seed in dimension 12;
  2. branch every surviving root with branch_search.py (which minimax-polishes
     good finals, recomputes every candidate independently and hands anything
     below 1/2 to certify_float.py);
  3. repeat until the time budget is spent, then write summary.json.

Nothing here is evidence by itself.  A configuration only counts once
certify_float.py (exact integer arithmetic) has passed it; the summary records
whether that happened.

usage:
    python3 lower_bound/lib/campaign.py OUTDIR --n 12 --N 841 --init INIT.txt
        --seed-start 5000000 --hours 0.2 --workers 4
        [--extra file|hypercube|gauss] [--anti 300] [--screen-batch 10]
        [--branch-minutes-per-root 4] [--jit 0.005] [--scale 3.4285714]

INIT has N rows (--extra file) or N-1 rows (--extra hypercube / gauss, where
every seed draws its own extra point).
"""
from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
FASTRIESZ = HERE / "fastriesz"
BRANCH = HERE / "branch_search.py"
CERTIFY = HERE / "certify_float.py"
ENV1 = dict(os.environ, OPENBLAS_NUM_THREADS="1", OMP_NUM_THREADS="1")
sys.path.insert(0, str(HERE))
from maxcos import indep_max  # noqa: E402


def parse(line: str) -> dict:
    return dict(kv.split("=", 1) for kv in line.split() if "=" in kv)


def screen(a, roots_dir: Path, lo: int, hi: int) -> list[dict]:
    """Run seeds lo..hi-1 through the first two stages; returns one record per seed."""
    cmd = [str(FASTRIESZ), str(a.n), str(a.N), str(a.init), str(roots_dir),
           "--extra", a.extra, "--adam", a.adam, "--scale", repr(a.scale),
           "--jit", repr(a.jit), "--start-stage", str(a.start_stage), "--stages", str(a.start_stage + 2),
           "--screen-anti", f"{a.start_stage + 1}:{a.anti}",
           "--save-stage", str(a.start_stage + 2), "--keep", "0", "--seeds", str(lo), str(hi)]
    r = subprocess.run(cmd, capture_output=True, text=True, env=ENV1)
    out = []
    for line in r.stdout.splitlines():
        if line.startswith("seed="):
            d = parse(line)
            out.append({"seed": int(d["seed"]), "status": d["status"],
                        "anti": int(d.get("anti", -1)), "final": float(d["final"])})
    if r.returncode != 0:
        print(f"fastriesz exited {r.returncode}: {r.stderr.strip()[-300:]}", file=sys.stderr, flush=True)
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("outdir", type=Path)
    ap.add_argument("--n", type=int, required=True)
    ap.add_argument("--N", type=int, required=True)
    ap.add_argument("--init", type=Path, required=True)
    ap.add_argument("--init-label", default=None,
                    help="name recorded for the starting configuration (e.g. d12-840); default: the file name")
    ap.add_argument("--seed-start", type=int, required=True)
    ap.add_argument("--hours", type=float, default=0.2)
    ap.add_argument("--workers", type=int, default=os.cpu_count() or 1)
    ap.add_argument("--extra", default="file", choices=["file", "hypercube", "gauss"])
    ap.add_argument("--adam", default="manifold")
    ap.add_argument("--scale", type=float, default=3.4285714)
    ap.add_argument("--jit", type=float, default=0.005)
    ap.add_argument("--anti", type=int, default=300,
                    help="near-antipodal pairs required after stage 1 for a seed to count as a root")
    ap.add_argument("--start-stage", type=int, default=0,
                    help="schedule stage at which fresh seeds enter (0 = exponent 8, 1 = 16, 2 = 32).  The "
                         "antipodal screen is applied after that stage and roots are saved one stage later.")
    ap.add_argument("--screen-batch", type=int, default=10, help="seeds per worker per screening round")
    ap.add_argument("--branch-minutes-per-root", type=float, default=4.0,
                    help="CPU-minutes of branching spent on each new root")
    ap.add_argument("--keep-below", type=float, default=0.5003,
                    help="keep the coordinates of each branch round's best candidate if it is below this "
                         "(every candidate below --deepen-below is kept as well)")
    ap.add_argument("--deepen-below", type=float, default=0.50004,
                    help="keep every candidate below this and branch again around each (0 disables)")
    ap.add_argument("--deepen-minutes", type=float, default=12.0,
                    help="wall-clock minutes of extra branching around each such candidate")
    ap.add_argument("--deepen-stage", type=int, default=4,
                    help="schedule stage at which a deepened candidate re-enters (4 = exponent 128)")
    a = ap.parse_args()

    for tool in (FASTRIESZ, HERE / "hingepol"):
        if not tool.exists():
            subprocess.run(["make", "-C", str(HERE), tool.name], check=True)
    out = a.outdir
    roots_dir = out / "roots"
    kept_dir = out / "kept"
    for d in (roots_dir, kept_dir):
        d.mkdir(parents=True, exist_ok=True)
    a.init = a.init.resolve()

    t0 = time.time()
    deadline = t0 + 3600 * a.hours
    next_seed = a.seed_start
    screened = 0
    anti_hist: dict[int, int] = {}
    roots: list[dict] = []
    rounds: list[dict] = []
    best = {"max": 9.0, "file": None}
    hits: list[dict] = []
    rnd = 0

    branch_failures: list[dict] = []

    def remaining() -> float:
        return deadline - time.time()

    def write_summary(final: bool) -> dict:
        """Write summary.json.  Called after every round, so that a shard killed at
        the job's time limit still leaves a summary next to whatever it kept."""
        summary = {
            "n": a.n, "N": a.N, "init": a.init_label or a.init.name, "extra": a.extra, "adam": a.adam,
            "scale": a.scale, "jit": a.jit, "start_stage": a.start_stage, "anti_threshold": a.anti,
            "seed_start": a.seed_start, "seed_end": next_seed,
            "workers": a.workers, "hours": a.hours, "elapsed_seconds": round(time.time() - t0, 1),
            "complete": final,
            "screened": screened, "roots": len(roots),
            "root_rate": (len(roots) / screened) if screened else None,
            "anti_histogram": {str(k): v for k, v in sorted(anti_hist.items())},
            "root_seeds": [(r["seed"], r["anti"]) for r in roots],
            "rounds": rounds, "best": best, "hits": hits, "branch_failures": branch_failures,
            "certified_below_half": any(h["certified"] for h in hits),
        }
        tmp = out / "summary.json.tmp"
        tmp.write_text(json.dumps(summary, indent=1) + "\n")
        os.replace(tmp, out / "summary.json")
        return summary

    def harvest(bdir: Path, rnd: int):
        """Read one branch_search directory: (best max, finished branches, promising candidates)."""
        nonlocal best
        round_best, finished, promising = 9.0, 0, []
        logf = bdir / "branch.log"
        if logf.exists():
            for line in logf.read_text().splitlines():
                rec = json.loads(line)
                if "indep_max" in rec:
                    finished += 1
                    round_best = min(round_best, rec["indep_max"])
        # keep the coordinates of near-misses and of every certified hit
        for cand in sorted(bdir.glob("cand_*.txt")):
            m = indep_max(cand)
            cert = cand.with_suffix(".cert.json")
            if m < 0.5:
                # certify again here so the verdict does not depend on branch_search
                c = subprocess.run([sys.executable, str(CERTIFY), str(cand), "--json-out", str(cert)],
                                   capture_output=True, text=True)
                ok = c.returncode == 0 and "CERTIFIED" in c.stdout
                hits.append({"file": cand.name, "round": rnd, "max": m, "certified": ok,
                             "certify_tail": c.stdout.strip().splitlines()[-1:]})
                shutil.copy(cand, kept_dir / cand.name)
                if cert.exists():
                    shutil.copy(cert, kept_dir / cert.name)
                print(f"*** candidate below 1/2: {cand.name} max={m!r} certified={ok}", flush=True)
            elif m < a.keep_below:
                shutil.copy(cand, kept_dir / cand.name)
                if m < a.deepen_below:
                    promising.append((cand.name, m))
            if m < best["max"]:
                best = {"max": m, "file": cand.name, "round": rnd}
        return round_best, finished, promising

    # A full branch costs about 30 s (dim 12) plus the polish; stop screening
    # when there would be no time left to follow a root up.
    while remaining() > 120:
        rnd += 1
        ranges = [(next_seed + i * a.screen_batch, next_seed + (i + 1) * a.screen_batch)
                  for i in range(a.workers)]
        next_seed += a.workers * a.screen_batch
        with ThreadPoolExecutor(max_workers=a.workers) as ex:
            results = list(ex.map(lambda r: screen(a, roots_dir, *r), ranges))
        if not any(results):
            # fastriesz produced no seed lines at all: a broken build or bad arguments.
            print("screening produced no results; aborting this shard", file=sys.stderr)
            write_summary(final=False)
            return 1
        new_roots = []
        for recs in results:
            for rec in recs:
                screened += 1
                b = min(rec["anti"] // 50 * 50, 600) if rec["anti"] >= 0 else -1
                anti_hist[b] = anti_hist.get(b, 0) + 1
                state = roots_dir / f"state_n{a.n}_N{a.N}_s{rec['seed']}_k{a.start_stage + 2}.txt"
                if rec["status"] == "done" and state.exists():
                    rec["state"] = state.name
                    new_roots.append(rec)
        print(f"[{time.time() - t0:7.0f}s] round {rnd}: screened {screened} seeds, "
              f"{len(new_roots)} new roots {[(r['seed'], r['anti']) for r in new_roots]}", flush=True)
        if not new_roots:
            continue
        if remaining() < 90:
            # No time left to branch them, but they were found: count them.
            for r_ in new_roots:
                r_["round"] = rnd
                r_["branched"] = False
                roots.append(r_)
            write_summary(final=False)
            continue
        hours = min(len(new_roots) * a.branch_minutes_per_root / a.workers / 60.0,
                    max(remaining() - 60, 30) / 3600.0)
        bdir = out / f"branch_r{rnd:04d}"
        states = [str(roots_dir / r["state"]) for r in new_roots]
        cmd = [sys.executable, str(BRANCH), states[0], str(bdir), "--n", str(a.n), "--N", str(a.N),
               "--root-stage", str(a.start_stage + 2), "--workers", str(a.workers), "--hours", repr(hours),
               "--scale", repr(a.scale), "--adam", a.adam, "--seed", str(a.seed_start // 1000 + rnd),
               "--root-prob", "0.6", "--keep-below", repr(a.deepen_below)]
        for s in states[1:]:
            cmd += ["--extra-root", s]
        r = subprocess.run(cmd, capture_output=True, text=True)
        tail = r.stdout.strip().splitlines()[-1:] or [""]
        if r.returncode != 0:
            branch_failures.append({"round": rnd, "returncode": r.returncode, "stderr": r.stderr.strip()[-300:]})
            print(f"branch_search exited {r.returncode} in round {rnd}: {r.stderr.strip()[-300:]}",
                  file=sys.stderr, flush=True)
        round_best, finished, promising = harvest(bdir, rnd)
        # Rare good basins are worth more than fresh roots: branch again around them.
        deepened = []
        # best first, and at most three per round so that one lucky round cannot eat the shard
        for cand_name, m in sorted(promising, key=lambda t: t[1])[:3]:
            if a.deepen_below <= 0 or remaining() < 300:
                break
            ddir = out / f"deepen_r{rnd:04d}_{len(deepened)}"
            dh = min(a.deepen_minutes * 60, max(remaining() - 120, 60)) / 3600.0
            dcmd = [sys.executable, str(BRANCH), str(kept_dir / cand_name), str(ddir), "--n", str(a.n),
                    "--N", str(a.N), "--root-stage", str(a.deepen_stage), "--save-stage", str(a.deepen_stage + 2),
                    "--screen-stage", "9", "--workers", str(a.workers), "--hours", repr(dh),
                    "--scale", repr(a.scale), "--adam", a.adam, "--sigmas", "1e-3,3e-3,1e-2",
                    "--keep-below", repr(a.deepen_below),
                    "--seed", str(a.seed_start // 1000 + 500 + rnd * 10 + len(deepened))]
            try:
                subprocess.run(dcmd, capture_output=True, text=True)
                dbest, dfin, _ = harvest(ddir, rnd)
            except Exception as exc:      # deepening is optional; never lose the shard over it
                print(f"deepening {cand_name} failed: {exc}", file=sys.stderr, flush=True)
                break
            deepened.append({"from": cand_name, "from_max": m, "finished_branches": dfin,
                             "best": dbest if dbest < 9 else None})
            shutil.rmtree(ddir, ignore_errors=True)
            print(f"[{time.time() - t0:7.0f}s] round {rnd}: deepened {cand_name} ({m:.9f}): "
                  f"{dfin} branches, best {dbest:.9f}", flush=True)
            round_best = min(round_best, dbest)
        for r_ in new_roots:
            r_["round"] = rnd
            roots.append(r_)
        rounds.append({"round": rnd, "roots": [r_["seed"] for r_ in new_roots], "finished_branches": finished,
                       "best": round_best if round_best < 9 else None, "deepened": deepened,
                       "branch_tail": tail[0][:200]})
        shutil.rmtree(bdir, ignore_errors=True)   # state files are large; summaries are kept
        print(f"[{time.time() - t0:7.0f}s] round {rnd}: {finished} finished branches, "
              f"round best {round_best:.9f}, campaign best {best['max']:.9f}", flush=True)
        write_summary(final=False)

    summary = write_summary(final=True)
    shutil.rmtree(roots_dir, ignore_errors=True)
    print("SUMMARY " + json.dumps({k: summary[k] for k in
                                   ("n", "N", "seed_start", "screened", "roots", "best", "certified_below_half")}))
    return 0


if __name__ == "__main__":
    sys.exit(main())
