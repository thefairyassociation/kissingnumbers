#!/usr/bin/env python3
"""Go-with-the-winners branching search on top of fastriesz.

A Riesz continuation is decided early: a state saved at a low exponent
(s = 16 .. 64) that lies in a good basin keeps landing in that basin when it
is re-run with small perturbations, while fresh starts reach it rarely (about
1 in 48 for the dimension-12 N = 841 calibration).  The final value still
varies from branch to branch.  This driver exploits both facts:

  * a pool of parent states, each a configuration saved after stage K;
  * each job perturbs a parent (Gaussian jitter of size sigma), re-runs the
    schedule from stage K with fresh Adam moments, saves its own state after
    stage K' (>= K) and is screened after stage --screen-stage against the
    current pool so that clearly worse branches stop early;
  * finished branches enter the pool as parents (at stage K'), ranked by
    their final maximum inner product; the pool keeps the best --pool.

Every finished candidate is recomputed independently (NumPy) from its file;
anything below 1/2 is handed to certify_float.py.  The optimiser's own numbers
are never used as evidence.

usage:
    python3 kissing/lib/branch_search.py ROOT_STATE.txt OUTDIR --n 12 --N 841
        --root-stage 2 [--save-stage 4] [--workers 4] [--hours 1]
        [--sigmas 3e-4,1e-3,3e-3] [--scale 3.4285714] [--adam manifold]
"""
from __future__ import annotations

import argparse
import heapq
import json
import os
import random
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import minimax_polish  # noqa: E402
FASTRIESZ = HERE / "fastriesz"
CERTIFY = HERE / "certify_float.py"


def indep_max(path: Path) -> float:
    X = np.loadtxt(path, comments="#", ndmin=2)
    if not np.all(np.isfinite(X)):
        return float("inf")
    X /= np.linalg.norm(X, axis=1, keepdims=True)
    G = X @ X.T
    np.fill_diagonal(G, -2)
    return float(G.max())


def parse(line: str) -> dict:
    return dict(kv.split("=", 1) for kv in line.split())


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("root")
    ap.add_argument("outdir")
    ap.add_argument("--n", type=int, required=True)
    ap.add_argument("--N", type=int, required=True)
    ap.add_argument("--root-stage", type=int, required=True,
                    help="number of schedule stages already applied to ROOT (0-based start)")
    ap.add_argument("--save-stage", type=int, default=4,
                    help="children save their state after this stage (1-based) to become parents")
    ap.add_argument("--screen-stage", type=int, default=7)
    ap.add_argument("--screen-quantile", type=float, default=0.5,
                    help="stop a branch whose screen-stage max is above this quantile of past ones")
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--hours", type=float, default=1.0)
    ap.add_argument("--sigmas", default="3e-4,1e-3,3e-3")
    ap.add_argument("--scale", type=float, default=3.4285714)
    ap.add_argument("--adam", default="manifold")
    ap.add_argument("--pool", type=int, default=8)
    ap.add_argument("--root-glob", default=None,
                    help="also use every file matching this glob as a root (re-scanned before each job), "
                         "e.g. the --save-stage states of a running held-start screen")
    ap.add_argument("--extra-root", action="append", default=[],
                    help="further root states (same --root-stage); roots are chosen uniformly")
    ap.add_argument("--polish-below", type=float, default=0.5005,
                    help="minimax-polish every finished candidate below this; rank by the polished value")
    ap.add_argument("--polish-rounds", type=int, default=30)
    ap.add_argument("--root-prob", type=float, default=0.25,
                    help="probability of branching from the root instead of the pool")
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()

    if not FASTRIESZ.exists():
        subprocess.run(["make", "-C", str(HERE), "fastriesz"], check=True)
    out = Path(a.outdir)
    out.mkdir(parents=True, exist_ok=True)
    rng = random.Random(a.seed)
    sigmas = [float(s) for s in a.sigmas.split(",")]
    log = open(out / "branch.log", "a")
    # pool entries: (final_max, parent_path, parent_stage)
    pool: list[tuple[float, str, int]] = []
    roots = [str(Path(r).resolve()) for r in [a.root] + a.extra_root]
    screen_vals: list[float] = []
    best = (9.0, None)
    t0 = time.time()
    job_id = a.seed * 1000000

    def threshold() -> float:
        if len(screen_vals) < 8:
            return 1.0
        return float(np.quantile(screen_vals, a.screen_quantile))

    def launch(jid: int):
        # tournament: favour the best parents, keep some breadth
        if a.root_glob:
            import glob
            for f in sorted(glob.glob(a.root_glob)):
                f = str(Path(f).resolve())
                if f not in roots:
                    roots.append(f)
        if not pool or rng.random() < a.root_prob:
            parent, pstage = rng.choice(roots), a.root_stage
        else:
            cands = rng.sample(pool, min(3, len(pool)))
            _, parent, pstage = min(cands)
        sigma = rng.choice(sigmas)
        thr = threshold()
        cmd = [str(FASTRIESZ), str(a.n), str(a.N), parent, str(out), "--extra", "file",
               "--adam", a.adam, "--scale", str(a.scale), "--start-stage", str(pstage),
               "--jit", repr(sigma), "--save-all", "--seeds", str(jid), str(jid + 1)]
        if a.save_stage > pstage:
            cmd += ["--save-stage", str(a.save_stage)]
        if a.screen_stage > pstage and thr < 1.0:
            cmd += ["--screen", f"{a.screen_stage}:{thr!r}"]
        env = dict(os.environ, OPENBLAS_NUM_THREADS="1", OMP_NUM_THREADS="1")
        r = subprocess.run(cmd, capture_output=True, text=True, env=env)
        line = r.stdout.strip()
        polished = None
        cand = out / f"cand_n{a.n}_N{a.N}_s{jid}.txt"
        if line and "status=done" in line and cand.exists():
            m = indep_max(cand)
            if m < a.polish_below:
                try:
                    X = minimax_polish.unit(cand)
                    Y = minimax_polish.polish(X, rounds=a.polish_rounds, verbose=False)
                    polished = minimax_polish.max_cos(Y)
                    if polished < m:
                        np.savetxt(cand, Y, fmt="%.17g",
                                   header=f"n={a.n} N={a.N} max_inner={polished:.17g} "
                                          f"minimax_polished_from={m:.17g}")
                except Exception as exc:  # a failed polish must not take the search down
                    print(f"polish of job {jid} failed: {exc}", file=sys.stderr, flush=True)
                    polished = None
        return jid, parent, pstage, sigma, thr, line, polished

    with ThreadPoolExecutor(max_workers=a.workers) as ex:
        futs = set()
        while True:
            while len(futs) < a.workers and time.time() - t0 < 3600 * a.hours:
                job_id += 1
                futs.add(ex.submit(launch, job_id))
            if not futs:
                break
            done = next(as_completed(futs))
            futs.remove(done)
            jid, parent, pstage, sigma, thr, line, polished = done.result()
            if not line:
                continue
            d = parse(line)
            stages = [float(x) for x in d["stages"].split(",")]
            k_screen = a.screen_stage - pstage - 1
            if 0 <= k_screen < len(stages):
                screen_vals.append(stages[k_screen])
            rec = {"job": jid, "parent": Path(parent).name, "pstage": pstage, "sigma": sigma,
                   "status": d["status"], "final": float(d["final"]), "stages": stages,
                   "screen_thr": thr, "t": round(time.time() - t0, 1)}
            cand = out / f"cand_n{a.n}_N{a.N}_s{jid}.txt"
            state = out / f"state_n{a.n}_N{a.N}_s{jid}_k{a.save_stage}.txt"
            keep_cand = False
            if d["status"] == "done" and cand.exists():
                m = indep_max(cand)  # after any polish: the file now holds the polished points
                rec["indep_max"] = m
                rec["polished"] = polished
                if state.exists():
                    pool.append((m, str(state), a.save_stage))
                if m < best[0]:
                    if best[1] and best[0] >= 0.5:
                        Path(best[1]).unlink(missing_ok=True)
                    best = (m, str(cand))
                    keep_cand = True
                if m < 0.5:
                    rec["HIT"] = True
                    keep_cand = True
                    subprocess.run([sys.executable, str(CERTIFY), str(cand),
                                    "--json-out", str(cand.with_suffix(".cert.json"))])
            if not keep_cand:
                cand.unlink(missing_ok=True)
            # the pool keeps the best parents; drop the state files that fall out
            pool.sort()
            for _, path, _ in pool[a.pool:]:
                if Path(path).parent == out:
                    Path(path).unlink(missing_ok=True)
            del pool[a.pool:]
            if d["status"] != "done":
                state.unlink(missing_ok=True)
            log.write(json.dumps(rec) + "\n")
            log.flush()
            print(f"job {jid} sigma={sigma:g} from {Path(parent).name}@{pstage}: {d['status']:8s} "
                  f"final={float(d['final']):.9f} best={best[0]:.9f} pool={[round(p[0], 6) for p in pool[:4]]} "
                  f"{time.time() - t0:7.0f}s", flush=True)
    print(f"done: best {best[0]:.12f} {best[1]}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
