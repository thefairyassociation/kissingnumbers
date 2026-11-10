#!/usr/bin/env python3
"""Merge the shard results of one search campaign into a committed record.

usage:
    python3 lower_bound/lib/campaign_collect.py SHARDS_DIR OUT_DIR [--spec SPEC.json] [--run-url URL]

SHARDS_DIR holds one directory per shard (`shard-<arm>-<k>/summary.json` and
`kept/`), as downloaded from the workflow artifacts.  OUT_DIR receives

    summary.json   every shard summary, plus totals per arm
    README.md      the same as a table
    hits/          coordinates + certificate of anything certified below 1/2
    near/          the best two near-misses per arm (coordinates)

Every kept coordinate file is recomputed here with NumPy, and anything below
1/2 is run through certify_float.py again; a "hit" is reported only if that
exact check passes in this process.

The kept files of a shard are examined whether or not the shard left a
summary: a job killed at its time limit can have certified something first.
Shards that the spec asked for but that delivered nothing are listed by name.

If OUT_DIR already holds a record (a manual re-run of the same campaign), the
new record goes to OUT_DIR/rerun-<run id>/ and the first one is left untouched.
"""
from __future__ import annotations

import argparse
import json
import re
import shutil
import subprocess
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
CERTIFY = HERE / "certify_float.py"
sys.path.insert(0, str(HERE))
from maxcos import indep_max  # noqa: E402

import numpy as np  # noqa: E402


def new_arm(n, N, init, extra) -> dict:
    return {"n": n, "N": N, "init": init, "extra": extra, "start_stage": 0,
            "shards": 0, "shards_without_summary": 0, "partial_shards": 0,
            "screened": 0, "roots": 0, "cpu_hours": 0.0,
            "anti_histogram": {}, "best": None, "certified_hits": [],
            "uncertified_below_half": [], "_kept": []}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("shards", type=Path)
    ap.add_argument("out", type=Path)
    ap.add_argument("--spec", type=Path)
    ap.add_argument("--run-url", default=None)
    a = ap.parse_args()
    campaign = a.out.name
    if (a.out / "summary.json").exists():
        run_id = (a.run_url or "").rstrip("/").rsplit("/", 1)[-1]
        if not re.fullmatch(r"[0-9]+", run_id):
            run_id = time.strftime("%Y%m%dT%H%M%SZ", time.gmtime())
        a.out = a.out / f"rerun-{run_id}"
        print(f"{campaign} already has a record; writing this run to {a.out}", file=sys.stderr)
    a.out.mkdir(parents=True, exist_ok=True)
    (a.out / "hits").mkdir(exist_ok=True)
    (a.out / "near").mkdir(exist_ok=True)

    spec = json.loads(a.spec.read_text()) if a.spec and a.spec.exists() else None
    spec_arms = {arm["id"]: arm for arm in spec["arms"]} if spec else {}
    expected = {(arm["id"], k) for arm in spec_arms.values() for k in range(int(arm["shards"]))}

    arms: dict[str, dict] = {}
    shards = []
    seen = set()
    for d in sorted(a.shards.glob("shard-*")):
        m = re.fullmatch(r"shard-(.+)-(\d+)", d.name)
        if not m:
            shards.append({"shard": d.name, "status": "unrecognised artifact name"})
            continue
        arm, k = m.group(1), int(m.group(2))
        seen.add((arm, k))
        sa = spec_arms.get(arm, {})
        kept = []
        for f in sorted((d / "kept").glob("cand_*.txt")) if (d / "kept").is_dir() else []:
            kept.append((indep_max(f), f))
        sj = d / "summary.json"
        s = None
        if sj.exists():
            try:
                s = json.loads(sj.read_text())
            except ValueError:
                s = None
        if s is None:
            # No usable summary: the job failed, was cancelled or hit its time limit.
            # Its kept files are still evidence and are examined below.
            A = arms.setdefault(arm, new_arm(sa.get("n"), sa.get("N"), sa.get("init"), sa.get("extra")))
            A["shards_without_summary"] += 1
            A["_kept"] += [(v, f, k) for v, f in kept]
            shards.append({"arm": arm, "shard": k, "status": "no summary (job failed, was cancelled or timed out)",
                           "kept": [(round(v, 12), f.name) for v, f in kept]})
            continue
        s["arm"], s["shard"] = arm, k
        s.pop("rounds", None)          # per-round detail stays in the artifact
        s["kept"] = [(round(v, 12), f.name) for v, f in kept]
        shards.append(s)
        # the shard only knows its init as a file name; the spec knows which configuration it was
        A = arms.setdefault(arm, new_arm(s["n"], s["N"], sa.get("init", s["init"]), s["extra"]))
        A["start_stage"] = int(s.get("start_stage", 0))
        A["shards"] += 1
        if s.get("complete") is False:
            A["partial_shards"] += 1
        A["screened"] += s["screened"]
        A["roots"] += s["roots"]
        A["cpu_hours"] += s["elapsed_seconds"] * s["workers"] / 3600.0
        for b, c in s["anti_histogram"].items():
            A["anti_histogram"][b] = A["anti_histogram"].get(b, 0) + c
        if s["best"]["file"] and (A["best"] is None or s["best"]["max"] < A["best"]["max"]):
            A["best"] = {"max": s["best"]["max"], "shard": k, "file": s["best"]["file"]}
        A["_kept"] += [(v, f, k) for v, f in kept]
    missing = sorted(expected - seen)

    for arm, A in arms.items():
        A["_kept"].sort(key=lambda t: t[0])
        for v, f, k in A["_kept"]:
            name = f"{arm}-{k}-{f.name}"
            if v < 0.5:
                dst = a.out / "hits" / name
                shutil.copy(f, dst)
                shape = np.loadtxt(f, comments="#", ndmin=2).shape
                if A["N"] is not None and shape != (A["N"], A["n"]):
                    # below 1/2, but not a configuration of the size this arm is about
                    A["uncertified_below_half"].append(
                        {"file": f"hits/{name}", "max": v,
                         "certify": [f"wrong shape {shape}, expected ({A['N']}, {A['n']})"]})
                    continue
                cert = dst.with_suffix(".cert.json")
                c = subprocess.run([sys.executable, str(CERTIFY), str(dst), "--json-out", str(cert)],
                                   capture_output=True, text=True)
                rec = {"file": f"hits/{name}", "max": v, "certify": c.stdout.strip().splitlines()[-1:]}
                if c.returncode == 0 and "CERTIFIED" in c.stdout:
                    A["certified_hits"].append(rec)
                else:
                    A["uncertified_below_half"].append(rec)
        near = [t for t in A["_kept"] if t[0] >= 0.5]
        for v, f, k in near[:2]:
            shutil.copy(f, a.out / "near" / f"{arm}-{k}-{f.name}")
        # a shard without a summary cannot report its best value; its kept files can
        if near and near[0][0] != float("inf") and (A["best"] is None or near[0][0] < A["best"]["max"]):
            A["best"] = {"max": near[0][0], "shard": near[0][2], "file": near[0][1].name}
        A["anti_histogram"] = dict(sorted(A["anti_histogram"].items(), key=lambda kv: int(kv[0])))
        A["cpu_hours"] = round(A["cpu_hours"], 1)
        A["root_rate"] = round(A["roots"] / A["screened"], 5) if A["screened"] else None
        del A["_kept"]

    record = {"campaign": campaign, "run_url": a.run_url, "spec": spec,
              "arms": arms, "shards": shards,
              "missing_shards": [f"{arm}-{k}" for arm, k in missing],
              "any_certified_hit": any(A["certified_hits"] for A in arms.values())}
    (a.out / "summary.json").write_text(json.dumps(record, indent=1) + "\n")

    L = [f"# Search campaign `{campaign}`", ""]
    if a.run_url:
        L += [f"Workflow run: {a.run_url}", ""]
    L += ["Written by `lower_bound/lib/campaign_collect.py`; the numbers are recomputed from the",
          "coordinate files, and a hit is listed only if `certify_float.py` passed it (exact integer",
          "arithmetic).", "",
          "| arm | n | N | shards | CPU-hours | seeds screened | roots | best max inner product | certified below 1/2 |",
          "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |"]
    for arm, A in arms.items():
        best = f"{A['best']['max']:.10f}" if A["best"] else "-"
        L.append(f"| {arm} | {A['n']} | {A['N']} | {A['shards']} | {A['cpu_hours']} | {A['screened']} | "
                 f"{A['roots']} | {best} | {'**YES** (' + str(len(A['certified_hits'])) + ')' if A['certified_hits'] else 'no'} |")
    L += ["", "Near-antipodal pairs at the screening point (bucketed by 50), per arm. The screen is applied",
          "after the first exponent stage a seed runs, which depends on the arm's entry stage:", ""]
    for arm, A in arms.items():
        L.append(f"* `{arm}` (after the s = {8 * 2 ** A['start_stage']} stage): "
                 + (", ".join(f"{b}+: {c}" for b, c in A["anti_histogram"].items()) or "no data"))
    below = [(arm, r) for arm, A in arms.items() for r in A["uncertified_below_half"]]
    if below:
        L += ["", "Below 1/2 in floating point but **not** certified (not hits):", ""]
        L += [f"* `{arm}`: `{r['file']}` ({r['max']:.12f})" for arm, r in below]
    nosum = [f"{s['arm']}-{s['shard']}" for s in shards if "status" in s and "arm" in s]
    partial = [f"{s['arm']}-{s['shard']}" for s in shards if s.get("complete") is False]
    odd = [s["shard"] for s in shards if "status" in s and "arm" not in s]
    if nosum:
        L += ["", f"Shards without a summary (their kept files were still checked; their seeds, roots and "
                  f"CPU time are not in the table): {', '.join(nosum)}."]
    if partial:
        L += ["", f"Shards that stopped before finishing (totals up to their last completed round): "
                  f"{', '.join(partial)}."]
    if missing:
        L += ["", f"Shards the spec asked for that delivered no artifact at all: "
                  f"{', '.join(f'{arm}-{k}' for arm, k in missing)}."]
    if odd:
        L += ["", f"Artifacts with an unrecognised name: {', '.join(odd)}."]
    L.append("")
    (a.out / "README.md").write_text("\n".join(L))
    for sub in ("hits", "near"):
        if not any((a.out / sub).iterdir()):
            (a.out / sub).rmdir()
    print("\n".join(L))
    return 0


if __name__ == "__main__":
    sys.exit(main())
