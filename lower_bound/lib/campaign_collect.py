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
"""
from __future__ import annotations

import argparse
import json
import re
import shutil
import subprocess
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
CERTIFY = HERE / "certify_float.py"


def indep_max(path: Path) -> float:
    X = np.loadtxt(path, comments="#", ndmin=2)
    if not np.all(np.isfinite(X)):
        return float("inf")
    X /= np.linalg.norm(X, axis=1, keepdims=True)
    G = X @ X.T
    np.fill_diagonal(G, -2)
    return float(G.max())


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("shards", type=Path)
    ap.add_argument("out", type=Path)
    ap.add_argument("--spec", type=Path)
    ap.add_argument("--run-url", default=None)
    a = ap.parse_args()
    a.out.mkdir(parents=True, exist_ok=True)
    (a.out / "hits").mkdir(exist_ok=True)
    (a.out / "near").mkdir(exist_ok=True)

    arms: dict[str, dict] = {}
    shards = []
    for d in sorted(a.shards.glob("shard-*")):
        m = re.fullmatch(r"shard-(.+)-(\d+)", d.name)
        sj = d / "summary.json"
        if not m or not sj.exists():
            shards.append({"shard": d.name, "status": "no summary (job failed or was cancelled)"})
            continue
        arm, k = m.group(1), int(m.group(2))
        s = json.loads(sj.read_text())
        s["arm"], s["shard"] = arm, k
        s.pop("rounds", None)          # per-round detail stays in the artifact
        kept = []
        for f in sorted((d / "kept").glob("cand_*.txt")) if (d / "kept").is_dir() else []:
            kept.append((indep_max(f), f))
        s["kept"] = [(round(v, 12), f.name) for v, f in kept]
        shards.append(s)
        A = arms.setdefault(arm, {"n": s["n"], "N": s["N"], "init": s["init"], "extra": s["extra"],
                                  "shards": 0, "screened": 0, "roots": 0, "cpu_hours": 0.0,
                                  "anti_histogram": {}, "best": None, "certified_hits": [],
                                  "uncertified_below_half": [], "_kept": []})
        A["shards"] += 1
        A["screened"] += s["screened"]
        A["roots"] += s["roots"]
        A["cpu_hours"] += s["elapsed_seconds"] * s["workers"] / 3600.0
        for b, c in s["anti_histogram"].items():
            A["anti_histogram"][b] = A["anti_histogram"].get(b, 0) + c
        if s["best"]["file"] and (A["best"] is None or s["best"]["max"] < A["best"]["max"]):
            A["best"] = {"max": s["best"]["max"], "shard": k, "file": s["best"]["file"]}
        A["_kept"] += [(v, f, k) for v, f in kept]

    for arm, A in arms.items():
        A["_kept"].sort(key=lambda t: t[0])
        for v, f, k in A["_kept"]:
            name = f"{arm}-{k}-{f.name}"
            if v < 0.5:
                dst = a.out / "hits" / name
                shutil.copy(f, dst)
                cert = dst.with_suffix(".cert.json")
                c = subprocess.run([sys.executable, str(CERTIFY), str(dst), "--json-out", str(cert)],
                                   capture_output=True, text=True)
                rec = {"file": f"hits/{name}", "max": v, "certify": c.stdout.strip().splitlines()[-1:]}
                if c.returncode == 0 and "CERTIFIED" in c.stdout:
                    A["certified_hits"].append(rec)
                else:
                    A["uncertified_below_half"].append(rec)
        for v, f, k in [t for t in A["_kept"] if t[0] >= 0.5][:2]:
            shutil.copy(f, a.out / "near" / f"{arm}-{k}-{f.name}")
        A["anti_histogram"] = dict(sorted(A["anti_histogram"].items(), key=lambda kv: int(kv[0])))
        A["cpu_hours"] = round(A["cpu_hours"], 1)
        A["root_rate"] = round(A["roots"] / A["screened"], 5) if A["screened"] else None
        del A["_kept"]

    record = {"campaign": a.out.name, "run_url": a.run_url,
              "spec": json.loads(a.spec.read_text()) if a.spec and a.spec.exists() else None,
              "arms": arms, "shards": shards,
              "any_certified_hit": any(A["certified_hits"] for A in arms.values())}
    (a.out / "summary.json").write_text(json.dumps(record, indent=1) + "\n")

    L = [f"# Search campaign `{a.out.name}`", ""]
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
    L += ["", "Near-antipodal pairs after the first exponent stage (bucketed by 50), per arm:", ""]
    for arm, A in arms.items():
        L.append(f"* `{arm}`: " + ", ".join(f"{b}+: {c}" for b, c in A["anti_histogram"].items()))
    failed = [s["shard"] for s in shards if "status" in s]
    if failed:
        L += ["", f"Shards without a summary: {', '.join(map(str, failed))}."]
    L.append("")
    (a.out / "README.md").write_text("\n".join(L))
    for sub in ("hits", "near"):
        if not any((a.out / sub).iterdir()):
            (a.out / sub).rmdir()
    print("\n".join(L))
    return 0


if __name__ == "__main__":
    sys.exit(main())
