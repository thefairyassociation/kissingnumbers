#!/usr/bin/env python3
"""Bounded 4-canonical versus 4-deformed source-faithful CPU pilot.

Each job uses exactly ``STEPS=35000`` and stops after Adam stage 4 or 6 via
``KISS_ADAM_BASE_END``.  ``KISS_FAITHFUL=1`` forces the published raw-Adam,
no-jitter semantics and independently draws the final uniform hypercube extra
inside the C optimizer.  Canonical controls are therefore genuinely distinct:
their optimizer seeds produce distinct recorded uniform extra indices; they
are not OpenMP-noise repeats of one deterministic input.
"""

from __future__ import annotations

import argparse
import json
import re
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(Path(__file__).resolve().parent))
from o4_breadth import make_seed, write_seed  # noqa: E402
from optimizer_env import clean_optimizer_env  # noqa: E402

STAGE_RE = re.compile(r"^s=([0-9.]+)\s+max=([0-9.eE+-]+).*nfev=(\d+)", re.MULTILINE)
# The extra may be uniform-random or pinned via KISS_FAITHFUL_EXTRA.
EXTRA_RE = re.compile(r"(?:uniform-random|pinned) hypercube extra index=(\d+)")


def _git_head() -> str | None:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True
        ).strip()
    except (OSError, subprocess.CalledProcessError):
        return None


def run_one(
    *,
    arm: str,
    seed: int,
    scale: float,
    outdir: Path,
    binary: Path,
    verifier: Path,
    base_end: int,
    threads: int,
) -> dict[str, object]:
    run_dir = outdir / "runs" / f"{arm}_seed{seed}"
    run_dir.mkdir(parents=True, exist_ok=True)
    seed_file = run_dir / "seed841.txt"
    X, seed_metadata = make_seed(
        arm=arm,
        seed=seed,
        scale=scale,
        extra_mode="uniform-random",
    )
    write_seed(seed_file, X, seed_metadata)
    # Build the environment explicitly rather than inheriting it: faithful mode
    # ignores legacy jitter, and an ambient KISS_FAITHFUL_EXTRA would pin every
    # arm of the pilot to the same hypercube extra, defeating the comparison.
    env = clean_optimizer_env(
        KISS_FAITHFUL="1",
        KISS_SOLVER="adam",
        KISS_LOSS="riesz",
        KISS_POLISH="0",
        KISS_ADAM_BASE_END=str(base_end),
        KISS_THREADS=str(threads),
        OMP_NUM_THREADS=str(threads),
        OPENBLAS_NUM_THREADS="1",
    )
    command = [str(binary), "12", "841", "35000", str(seed), str(seed_file)]
    started = time.monotonic()
    proc = subprocess.run(command, env=env, text=True, capture_output=True)
    elapsed = time.monotonic() - started
    (run_dir / "optimizer.stdout").write_text(proc.stdout)
    (run_dir / "optimizer.stderr").write_text(proc.stderr)
    candidate = Path(f"{seed_file}.riesz.s{seed}.out")
    record: dict[str, object] = {
        "arm": arm,
        "seed": seed,
        "command": command,
        "returncode": proc.returncode,
        "elapsed_seconds": round(elapsed, 3),
        "source_seed_metadata": seed_metadata,
        "optimizer_seed": seed,
        "settings": {
            "faithful": True,
            "steps": 35000,
            "base_end": base_end,
            "threads": threads,
            "jitter": 0.0,
            "extra_mode": "uniform-random (drawn by C faithful mode)",
        },
        "stages": [
            {"s": float(s), "max_ip_reported": float(mx), "nfev": int(nfev)}
            for s, mx, nfev in STAGE_RE.findall(proc.stderr)
        ],
        "optimizer_extra_index": (
            int(EXTRA_RE.search(proc.stderr).group(1))
            if EXTRA_RE.search(proc.stderr)
            else None
        ),
        "candidate": str(candidate),
    }
    if proc.returncode != 0 or not candidate.exists():
        record["error"] = "optimizer failed or did not write candidate"
        return record

    check = subprocess.run(
        [sys.executable, str(verifier), str(candidate)],
        text=True,
        capture_output=True,
        check=False,
    )
    (run_dir / "independent_verify.stdout.json").write_text(check.stdout)
    (run_dir / "independent_verify.stderr").write_text(check.stderr)
    record["verify_returncode"] = check.returncode
    if check.returncode != 0:
        record["error"] = "independent verifier failed"
    else:
        try:
            record["verification"] = json.loads(check.stdout)["results"][0]
        except (json.JSONDecodeError, KeyError, IndexError, TypeError) as exc:
            record["error"] = f"could not parse verifier output: {exc}"
    return record


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--binary", type=Path, default=ROOT / "kissing" / "lib" / "riesz2")
    parser.add_argument("--verifier", type=Path, default=ROOT / "kissing" / "lib" / "verify_o4_841.py")
    parser.add_argument("--outdir", type=Path, default=ROOT / "kissing" / "logs" / "o4-faithful-pilot")
    parser.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2, 3])
    parser.add_argument("--arms", nargs="+", choices=["canonical", "deformed"], default=["canonical", "deformed"])
    parser.add_argument("--scale", type=float, default=0.006)
    parser.add_argument("--base-end", type=int, choices=[4, 6], default=4)
    parser.add_argument("--threads", type=int, default=1)
    args = parser.parse_args()
    if not args.binary.exists():
        parser.error(f"missing binary {args.binary}; run make -C kissing/lib riesz2")
    if not args.verifier.exists():
        parser.error(f"missing verifier {args.verifier}")
    args.outdir.mkdir(parents=True, exist_ok=True)
    records = []
    for arm in args.arms:
        for seed in args.seeds:
            print(f"RUN faithful arm={arm} seed={seed} base_end={args.base_end}", flush=True)
            record = run_one(
                arm=arm,
                seed=seed,
                scale=args.scale,
                outdir=args.outdir,
                binary=args.binary,
                verifier=args.verifier,
                base_end=args.base_end,
                threads=args.threads,
            )
            records.append(record)
            print(json.dumps(record, sort_keys=True), flush=True)
    payload = {
        "source_commit": _git_head(),
        "pilot": {
            "faithful": True,
            "steps": 35000,
            "base_end": args.base_end,
            "arms": args.arms,
            "seeds": args.seeds,
            "scale": args.scale,
            "threads": args.threads,
            "extra_mode": "uniform-random",
        },
        "records": records,
    }
    summary = args.outdir / "summary.json"
    summary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    print(f"wrote {summary}")


if __name__ == "__main__":
    main()
