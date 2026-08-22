#!/usr/bin/env python3
"""Audited CPU multi-start driver for the fixed dim-12, N=841 calibration.

Each worker runs the C optimiser with one OpenMP thread so independent basins
use the available cores.  Candidate files are re-normalised and checked with a
fresh NumPy Gram product; the optimiser's ``feasible`` flag is never trusted.
Near-threshold coordinates and per-run logs are retained in ``outdir`` and the
JSON summary is rewritten after every completed run, so an interrupted search
can resume without losing finished basins.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import subprocess
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from optimizer_env import clean_optimizer_env  # noqa: E402


@dataclass
class Result:
    seed: int
    returncode: int
    verified_max_inner: float | None
    finite: bool
    shape_ok: bool
    max_row_norm_error: float | None
    basin_signature: str | None
    kept_candidate: str | None
    log: str
    error: str | None = None


def verify_candidate(path: Path) -> tuple[float, float, str]:
    """Return independently recomputed max-IP, pre-normalisation norm error,
    and a rotation/permutation-invariant near-contact fingerprint.
    """
    X = np.loadtxt(path)
    if X.shape != (841, 12):
        raise ValueError(f"expected (841, 12), got {X.shape}")
    if not np.isfinite(X).all():
        raise ValueError("candidate contains non-finite coordinates")
    norms = np.linalg.norm(X, axis=1)
    if not np.all(norms > 1e-15):
        raise ValueError("candidate contains a zero row")
    norm_error = float(np.max(np.abs(norms - 1.0)))
    X = X / norms[:, None]
    gram = X @ X.T
    upper = gram[np.triu_indices(841, 1)]
    max_inner = float(upper.max())

    # A compact invariant for separating numerically different basins.  The
    # 4096 largest inner products capture the near-contact structure without
    # depending on point order or a global orthogonal transformation.
    top = np.partition(upper, -4096)[-4096:]
    top.sort()
    signature = hashlib.sha256(np.round(top, 10).tobytes()).hexdigest()[:20]
    return max_inner, norm_error, signature


def optimizer_env(spec: dict[str, Any]) -> dict[str, str]:
    """Worker environment with every optimizer flag pinned, not inherited.

    Inheriting the ambient environment silently reinterpreted the run: a
    leftover KISS_FAITHFUL=1 makes every default 120000-step worker fail
    riesz.c's exact-35000-step guard, and KISS_LOSS=ip changes the objective
    without changing the summary.
    """
    settings = {
        "KISS_THREADS": str(spec["threads"]),
        "OMP_NUM_THREADS": str(spec["threads"]),
        "OPENBLAS_NUM_THREADS": "1",
        "KISS_SOLVER": "adam",
        "KISS_LOSS": "riesz",
        "KISS_FAITHFUL": "0",
        "KISS_ADAM_RAW": "0",
        "KISS_ADAM_POLISH": "0",
        "KISS_ADAM_POLISH_ONLY": "0",
        "KISS_PENALTY_ONLY": "0",
        "KISS_JIT": str(spec["jit"]),
    }
    if spec["base_end"] is not None:
        settings["KISS_ADAM_BASE_END"] = str(spec["base_end"])
    if not spec["polish"] or spec["base_end"] is not None:
        settings["KISS_POLISH"] = "0"
    return clean_optimizer_env(**settings)


def run_one(spec: dict[str, Any]) -> Result:
    seed = int(spec["seed"])
    seedfile = Path(spec["seedfile"])
    outdir = Path(spec["outdir"])
    log_path = outdir / f"seed_{seed}.log"
    source_candidate = Path(f"{seedfile}.riesz.s{seed}.out")
    kept_path = outdir / f"candidate_seed_{seed}.txt"

    if kept_path.exists():
        try:
            mx, norm_error, signature = verify_candidate(kept_path)
            return Result(
                seed, 0, mx, True, True, norm_error, signature,
                str(kept_path), str(log_path), None,
            )
        except Exception:
            pass

    env = optimizer_env(spec)
    command = [
        str(spec["binary"]), "12", "841", str(spec["steps"]),
        str(seed), str(seedfile),
    ]
    proc = subprocess.run(command, env=env, text=True, capture_output=True)
    log_path.write_text(
        f"command={' '.join(command)}\nreturncode={proc.returncode}\n\n"
        f"--- stdout ---\n{proc.stdout}\n--- stderr ---\n{proc.stderr}"
    )

    if proc.returncode != 0:
        return Result(
            seed, proc.returncode, None, False, False, None, None, None,
            str(log_path), "optimizer returned nonzero",
        )
    try:
        mx, norm_error, signature = verify_candidate(source_candidate)
    except Exception as exc:
        return Result(
            seed, proc.returncode, None, False, False, None, None, None,
            str(log_path), str(exc),
        )

    kept: str | None = None
    if mx < float(spec["keep_threshold"]):
        shutil.copy2(source_candidate, kept_path)
        kept = str(kept_path)
    return Result(
        seed, proc.returncode, mx, True, True, norm_error, signature, kept,
        str(log_path), None,
    )


def file_identity(path: Path) -> str:
    """Content hash of an input, so resuming cannot silently change it."""
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()[:20]


def run_settings(args: argparse.Namespace) -> dict[str, Any]:
    """Everything that defines the experiment.

    The seed file and the optimizer binary are part of this: without them,
    reusing an outdir with a different seed or a rebuilt binary skipped the
    finished seeds and reported their old verified maxima as results of the
    new experiment.
    """
    return {
        "steps": args.steps,
        "jit": args.jit,
        "threads_per_worker": args.threads,
        "workers": args.workers,
        "keep_threshold": args.keep_threshold,
        "base_end": args.base_end,
        "polish": args.polish,
        "seedfile_sha256": file_identity(args.seedfile),
        "binary_sha256": file_identity(args.binary),
    }


def write_summary(path: Path, args: argparse.Namespace, results: list[Result]) -> None:
    ordered = sorted(results, key=lambda result: result.seed)
    valid = [r for r in ordered if r.verified_max_inner is not None]
    payload = {
        "benchmark": {"dimension": 12, "count": 841, "threshold": 0.5},
        "settings": run_settings(args),
        "completed": len(ordered),
        "best_verified_max_inner": (
            min(r.verified_max_inner for r in valid) if valid else None
        ),
        "passed": any(r.verified_max_inner < 0.5 for r in valid),
        "distinct_signatures": len({r.basin_signature for r in valid}),
        "results": [asdict(result) for result in ordered],
    }
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    temporary.replace(path)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("seedfile", type=Path)
    parser.add_argument("--binary", type=Path, default=Path("kissing/lib/riesz2"))
    parser.add_argument("--outdir", type=Path, default=Path("/tmp/kissing-scratch/d12-multistart"))
    parser.add_argument("--start-seed", type=int, default=1)
    parser.add_argument("--runs", type=int, default=4)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--threads", type=int, default=1)
    parser.add_argument("--steps", type=int, default=120_000)
    parser.add_argument("--jit", type=float, default=0.03)
    parser.add_argument("--keep-threshold", type=float, default=0.5015)
    parser.add_argument(
        "--base-end",
        type=int,
        choices=range(1, 14),
        metavar="STAGE",
        help=(
            "stop before this zero-based Adam stage (for cheap basin screening); "
            "stage 4 stops after s=64 and automatically disables polishing"
        ),
    )
    parser.add_argument(
        "--polish", action=argparse.BooleanOptionalAction, default=True,
        help="run the threshold-penalty phase after the complete Adam schedule",
    )
    args = parser.parse_args()

    if not args.seedfile.exists():
        parser.error(f"seed file does not exist: {args.seedfile}")
    if not args.binary.exists():
        parser.error(f"optimizer binary does not exist: {args.binary}")
    if min(args.runs, args.workers, args.threads, args.steps) < 1:
        parser.error("runs, workers, threads and steps must all be positive")
    args.outdir.mkdir(parents=True, exist_ok=True)
    summary_path = args.outdir / "summary.json"

    expected_settings = run_settings(args)
    results: list[Result] = []
    if summary_path.exists():
        try:
            previous = json.loads(summary_path.read_text())
            previous_settings = previous.get("settings", {})
            if previous_settings != expected_settings:
                differing = sorted(
                    key for key in set(previous_settings) | set(expected_settings)
                    if previous_settings.get(key) != expected_settings.get(key)
                )
                parser.error(
                    f"existing {summary_path} was written with different "
                    f"settings ({', '.join(differing)}); choose another outdir"
                )
            results = [Result(**item) for item in previous.get("results", [])]
        except (json.JSONDecodeError, TypeError) as exc:
            parser.error(f"cannot resume invalid {summary_path}: {exc}")

    common = {
        "seedfile": str(args.seedfile.resolve()),
        "binary": str(args.binary.resolve()),
        "outdir": str(args.outdir.resolve()),
        "steps": args.steps,
        "jit": args.jit,
        "threads": args.threads,
        "keep_threshold": args.keep_threshold,
        "base_end": args.base_end,
        "polish": args.polish,
    }
    # A nonzero exit or an unreadable candidate is a failure to retry, not a
    # finished basin.  Keeping those in the skip set meant a resumed search
    # silently abandoned every seed that had errored once.
    completed_seeds = {
        result.seed for result in results if result.verified_max_inner is not None
    }
    retrying = [result for result in results if result.verified_max_inner is None]
    if retrying:
        print(
            f"retrying {len(retrying)} seed(s) with no verified candidate: "
            + ", ".join(str(result.seed) for result in sorted(
                retrying, key=lambda r: r.seed))
        )
        results = [r for r in results if r.verified_max_inner is not None]
    specs = [
        {**common, "seed": seed}
        for seed in range(args.start_seed, args.start_seed + args.runs)
        if seed not in completed_seeds
    ]
    with ProcessPoolExecutor(max_workers=args.workers) as executor:
        futures = {executor.submit(run_one, spec): spec["seed"] for spec in specs}
        for future in as_completed(futures):
            result = future.result()
            results.append(result)
            write_summary(summary_path, args, results)
            print(json.dumps(asdict(result), sort_keys=True), flush=True)
            if result.verified_max_inner is not None and result.verified_max_inner < 0.5:
                print(f"PASS seed={result.seed} verified={result.verified_max_inner:.17g}")

    best = min(
        (r.verified_max_inner for r in results if r.verified_max_inner is not None),
        default=None,
    )
    print(f"summary={summary_path} best={best}")
    sys.exit(0 if best is not None and best < 0.5 else 2)


if __name__ == "__main__":
    main()
