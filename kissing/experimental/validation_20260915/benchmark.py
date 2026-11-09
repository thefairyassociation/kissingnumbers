#!/usr/bin/env python3
"""Bounded, warmed CPU timing comparison; not a calibration search.

Each repetition resets the same canonical-core input and zero moments.
Three shortened stages exercise different exponents. C B=1 and Torch B=1
are matched workloads. B>1 is throughput evidence, not identical trajectories:
the true batch mean changes Adam's effective epsilon relative to B=1.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import platform
import statistics
import subprocess
import sys
import time

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))
from kissing.experimental.batched841 import kernel, run
from kissing.experimental.batched841.test_parity import _scipy_blas_flags


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("outdir", type=Path)
    parser.add_argument("--steps", type=int, default=50)
    parser.add_argument("--repetitions", type=int, default=3)
    args = parser.parse_args()
    if not 1 <= args.steps <= 200 or not 1 <= args.repetitions <= 10:
        parser.error("bounded timing requires 1..200 steps/stage and 1..10 repetitions")
    args.outdir.mkdir(parents=True, exist_ok=False)
    env = os.environ.copy()
    env.update(OMP_NUM_THREADS="1", OPENBLAS_NUM_THREADS="1")
    binary = args.outdir / "bench_c"
    subprocess.run(["gcc", "-std=c11", "-O3", "-o", str(binary),
                    str(Path(__file__).with_name("bench_c.c")), *_scipy_blas_flags()], check=True)
    core, digest, _ = run._load_core(None)
    raw = run._new_batch(core, 1, "cpu", run._make_generator("cpu", 51)).numpy()
    raw[0].tofile(args.outdir / "raw.bin")
    np.savetxt(args.outdir / "initial.txt", raw[0], fmt="%.17g")
    schedule = [(s, args.steps, lr) for s, _, lr in
                (kernel.SEARCH_SCHEDULE[0], kernel.SEARCH_SCHEDULE[6], kernel.SEARCH_SCHEDULE[-1])]
    schedule_file = args.outdir / "schedule.txt"
    schedule_file.write_text("".join(f"{s} {steps} {lr}\n" for s, steps, lr in schedule))
    results = []
    endpoints = {}
    torch.set_num_interop_threads(1)
    for threads in (1, 4):
        for engine, batch in (("C", 1), ("Torch", 1), ("Torch", 2), ("Torch", 4)):
            torch.set_num_threads(threads)
            samples = []
            for repetition in range(-1, args.repetitions):  # first is warmup
                if engine == "C":
                    output = args.outdir / f"c-t{threads}.txt"
                    proc = subprocess.run([str(binary), str(args.outdir / "raw.bin"),
                                           str(schedule_file), str(output), str(threads)],
                                          env=env, check=True, text=True, capture_output=True)
                    stages = [json.loads(line) for line in proc.stdout.splitlines()]
                    durations = [stage["seconds"] for stage in stages]
                    final = np.loadtxt(output)
                else:
                    state = kernel.initial_state(torch.from_numpy(np.repeat(raw, batch, axis=0)))
                    durations = []
                    for s, steps, lr in schedule:
                        start = time.perf_counter()
                        for _ in range(steps):
                            kernel.adam_step(state, s, lr)
                        durations.append(time.perf_counter() - start)
                    final = kernel.normalized_view(state["raw"])[0].numpy()
                if not np.isfinite(final).all():
                    raise RuntimeError("invalid timed endpoint")
                if repetition >= 0:
                    samples.append(durations)
            medians = [statistics.median(sample[i] for sample in samples) for i in range(len(schedule))]
            total_median = statistics.median(sum(sample) for sample in samples)
            result = dict(engine=engine, batch=batch, threads=threads, samples_seconds=samples,
                          median_stage_seconds=medians, median_total_seconds=total_median,
                          candidate_updates_per_second=batch*args.steps*len(schedule)/total_median)
            results.append(result)
            endpoints[(engine,batch,threads)] = final
            print(json.dumps(result), flush=True)
    agreement = []
    for threads in (1,4):
        delta = float(np.max(np.abs(endpoints[("C",1,threads)]-endpoints[("Torch",1,threads)])))
        agreement.append(dict(threads=threads, max_coordinate_difference=delta))
    record = dict(python=platform.python_version(), torch=torch.__version__, numpy=np.__version__,
                  cuda_available=torch.cuda.is_available(), platform=platform.platform(),
                  cpu_quota=Path('/sys/fs/cgroup/cpu.max').read_text().strip(),
                  memory_limit_bytes=Path('/sys/fs/cgroup/memory.max').read_text().strip(),
                  core_sha256=digest, raw_sha256=hashlib.sha256(raw.tobytes()).hexdigest(),
                  kernel_sha256=hashlib.sha256(Path(kernel.__file__).read_bytes()).hexdigest(),
                  c_source_sha256=hashlib.sha256((ROOT/'kissing/lib/riesz.c').read_bytes()).hexdigest(),
                  schedule=schedule, repetitions=args.repetitions, warmup_repetitions=1,
                  results=results, b1_endpoint_agreement=agreement)
    (args.outdir/'benchmark.json').write_text(json.dumps(record,indent=2)+'\n')


if __name__ == '__main__':
    main()
