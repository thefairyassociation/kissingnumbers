#!/usr/bin/env python3
"""One preselected, matched B=1 full-search calibration per engine.

Both use the canonical 840 rows and the same seed-51 Torch hypercube draw.
The C CLI is pinned to that vertex. No fixture endpoint is used as a seed,
no early screen is used, and no handoff or polish is silently applied.
Each subprocess has a 1,200-second wall limit. This is two runs, not a
statistical comparison or a reproduction of authors-scale GPU breadth.
"""
from concurrent.futures import ThreadPoolExecutor
import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time

import numpy as np

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))
from kissing.experimental.batched841 import run


def measure(path):
    coordinates = np.loadtxt(path)
    if coordinates.shape != (841, 12) or not np.isfinite(coordinates).all():
        raise RuntimeError(f"invalid coordinates: {path}")
    norms = np.linalg.norm(coordinates, axis=1)
    if not np.isfinite(norms).all() or np.any(norms == 0):
        raise RuntimeError("invalid coordinate norms")
    normalized = coordinates / norms[:, None]
    gram = normalized @ normalized.T
    np.fill_diagonal(gram, -np.inf)
    pair = np.unravel_index(np.argmax(gram), gram.shape)
    return dict(shape=list(coordinates.shape), max_inner_product=float(gram[pair]),
                max_pair=[int(x) for x in pair], max_norm_error=float(np.max(np.abs(norms-1))),
                strictly_below_half=bool(gram[pair] < .5),
                sha256=hashlib.sha256(Path(path).read_bytes()).hexdigest())


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("outdir", type=Path)
    args = parser.parse_args()
    args.outdir = args.outdir.resolve()
    args.outdir.mkdir(parents=True, exist_ok=False)
    core, digest, _ = run._load_core(None)
    initial = run._new_batch(core, 1, "cpu", run._make_generator("cpu", 51)).numpy()[0]
    pin = sum(int(bit) << k for k, bit in enumerate(initial[-1] > 0))
    seed = args.outdir / "initial.txt"
    np.savetxt(seed, initial, fmt="%.17g")
    env = {key:value for key,value in os.environ.items() if not key.startswith("KISS_")}
    env.update(OMP_NUM_THREADS="1", OPENBLAS_NUM_THREADS="1", KISS_THREADS="1")
    subprocess.run(["make", "-C", str(ROOT/'kissing/lib'), "riesz-tools"], check=True,
                   env=env, stdout=subprocess.DEVNULL)
    config = dict(seed=51, batch=1, steps=35000, threads=1, core_sha256=digest,
                  pinned_extra_index=pin, initial_sha256=hashlib.sha256(seed.read_bytes()).hexdigest(),
                  timeout_seconds_per_engine=1200, concurrent_calibration_jobs=2,
                  schedule=run.normalized_schedule())
    (args.outdir/'plan.json').write_text(json.dumps(config,indent=2)+'\n')

    def job(engine):
        local_env = env.copy()
        if engine == 'C':
            local_env.update(KISS_FAITHFUL="1", KISS_FAITHFUL_EXTRA=str(pin))
            command = [str(ROOT/'kissing/lib/riesz2'), '12', '841', '35000', '51', str(seed)]
            output = Path(str(seed)+'.riesz.s51.out')
        else:
            output = args.outdir/'torch.txt'
            command = [sys.executable, str(ROOT/'kissing/experimental/batched841/run.py'),
                       '--seed-file', str(seed), '--seed', '51', '--batch-size', '1',
                       '--full', '--threads', '1', '--checkpoint', str(args.outdir/'torch-checkpoint'),
                       '--output', str(output)]
        start = time.perf_counter()
        with (args.outdir/f'{engine}.stdout.log').open('w') as stdout, (args.outdir/f'{engine}.stderr.log').open('w') as stderr:
            try:
                process = subprocess.run(command, env=local_env, cwd=ROOT, stdout=stdout,
                                         stderr=stderr, timeout=1200, check=False)
                record = dict(engine=engine, returncode=process.returncode,
                              wall_seconds=time.perf_counter()-start, timed_out=False)
            except subprocess.TimeoutExpired:
                record = dict(engine=engine, wall_seconds=time.perf_counter()-start, timed_out=True)
        if record.get('returncode') == 0:
            record['readback'] = measure(output)
            if engine == 'C':
                shutil.copyfile(output, args.outdir/'c.txt')
                record['header'] = [line for line in output.read_text().splitlines() if line.startswith('#')]
            else:
                record['metadata'] = json.loads(output.with_suffix('.json').read_text())
        (args.outdir/f'{engine}-result.json').write_text(json.dumps(record,indent=2)+'\n')
        print(json.dumps(record),flush=True)
        return record

    with ThreadPoolExecutor(max_workers=2) as pool:
        records = list(pool.map(job, ['C','Torch']))
    (args.outdir/'calibration.json').write_text(json.dumps(dict(plan=config,results=records),indent=2)+'\n')
    if any(record.get('returncode') != 0 or record['timed_out'] for record in records):
        raise SystemExit(1)


if __name__ == '__main__':
    main()
