# Bounded CPU validation, 2026-09-15

This follow-up evaluates the additive engine in draft PR #10. It changes no
established solver or prior research artifact. Its measurements concern this
CPU environment only; CUDA was unavailable (`torch.cuda.is_available() == False`,
CPU-only PyTorch build, no NVIDIA device or `nvidia-smi`).

## Timing protocol

`benchmark.py` uses the canonical 840-point core plus one seed-51 hypercube
extra. C and Torch B=1 receive identical raw coordinates and zero moments.
The C adapter calls the unchanged `adam_raw_stage` directly, once per stage;
there is no per-update subprocess or output overhead. Both use float64,
epsilon 1e-8, and persistent moments across the three stages.

Each timed repetition has 50 updates at each of s=8, 512, and 40000, using
their published learning rates. This deliberately shortened schedule tests
throughput and numerical agreement, not discovery. There is one discarded
warmup and three measured repetitions, each starting from fresh state.
The rate uses the median of the three total times, not the sum of stage
medians. Process startup, input parsing, and final coordinate output are
excluded. Kernel validation and C's extra end-of-stage evaluation are included.
The timing probes ran sequentially before either full calibration started.

| Engine | Batch | Threads | Candidate updates/second |
| --- | ---: | ---: | ---: |
| Existing C | 1 | 1 | 122.68 |
| New Torch | 1 | 1 | 61.93 |
| New Torch | 2 | 1 | 55.07 |
| New Torch | 4 | 1 | 54.54 |
| Existing C | 1 | 4 | 140.46 |
| New Torch | 1 | 4 | 88.86 |
| New Torch | 2 | 4 | 89.70 |
| New Torch | 4 | 4 | 118.53 |

One batched update advances B candidates, so these rates include that factor.
B>1 uses repeated initial coordinates to keep the workload controlled. The
true batch-mean gradient changes Adam's effective epsilon compared with B=1;
the B>1 probes are not equivalent trajectory comparisons or independent trials.
C uses the indicated OpenMP thread count and one OpenBLAS thread; Torch uses
the indicated intra-op count and one inter-op thread.

C and Torch B=1 final coordinates differ by at most 1.99e-9 after the 150
shortened updates. This supports short-run numerical agreement; it does not
promise the same long nonlinear trajectory. Four-thread timings have visible
variation, and three repetitions are not a rigorous performance study.
Full samples, source hashes, versions, and resource limits are in
`results/benchmark.json` (8 CPU quota, 20 GiB memory).

## Full-search protocol

`calibrate.py` preselects one B=1, 35,000-update search for each engine, with
the same canonical core and the same seed-51 extra. C pins hypercube vertex
3254; Torch reads the same initial file. The known published 841 endpoint is
never an initializer. There is no early screen, jitter, handoff, or polish.
Both jobs use one thread and run concurrently, each with a 1,200-second wall
limit. Their end-to-end times therefore are not isolated throughput estimates.

The C file's legacy second header line prints the enabled `polish` option.
The following faithful-provenance header records what actually ran:
`polish=0 polish_updates=0 gram_handoff=0`. No polishing was performed.

Both processes returned successfully after all 13 stages and 35,000 updates.
Independent readback of the saved coordinates gives:

| Engine | Updates | End-to-end seconds | Final maximum inner product | Valid kissing configuration? |
| --- | ---: | ---: | ---: | --- |
| Existing C | 35,000 | 373.68 | 0.5342891714277646 | No |
| New Torch | 35,000 | 630.22 | 0.5343313313187009 | No |

The requirement is at most 0.5. Exact rational checks of a violating pair in
each file confirm the failures; they are not floating-point boundary cases.
The C maximum occurs at zero-based pair (638, 820), the Torch maximum at
(141, 465). The similar final scores do not imply identical trajectories.
One matched start cannot estimate recovery probability or prove one solver
has better search quality. No impossibility or improved lower bound follows.

`results/calibration.json` retains the plan, completion metadata, file hashes,
and independent numerical readback. `results/c.txt` and `results/torch.txt`
retain both endpoints; `results/independent_readback.json` contains exact
rational counterexamples. `results/C.stderr.log` retains C's stage trace.
Optimizer checkpoints and compiled binaries are not part of this report.

This does validate the runner's complete real-size schedule and terminal
output lifecycle. It does **not** satisfy the independent 841 recovery gate.

## Reproduce

From the repository root, with the existing C build dependencies plus NumPy,
PyTorch, SciPy, and `scipy-openblas32` installed:

```sh
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 python kissing/experimental/validation_20260915/benchmark.py /tmp/kiss-benchmark-new
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 python kissing/experimental/validation_20260915/calibrate.py /tmp/kiss-calibration-new
```

Output directories must not already exist. The calibration driver compiles
the existing C tools and records the plan before starting either search.
Different library builds or platforms may produce different trajectories.

The saved final coordinate files can be checked without rerunning the search:

```sh
OPENBLAS_NUM_THREADS=1 python kissing/experimental/validation_20260915/verify_saved.py
```

This reads the full coordinates, verifies their recorded hashes, recomputes
the maximum inner product across every pair, and checks the reported schedule
completion. It also parses a violating pair directly as rational numbers and
proves its normalized inner product exceeds one half using exact arithmetic.
That rejection applies only to the saved configuration.

## Decision boundary

There is no measured CPU speed advantage for the new batched implementation
in these probes. At four threads its best measured rate is about 16% below C;
at one thread B=1 is about twice as slow. Do not scale a CPU campaign on the
assumption that this implementation is faster. The checkpointing and exact
certificate tools remain useful engineering additions.

Neither full search recovered 841, so this bounded round provides no positive
search result to justify increasing the CPU budget. Retain the experiment
and its negative evidence rather than presenting more search volume as
mathematical progress.

A small GPU correctness, memory, and warmed-throughput check would be the next
useful performance experiment if a GPU becomes available. No GPU performance
or independent-search success rate can be inferred from these CPU results.
