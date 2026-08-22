# Handoff: make the spherical-code optimiser good enough to matter

## Why this exists

The repo tries to beat known kissing-number lower bounds in dimensions 12, 13, 14
(`kissing/README.md` has the full state). Nothing was beaten. The structural
results there are solid, but every *numerical* result is currently worthless,
for one measurable reason:

> On dimension 12 with N = 841 — a case where a solution **provably exists**,
> because Takhanov–Assylbekov–Yun reached max inner product 0.499999937751 —
> the best optimiser run in this repo reaches **0.500477238629** (independently recomputed), still above the required threshold.

So when the same machinery reaches 0.5088 for 1155 points in dimension 13, that
says nothing about whether 1155 points exist. Closing that gap is the job.

## The task

Get the dimension-12 N = 841 calibration **below 0.5**. That is the whole
pass/fail criterion. Everything else is optional.

```bash
python3 kissing/lib/seed841.py /tmp/kissing-scratch/cl840_841.txt --mode hypercube
./kissing/lib/riesz2 12 841 120000 51 /tmp/kissing-scratch/cl840_841.txt
```

Scoreboard so far (all on this exact seed):

| configuration | dim-12 N=841 |
| --- | --- |
| Takhanov et al. (published, the target) | 0.499999937751 |
| old hand-rolled gradient descent + Armijo | 0.50519 |
| BLAS + L-BFGS branch, 18 short restarts | 0.51123 |
| BLAS + L-BFGS branch, equal wall clock | 0.51113 |
| BLAS + L-BFGS branch, finer homotopy (`KISS_SMUL=1.04`) | 0.51392 |
| BLAS engine + `KISS_SOLVER=gd` | 0.509619958637 |
| BLAS engine + published-schedule Adam | **0.500477238629** ← best here, still not feasible |

Only after 841 goes below 0.5 is it worth re-running dimension 13 at N = 1155.

## What is already done, and is good

`cursor/optimizer-blas-lbfgs-e452` replaced the O(N^2 n) pair loops with one BLAS
`dgemm` for the Gram matrix plus an OpenMP pair loop. Reviewed in
`kissing/REVIEW_optimizer_branch.md`:

* **18.3x faster**, measured on identical work (33803 ms -> 1850 ms).
* **Correct** — its own `KISS_SELFTEST=1` agrees with a naive reference to
  `1.6e-14`, and the reported maximum was independently recomputed from the
  written output file (matched to 1e-16, unit norms).

Do not redo that work. Build on it.

## Cheap GD hypothesis — tested, negative

The L-BFGS inner solve appears to be a **regression**, not an improvement. Every
variant of it lands worse than the crude gradient descent it replaced, and the
branch's own README notices the direction without drawing the conclusion:
*"fully converging L-BFGS at each exponent is worse (0.534) — keep the inner
solve inexact."*

The likely reason: the Riesz minimiser at finite exponent `s` is **not** a
best-packing configuration. A sloppy, under-converged descent wanders and
explores; a good inner solver converges precisely to the wrong object.

`KISS_SOLVER=gd` was run on the fixed 120000/51 hypercube benchmark and reached
0.509619958637, independently recomputed from the output file. It improves on
L-BFGS but does not recover the old 0.50519 baseline. A manifold-Adam port of the
published exponent/LR schedule reached 0.500477238629. See
`kissing/CALIBRATION_optimizer.md` for exact measurements and caveats.

## Multi-start infrastructure now available

`kissing/lib/multistart_d12_841.py` runs independent one-thread basins across
the four cores, recomputes every result from its written coordinates, retains
near-threshold candidates, fingerprints their near-contact Gram structure, and
checkpoints a JSON summary after every run. Use `--base-end 4` for the measured
`s=64` screen: over seeds 40..87, the known strong seed 51 was already at
0.537003 while all 47 other seeds were at least 0.550721. A cutoff of 0.545
therefore avoids spending the complete 120000-step budget on clearly ordinary
basins.

`kissing/lib/hypercube_classes.py` covers the signed interaction classes of the
authors' random hypercube extra point. The class omitted by the deterministic
seed was tested and was negative; do not spend more CPU simply enumerating its
2048 tied vertices. The raw-parameter Adam path and its batch-equivalent epsilon
were also bounded and negative. The measured `0.545` screen separation is an
empirical result for this fixed legacy schedule, not a safe cutoff for the
authors' different search protocol. Do not use it to prune faithful runs.

## Source-fidelity CPU preparation

The repository now has an opt-in `KISS_FAITHFUL=1` path for the published
N=841 search semantics. It keeps the first 840 seed rows exactly, replaces only
the extra row with a seeded uniform hypercube draw, optimizes raw Adam
parameters through normalized loss views, and enforces the published 13-stage,
35,000-step exponent/learning-rate schedule. It also disables legacy jitter,
relative-weight pruning, and penalty-polish fallthrough. See
`lib/FAITHFUL_841.md` and run `python3 lib/test_faithful_841.py` from this
directory.

The successful public coordinates are checked in as an attributed regression
fixture. The test independently recovers max-IP `0.499999937751...`, verifies
17-digit serialization, reproduces the authors' `%.10f` decimal-Gram to
rank-12 eigendecomposition handoff below `0.5`, and checks that invalid or
non-finite faithful states serialize no candidate. This proves the verifier and
handoff can preserve the known witness. It is **not** an independent search
recovery and does not satisfy the calibration gate above.

`STRUCTURAL_841.md` records a separate numerical audit. The published endpoint
has extensive near-antipodal structure but is globally far, under natural
labels, from both the canonical 840 core and a small theorem-valid O(4)
deformation. A paired four-seed faithful early-stage pilot consequently kept
the canonical core as the control: the O(4) arm lost all four comparisons and
neither arm approached `0.5`. O(4) remains an optional breadth arm, not the
preferred initializer (`lib/O4_BREADTH.md`).

The C path is deliberately one candidate at a time on CPU. It does not
reproduce batched GPU arithmetic. The paper reports 64 independent batches;
separately, the public source defaults to `B=512` and 100 macro repeats, or
51,200 initializations if those defaults are used. Do not conflate those two
claims. With GPU work deferred, the useful CPU deliverable is now the faithful
protocol, regression gate, structural fingerprints, and independently checked
seed/handoff tooling—not additional polishing of the current `0.5006` basin.

## Traps — please read, two of these already produced false results

1. **Never build `riesz.c` with `-ffast-math` or `-Ofast`.** They imply
   finite-math-only, which compiles away the guard `if(!(m>-1.5)) return -1.0;`.
   That guard exists because `pow(r2, -s/2)` overflows to `+inf` at large
   exponents, the configuration goes `NaN`, the maximum inner product then reads
   as `-2`, and the run reports **FEASIBLE**. This happened twice already. In a
   speed-focused rewrite it comes back as a *false record claim*.
   (`opt.c` may keep `-ffast-math`; it has no `pow`.)
2. **Do not commit the compiled binary.** A committed `riesz2` is newer than
   `riesz.c`, so `make riesz2` prints `'riesz2' is up to date` and silently does
   nothing. That is how a stale binary shadows every change you make.
3. **Never believe the optimiser's own `feasible=1`.** Recompute from the written
   `.out` file:
   ```python
   X = np.loadtxt(path); X /= np.linalg.norm(X, axis=1, keepdims=True)
   G = X @ X.T; np.fill_diagonal(G, -9); print(G.max())
   ```
   And for anything that looks like a real result, `kissing/lib/verify_exact.py`
   is the only arbiter.
4. **A numerical hit is not a result.** Beating a record requires exact
   coordinates verified in exact arithmetic. A float configuration at 0.4999 is a
   lead, not an answer — its algebraic structure still has to be identified.
5. **Do not "fix" the calibration by making it easier.** The seed, the target and
   N = 841 are the benchmark. If it cannot be beaten, say so — a *meaningful*
   negative ("this optimiser matches the published method and still cannot find
   1155 in dim 13") is a genuinely useful outcome and much better than a
   fabricated one.

## Building where there is no OpenBLAS

`riesz`/`riesz2` are build products, not committed binaries. The `Makefile`
prefers a system OpenBLAS via `pkg-config`, then falls back automatically to
`scipy-openblas32`, whose symbols are prefixed and are remapped with `-D`
flags. So on a container without `libopenblas` the whole build story is:

```bash
python3 -m pip install scipy-openblas32     # or install a system OpenBLAS
make -C kissing/lib riesz-tools             # builds riesz and riesz2
```

`make -C kissing/lib blas-check` reports whether the Riesz targets can be
built without building anything.

The BLAS-free verification tools (`mis8`, `clique`, `gclique`, `addable`,
`opt`, `opt2`, `signmis`) never needed OpenBLAS and build on their own:

```bash
make -C kissing/lib verify-tools
```

`bash kissing/lib/selftest.sh` builds `riesz2` when it is missing, and skips
only the BLAS engrad check -- still passing the rest -- if it cannot.

## Where the remaining speed is (it is not more BLAS)

`KISS_PROFILE=1` on dimension 13, N = 1155:

```
profile nfev=2916  gemm=7.324s  pairs=54.748s  CX=44.313s
```

The `dgemm` is **7%** of runtime. At `n = 13 << N = 1155` the N x N Gram matrix is
10 MB against 15k doubles of input — roughly 89x the memory traffic — so the pair
loop and the coefficient-matrix pass are memory-bound. Any further speedup has to
come from blocking, or from not materialising the full N x N matrix at all.

## Ground rules

* Four cores, ~15 GB RAM, no GPU. The container is reclaimed when the chat
  closes; a `SessionStart` hook (`.claude/hooks/session-start.sh`) rebuilds and
  resumes. Each search round is independent, so a restart costs one round.
* Do not cite anything you cannot link and state precisely.
* Report failures. Say plainly which things you tried and which did not work.
