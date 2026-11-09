# Review: `cursor/optimizer-blas-lbfgs-e452`

Reviewed at `d14bb62`, against `claude/kissing-number-dim-13-97lh52`.

**Verdict: the speedup is real and correct, one packaging defect blocks it on this
box, and the tuning conclusion in the README is not supported by the data.**

## What is right

| check | result |
| --- | --- |
| gradient correctness | their `KISS_SELFTEST=1`: naive reference vs BLAS agree, `\|dG\| = 1.6e-14` |
| honest reporting | reported `best_max_inner` recomputed independently from the written output file: matched to 1e-16, all rows unit norm |
| `-ffast-math` trap | avoided — `Makefile` uses `-O3 -fopenmp`, so the `NaN` guard survives |
| the guard itself | `if(!(m>-1.5)) return -1.0;` present in **both** loss paths |
| what gets reported | the true max Gram off-diagonal, not the smooth-max surrogate used as the loss |

**Measured speedup: 18.3x** on identical work (400 iterations, 2 continuation
stages, dim 13 N=1155): 33803 ms for the hand-rolled loops on one core vs
1850 ms for BLAS + OpenMP on four. That is essentially the whole ~19x that was
predicted to be available.

`lib/seed841.py` is a good addition on its own: it seeds the extra point at the
best hypercube vertex `(+-1)^12/sqrt(12)`, which is what Takhanov et al. do. The
previous calibration used a *random* extra point, which was simply wrong.

## What blocks it

1. **It does not build or run here.** This box has no OpenBLAS. The committed
   binary dies with `libopenblas.so.0: cannot open shared object file`, and
   `make riesz2` reports `'riesz2' is up to date` — because the committed binary
   is newer than the source, so it silently does nothing instead of failing.
   Testing it at all required hand-mapping `cblas_* -> scipy_cblas_*` against the
   OpenBLAS that ships inside numpy/scipy (`pip install scipy-openblas32`).
   Either vendor that fallback in the `Makefile`, or stop committing the binary
   so a stale artifact cannot shadow a real build.
2. **`riesz2` was overwritten** (`riesz2: riesz` / `cp -f riesz riesz2`), so the
   0.50519 baseline the branch measures itself against is no longer reproducible
   from the tree.

## Where the conclusion is wrong

The README says the remaining gap on the dim-12 841 calibration "is about the
basin ..., not the O(N^2 n) loops". Every variant of the new solver measured here
lands *worse* than the old one:

| configuration | dim-12 N=841 |
| --- | --- |
| old hand-rolled GD + Armijo (baseline) | **0.50519** |
| branch, as documented (18 short restarts) | 0.51123 |
| branch, 4 restarts at equal wall clock | 0.51113 |
| branch, finer homotopy `KISS_SMUL=1.04`, 400k steps | 0.51392 |

The regression tracks the **L-BFGS inner solve**, not the basin. The branch's own
README already sees the direction — "fully converging L-BFGS at each exponent is
worse (0.534) — keep the inner solve inexact" — but stops one step short: an
under-converged sloppy descent *explores*, while a good inner solver converges
precisely to the wrong object, because the Riesz minimiser at finite `s` is not a
best-packing configuration. Turning the inner solver up is the regression.

**Recommendation: keep the engine, drop L-BFGS.** `KISS_SOLVER=gd` is already
preserved in the code; BLAS speed with the old sloppy inner solve is the
combination that should be benchmarked, and it is the one experiment the branch
did not run.

## Where the next speedup is (not more BLAS)

Their `KISS_PROFILE=1` on dim 13, N=1155:

```
profile nfev=2916  gemm=7.324s  pairs=54.748s  CX=44.313s
```

The `dgemm` they optimised is **7%** of runtime. For `n = 13 << N = 1155` the
N x N Gram is 10 MB against 15k doubles of input — about 89x the memory traffic —
so the pair loop and the coefficient-matrix pass are memory-bound. Further gains
have to come from blocking / not materialising the full N x N matrix, not from
more BLAS.
