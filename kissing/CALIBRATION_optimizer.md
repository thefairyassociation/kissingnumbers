# Dimension-12 N=841 optimizer calibration

## Verdict

**Pass criterion not met.** The best independently recomputed maximum inner
product obtained in this pass was

```
0.5004772386288089
```

for the fixed 841-point hypercube-seeded calibration. This is a large improvement
over the previous 0.50519 result, but it is still not below 0.5 and must not be
reported as feasible.

## Fixed benchmark

```bash
python3 kissing/lib/seed841.py /tmp/kissing-scratch/cl840_841.txt --mode hypercube
./kissing/lib/riesz2 12 841 120000 51 /tmp/kissing-scratch/cl840_841.txt
```

The final file was checked independently rather than trusting `feasible`:

```python
X = np.loadtxt(path)
X /= np.linalg.norm(X, axis=1, keepdims=True)
G = X @ X.T
np.fill_diagonal(G, -9)
print(G.max())
```

All 841 rows were finite; the maximum norm error after normalization was
`2.22e-16`. The recomputed maximum was
`0.5004772386288089`.

## Measurements

| solver/configuration | independently recomputed max inner product |
| --- | ---: |
| reviewed BLAS + L-BFGS branch | 0.51123 |
| BLAS engine + projected GD, fixed 120000/51 benchmark | 0.5096199586372459 |
| BLAS engine + Adam, published exponent/LR schedule scaled to 120000 steps | **0.5004772386288089** |
| required threshold | **< 0.5** |

The Adam trajectory was basin-sensitive: repeated runs could land materially
worse because OpenMP reductions perturb the path at floating-point scale. The
0.500477 result is therefore a measured best, not a claim that every run
reproduces that number.

## What changed

* Imported the reviewed BLAS/OpenMP engine and retained `KISS_SOLVER=gd` and
  `KISS_SOLVER=lbfgs`.
* Added Adam with the exponent and learning-rate schedule published by
  Takhanov–Assylbekov–Yun.  Adam is the solver this calibration uses (`run_d12_841.sh` sets `KISS_SOLVER=adam`); it is not `riesz.c`'s default, which is the legacy geometric-homotopy GD.
* Added an automatic `scipy-openblas32` Makefile fallback without
  `-ffast-math` or `-Ofast`.
* Removed tracked `riesz` / `riesz2` binaries so stale executables cannot
  shadow source changes.
* Made the seed reader reject truncated inputs and accept the optimizer's own
  commented `.out` files.
* Preserved the independent BLAS-vs-naive self-test; measured gradient error was
  `1.55e-14`.

## Source fidelity

The published search used Adam in 64 batches and the exponent/LR schedule now
encoded in `riesz.c`; its separate ultra-high-exponent polishing stage starts
from candidates already below 0.501. See the
[paper](https://arxiv.org/abs/2606.18984) and
[authors' code](https://github.com/k-nic/841_in_12D).

No binary or float configuration is committed, and no numerical result here is
claimed as an exact kissing configuration.

## Follow-up bounded tests (2026-08-22)

The calibration still does **not** pass.  A fresh four-thread run of the fixed
seed, with the existing threshold-penalty phase disabled so the Adam basin could
be measured on its own, ended at `0.5007033607244914`:

```bash
KISS_POLISH=0 KISS_THREADS=4 OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=1 \
  ./kissing/lib/riesz2 12 841 120000 51 \
  /tmp/kissing-scratch/cl840_841.txt
```

The written 841 by 12 output was finite and, after independent row
normalisation and a fresh NumPy Gram product, reproduced
`0.5007033607244914`.  The largest row-norm error before that normalisation was
`3.33e-16`.

The cheap follow-up hypotheses were bounded and negative:

| experiment | independently recomputed max inner product / cutoff |
| --- | ---: |
| five ultra-high Riesz exponents, 20,000 Adam steps each, LR scale 100 | `0.5006379823191830` |
| local `1e-4` kick, resume published tail at `s=1024` | `0.5006912110923703` |
| local `1e-3` kick, resume published tail at `s=1024` | `0.5006912214674577` |
| local `1e-3` kick, resume earlier at `s=256` | `0.5007043564409575` |
| exact structured seed with no jitter, stopped at `s=64` | `0.552076735372` |
| jitter `0.01`, stopped at `s=64` | `0.553212520746` |
| jitter `0.05`, stopped after dominated `s=32` | `0.574265236111` |
| three-thread reduction order, stopped at `s=512` | `0.506491838310` |
| two-thread reduction order, stopped at `s=512` | `0.506449770014` |
| four-thread retry, stopped at `s=512` | `0.506348517437` |

The five-stage high-exponent run used the exponent sequence in the authors'
`polish_841.py`, but deliberately short stages and a larger learning-rate scale
to test whether this checkpoint had room to move.  It improved almost entirely
in the first stage (`0.500638863699`) and plateaued at
`0.500637982319`; it did not approach `0.5`.  All completed local-tail outputs
in the table were independently recomputed from their coordinate files.  No
internal `feasible` flag was used as evidence.

These measurements rule out polishing this particular basin harder and the
tested small local restarts.  They do not prove that the fixed calibration
cannot pass: the published search used a large batch of independently perturbed
basins, while the command here follows one extremely basin-sensitive path.

## Audited CPU multi-start follow-up

The authors' source exposes two details that a faithful continuation needs to
respect.  First, their Adam parameters are raw coordinates: a normalised view is
used in the loss, but the parameter itself is not retracted.  Second, the loss
is averaged over 512 candidates.  The latter changes the effective Adam epsilon
(`1e-8` in PyTorch is equivalent to `512e-8` after undoing the batch gradient
scale), so `riesz.c` now exposes `KISS_ADAM_EPS` for controlled fidelity tests.
The batch-equivalent raw-Adam variant was bounded and negative; it entered a
different basin but did not approach the manifold-Adam result.

The 4096 possible hypercube extras all tie under the absolute compatibility
score used by `seed841.py`.  Their *signed* inner-product multiplicities split
into three fingerprints of sizes 1024, 2048 and 1024; the first and third are
antipodal, leaving two genuine start types.  `lib/hypercube_classes.py` writes a
representative of each fingerprint.  The previously uncovered 2048-vertex type
was tested under both the published raw schedule and the longer manifold
schedule and fell into the ordinary, much worse basin.

`lib/multistart_d12_841.py` now provides the missing auditable search layer.  It
runs independent one-thread basins across the available cores, independently
normalises and recomputes every Gram maximum, retains near-threshold coordinate
files, fingerprints the near-contact structure, and rewrites a JSON checkpoint
after every completed run.  It also supports a stage-bounded screen:

```bash
python3 kissing/lib/multistart_d12_841.py \
  /tmp/kissing-scratch/cl840_841.txt \
  --start-seed 40 --runs 24 --workers 4 --threads 1 \
  --steps 120000 --base-end 4 --keep-threshold 0.545 \
  --outdir /tmp/kissing-scratch/d12-screen
```

At the end of `s=64`, seeds 40 through 87 produced 48 independently verified,
distinct numerical signatures.  Seed 51 alone entered the strong basin at
`0.5370032857173406`; the other 47 ended in
`[0.5507207182212339, 0.5567214654646567]`.  This clean separation makes
`0.545` a measured early-pruning cutoff for this fixed schedule.  A fresh full
four-thread seed-51 run, including threshold polishing, independently verified
`0.5006023521255724` with maximum pre-normalisation row-norm error `2.22e-16`.
It remains above the pass threshold.

Promoting the same screened seed with one thread produced a distinct signature
but a worse independently verified result, `0.5008491496317989` (row-norm error
`3.33e-16`). Two final bounded objective changes on that retained checkpoint
were also negative: high-exponent smooth-max-inner-product Adam did not improve
the starting maximum at all, and a fixed `KISS_PENALTY_TARGET=0.5` reduced its
hinge loss only from `8.697e-4` to `8.676e-4` without improving the maximum.
The fixed-target mode preserves its iterate between rounds; the default adaptive
threshold behaviour is unchanged.

## Source-faithful CPU mode and published-witness regression

`KISS_FAITHFUL=1` now isolates the authors-style N=841 protocol from all legacy
optimizer behavior. It requires dimension 12, 841 points, a seed file, and a
35,000-step budget; preserves the first 840 seed rows; randomizes only the final
hypercube row; uses raw Adam parameters with normalized loss views; and applies
the exact 13-stage exponent, learning-rate, and iteration schedule. Gaussian
jitter, the legacy Riesz relative-weight cutoff, and penalty-polish fallthrough
are disabled. Full invocation and limitations are in `lib/FAITHFUL_841.md`.

The public successful coordinates are retained in
`lib/testdata/authors_841_coordinates.txt` as an attributed regression fixture.
`python3 lib/test_faithful_841.py` independently verifies max-IP
`0.4999999377514321`, lossless 17-digit coordinate serialization, and the
authors' search-to-polish handoff: write the Gram matrix at `%.10f`, reload and
symmetrize it, reconstruct rank-12 coordinates, and renormalize. The resulting
maximum remains `0.4999999377601447`. Faithful C stages also reject invalid,
non-finite, and overflowed states without writing a candidate.

This is a verifier and handoff regression, not a successful independent search.
The strict pass criterion remains unrecovered. The CPU executable handles one
candidate and therefore does not reproduce the authors' batched GPU arithmetic.
The paper's 64 independent batches and the public source defaults of `B=512`
and 100 macro repeats are separate facts; only the latter implies 51,200
initializations when those defaults are used.

The structural audit in `STRUCTURAL_841.md` finds that the public endpoint is
not a small natural-label displacement of the canonical 840 core. It has 417
mutual near-antipodal pairs covering 834 points, but natural-label Procrustes
RMS displacement `0.8725625` and poor local O(4) fits. A source-faithful paired
pilot tested four canonical and four small theorem-valid O(4)-deformed cores
through exponent 64. The deformation was worse in every pair:

| seed | canonical max-IP | O(4)-deformed max-IP |
| ---: | ---: | ---: |
| 0 | 0.551982631295 | 0.555039435016 |
| 1 | 0.553314595917 | 0.555247647193 |
| 2 | 0.552171589798 | 0.555695728485 |
| 3 | 0.552429102547 | 0.554316230400 |

No pilot candidate was below `0.5`. Keep O(4) as an optional breadth arm, not a
preferred initializer; `lib/O4_BREADTH.md` documents the implementation and
the corrected comparison.

## 2026-10-02: faster engine, basin fingerprints, branching

**Pass criterion still not met.**  Best independently recomputed maximum for
the fixed 841-point calibration: `0.500164723796735` straight from the search,
and `0.5001013` after a local minimax polish of that candidate (the polish
converges there: the bottom of that basin is above 1/2).  The previous best
recorded here was `0.500477`; a rerun of the old engine in this session gave
`0.500245`.

### Engine

`lib/fastriesz.c` evaluates the same loss as `riesz.c` (log of the Riesz
s-energy on normalised rows, published exponent/LR schedule, raw or manifold
Adam).  Power-of-two exponents are computed by repeated squaring of
`r2min/r2`; for `s >= 256` a Verlet neighbour list holds every pair whose term
is within `e^-40` of the largest, rebuilt whenever `2*max displacement +
growth of the cutoff radius` exceeds the skin, so no contributing pair is ever
dropped.  `test_fast_tools.py` checks loss and gradient against a NumPy
reference at s = 8, 64, 1024 (both paths) and 40000 to `1e-10` relative;
`--check K` compares list and full evaluations during a live run (observed
`|dG|/|G| <= 2e-12` over a full schedule).  One full 35,000-step candidate
takes 28 s on one core; the old engine's equivalent is roughly 470
core-seconds.

### What the protocols actually produce

Fingerprints (`max`, fraction of pair inner products within `1e-2` of a
canonical-840 value, pairs below `-0.99`):

| configuration | max | canonical fraction | antipodal pairs |
| --- | ---: | ---: | ---: |
| canonical 840 core | 0.5 | 1.000 | 372 |
| published 841 witness | 0.49999994 | 0.724 | 387 |
| old-engine strong basin (seed 51, 3 threads) | 0.500245 | 0.721 | 388 |
| best branch, minimax-polished | 0.500101 | 0.728 | 388 |
| authors' schedule from `s=8`, exact core (24 runs) | 0.5312-0.5344 | 0.37 | 11-56 |
| same schedule entered at `s=32` ("gentle") | 0.5042-0.5096 | 0.83 | 372 |

* The authors' schedule run one candidate at a time on CPU **melts** the 840
  core: 20 of 24 runs ended in `[0.53412, 0.53436]`, the best at `0.53116`.
* Entering the same schedule at a higher exponent keeps the core intact but
  **uniformly strained**: tens of thousands of contacts all sit at `0.5042`.
  A hinge polish at `t = 1/2` then trades those small violations for a single
  large one (`E = 0.0298`, max `0.5398`): a dead end.
* The rare strong basin of the legacy protocol (manifold Adam, jitter 0.03,
  120000-step scaled schedule) has **the same fingerprint as the published
  witness**.  It is the right family; the problem is the member.
* A fresh 48-seed screen of the legacy protocol with `fastriesz` (seeds 0..47)
  found **no** strong basin (s=64 maxima `0.55016 .. 0.55789`), consistent
  with the earlier 1-in-48 estimate.

### Branching

Seed 51's state after `s=16` (one thread, old engine) was re-run with
`fastriesz --start-stage 2 --jit sigma` (fresh Adam moments).  Branches with
`sigma` in `1e-3 .. 1e-2` stay in the strong family essentially always; the
s=512 value already predicts the final one, so `branch_search.py` screens
branches there against the running median.  Over the two runs (about 40
finished branches) the finals fall on **discrete levels**, e.g.
`0.500165-0.500188` (three hits), `0.500269`, `0.500354-0.500369`,
`0.50119-0.50128`, `0.5014-0.5015`, `0.5018-0.5020`.  Branches from `s=64`
states land next to their parents.  The best level of this root is above
1/2.

### Feasibility and basin bottoms

`hingepol` (L-BFGS on `sum (g - t)_+^2`) decides feasibility: on the
0.500245 basin it converges to `E = 8.98e-5 > 0` (same value as SciPy's
L-BFGS), so that basin holds no configuration with max `<= 1/2`.
`minimax_polish.py` (hinge continuation `t = max - delta`) gives basin
bottoms: `0.500245 -> 0.500236`, `0.500165 -> 0.5001013`.

A sequential-LP minimax polish was also tried and dropped: at N = 841 the
linearised problem has about 15,000 nearly tight rows and HiGHS needed over
100,000 dual-simplex pivots (and IPX stalled in its basis preconditioner) for
a single step.

### Certification

`lib/certify_float.py` turns any float configuration with max cosine
*strictly* below 1/2 into an exact proof (integer rounding at `2^40`, exact
test `4<x,y>^2 <= (1-2d)^2 |x|^2 |y|^2`).  The published witness certifies with
`d = 6e-8`; the certificate is `dim12/configs/takhanov841_certified.json`.
So a numerical hit below 1/2 -- in any dimension -- is immediately provable;
only tight configurations still need algebraic coordinates.

### Multi-root search and the 0.50010 floor (end of the 2026-10-02 session)

The first exponent stage either melts the core into a disordered state
(110-240 pairs with cosine below -0.99 after `s = 8`) or re-crystallises it
into a new antipodal-rich structure (401-406 pairs; the canonical core has
372).  Only re-crystallised starts reach the witness's family, and the split
is visible after 6 s, long before the maximum inner product shows anything
(about 0.60 either way).  Entering the schedule at `s = 16` instead keeps the
original core rigid (exactly 372 pairs) and leads to the strained 0.504 family.

Screen used for the final run (jitter 0.005 raises the re-crystallisation rate
from about 1/48 at the old 0.03 to about 1/20):

```bash
fastriesz 12 841 cl840_841.txt ROOTS --extra file --adam manifold \
  --scale 3.4285714 --jit 0.005 --stages 2 --screen-anti 1:300 \
  --save-stage 2 --keep 0 --seeds 2000 2600
```

550 seeds gave 28 re-crystallised roots (5.1%).  Together with seed 51 and
the five roots of an earlier batch, `branch_search.py` ran 92 branch jobs over
38 parents (61 completed, 31 stopped at the `s = 512` screen).  Best level per
root, after the local minimax polish:

| root | branches | best max inner product |
| --- | ---: | ---: |
| seed 51 (old engine, one thread) | ~70 over the session | **0.5001014** |
| fastriesz seed 2280 | 3 | 0.5001039 |
| fastriesz seed 2055 | 4 | 0.5001406 |
| fastriesz seed 2180 | 2 | 0.5001527 |
| fastriesz seed 2399 | 1 | 0.5002536 |
| other roots | 1-17 | 0.50026 - 0.503 |

**No branch went below 1/2.**  Independent roots reach the same floor,
about `0.50010`, within a few branches, which suggests a common defect type
for this family.  The published witness (0.49999994) is in the same family by
fingerprint, so it must use a different, rarer defect arrangement.  That is
the open question the calibration now reduces to.

The best result reproduces exactly from these commands (one thread, so
deterministic; checked end to end at the end of the session):

```bash
python3 kissing/lib/seed841.py cl840_841.txt --mode hypercube
KISS_ADAM_BASE_END=2 KISS_POLISH=0 KISS_SOLVER=adam KISS_LOSS=riesz KISS_INNER=16 \
  KISS_JIT=0.03 KISS_THREADS=1 OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
  ./kissing/lib/riesz2 12 841 120000 51 cl840_841.txt        # root: s=16 state
OPENBLAS_NUM_THREADS=1 ./kissing/lib/fastriesz 12 841 cl840_841.txt.riesz.s51.out OUT \
  --extra file --adam manifold --scale 3.4285714 --start-stage 2 --jit 0.001 \
  --save-all --seeds 11000008 11000009                      # best 0.500165121392458
python3 kissing/lib/minimax_polish.py OUT/cand_n12_N841_s11000008.txt polished.txt \
  --rounds 30                                               # -> 0.500101436054909
```

Following this repository's convention, no float configuration from the
search is committed; the commands above regenerate it.

Things tried and dropped in this session: a remove-k/reinsert surgery
(`surgery.py --track-max`) on the best basin (13 moves, closest 0.5001016,
never below the 0.5001014 it started from); branching from the `s = 8` state
with larger kicks (10 runs, 0.50076-0.50229); truncating the schedule at
`s = 4096` and polishing instead (same level, but the polish costs more than
the truncated tail saves); lower first-stage learning rates (no effect on the
re-crystallisation rate).
