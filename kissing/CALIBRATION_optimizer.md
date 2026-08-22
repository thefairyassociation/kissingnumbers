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
