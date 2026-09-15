# Additive search and verification tools

These opt-in tools extend the 841-point calibration workflow. They do not
change any existing solver, initializer, certificate, result, or default.
See [the repository review](REVIEW_2026_09_15.md) for scope and limitations.

## Exact verification of decimal directions

This checker needs only Python's standard library. It treats each coordinate
string as an exact rational, normalizes each direction algebraically, and
checks every pair using Python integers. No floating-point tolerance is used.

```bash
python3 kissing/experimental/decimal_certificate/check.py \
  kissing/lib/testdata/authors_841_coordinates.txt \
  --expected-dimension 12 --expected-count 841 --strict --compact
```

The existing published fixture passes all 353,220 pairs. This reproduces a
known configuration; it is not a new search recovery or lower bound. The
checker can also be applied to future finite-decimal candidate files. A
candidate still needs the intended count and dimension; a certificate alone
does not establish that a construction is new.

See [the exact comparison and input format](decimal_certificate/README.md).

## Batched raw Adam

The new [kernel](batched841/KERNEL.md) uses float64 PyTorch tensors of shape
`(B, N, d)`. It preserves the existing C search schedule, raw updates,
within-macro Adam moments, and batch-mean gradient scaling. Its active
distance-clamp derivative follows C, with the documented difference from
ordinary upstream PyTorch autograd at that boundary.

PyTorch and NumPy are optional dependencies for this experiment. CPU validation
used PyTorch `2.14.0+cpu` and NumPy `2.3.5`. The C parity harness also needs a C
compiler and OpenBLAS, using the existing build's system or
`scipy-openblas32` configuration. No compiled binaries are committed.

GPU execution and discovery-scale throughput have not been validated here.
Small CPU agreement checks cannot predict successful 841-point recovery.
The published fixture remains a verification fixture, never a discovery seed
for claiming independent recovery.

The [resumable runner](batched841/RUNNER.md) starts from the existing canonical
840-point core and draws a fresh random extra row for each candidate. A short
run and continuation, using fresh output paths, are:

```bash
python3 kissing/experimental/batched841/run.py \
  --seed 51 --batch-size 4 --max-updates 20 \
  --checkpoint /tmp/batched841-run --output /tmp/batched841-first.txt
python3 kissing/experimental/batched841/run.py \
  --seed 51 --batch-size 4 --max-updates 40 --resume \
  --checkpoint /tmp/batched841-run --output /tmp/batched841-resumed.txt
```

The stopping bound is cumulative, per candidate. Checkpoints retain raw
coordinates, both Adam moments, the update position, and random-number state
in one atomic NPZ file. Exported coordinates are reopened and independently
checked with NumPy. Use the existing decimal-Gram preparer before separate
polishing; this runner does not perform the handoff or polish itself.

## Checks

Run from the repository root:

```bash
python3 -m unittest discover -s kissing/experimental/decimal_certificate -p 'test_*.py'
python3 -m unittest kissing.experimental.batched841.test_kernel
python3 -m unittest kissing.experimental.batched841.test_runner kissing.experimental.batched841.test_runner_integrity
python3 kissing/experimental/batched841/test_parity.py
```

The historical regression remains available separately:

```bash
make -C kissing/lib riesz-tools
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 python3 kissing/lib/test_faithful_841.py
```
