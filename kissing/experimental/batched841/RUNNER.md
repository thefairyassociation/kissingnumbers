# Batched 841 runner

`run.py` is an opt in, bounded driver for four (or another explicitly chosen
batch size) independent N=841 candidates. It keeps the first 840 rows of the
seed file unchanged at initialization and draws one fresh uniform
`(±1)^12/sqrt(12)` row per candidate. Each macro repeat starts from that fixed
core with fresh Adam moments. Moments continue through all 13 stages within a
macro. The kernel uses the batch mean loss and updates raw coordinates through
the normalized loss view.

The kernel's `SEARCH_SCHEDULE` is checked against the published 35,000 update
schedule. A full run must be explicit:

```bash
python3 kissing/experimental/batched841/run.py \
  --seed-file /tmp/cl840_841.txt --seed 51 \
  --batch-size 4 --macro-repeats 1 --device cpu --dtype float64 \
  --full --checkpoint /tmp/batched841-51 \
  --output /tmp/batched841-51.txt
```

For a safe smoke run or a bounded diagnostic, pass `--max-updates` instead of
`--full`:

```bash
python3 kissing/experimental/batched841/run.py \
  --max-updates 4 --checkpoint /tmp/batched841-smoke \
  --output /tmp/batched841-smoke.txt
```

Resume with the same immutable configuration and a stopping bound at least as
large as the checkpoint cursor:

```bash
python3 kissing/experimental/batched841/run.py \
  --max-updates 35000 --resume --checkpoint /tmp/batched841-smoke \
  --output /tmp/batched841-smoke-resumed.txt
```

The checkpoint is a single atomically replaced NPZ. It stores `raw`, `m`, `v`,
`step`, the torch generator state, the macro/stage/in-stage cursor, and a JSON
metadata payload. `allow_pickle=False` is always used on readback. Seed digest,
shape, dtype, finite values, nonnegative second moments, cursor arithmetic,
RNG device, kernel checksum, PyTorch version, and immutable configuration are checked before resume. Increasing
the stopping bound is allowed; rewinding it or changing the seed, batch size,
macro count, device, dtype, thread count, or schedule is refused.

The output is normalized and immediately reopened for an independent NumPy
Gram recomputation. Its JSON sidecar labels a complete schedule as `full` and
a bounded run as `diagnostic`; it contains no exact or record claim. The
runner does not perform the authors' decimal `%.10f` Gram handoff or polish.
Use `kissing/lib/prepare_841_polish.py` between a completed search and any
separate polish invocation.

Every CLI run requires an explicit `--full` or `--max-updates` choice. A bound
of 35,000 updates can itself cover a complete macro. Existing output and
checkpoint paths are refused on a fresh invocation to prevent accidental
replacement.
