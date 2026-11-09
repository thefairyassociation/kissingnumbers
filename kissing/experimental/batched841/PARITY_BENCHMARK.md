# Batched 841 parity and bounded CPU benchmark

`test_parity.py` checks the experimental `kernel.py` against the existing C
implementation in `kissing/lib/riesz.c`.  It compiles
`oracle_raw_adam.c` in a temporary directory; the helper includes `riesz.c`
with its `main` renamed and calls the actual static `adam_raw_stage` one update
at a time.  No optimizer source is edited and no generated binary is tracked.

Run the short regression from the repository root after the kernel is present:

```bash
python3 kissing/experimental/batched841/test_parity.py
```

When the experimental runner is available, add its path to cross-check the
runner's schedule and normalized candidate coordinates against the same kernel:

```bash
python3 kissing/experimental/batched841/test_parity.py \
  --runner kissing/experimental/batched841/run.py
```

The regression covers:

* six B=1 raw Adam updates at `s=8`, `s=512`, and `s=40000`, retaining `m`,
  `v`, and the bias-correction step across stage boundaries;
* two updates on the real `(N, d)=(841, 12)` fixture through the unchanged C
  `adam_raw_stage`, covering the large pair matrix and cancellation path;
* an optional runner cross-check for the schedule and normalized coordinates;
* B=2 candidate losses and gradients, including the true batch mean (`1/B`)
  and one-step Adam's `sqrt(v_hat) + 1e-8` denominator;
* zero or too-small rows, non-finite inputs, and non-finite Adam moments; and
* a distinct near-coincident pair below the C `1e-12` squared-distance floor.

The near-floor probe is intentional.  C retains the derivative coefficient
`s * weight / r2` after flooring `r2`; ordinary autograd through
`torch.clamp_min` would produce a zero derivative below the floor.  The kernel
uses a straight-through floor so this source convention is tested directly.

The bounded benchmark is opt-in:

```bash
python3 kissing/experimental/batched841/test_parity.py \
  --benchmark --benchmark-steps 2
```

One run in this environment used the published 841 coordinate fixture with
Torch `2.14.0+cpu`, one CPU thread, and two updates at `s=8`:

| candidates B | total | per update | final loss |
| ---: | ---: | ---: | ---: |
| 1 | 0.029345 s | 0.014673 s | 11.0539834 |
| 2 | 0.067125 s | 0.033563 s | 11.0539834 |

The run reported `GPU available=False`.  These numbers are a short CPU timing
probe for the kernel and are machine-dependent.  No GPU timing, long search,
search recovery, or record claim is implied by this report.
