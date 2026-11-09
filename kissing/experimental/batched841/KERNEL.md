# Batched faithful 841 kernel

`kernel.py` provides the small functional core needed to run independent
faithful 841 candidates as one float64 PyTorch batch.  Inputs have shape
`(B, N, d)`, and the caller chooses CPU or CUDA by placing the input tensor on
the desired device.  The exported schedule is the exact 13-stage schedule in
`kissing/lib/riesz.c`:

```python
from kissing.experimental.batched841 import (
    SEARCH_SCHEDULE,
    adam_step,
    initial_state,
)

state = initial_state(raw_float64_batch)
for s, steps, lr in SEARCH_SCHEDULE:
    for _ in range(steps):
        adam_step(state, s, lr)
```

Each loss normalises rows to a view, sums only the upper triangle `i < j`, and
uses `logsumexp(-(s/2) * log(clamp(||zi-zj||², min=1e-12)))`.  The scalar is
the mean across candidates, so the returned gradient is divided by `B` once.
Adam moments persist across stages.  Updates apply directly to raw coordinates,
with no normalisation or retraction after a step, and use beta `(0.9, 0.999)`
and epsilon `1e-8`.

The clamp follows the current C implementation's value *and hand-coded
 derivative*: a clamped pair still receives `s * weight / r2` in the derivative.
Vanilla autograd through `torch.clamp` would instead give zero derivative while
the clamp is active.  The kernel computes the derivative explicitly to retain
C parity.  Tests therefore compare against an independent C-style oracle and
use finite differences on candidates whose pair distances are above the floor.
Inputs with NaN, infinity, a zero or too-small row (norm at most `1e-12`),
invalid shape, or non-float64 dtype are rejected.

The protocol provenance is the public [`search_841_riesz.py`](https://github.com/k-nic/841_in_12D/blob/main/search_841_riesz.py) source, read at blob `2a3e8ce119c20ddc3aedf00f44f8600ad91d8e23`, together with the unchanged local [faithful protocol](../../lib/FAITHFUL_841.md) and [C implementation](../../lib/riesz.c).
