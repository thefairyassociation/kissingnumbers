# Faithful N=841 calibration mode

`riesz.c` keeps its historical seeded CPU behaviour by default: the unmarked
binary runs the geometric-homotopy gradient descent that the old committed
`riesz2` implemented, and `KISS_SOLVER=lbfgs` / `KISS_SOLVER=adam` are opt-in.
Every legacy script in `kissing/lib` pins `KISS_SOLVER=gd` explicitly so none
of them depends on that default.  The published Takhanov--Assylbekov--Yun
search semantics are available explicitly with `KISS_FAITHFUL=1`, which
selects Adam on its own:

```bash
make -C kissing/lib riesz2
python3 kissing/lib/seed841.py /tmp/cl840_841.txt --mode hypercube

KISS_FAITHFUL=1 \
KISS_THREADS=4 OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=1 \
./kissing/lib/riesz2 12 841 35000 51 /tmp/cl840_841.txt
```

Faithful search mode:

* requires `n=12`, `N=841`, a seed file, and exactly `35000` search steps;
* imports the seed's first 840 rows without Gaussian jitter;
* replaces only row 841 with a deterministic uniform `(±1)^12/sqrt(12)` draw;
* forces raw Adam on a normalized loss view;
* uses the exact published stage schedule
  `8:1000,16:1000,32:1000,64:2000,128:2000,256:2000,512:2000,
  1024:4000,2048:4000,4096:4000,10000:4000,20000:4000,40000:4000`;
* retains every term in the Riesz log-sum-exp, matching the authors' stable
  `torch.logsumexp` rather than the legacy relative-weight cutoff; and
* never enters the legacy threshold-penalty polish.

The C executable is a one-candidate CPU mode.  It does not reproduce the
authors' 512-way batched gradient or 100 macro repeats; those require the
authors' GPU program or a separate `(B,N,d)` implementation.  Thus a CPU
faithful run is a protocol/fidelity check and a bounded screening tool, not a
replacement for the full discovery experiment.

`KISS_ADAM_BASE_END` may truncate the schedule to a prefix for an explicitly
bounded diagnostic. Candidate headers record the stage interval, actual Adam
update count, and `full_schedule=0|1`; only stages 0 through 12 with 35,000
updates are labelled full. Faithful mode rejects nonzero
`KISS_ADAM_BASE_START`, because jumping to a later stage with fresh Adam
moments is not a continuation of the published optimizer state.

The authors' search-to-polish handoff is not a direct-coordinate handoff.  The
search writes the Gram matrix at `%.10f`; `polish_841.py` reloads that decimal
matrix, symmetrises it, reconstructs rank-12 coordinates by eigendecomposition,
and normalises the rows.  Prepare that handoff explicitly before using the C
polish-only path:

```bash
python3 kissing/lib/prepare_841_polish.py candidate.txt \
  --gram-out /tmp/candidate_gram_10dp.txt \
  --coordinates-out /tmp/candidate_polish_coords.txt
```

Then run the separate authors-style polish on the reconstructed coordinates
without starting a search:

```bash
KISS_FAITHFUL=1 KISS_ADAM_POLISH=1 KISS_ADAM_POLISH_ONLY=1 \
KISS_THREADS=4 OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=1 \
./kissing/lib/riesz2 12 841 35000 51 /tmp/candidate_polish_coords.txt
```

Faithful polish uses the authors' `1e-14` distance clamp.  Their `lr/10`
factor is already folded into `polish_lr[]` in `riesz.c`, which stores the
effective rates `{5e-10, 2e-10, 1e-10, 5e-11, 2e-11}`; it is not applied a
second time.  `test_faithful_841.py` measures the displacement of a single
polish update to keep that honest.  Polish returns the final state of each
stage, as `polish_841.py` does, so a bounded polish can report a slightly
worse max-IP than its input -- that is the polished configuration, not a
regression.  Set `KISS_ADAM_POLISH_STEPS` and `KISS_ADAM_POLISH_STAGES` for a
bounded run.
It runs the requested fixed schedule even when the input already has max-IP
below `0.5`, matching the purpose of the authors' margin-improving polish.
The preparer independently verifies both the decimal Gram and reconstructed
coordinates, but this remains a numerical handoff; independently recompute the
final normalized Gram maximum and use `kissing/lib/verify_exact.py` before any
claim.  C's one-candidate optimizer still differs from the authors' batched
PyTorch execution and is not bit-for-bit source exact.  Faithful stages abort
on non-finite or invalid-row states and serialize no candidate.  Never compile
this code with `-ffast-math` or `-Ofast`.

The deterministic regression checks use the published successful coordinates
in `kissing/lib/testdata/authors_841_coordinates.txt`:

```bash
python3 kissing/lib/test_faithful_841.py
```

They independently verify the `0.499999937751...` maximum, round-trip
17-digit serialization, reproduce the `%.10f` Gram/eigendecomposition handoff,
run a one-step bounded polish smoke test, and exercise non-finite aborts when
`kissing/lib/riesz2` exists.
