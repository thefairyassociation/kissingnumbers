# O(4) breadth initializer: fidelity implementation and bounded pilot

This change is based on commit `62cf6d2` (`Add opt-in faithful N=841
optimizer mode`).  It was developed in a fresh isolated checkout of that
commit.  No `repo_audit` files were used or modified.

## Implementation

* `o4_breadth.py` is repo-native and imports the construction from
  `kissing/dim12/constructions/clebsch840.py`, so it works from a normal
  checkout.  It needs only NumPy.  It generates canonical
  or Theorem-2 O(4)-deformed 840 cores, audits the complete core before adding
  row 841, and emits the existing plain-text `841 x 12` seed format.
* The default extra mode is `uniform-random`, with a recorded hypercube index
  and metadata sidecar.  `deterministic-best` remains available only as an
  explicitly labelled legacy diagnostic mode.
* `riesz.c` now records the actual faithful C-side uniform-random extra index
  both in stderr and in the candidate header.  Faithful mode itself already
  replaces the input's final row with this extra, matching the authors' search
  semantics.
* `verify_o4_841.py` is an independent read-back verifier.  It recomputes the
  core and full 841 Gram maxima, contact tails, histograms, and basin
  fingerprints without importing the generator.
* `run_o4_faithful_pilot.py` generates the seeds, invokes
  `KISS_FAITHFUL=1`, uses exactly `35000` steps, supports `--base-end 4` or
  `--base-end 6`, records the actual C-side uniform extra, and independently
  verifies every output.
* `test_o4_breadth.py` covers identity equivalence, theorem row-`l1` bounds,
  direct 840-core validity, extra-mode metadata, seed round-trip, and the
  faithful C source contract.  `--binary-smoke` additionally executes a
  stage-4 faithful run.

## Self-tests

From the repository root:

```bash
make -C kissing/lib riesz2
python3 kissing/lib/test_o4_breadth.py
python3 kissing/lib/test_o4_breadth.py --binary-smoke
python3 kissing/lib/test_faithful_841.py --skip-binary
```

Observed results:

```text
identity equivalence: PASS
theorem row-l1 and direct 840-core audits: PASS
uniform-random and legacy deterministic-best extra modes: PASS
841 coordinate + metadata round-trip: PASS
faithful C source contract: PASS
faithful binary stage-4 smoke: PASS
PASS: O(4) breadth self-tests
```

The identity test is exact at float serialization level:
`deformed_core(I, I)` and the checked-in canonical 840 array are identical.
The four test deformations all satisfy the conservative row-`l1` limit
`1.055 < 3/(2 sqrt(2)) = 1.0606601717798212` and independently recompute to
maximum inner product at most `0.5000000000000003`.

## Source-faithful pilot

Only one small pilot was run: four canonical controls and four deformed cores,
seeds `0,1,2,3`, O(4) scale `0.006`, one OpenMP thread, no polish, and
`KISS_ADAM_BASE_END=4` (stages `s=8,16,32,64`).  Every job used the exact
35,000-step guard:

```bash
python3 kissing/lib/run_o4_faithful_pilot.py \
  --outdir kissing/logs/o4-faithful-pilot \
  --base-end 4 --threads 1 --seeds 0 1 2 3 \
  --arms canonical deformed --scale 0.006
```

The runner set:

```text
KISS_FAITHFUL=1
KISS_SOLVER=adam
KISS_LOSS=riesz
KISS_POLISH=0
KISS_ADAM_BASE_END=4
STEPS=35000
KISS_JIT unset (faithful mode: no jitter)
```

Because this is a four-stage prefix, each candidate header records
`search_stage_start=0`, `search_stage_end=4`, `search_updates=5000`, and
`full_schedule=0`. It is deliberately not labelled as a completed 35,000-update
search.

The faithful stage evaluation counts were exactly `1002, 2003, 3004, 5005`
(the initial evaluation plus the cumulative schedule work), confirming that
the pilot did not silently scale the schedule down to the old legacy
`steps/35000` behavior.  The four canonical controls had distinct actual
faithful uniform-extra indices `3250, 3702, 4095, 2564`; therefore these are
not four OpenMP-reduction repeats of one identical initialization.  The same
optimizer seeds intentionally give the same extra indices in the paired
deformed runs, isolating the core deformation.

All eight outputs were read back by `verify_o4_841.py`.  They were finite,
had maximum pre-normalization row-norm error at most `4.44e-16`, and produced
eight distinct full-configuration basin fingerprints.  No candidate was below
`0.5`.

| seed | canonical max-IP | deformed max-IP | deformed − canonical | faithful extra index |
|---:|---:|---:|---:|---:|
| 0 | `0.551982631295` | `0.555039435016` | `+0.003056803721` | `3250` |
| 1 | `0.553314595917` | `0.555247647193` | `+0.001933051277` | `3702` |
| 2 | `0.552171589798` | `0.555695728485` | `+0.003524138687` | `4095` |
| 3 | `0.552429102547` | `0.554316230400` | `+0.001887127853` | `2564` |

The deformed arm lost all four paired comparisons in this source-faithful
early-stage pilot.  This updates the earlier exploratory conclusion: the
previous legacy pilot's apparent O(4) advantage was hypothesis-generating
only.  It used manifold Adam, a non-faithful shortened schedule, and legacy
extra handling; it should not be used to prioritize O(4) for the authors'
search.  O(4) remains a valid optional breadth arm and is worth retaining for
future GPU testing, but the canonical control should remain primary and the
deformation is not currently an evidence-backed improvement.

The `--base-end 6` option is implemented for a later bounded follow-up, but was
not run here in order to keep this correction to one 4-vs-4 pilot as requested.

## Artifact location

The complete pilot logs, seed metadata, candidate coordinates, and independent
verification JSON are retained outside the commit in the scratch artifact
directory.  They are diagnostic outputs only and are intentionally not added
to Git.
