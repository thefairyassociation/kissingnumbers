# Where this project stands (plain English)

*Last updated 2026-11-10. Technical detail: [`lower_bound/README.md`](lower_bound/README.md)
and [`upper_bound/README.md`](upper_bound/README.md). What happened to the old
branches and pull requests: [`docs/HISTORY.md`](docs/HISTORY.md).*

## The problem

The **kissing number** in dimension *n* is the largest number of unit spheres
that can all touch one central unit sphere without overlapping. Equivalently:
the most unit vectors in Rⁿ with every pairwise inner product at most 1/2.
The exact answer is known only in dimensions 1, 2, 3, 4, 8 and 24. In every
other dimension there is a range: someone has *built* this many, and someone
has *proved* you cannot have more than that many.

The repository attacks both ends, in two separate directories:

* `lower_bound/` — build a configuration with more spheres than the record;
* `upper_bound/` — prove a smaller ceiling than the record.

| dim | record lower | record upper | best verified here (lower / upper) |
| ---: | ---: | ---: | --- |
| 12 | 841 | 1355 | 841 (the published one, certified exactly) / 1416 |
| 13 | 1154 | 2064 | 1154 (reproduction) / 2233 |
| 14 | 1932 | 3174 | 1932 (reproduction) / 3492 |

**No record has been beaten, in either direction. The problem is not
solved.** That is the honest headline. What follows is what was actually
gained.

## What changed in the 2026-11-09/10 session

### 1. The repository is one tree again

The work had been spread over 17 side branches and 7 open pull requests, most of it
never merged; `main` held the first two days only. Everything is now merged
into a single branch with the full history of each, split into
`lower_bound/` and `upper_bound/`. Stray output files, committed binaries, a
broken session hook and thirteen single-use workflow files are gone. Nothing
was deleted on GitHub: `docs/HISTORY.md` lists every branch and PR, what it
contained, and why it is now safe to close.

### 2. Lower bound: one open question closed by proof

Dimension 13's record (1154) is built from layers; two of them have 168 points
each. An earlier agent set up a search for a 169-point layer inside a natural
finite family of 4248 candidate directions — one more point there would have
given 1155 and a new record. The search ran out of time, twice, without an
answer.

It turns out no search is needed. A short argument, checked by a script in
exact arithmetic in about a second, proves **the maximum is exactly 168**
(`lower_bound/dim13/prove_joint_layer_168.py`). That route to 1155 is closed
for good, rather than left as "timed out".

### 3. Lower bound: the numerical search now runs at a hundred times the scale

The main open front is numerical: start from a record configuration plus one
extra point and let an optimiser try to make room. The test case is dimension
12 with 841 points, where a solution is *known* to exist; until the pipeline
can rediscover one there ("the calibration"), its failures in dimension 13
mean nothing. The October session got to 0.5001014 (needs to be below 0.5)
using about 2 CPU-hours.

This session wrapped that pipeline into a single driver and one reusable
GitHub Actions workflow, which runs it on 20 free machines in parallel and
commits a verified summary back to the repository.

The first full-scale run (`lower_bound/campaigns/c02-scale/`, 340 CPU-hours):

| run | starts tried | usable starts | best max inner product (need < 0.5) |
| --- | ---: | ---: | ---: |
| October session | 550 | 28 | 0.5001014 |
| campaign, same recipe | 28,440 | 1,200 | 0.5000897 |
| campaign, extra point placed at random | 10,480 | 358 | **0.5000060** |

What that says:

* **The calibration is still not passed.** Nothing went below 1/2, so nothing
  was certified.
* The "floor" near 0.5001 that October's 28 starts suggested is real: almost
  every one of 1,558 starts ends there.
* But the floor has a tail. One start came **16 times closer** than anything
  before (0.5000060; its basin bottoms out at 0.5000059 after local
  polishing and re-branching, so that particular one is not a solution).
  The published 841 sits at 0.49999994. Rare starts do better, which is what
  one would expect if the published configuration is a rare one, and it means
  more starts of the right kind are worth running rather than more tuning.
* Placing the extra point at random each time finds these rarer outcomes;
  the fixed placement did not in three times as many starts.
* Dimension 14 (first numerical attempt here at 1933 points): the record
  configuration survives the first stage intact, but 296 full runs all end
  near 0.519. Not close.
* Dimension 13 did not run in that campaign (a missing dependency in the
  workflow, since fixed).

A second campaign, `c03-tail`, was started on 2026-11-10 with every dimension-12
shard on the random-placement recipe, an extra step that re-branches around any
start that lands below 0.50004, and the dimension-13 arm repaired. Its record
will appear in `lower_bound/campaigns/c03-tail/` when it finishes.

### 4. Upper bound: a new, separate track with machine-checked proofs

`upper_bound/` did not exist before. It now contains:

* **Exact certificates for the classical linear-programming bounds in
  dimensions 3 to 24**, each a small file of fractions that a
  standard-library-only program verifies with no floating point at all. They
  include the two famous sharp cases: exactly **240** in dimension 8 and
  exactly **196560** in dimension 24.
* **Exact certificates that k(3) ≤ 12 and k(4) ≤ 24** from the stronger
  "three-point" semidefinite method. Linear programming cannot do either (it
  stops at 13 and 25). Together with the known 12- and 24-point
  configurations these are complete machine-checked determinations of the
  kissing number in dimensions 3 and 4 — the second is Musin's 2008 theorem —
  resting on two classical positivity theorems that the verifier states.
  The pipeline that produced them — solve numerically, round to fractions,
  repair the identities exactly, verify independently — is the same one the
  published records in higher dimensions rest on, at small scale.

All of these are known theorems. Nothing here lowers a record. The published
upper bounds in dimensions 9–23 come from the same three-point method at far
higher degree; `upper_bound/README.md` lists what is missing to get there
(symmetry reduction first).

## What is solid and what is not

* **Solid:** the exact reproductions of the three record configurations; the
  exact certificate of the published 841; the structural results in
  `lower_bound/README.md` (the 1154 and the 1932 cannot be extended by one
  point, cannot be improved inside their own frameworks, and so on); the new
  168 proof; every upper-bound certificate.
* **Not evidence either way:** every "the optimiser did not find it" result,
  in any dimension, until the dimension-12 calibration passes.
* **Timed out, not excluded:** the one-tetrad-deletion searches in dimension
  13 (`lower_bound/dim13/README.md`).

## How to check things yourself

```bash
python3 -m pip install numpy scipy scipy-openblas32 sympy
```

```bash
make -C lower_bound/lib all
```

```bash
python3 lower_bound/lib/verify_exact.py lower_bound/dim13/configs/ze99_1154_exact.json
```

```bash
python3 lower_bound/lib/certify_float.py lower_bound/lib/testdata/authors_841_coordinates.txt
```

```bash
python3 lower_bound/dim13/prove_joint_layer_168.py
```

```bash
python3 upper_bound/lp/verify_certificate.py upper_bound/certificates/lp/*.json
```

```bash
python3 upper_bound/sdp/verify_certificate.py upper_bound/certificates/sdp/*.json
```

## What to do next

1. **Read campaign `c03-tail`.** If it certifies a configuration below 1/2 in
   dimension 12, the calibration is passed and the same pipeline should be
   pointed at dimension 13 at full scale. If not, compare how many starts fall
   in the tail with c02 (1 in 358) to judge whether a tenfold larger run is
   worthwhile.
2. **Settle the tetrad-deletion question in dimension 13 by argument, not by
   search.** It is the last "timed out" entry in `lower_bound/dim13/README.md`.
   The 168 proof shows the style: bound the sign-vector code by counting the
   subcubes each extra direction forbids. The deletion family has six more
   direction types, so the case analysis is longer, but it is the same game.
3. **Upper bound: divide out the symmetry in the three-point program.** That is
   what stands between the current code and higher degrees; see "What it
   would take" in `upper_bound/README.md`.
4. **On GitHub (owner's call):** merge the consolidated branch into `main`,
   then close PRs #4–#10 and delete the old branches. `docs/HISTORY.md`
   explains why that loses nothing.

## Housekeeping

* Compiled tools are build products and are git-ignored; `make` rebuilds
  them. Never compile the optimisers with `-ffast-math`
  (`lower_bound/docs/HANDOFF_optimizer.md`, trap 1).
* Long searches write to `lower_bound/runs/` (git-ignored) or run as
  campaigns; only summaries and certified results are committed.
* To start a campaign, edit `.github/search-campaign.json` and push. Hosted
  runners are free for a public repository; a campaign of 20 five-hour shards
  uses about 100 runner-hours.
