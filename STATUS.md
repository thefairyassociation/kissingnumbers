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
| 12 | 841 | 1355 | 841 (found independently here, certified exactly; ties the record) / 1416 |
| 13 | 1154 | 2064 | 1154 (reproduction) / 2233 |
| 14 | 1932 | 3174 | 1932 (reproduction) / 3492 |

**No record has been beaten, in either direction. The problem is not
solved.** That is the honest headline. The best single result is that the
search pipeline here has now **found an 841-point configuration in dimension
12 by itself** — equal to the record, not above it. What follows is what was
actually gained.

## What changed in the 2026-11-09/10 session

### 1. The repository is one tree again

The work had been spread over 17 side branches and 7 open pull requests, most of it
never merged; `main` held the first two days only. Everything is now merged
into a single branch with the full history of each, split into
`lower_bound/` and `upper_bound/`. Stray output files, committed binaries, a
broken session hook and thirteen single-use workflow files are gone. Nothing
was deleted on GitHub: `docs/HISTORY.md` lists every branch and PR, what it
contained, and why it is now safe to close.

### 2. Lower bound: two open questions closed by proof

Dimension 13's record (1154) is built from layers; two of them have 168 points
each. An earlier agent set up a search for a 169-point layer inside a natural
finite family of 4248 candidate directions — one more point there would have
given 1155 and a new record. The search ran out of time, twice, without an
answer.

It turns out no search is needed. A short argument, checked by a script in
exact arithmetic in about a second, proves **the maximum is exactly 168**
(`lower_bound/dim13/prove_joint_layer_168.py`). That route to 1155 is closed
for good, rather than left as "timed out".

The last "timed out" search in that directory went the same way on
2026-11-10. It asked: if one of the 51 building blocks of the record is
*removed* (losing 16 points), does that free enough room for a layer of 177
points instead of 168, which would more than pay for the loss? A second
argument of the same style, `lower_bound/dim13/prove_drop_tetrad_176.py`,
shows the answer is no: after any such removal a layer holds **at most 176**,
and 176 only brings the total back to exactly 1154. This one is not a
pencil-and-paper proof: it reduces the question to 28 small combinatorial
problems that two unrelated solvers (CP-SAT and HiGHS) bound in the same
way, plus four classical values from coding theory. It is a
computer-assisted bound, with the record in
`lower_bound/dim13/results/ze99_drop_tetrad/BOUND.json`.

### 3. Lower bound: the numerical search passed its test — it found an 841

The main open front is numerical: start from a record configuration plus one
extra point and let an optimiser try to make room. The test case is dimension
12 with 841 points, where a solution is *known* to exist; until the pipeline
could rediscover one there ("the calibration"), its failures in dimension 13
meant nothing. The October session got to 0.5001014 (needs to be below 0.5)
using about 2 CPU-hours.

This session wrapped that pipeline into a single driver and one reusable
GitHub Actions workflow, which runs it on 20 free machines in parallel and
commits a verified summary back to the repository.

Two full-scale runs followed (`lower_bound/campaigns/c02-scale/` and
`c03-tail/`, about 340 and 400 CPU-hours):

| run | starts tried | usable starts | best max inner product (need < 0.5) |
| --- | ---: | ---: | ---: |
| October session | 550 | 28 | 0.5001014 |
| c02, same recipe | 28,440 | 1,200 | 0.5000897 |
| c02, extra point placed at random | 10,480 | 358 | 0.5000060 |
| c03, extra point placed at random | 48,160 | 1,685 | **0.4999968 — certified** |

What that says:

* **The calibration is passed.** Campaign c03 produced an 841-point
  configuration in dimension 12 whose largest inner product is 0.4999968. The
  exact certifier accepts it (all 353,220 pairs at most 1/2 − 3·10⁻⁶, proved in
  integer arithmetic), on the GitHub runner and again locally. It is stored as
  `lower_bound/dim12/configs/found_841_c03.txt` with its certificate beside it.
* **It ties the record; it does not beat it.** 841 is the known lower bound.
  What is new is only that this repository's own pipeline now reaches it from
  the classical 840 with no help from the published coordinates.
* **It is not the published configuration.** It has 378 near-opposite pairs of
  points where the published one has 387, so no rotation maps one to the
  other, and it clears 1/2 by about fifty times as much (3·10⁻⁶ against
  6·10⁻⁸). So there is more than one 841.
* **It is rare.** Out of 1,685 usable starts, a single batch got there (two
  certified outputs, close variants of each other; the better one is the
  stored file). Almost every other start ends on the "floor" near
  0.5001; across both campaigns about 1 start in 500 lands within 10⁻⁵ of
  1/2. This is why the October runs, with 28 starts, could not have found it.
* Placing the extra point at random each time is what reaches these rare
  outcomes; the fixed placement did not in 1,200 starts.
* **Dimension 13 (1155 points) got no usable starts in c03**: 13,200 tries, 0
  kept. The first optimisation stage that makes dimension 12 work (it lets
  the 840 melt and re-form) destroys the 1154 completely, every time, and it
  never re-forms. Entering one stage later keeps about three quarters of the
  structure, two stages later keeps all of it; the driver can now start there
  (`--start-stage`), and campaign c04 uses that.
* Dimension 14 (first numerical attempt here at 1933 points, c02): the record
  configuration survives the first stage intact, but 296 full runs all end
  near 0.519. Not close.

Campaign `c04-beyond`, started 2026-11-10, points the now-validated pipeline
at the actual open questions: **842 points in dimension 12** starting from the
new 841, and **1155 points in dimension 13** from the two later entry stages.
Its record will appear in `lower_bound/campaigns/c04-beyond/`. Nobody should
expect a hit: the authors of the 841 report that their own 842 attempts
stalled at 0.5009.

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
  exact certificates of the published 841 and of the 841 found here; the
  structural results in
  `lower_bound/README.md` (the 1154 and the 1932 cannot be extended by one
  point, cannot be improved inside their own frameworks, and so on); the new
  168 proof; every upper-bound certificate.
* **Solid if you trust two independent solvers:** the 176 bound for the
  tetrad-deletion family.
* **Weak evidence at most:** every "the optimiser did not find it" result.
  The calibration shows the pipeline *can* find a rare solution in dimension
  12 given about 1,700 good starts; it says nothing about how many starts
  dimension 13 would need, and there the starts are of a different kind.
* **Timed out, not excluded:** nothing in dimension 13 any more. Both
  searches that had timed out are now replaced by proofs.

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
python3 lower_bound/lib/certify_float.py lower_bound/dim12/configs/found_841_c03.txt
```

```bash
python3 upper_bound/lp/verify_certificate.py upper_bound/certificates/lp/*.json
```

```bash
python3 upper_bound/sdp/verify_certificate.py upper_bound/certificates/sdp/*.json
```

## What to do next

1. **Read campaign `c04-beyond`** (842 points in dimension 12; 1155 points in
   dimension 13 from the later entry stages). A certified hit in either arm
   would be a new record and must be re-certified locally before anyone says
   so. The likely outcome is a set of best values above 1/2; compare the
   dimension-13 ones with 0.506587, the best so far.
2. **Find a way to get "good starts" in dimension 13.** Dimension 12 works
   because the 840 melts and re-forms into a new ordered structure with room
   for one more point. The 1154 never re-forms. Without an analogue of that
   step, more computing time in dimension 13 is unlikely to help. Ideas not
   yet tried: melt only one layer and hold the rest; start from the 1146-point
   PackingStar configurations instead; remove a few points before adding more.
3. **Understand the new 841.** It differs from the published one. Reading off
   its structure (which of the 840's pairs survive, where the extra point
   sits) might show how to make an 842, or how to write an 841 down exactly.
4. **Upper bound: divide out the symmetry in the three-point program.** That is
   what stands between the current code and higher degrees; see "What it
   would take" in `upper_bound/README.md`.
5. **On GitHub (owner's call):** merge the consolidated branch into `main`,
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
