# Where this project stands (plain English)

*Last updated 2026-10-02.  The detailed technical report is
[`kissing/README.md`](kissing/README.md); this page is the short version.*

## What the project is trying to do

The **kissing number** in dimension *n* is the largest number of unit spheres
that can all touch one central unit sphere without overlapping.  Equivalently:
the most unit vectors you can place in R^n with every pairwise angle at least
60 degrees (every inner product at most 1/2).  Nobody knows the exact answer in
dimensions 12, 13 or 14 -- only a range.  The repo tries to **raise the lower
end of the range**: find an explicit configuration with more points than the
current record, and *prove* it in exact arithmetic.

| dim | best known (record) | upper bound | best this repo has verified exactly |
| --- | --- | --- | --- |
| 12 | 841 (Takhanov–Assylbekov–Yun 2026) | 1355 | **841** (their configuration, certified exactly here — see below) |
| 13 | 1154 (Zinoviev–Ericson 1999) | 2064 | 1154 (reproduction) |
| 14 | 1932 (Ganzhinov 2025) | 3174 | 1932 (reproduction) |

**No record has been beaten.**  That is the honest headline, and it is
unchanged by the latest session.

## The two kinds of work in the repo

1. **Structural / exact work (solid).**  The existing record configurations in
   dimensions 13 and 14 were rebuilt, decoded and shown to be "stuck": no point
   can be added and no point can move on its own.  Several natural families of
   constructions were ruled out with proofs or exhaustive searches.  This part
   is finished and reliable (`kissing/README.md`, sections 1–3).

2. **Numerical search (the open front).**  Optimise point positions on the
   sphere and hope to squeeze in one more point.  The catch: the repo's
   optimiser had never been shown to work.  The test is dimension 12 with 841
   points, where a solution is *known* to exist.  Until the optimiser can
   rediscover a sub-1/2 configuration there, its failures in dimension 13 mean
   nothing.  That test is called "the calibration" throughout the repo.

## What changed in the 2026-10-02 session

* **The published 841 is now certified exactly.**  Any float configuration
  whose largest inner product is *strictly* below 1/2 can be rounded to
  integers and checked with exact integer arithmetic.  No algebra needed.  The
  authors' 841 passes with every cosine at most 1/2 − 6×10⁻⁸.
  Tool: `kissing/lib/certify_float.py`; certificate:
  `kissing/dim12/configs/takhanov841_certified.json`.  This also means a
  future numerical hit in dimension 13 would be immediately provable.
* **A search engine about 17× faster per CPU core** (`kissing/lib/fastriesz.c`).
  Same mathematics as the old optimiser, checked against an independent
  reference to 1e-12.
* **A feasibility test and a basin-bottom test.**  `hingepol` decides whether a
  configuration can be pushed to max inner product ≤ 1/2 without leaving its
  basin; `minimax_polish.py` finds how low its basin goes.
* **Why the calibration kept failing, now understood much better.**
  Fingerprinting configurations (how many inner products stay at the 840's
  exact values, how many antipodal pairs) shows:
  * the authors' own schedule, run on CPU, melts the 840 core and always ends
    near **0.534** — the wrong family;
  * starting gently keeps the core rigid and ends near **0.504** — also wrong
    (uniformly strained);
  * the rare "strong basin" the old code found once in 48 tries has the
    **same fingerprint as the published 841** — the right family.
* **The very first stage is a coin flip that decides the family.**  In the
  first exponent stage the 840 core either melts into a disordered state or
  re-crystallises into a new structure with *more* antipodal pairs than it
  started with; only the second kind leads to the published 841's family.
  Counting antipodal pairs tells which after 6 seconds (the max inner product
  cannot).  A smaller random kick at the start makes the good outcome several
  times more likely, so good starting points went from about 25 CPU-minutes
  each to about one.
* **Branching search** (`kissing/lib/branch_search.py`): re-running a re-crystallised
  start from an early checkpoint with small perturbations stays in the right
  family; finals land on a few discrete levels per start.  Each good result is
  then pushed to the bottom of its basin by `minimax_polish.py`.

  | | best max inner product for 841 points in R^12 (need < 0.5) |
  | --- | --- |
  | old optimiser, best ever recorded | 0.500477 |
  | old optimiser, rerun this session | 0.500245 |
  | **new pipeline** | **0.5001014** |

  Still above 0.5, so the calibration is **still not passed** — but the gap
  shrank by about 80%, and the remaining question is now concrete: does one
  of the (cheaply findable) re-crystallised starts have a level below 1/2?  The published
  841 proves at least one member of the family does.
* **Dimension 13 with 1155 points** (the actual target): the new engine
  reaches **0.506671** (previous best here 0.5088), but that is a uniformly
  strained copy of the 1154 record, the dimension-13 analogue of the dead-end
  0.504 family above.  Unlike dimension 12, the 1154 record melts in the
  first stage every single time and never re-crystallises, so the trick does
  not transfer directly.  No evidence either way about whether 1155 exists.

## How to check things yourself

```bash
python3 -m pip install numpy scipy scipy-openblas32 sympy
make -C kissing/lib riesz-tools verify-tools
bash kissing/lib/selftest.sh                      # old self-checks
python3 kissing/lib/test_fast_tools.py            # new tools
python3 kissing/lib/certify_float.py kissing/lib/testdata/authors_841_coordinates.txt
```

## What to do next

1. **More good starts, not more branches.**  Each re-crystallised start has its own set
   of reachable levels.  Find many (6-second screen on the antipodal-pair
   count) and branch each; one of them may contain a level below 1/2.  The
   exact commands are in `kissing/README.md`, "If someone picks this up".
2. Once the calibration passes, run the identical pipeline on dimension 13
   with 1155 points (seed files and commands in
   `kissing/CALIBRATION_optimizer.md`).
3. The combinatorial ideas for dimension 14 listed at the end of
   `kissing/README.md` are untouched and independent of all of the above.

## Housekeeping notes

* The files `1.out` … `7.out`, `riesz_1.out`, `riesz_5.out` at the top level
  are old optimiser outputs committed by earlier sessions; they are kept
  as-is (nothing depends on them).
* Compiled binaries are build products and are git-ignored; `make` rebuilds
  them.  Never compile the optimisers with `-ffast-math` (see
  `kissing/HANDOFF_optimizer.md`, trap 1).
