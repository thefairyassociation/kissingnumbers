# Lower bounds: kissing configurations in dimensions 12, 13, 14

*This is the construction side of the repository. Upper bounds are in
[`../upper_bound/`](../upper_bound/README.md); the short plain-English status
of both is [`../STATUS.md`](../STATUS.md); per-dimension indexes are in
[`dim13/README.md`](dim13/README.md), `dim12/README.md` and `dim14/README.md`.*

**Goal.** Beat a known kissing-number lower bound in dimension 13 (priority), 12 or 14:
an explicit set of unit vectors in R^n with all pairwise inner products <= 1/2,
strictly larger than the record, verified in exact arithmetic.

**Outcome: no record was beaten.** What follows is the verified state, the
constructions that were built and checked, and the families that were ruled out
(with the reason in each case).

## Live records

Fetched 2026-08-21 from [Henry Cohn's table](https://cohn.mit.edu/kissing-numbers)
and cross-checked against the MIT DSpace copy
([hdl.handle.net/1721.1/153312](https://hdl.handle.net/1721.1/153312), last
compiled June 2026). Re-checked 2026-11-09: all three rows are unchanged.

| dim | lower | upper | lower-bound source |
| --- | --- | --- | --- |
| 12 | 841 | 1355 | Takhanov–Assylbekov–Yun 2026, [arXiv:2606.18984](https://arxiv.org/abs/2606.18984) — numerical (max inner product 0.499999937751) |
| 13 | **1154** | 2064 | Zinoviev–Ericson 1999, Problems Inform. Transmission 35, 287–294, [mathnet ppi457](https://www.mathnet.ru/eng/ppi457) |
| 14 | 1932 | 3174 | Ganzhinov, *Highly symmetric lines*, Linear Algebra Appl. 722 (2025) 12–37, [arXiv:2207.08266](https://arxiv.org/abs/2207.08266) |

Upper bounds: de Laat–Leijenhorst 2024, [doi:10.1007/s12532-024-00264-w](https://doi.org/10.1007/s12532-024-00264-w).

## Best verified counts produced here

All three are **reproductions of known constructions**, not new results.

| dim | count | field | max off-diagonal | verifier output |
| --- | --- | --- | --- | --- |
| 12 | 840 | Q(sqrt 2) | exactly 1/2 | `dim12/verifier_output.txt`, `logs/verify_d12_840.txt` |
| 13 | 1154 | Q(sqrt 3) | exactly 1/2, 59640 tight pairs | `dim13/verifier_output.txt` |
| 14 | 1932 | Q (integer, norm^2 = 8) | exactly 4/8 = 1/2, 159264 tight pairs | `logs/verify_d14_1932.txt` |

Verify with

```bash
python3 lower_bound/lib/verify_exact.py lower_bound/dim13/configs/ze99_1154_exact.json
python3 -c "import sys;sys.path.insert(0,'lower_bound/lib');from verify_exact import verify_integer;\
verify_integer('lower_bound/dim14/configs/ganzhinov_1932_exact.json')"
```

**Dimension 12, 841 points — certified exactly here, configuration not ours.**
The published Takhanov–Assylbekov–Yun coordinates (`lib/testdata/authors_841_coordinates.txt`)
have maximum cosine `0.4999999377...`, strictly below 1/2, so they need no
algebraic structure to be proved: `lib/certify_float.py` rounds them to integer
vectors at scale `2^40` and checks `<x,y> <= 0` or `4<x,y>^2 <= (1-2d)^2 |x|^2|y|^2`
for all 353220 pairs in exact integer arithmetic, with `d = 6e-8`.  Certificate:
`dim12/configs/takhanov841_certified.json`; output: `dim12/certify_841_output.txt`.
So the dimension-12 record is now *exactly* verified in this repository, and
any future numerical hit strictly below 1/2 (in any dimension) is immediately
a proof.

```bash
python3 lower_bound/lib/certify_float.py lower_bound/lib/testdata/authors_841_coordinates.txt
```

`verify_exact.py` uses `fractions.Fraction` in Q(sqrt D) with an exact
comparison for `a + b*sqrt(D) <= c`. The integer fast path uses numpy int64
products — exact, not floating point — after *proving* the no-overflow bound
`n * max|c|^2 < 2^62`, and independently re-computes 20000 random entries with
Python big integers.

## Structural results established

### 1. ZE99's 1154 is maximal, and every layer of it is individually optimal

In unit coordinates the 1154 configuration has exactly five heights
`x_13 in {0, +-1/2, +-1}`, giving the layered decomposition

> `N(13) >= 2 + |A| + |B_+| + |B_-|`, with `A` a **1/2-code** in R^12 (height 0),
> `B_+-` **1/3-codes** (height +-1/2), cross condition `cos(A,B) <= 1/sqrt3`, and
> **no** constraint between `B_+` and `B_-`.

ZE99 realises `2 + 816 + 168 + 168`. All three thresholds are attained exactly
(`dim13/v2/layers.py`). Each piece is at its own combinatorial optimum:

* `A` = 816 = 51 x 16 tetrads `(+-1/2)^4`; 51 = `A(12,4,4)`, the maximum number of
  4-subsets of a 12-set with pairwise intersections <= 2.
* `B` = 24 axials `+-e_i` + 144 diamonds `(+-1)^12/sqrt12`; 144 = `A(12,4)` = `A(11,3)`.
* The 48 irrational vectors `(+-2 sqrt3 e_i, +-2)` beat the 24 axials that would
  otherwise sit at height 0: the all-rational version is 816+24+288+2 = **1130**.

Three independent negative checks:

* **Hole search** (`lib/holes.py`). `P = {u : <v_i,u> <= 1/2}` is convex and
  contains 0, so a 1155th unit vector exists iff some point of `P` has norm >= 1.
  Maximising `||u||` over `P` by the LP fixed-point iteration `u <- argmax_P <u,.>`
  from 250 random starts gives **0.9014 < 1**: nothing can be added.
* **Local jamming** (`dim13/v2/jamming.py`). For every point, `0` is not strictly
  outside the convex hull of the tangent projections of its tight neighbours:
  **no single point can move**. (Same verdict for the dim-12 840 and the dim-14
  1932.)
* **Enlarging B** (`dim13/v2/completeB.py`). One extra point in `B` would be worth
  **two** (it goes into both `B_+` and `B_-`) and give 1156. With `A` = the 816
  tetrads the same LP gives max norm **0.7232**; with `A` = the 840
  (tetrads + axials) and `B` = the 144 diamonds it gives **0.8327**. Five randomised
  greedy completions of the 144 diamonds all terminate at exactly **168**.
* **K(12,1/3) = 168 numerically.** The penalty optimiser seeded with the known 168
  plus one point, over many restarts, converges to max inner product
  **0.33335 > 1/3** for 169 points.

So 1154 can only be beaten in this framework by *moving several points at once*.
And the framework has little headroom **given the current records**: with the best
known values `K(12) = 841` and `K(12,1/3) = 168` it yields at most
`2 + 841 + 2*168 = 1179`, so pushing a pole-based layered construction past that
would first require improving one of those two numbers. (This is a ceiling
relative to today's records, not a proof: the Delsarte LP bounds computed by
`lib/lpbound.py` are much weaker — `M(12,1/2) <= 1416.1`, `M(12,1/3) <= 257.5`,
giving only `<= 1933`. That code reproduces `M(8,1/2) = 240` exactly and
`M(13,1/2) <= 2233.6`, the classical Odlyzko–Sloane value, so the machinery is
sound; the bounds themselves are simply not tight in this range.)

**A calibration worth recording.** The same two tests applied to configurations
whose record *was* subsequently broken:

| configuration | hole search `max ||u||` | locally jammed | beaten later? |
| --- | --- | --- | --- |
| dim 12, 840 (Leech–Sloane / Clebsch) | 0.8164 | yes | **yes** — 841 by Takhanov et al. 2026, by deforming the O(4) family inside the 48-system |
| dim 13, 1154 (ZE99) | 0.9014 | yes | not here |
| dim 14, 1932 (Ganzhinov) | 0.9354 | yes | not here |

So "no hole and locally jammed" does **not** imply optimality — the dim-12 840 has
both properties and is still beatable by a collective deformation. This is why the
remove-k/add-(k+1) continuation search (`lib/shakeloop.sh`) is the right residual
attack, and it is what is left running.

### 2. Ganzhinov's dim-14 1932, decoded

Re-derived from his published coordinates (`lib/analyze_g14.py`): in the norm-8
shell of Z^14 it is

* **364** vectors `(+-2,+-2,0^12)` — every one of these is compatible with every
  other vector in the shell, so any configuration here should take all of them;
* **1568** vectors `(+-1^8, 0^6)` on **49 supports x 32 sign patterns**.

The 14 coordinates split 7+7; every support has 4 coordinates in each half, the
7 half-blocks are the complements of the **Fano plane** lines, and the 49 supports
are all products of them, indexed by the edges of `K_{7,7}`. Support pairs meet in
4 (882 pairs) or 6 (294 pairs); the t=6 pairs are exactly "same row or same column".

The 49 sign codes are all **cosets of one global linear code `V <= F_2^14` of
dimension 5** (`lib/g14_global.py`), weight distribution {0:1, 6:7, 7:16, 8:7, 14:1}.

That yields a complete framework (`lib/gcode.py`), **validated by rebuilding a
valid 1932 from scratch** out of `V` + supports + solved coset representatives:

* a support `S` is usable iff every nonzero codeword of `V` meets `S` in >= 2
  coordinates (then `|C_S| = 2^dim V` with min distance >= 2);
* for a pair with `t = |S & T|`: `t <= 4` is free; `t in {5,6}` needs
  `dim V|_(S&T) < t`; `t >= 7` is impossible;
* the coset representatives must satisfy `<a, c_S + c_T> = 1` for some
  `a in V^perp` supported inside `S & T` — an **F_2 linear system**, solved
  exactly by elimination plus a search over its solution space.

**The 49 supports are exactly the Fano-plane product.** Splitting the 14
coordinates 7+7, take in each half the seven complements of the lines of the
Fano plane (lines `{0,1,3}+i mod 7`); the 49 supports are all 49 products
`P u Q`. Reproduced independently in `dim14/fano50c.py`: the same 49 sets, the
same intersection profile (4 for 882 pairs, 6 for 294).

**Which global codes work is exactly characterisable.** With `S^c = L u L'` a
pair of Fano lines, goodness of `V` for the whole design is

> `def(v_A) + def(v_B) >= 2` for every nonzero `v = (v_A, v_B)` in `V`,
> where `def(x) = wt(x) - max_L |x & L|` over the seven Fano lines.

Ganzhinov's `V` satisfies it (weights 6^7 7^16 8^7 14^1; the all-ones word is
present, which is exactly antipodality of the vector set). Griesmer caps the
minimum weight of a `[14,5]` code at 6, and containing `1^14` then forces every
other codeword into weight 6..8 — so the admissible `V` form a small explicit
family.

**Ganzhinov's 1932 is maximal, and the design is not extendable.**

* `lib/addable.c` checks all 768768 weight-8 candidates against the 1568 and
  finds **0** addable (0.7 s).
* Over all 3003 8-subsets (`dim14/why_no_50th.py`), the obstruction to a 50th
  support with *his* `V` splits as 2392 blocked because some `t=5` pair has
  `V|_X` equal to all of `F_2^X`, 455 not good for `V`, 107 meeting a support in
  >= 7, **0 addable**.
* `dim14/fano50c.py` enumerated **127190 distinct admissible codes `V`**; 10978 of
  them give a consistent 49-support coset system, and **none admits a 50th
  support**.
  (A 50th would give 1600 weight-8 vectors and 1964 > 1932.)
* That blocker does not depend on the sign code chosen for the new support:
  disjointness needs `U|_X + V|_X != F_2^X`, and where `V|_X` is already all of
  `F_2^X` no `U` can help — matching the exhaustive `addable` result.

Since the maximum support family with intersections in {2,3,4,6} is exactly 49
(`lib/clique.c` finds 49, matching the design), beating 1568 needs `t=5` pairs —
hence a different `V` — or `dim V = 6` with >= 25 supports. Cliques of up to
**72** supports do exist in the structural graph for other minimum-weight-6
codes, but the F_2 coset system then collapses: the greedy consistent subfamily
stops at 11-13 supports (352-416 weight-8 vectors).

### 3. The norm-8 shell of Z^13 cannot reach 1154 — ruled out

The natural dim-13 analogue of Ganzhinov's model is 312 D-roots plus weight-8
vectors; beating 1154 needs **843** weight-8 vectors. It cannot happen:

* In `[13]` two 8-subsets always meet in >= 3, and **at most 3** supports can
  pairwise meet in <= 4. (Complements are 5-sets pairwise meeting in <= 1; a point
  lies in at most 3 such blocks, and the degree count forces `2a + b >= 22`
  together with `3a + b <= 21`, a contradiction for 4 blocks.)
* So essentially every support pair is constrained, and for both `t=5` and `t=6`
  the annihilator is 1-dimensional — two kernel codewords would sum to one
  disjoint from a support, contradicting goodness. Every pair is therefore a
  **forced** F_2 equation.
* With `k = 13 - dim V` unknowns per support, consistency needs
  `C(m,2) <~ k*m`, i.e. `m <~ 2k+1`:

| dim V | signs/support | max m | max weight-8 | max total |
| --- | --- | --- | --- | --- |
| 5 | 32 | 17 | 544 | 856 |
| 6 | 64 | 15 | 960 | 1272 (search reaches m = 7, i.e. 760) |
| 7 | 128 | 3 (dim V = 7 forces every pair to `t <= 4`) | 384 | 696 |

Best actually built and Gram-checked in this model: **760**. A coordinate section
of Ganzhinov's 1932 gives 312 + 672 = 984. The maximum support family with
intersections in {4,6} is 21 (vs 49 in dim 14), giving 21 x 32 + 312 = 984 again.

An explicit `dim V = 6` example: `V = S_7 (+) S_7` (two copies of the [7,3,4]
simplex code) makes *all* 49 Fano-product supports good in dim 14 with 64 signs
each, but three supports in one row share the same functional `(0, 1_Q)`, so their
three equations sum to `0 = 1`. At most 2 per row and per column survive — 14
supports, 896 + 364 = 1260 < 1932. Bigger sign codes cost supports faster than
they gain.

## Numerical searches, and an honest calibration

Two continuation schemes are implemented and run against dimension 13:

* `lib/opt.c` (`opt2`) — penalty method `sum max(0, g_ij - t)^2` with the
  threshold `t` annealed down towards 1/2 each time the energy is driven to zero;
* `lib/riesz.c` — Riesz-energy continuation, minimising `sum ||x_i - x_j||^{-s}`
  for a geometrically increasing exponent starting near the logarithmic energy.
  This is the scheme Takhanov–Assylbekov–Yun used to get 841 from the classical
  840 in dimension 12.  The unmarked binary runs the geometric-homotopy gradient
  descent described here; `KISS_SOLVER=lbfgs` and `KISS_SOLVER=adam` select the
  BLAS L-BFGS and the published Adam schedule instead, and `KISS_FAITHFUL=1`
  runs the source-faithful N=841 protocol (`lib/FAITHFUL_841.md`).

Both are driven by `lib/shake2.py`, which removes k points and re-inserts k+1 at
the **deepest holes** of what remains (found by LP) rather than at random; that
change alone moved the dimension-13 results from max inner product 0.69 to 0.51.

Best reached so far for 1155 points in R^13: **0.5088** (needs <= 0.5), over
several hundred shake-and-relax rounds with k in 1..8 removed points.

**But the calibration says not to read much into that.** Seed the same machinery
with the classical dimension-12 840 plus one extra point — a case where a solution
provably exists, since Takhanov et al. reached 0.499999937751:

| optimiser | dim 12, N = 841 (solution exists) | dim 13, N = 1155 (unknown) |
| --- | --- | --- |
| penalty continuation, crude step rule | ~0.52 | 0.5088 |
| Riesz continuation + Armijo line search (`KISS_SOLVER=gd`, the default) | **0.50519** | 0.5107 |
| BLAS engine + Adam on the published schedule (`KISS_SOLVER=adam`) | **0.500477** | not run |
| `fastriesz` + fingerprint screen + branching + minimax polish (2026-10-02) | **0.5001014** | 0.506587 |
| the same pipeline as a campaign, random extra point, 1,685 re-crystallised roots (2026-11-10) | **0.4999968 — certified** | 0.506009 (entering at `s = 16`; the `s = 8` entry gives no usable start) |

The Adam row is the newer calibration; see `docs/CALIBRATION_optimizer.md` for how it
was measured and `lib/FAITHFUL_841.md` for the opt-in source-faithful protocol.
Those two were above 0.5; the last row is the first to get below (see the
2026-11-10 update further down).

Until that last row the two columns were close, and the left one is a case
where the answer is *yes*. So a dimension-13 near-miss at 0.5088 was **not**
evidence that 1155 points fail to exist, and it still is not much: the
dimension-12 success took about 1,700 starts of a kind that dimension 13 has
not produced at all. The structural results above stand on their own; the
numerical ones do not.

### 2026-10-02 update: what the calibration now says

Details and every number are in `docs/CALIBRATION_optimizer.md` (last section);
the short version:

* **The first exponent stage is a coin flip that decides everything.**  After
  `s = 8` the 840 core has either *melted* into a disordered state (110-240
  near-antipodal pairs left, inner products far from the canonical values) or
  *re-crystallised* into a new antipodal-rich structure (401-406 pairs, more
  than the canonical 372; canonical-value fraction about 0.6).  Entering the
  schedule at `s = 16` instead keeps the original core rigid (exactly 372
  pairs) and leads to the strained 0.504 family, so the melt-and-recrystallise
  step at `s = 8` is essential.  The max inner product cannot tell the two apart at that point
  (about 0.60 either way), but the antipodal-pair count separates them
  perfectly, so `fastriesz --screen-anti 1:300` rejects a melted start after
  6 s instead of 30 s.  A smaller start jitter (0.005 instead of 0.03) raises
  the re-crystallisation rate from about 1 in 48 to about 1 in 10-25.
* **Re-crystallised starts lead to the published witness's family.**  Fingerprints
  (fraction of inner products within 0.01 of a canonical-840 value; number of
  near-antipodal pairs): witness 0.724 / 387, best branch 0.728 / 388, versus
  0.37 / 11-56 for melted runs and 0.83 / 372 for a rigid "gentle" start (which
  ends uniformly strained at 0.5042).  Under natural labels these runs are
  heavily rearranged -- half of the canonical antipodal pairs are broken and
  48-system points pair with bridges -- so the family is an 840-like
  structure *re-crystallised* with new labels plus a defect for the extra
  point.  That reconciles `docs/STRUCTURAL_841.md` ("far from the core under
  natural labels") with the witness's canonical inner-product spectrum.
* **Branching** from a re-crystallised state at `s = 16` stays in the family; finals land
  on discrete levels per root.  Best level so far: `0.500165` from the search,
  `0.5001014` after the local minimax polish -- the bottom of that basin.
  Still above 1/2: **the calibration is not passed.**
* **Dimension 13.**  Melting is not a coin flip there: ZE99 + 1 melts at
  `s = 8` in every run tried (antipodal pairs 577 -> 65-102), even at a tenth
  of the learning rate, and it never re-crystallises the way the dimension-12
  core does: 44 first-stage screens at jitter 0.005 plus 36 earlier runs, none
  above 102 near-antipodal pairs.  Entering the schedule at `s = 16` keeps it
  partially
  (443-490 pairs) and ends at 0.5097; entering at `s = 32` keeps ZE99 rigid and
  ends on two discrete strained levels, `0.506671` and `0.506738` (the first
  minimax-polishes to `0.506587`) -- the best values this repository has
  reached for 1155 points in R^13 (previously 0.5088), but a uniformly strained family like dimension 12's 0.5042, not a
  promising one.

### 2026-11-10 update: the calibration is passed

Campaign `c03-tail` ([`campaigns/c03-tail/`](campaigns/c03-tail/README.md))
ran the recipe above at scale: 16 five-hour shards, 320 CPU-hours, 48,160
first-stage screens, 1,685 re-crystallised roots, each branched for about four
minutes. The one change from October is that the extra point is a *random*
hypercube vertex direction for every seed instead of a fixed one.

* **Result: an 841-point configuration with max inner product 0.4999968,
  certified.** `certify_float.py` proves all 353,220 pairs are at most
  1/2 − 3·10⁻⁶ in exact integer arithmetic; it was run on the runner, again by
  the collect job, and again locally. The file is
  [`dim12/configs/found_841_c03.txt`](dim12/configs/found_841_c03.txt), the
  certificate `found_841_c03_certified.json`. A sibling branch of the same
  round gave a second certified output (0.4999972), a close variant.
* **This equals the record (841). It is not a new one.**
* **It is a different 841.** Fingerprint against the published coordinates:

  | | max inner product | near-antipodal pairs (cos < −0.99) | pairs above 0.4999 |
  | --- | ---: | ---: | ---: |
  | published (Takhanov–Assylbekov–Yun) | 0.49999994 | 387 | 9,116 |
  | found here | 0.49999680 | 378 | 8,022 |

  The pair counts are invariant under rotation and relabelling, so the two
  are not congruent. The margin below 1/2 is about fifty times larger here.
* **How rare.** Final values by root, c02 and c03 together (2,043 roots with a
  random extra point): one below 1/2, three more within 10⁻⁵ above it
  (0.50000027, 0.5000060, 0.5000071), the rest on the 0.5001 floor or
  higher. The fixed extra point of October and of c02's first arm gave nothing
  below 0.5000897 in 1,200 roots.
* **Dimension 13 got nothing from the same recipe**: 13,200 screens, no root.
  After `s = 8` the 1154 keeps 50–150 of its 577 antipodal pairs and never
  re-forms (threshold was 250). Entering at `s = 16` keeps 431–464 pairs and
  entering at `s = 32` keeps 566–577. `campaign.py --start-stage 1|2` now
  screens and branches from those entries; campaign c04 ran both.
* **c04, the first use of the validated pipeline on open cases, found no
  record.** 842 points in dimension 12 from the found 841: 1,680 roots, best
  0.5009716, and every shard's best lies in 0.50097–0.50104 — a floor ten
  times higher than the 841 floor and with no tail (the authors of the 841
  report 0.50090 for their own 842). 1155 points in dimension 13: 1,296 roots
  entering at `s = 16`, best 0.5060090; 492 roots entering at `s = 32`, best
  0.5063593; per-shard bests all in 0.5060–0.5072. The previous best was
  0.506587.

A second flaw the calibration exposed: with a seed file the Riesz run was fully
deterministic, so two different "restarts" returned bit-identical answers
(0.50518677246961574 twice). Restarts now jitter the starting configuration
(`KISS_JIT`, default 0.03), which is what makes multi-seed search mean anything.

A related bug worth recording: `pow(r2, -s/2)` overflows to `+inf` at large
exponents, turning the configuration into `NaN`, whereupon the maximum inner
product reads as `-2` and the run reports **feasible**. Two runs did exactly that
before the rescaled evaluation `(r2/r2min)^(-s/2)` and a `NaN` guard were added.
Every configuration reported here is checked by the exact verifier, not by the
optimiser's own claim.

## Families tried, and why they were dropped

| # | family | verdict |
| --- | --- | --- |
| 1 | ZE99 exact reproduction (dim 13) | baseline 1154, re-verified in Q(sqrt 3) |
| 2 | Adding a point to ZE99 | impossible: hole-search max norm 0.9014 |
| 3 | Enlarging the 1/3-code layer B | impossible with A fixed (0.7232 / 0.8327); 169-point 1/3-codes in R^12 not found numerically |
| 3b | A 169-point layer drawn from the 4248 primitive directions of magnitude <= 2 compatible with the tetrads | **proved impossible**: the maximum is exactly 168 (`dim13/prove_joint_layer_168.py`; the CP-SAT search for it had timed out) |
| 3c | A 25th "auxiliary" (irrational) direction per layer | exhaustive among ternary and magnitude-<=2 directions: exactly 24 exist; the continuous search (two seeds, 44 and 46 runs before the session restarted, 2026-11-09) found 24 again among the sampled polytope vertices and got no closer to 25 than a constraint violation of 0.204 |
| 3d | Delete one tetrad support and fit a 177-point layer | **proved impossible, given the solvers' bounds**: at most 176 per layer in every one of the seven orbits, so at most 1154 in total (`dim13/prove_drop_tetrad_176.py`; CP-SAT and HiGHS agree; the original CP-SAT search had timed out) |
| 3e | Steiner-switch of the diamonds plus one integer vector | exhaustive over the 8192-vector shell: none |
| 4 | Enlarging A with B fixed | tetrads are `A(12,4,4)`-optimal; no vector of the region admits a 52nd 4-subset, and triples/5-sets/6-sets all violate `sum_{i in S}|b_i| <= 2/sqrt3` |
| 5 | Norm-8 shell of Z^13 (D-roots + weight-8) | ruled out, see section 3; ceiling ~984, best built 760 |
| 6 | Generic max-independent-set on all 329472 weight-8 vectors of Z^13 | local search reaches 404 (+312 = 716); far below the structured 984 |
| 7 | Global-linear-code framework in dim 13, dim V in {4,5,6,7} | best Gram-verified 760 |
| 8 | Coordinate and hyperplane sections of Ganzhinov's 1932 | 984 |
| 9 | Extending Ganzhinov's 1932 (dim 14) | 1568 weight-8 is maximal (0 of 768768 addable); 0 of 3003 supports addable |
| 10 | dim-14 global-code search over minimum-weight-6 codes V | structural cliques up to 72 supports, but F_2 coset consistency collapses to 11-13 |
| 10b | dim-14 Fano-design code enumeration (836 admissible V) | none admits a 50th support |
| 11 | Numerical continuation from ZE99 + 1 point, and remove-k/add-(k+1) shaking | ongoing; no feasible 1155 found |
| 11b | PackingStar's two 1146-point configurations (dim 13) | reconstructed and certified exactly; 1146 < 1154, and their E6 x E7 fibre core is capped at 1008 |
| 12 | Highly symmetric (group-orbit) configurations in R^13 from PSL(2,13) / PGL(2,13) on the 13-dimensional standard representation | implemented (`dim13/v2/orbits.py`); orbit optimisation did not reach max cos <= 1/2 |

## Running the checks

`.github/workflows/checks.yml` runs these on every push to `main` and on every
pull request; they take a few minutes.  Locally, from the repository root:

```bash
python3 -m pip install numpy scipy-openblas32   # or a system OpenBLAS
make -C lower_bound/lib all

bash lower_bound/lib/selftest.sh                          # BLAS engrad vs reference loops
python3 lower_bound/lib/test_fast_tools.py                # fast engine, hinge polish, float certifier
python3 lower_bound/dim13/prove_joint_layer_168.py        # the 168-point layer proof
python3 lower_bound/lib/test_faithful_841.py              # source-faithful N=841 protocol
python3 lower_bound/lib/test_o4_breadth.py --binary-smoke # O(4) breadth seeds
python3 lower_bound/lib/analyze_published_841.py          # structural audit of the witness
```

`make -C lower_bound/lib blas-check` reports whether the Riesz targets can be built
without building anything.  The verification tools (`mis8`, `clique`, ...) do
not link BLAS and build on their own with `make -C lower_bound/lib verify-tools`.

What the checks are actually protecting, since the numbers in this file depend
on it: that `riesz.c`'s unmarked default stays the geometric-homotopy GD the
scoreboard was measured with, that the faithful N=841 path keeps the published
schedule and the authors' polish rates, that a polish result reaches the
candidate file at all, that the non-finite guards still fail closed, and that
the build does not start requiring OpenBLAS for targets that never needed it.
Each of those has been wrong at least once.

Two things to know when running locally:

* `mis8` rewrites `logs/mis8_n14_s1.txt` as a side effect of the self-check, so
  a clean tree comes back dirty.  Restore it with
  `git checkout -- lower_bound/logs/mis8_n14_s1.txt`.  CI asserts that this is the
  *only* file the checks modify.
* Every C tool is a build product; none is committed. `make -C lower_bound/lib all`
  builds them, and `make clean` removes them.

## Code

| file | what it does |
| --- | --- |
| `lib/verify_exact.py` | exact Gram verifier: Q(sqrt D) via `Fraction`, plus a proven-safe exact integer path |
| `lib/holes.py` | LP hole search — can a point be added at all? |
| `lib/opt.c` | penalty-method spherical-code optimiser, adaptive step + basin hopping (`KISS_T` sets the threshold) |
| `lib/multistart_d12_841.py` | resumable CPU multi-start for the N=841 calibration; independently verifies every output and retains distinct near-threshold basins |
| `lib/hypercube_classes.py` | writes representatives of the signed interaction classes of the published hypercube extra-point starts |
| `lib/FAITHFUL_841.md`, `lib/test_faithful_841.py` | opt-in one-candidate CPU reproduction of the published N=841 schedule, plus the known-witness regression suite |
| `lib/prepare_841_polish.py` | reproduces the published `%.10f` decimal-Gram, rank-12 eigendecomposition search-to-polish handoff |
| `lib/analyze_published_841.py`, `docs/STRUCTURAL_841.md` | numerical structure and basin-fingerprint audit of the public 841-point witness |
| `lib/o4_breadth.py`, `lib/run_o4_faithful_pilot.py`, `lib/verify_o4_841.py` | theorem-valid O(4) breadth seeds, paired faithful pilot runner, and independent read-back verification |
| `lib/shake.py`, `lib/shakeloop.sh` | remove k points / add k+1 / re-optimise (the dim-12 840 -> 841 method) |
| `lib/fastriesz.c` | many-start Riesz continuation, ~17x the per-core throughput of `riesz.c` (squaring for power-of-two exponents, Verlet neighbour lists); `--start-stage`/`--save-stage` for branching |
| `lib/branch_search.py` | go-with-the-winners branching on top of `fastriesz`; independently recomputes every candidate and certifies anything below 1/2 |
| `lib/hingepol.c` | L-BFGS on `sum (g_ij - t)_+^2`: `E = 0` iff every cosine is <= t (feasibility test) |
| `lib/minimax_polish.py` | local minimax descent by hinge continuation (`t = max - delta`): the bottom of a candidate's basin |
| `lib/surgery.py` | remove the most-violated points, relax, reinsert at the deepest holes, re-polish |
| `lib/certify_float.py` | **exact** certificate for a float configuration whose max cosine is strictly below 1/2 (integer rounding at 2^40, exact angle test) |
| `lib/test_fast_tools.py` | regression tests for the four tools above against NumPy / `Fraction` references |
| `lib/mis8.c` | max independent set over all `2^8 * C(n,8)` weight-8 vectors, implicit adjacency |
| `lib/addable.c` | is a weight-8 configuration maximal? |
| `lib/signmis.c` | dense-bitset MIS over (support, sign) pairs for a fixed support family |
| `lib/clique.c` | max clique on k-subsets with prescribed pairwise intersections |
| `lib/gclique.c` | max clique on an arbitrary graph |
| `lib/Makefile` | builds everything; `make riesz-tools` needs OpenBLAS (or `pip install scipy-openblas32`), `make verify-tools` does not, `make blas-check` reports which |
| `lib/gcode.py`, `lib/pipeline.py` | the global-linear-code framework and the F_2 coset solver |
| `lib/analyze_g14.py`, `lib/g14_global.py` | decoding Ganzhinov's dim-14 configuration |
| `dim13/v2/`, `dim14/` | the dimension-specific searches |
| `logs/progress.log` | append-only attempt log |

Benchmarks that pin the machinery down: `mis8` loads Ganzhinov's 1568 with zero
violations; `gcode` rebuilds a valid 1932 from `V` + supports + solved cosets;
`clique` returns exactly 49 for the dim-14 {4,6} support design and 21 for dim 13.

## Search campaigns

Long numerical searches no longer run as background loops in a session (they
do not survive a container restart). They run as *campaigns*:

* `lib/campaign.py` is one time-budgeted shard: screen fresh seeds through the
  first exponent stage, keep the ones whose core re-crystallises, branch them
  with `branch_search.py`, minimax-polish, and hand anything below 1/2 to the
  exact certifier;
* `.github/workflows/search.yml` runs many shards in parallel on the free
  GitHub-hosted runners. What to run is described by
  `.github/search-campaign.json`; pushing a change to that file starts a
  campaign, and its last job commits the result to `campaigns/<name>/`.
  Because a merge is also such a push, a push starts nothing on the default
  branch, on a branch other than the one the spec names (optional
  `"branch"` key), or when that directory already exists. "Run workflow"
  starts a campaign regardless; a second run of a recorded campaign is
  written to `campaigns/<name>/rerun-<run id>/`;
* `lib/campaign_collect.py` merges the shards. It recomputes every kept
  configuration and re-runs the exact certifier itself, so a "hit" in a
  campaign record means the certifier passed, not that an optimiser said so.

To run one shard locally:

```bash
lower_bound/lib/campaign_init.sh d12-841 /tmp/init.txt
python3 lower_bound/lib/campaign.py /tmp/out --n 12 --N 841 --init /tmp/init.txt \
    --seed-start 1 --hours 0.2 --workers 4
```

Records of finished campaigns are in [`campaigns/`](campaigns/):

| campaign | what it ran | outcome |
| --- | --- | --- |
| `c01-smoke` | one short shard per arm | plumbing test only |
| `c02-scale` | dim 12 / 841, fixed and random extra point; dim 14 / 1933 | best 0.5000060 (dim 12), 0.518981 (dim 14); nothing certified |
| `c03-tail` | dim 12 / 841, random extra point, 16 shards; dim 13 / 1155, 4 shards | **841 found and certified** (0.4999968); dim 13: no roots |
| `c04-beyond` | dim 12 / **842** from the found 841; dim 13 / 1155 entering at `s = 16` and `s = 32` | nothing certified. 842: best 0.5009716, all six shards within 0.50097–0.50104. Dim 13: best 0.5060090 (`s = 16`), 0.5063593 (`s = 32`) |
| `c05-two-points` | dim 12 / **842** from the classical 840 plus one fixed and one random extra point, entering at `s = 8` | started 2026-11-10 |

## If someone picks this up

The three things that look most worth attacking next, in order:

1. **Done 2026-11-10: the calibration passed** (see the update above). What
   follows is the October text, kept for the record. The open item it leaves
   is the dimension-13 analogue: a way to get re-crystallised starts from the
   1154, which melts and never re-forms.

   *Finish the calibration with many re-crystallised roots.*  *(Updated 2026-10-02;
   the original item asked for "a better optimiser", and the 2026-08 notes
   showed L-BFGS was a regression.)*  The machinery now exists and is fast:
   screen fresh starts on the antipodal-pair count after `s = 8`
   (`fastriesz --stages 2 --screen-anti 1:300 --save-stage 2 --jit 0.005`),
   then run `branch_search.py` over all such roots and minimax-polish the good
   finals.  Each root has its own discrete set of final levels; the best seen
   so far bottoms out at 0.5001014.  The open question is simply whether some
   root has a level below 1/2 -- the published witness proves at least one
   member of this family does.  Only after that is the dimension-13 numerical
   evidence worth anything either way.
2. ~~**Dimension 14 with `dim V = 6`.**~~ **Closed — see below.** This was
   recommended here first and it is wrong.
3. ~~**Non-global sign codes.**~~ **Closed — see below.**

### Both dimension-14 leads above are now ruled out

*(`dim14/dimv6.py`, `dim14/diag_dimv6.py`, `dim14/dimv6_verdict.py`,
`dim14/why_global_v.py`)*

**`dim V = 6` caps at 9 supports, and the reason is not the F_2 system.** The
earlier claim here — "what collapses is the F_2 coset system" — was wrong. Of the
403 good candidates rejected after growing a family, **zero** failed on the F_2
system (8 constrained pairs against 72 unknowns; nowhere near saturated). 362
failed a *structural* test and 41 met a member in >= 7.

The structural test is the asymmetry that makes Ganzhinov's construction work:
**a pair at intersection `t` is automatically fine whenever `dim V < t`.** At
`dim V = 5` the `t = 6` pairs are free and only `t = 5` needs a witness — hence
49 supports. At `dim V = 6` the `t = 6` pairs need a witness too: some nonzero
`a` in `V^perp` (now only dimension 8) supported inside a 6-element set. Its
low-weight words cover roughly 70% of 6-sets and 31% of 5-sets, and *every* one
of the `C(m,2)` pairs has to pass, so `m` collapses.

Measured over **1315** sampled dimension-6 codes: best family **9 supports**
(9 x 64 + 364 = 940, against the 25 supports needed for 1964). Selecting codes
whose dual is rich in low-weight words does not help —
correlation(low-weight words in `V^perp`, family size) = **0.06**. Doubling the
sign code costs far more than a factor two in supports: 49 x 32 = 1568 beats
9 x 64 = 576.

**Per-support codes are not a real degree of freedom either.** Two sign sets
project onto `X = S & T` as cosets `a + A` and `b + B`; disjoint cosets of
*different* subspaces require `A + B != F_2^X`, and for `t = 6` with 32 signs
each (so `dim A = dim B = 5` in `F_2^6`) that forces `A = B`. Checked on the
record configuration: all **294** constrained pairs have `dim A = dim B = 5` and
`A = B`, every time. Per-support codes could only differ by projecting to
*smaller* subspaces — i.e. by carrying fewer signs, which loses more than it
gains. So one global `V` is close to forced, not a restriction that was costing
anything.

**What that leaves.** Inside the norm-8 shell of Z^14 the record now looks
closed from every side tried: the 1568 weight-8 vectors are maximal
(exhaustive over all 768768 candidates), the 49-support Fano design admits no
50th support over 127190 enumerated codes, `dim V = 6` caps at 9 supports, and
per-support codes buy nothing. Beating 1932 needs a different shell or a
genuinely different construction, not a better search inside this one.
