# Dimension 13 — what has been established here

Live records (fetched 2026-08-21, https://cohn.mit.edu/kissing-numbers, cross-checked
by web search against the MIT DSpace copy of Cohn's table, updated June 2026):

| dim | lower | upper | lower-bound source |
| --- | --- | --- | --- |
| 12 | 841 | 1355 | Takhanov–Assylbekov–Yun 2026 (numerical) |
| 13 | **1154** | 2064 | Zinoviev–Ericson 1999 |
| 14 | 1932 | 3174 | Ganzhinov, Linear Algebra Appl. 722 (2025) 12–37 |

## 1. ZE99 (1154) re-verified exactly, and it is *maximal*

`lower_bound/lib/verify_exact.py` re-checks the 1154 configuration in exact
Q(sqrt 3) arithmetic: norm^2 = 16 throughout, max off-diagonal exactly 8 = 16/2,
59640 tight pairs, all vectors distinct.  **Reproduction, not progress.**

Structure (norm^2 = 16, coordinate 13 singled out; heights are x_13/4):

| layer | height | vectors | count |
| --- | --- | --- | --- |
| tetrads | 0 | `(+-2)^4` on 51 4-subsets of the first 12 coords | 816 |
| diamonds | +-1/2 | `((+-1)^12, +-2)` | 288 |
| irrationals | +-1/2 | `(+-2 sqrt3 e_i, +-2)` | 48 |
| poles | +-1 | `+-4 e_13` | 2 |

In unit terms this is exactly the layered decomposition

    N(13) >= 2 + |A| + |B_+| + |B_-|

with A a 1/2-code in R^12 (height 0), B_+- 1/3-codes (height +-1/2), and the
cross condition cos(A, B) <= 1/sqrt 3.  ZE99 = 2 + 816 + 168 + 168.
Each ZE99 layer is at its own combinatorial optimum:
`A(12,4,4) = 51` tetrad supports (=> 816) and `A(12,4) = 144` diamond signs per
height sign (=> 288); the 48 irrationals beat the 24 axials `+-e_i` that would
otherwise sit at height 0 (816+24+288+2 = 1130, the rational configuration).

**Hole search** (`lower_bound/lib/holes.py`): the polytope
`P = {u : <v_i,u> <= 1/2}` of the 1154 configuration has
`max ||u|| over P = 0.9014 < 1`, so **no 1155th unit vector can be added**
without moving existing points.

## 2. The norm-8 integer shell of Z^13 — RULED OUT

Ganzhinov's dim-14 record lives in the norm-8 shell of Z^14:
364 vectors `(+-2,+-2,0^12)` (compatible with everything) plus 1568 vectors
`(+-1^8, 0^6)`.  Re-derived here from his coordinates and re-verified
(`lower_bound/lib/analyze_g14.py`): 49 supports x 32 sign patterns, supports pairwise
meeting in 4 or 6, the 14 coordinates splitting 7+7 with the 49 supports indexed
by the edges of K_{7,7}.

The 49 sign codes are *all cosets of one global linear code* `V <= F_2^14` of
dimension 5 (`lower_bound/lib/g14_global.py`), weight distribution
{0:1, 6:7, 7:16, 8:7, 14:1}.  That gives a complete framework
(`lower_bound/lib/gcode.py`), validated by rebuilding a valid 1932 configuration
from scratch:

* supports S: every nonzero codeword of V meets S in >= 2 coordinates;
* pair (S,T), `t = |S & T|`: `t <= 4` free; `t in {5,6}` needs
  `dim V|_(S&T) < t`; `t >= 7` impossible;
* coset reps c_S with `<a, c_S + c_T> = 1` for some `a` in `V^perp` supported
  inside `S & T` — an **F_2 linear system**.

**Why dim 13 cannot reach 1154 this way.**  In `[13]` two 8-subsets always meet
in >= 3, and at most **3** supports can pairwise meet in <= 4 (a counting
argument on the complementary 5-sets: a point lies in <= 3 blocks, forcing
`2a + b >= 22` and `3a + b <= 21`).  So essentially *every* pair of supports is
constrained, and for both `t = 5` and `t = 6` the annihilator is
1-dimensional (two kernel codewords would sum to one disjoint from a support),
so every pair contributes a **forced** equation.  With `k = 13 - dim V`
unknowns per support, consistency needs `C(m,2) <~ k*m`, i.e. `m <~ 2k+1`:

| dim V | signs/support | max m | max weight-8 | max total |
| --- | --- | --- | --- | --- |
| 5 | 32 | 17 | 544 | 856 |
| 6 | 64 | 15 | 960 | 1272 (search reaches m=7 -> 760) |
| 7 | 128 | 3 (all pairs forced to t<=4) | 384 | 696 |

Best actually constructed and Gram-verified in this model: **760**.
A coordinate section of Ganzhinov's 1932 gives 312 + 672 = 984.
The model does not reach 1154.

## 3. Search machinery in this repo

| file | what it does |
| --- | --- |
| `lower_bound/lib/verify_exact.py` | exact Gram verifier (Q and Q(sqrt D)) |
| `lower_bound/lib/opt.c` | penalty-method spherical-code optimiser with basin hopping |
| `lower_bound/lib/mis8.c` | max independent set over all 2^8 * C(n,8) weight-8 vectors |
| `lower_bound/lib/signmis.c` | dense-bitset MIS over (support, sign) pairs |
| `lower_bound/lib/clique.c` | max clique on k-subsets with prescribed intersections |
| `lower_bound/lib/gclique.c` | max clique on an arbitrary graph |
| `lower_bound/lib/gcode.py` | the global-linear-code framework + F_2 coset solver |
| `lower_bound/lib/holes.py` | LP hole search (can a point be added?) |

Benchmarks: `mis8` loads Ganzhinov's 1568 with zero violations; `gcode`
rebuilds a valid 1932 from V + supports + solved cosets; `clique` finds exactly
49 for the dim-14 {4,6} support design and 21 for dim 13.
