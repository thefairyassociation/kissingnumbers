# Upper bounds

This tree is about proving that *no more than* a certain number of spheres can
touch a central one. It is separate from [`../lower_bound/`](../lower_bound/README.md),
which builds configurations, and shares no code with it.

**Nothing here improves a known upper bound.** What exists is a verified base:
the classical linear-programming bounds in dimensions 3–24 and one
semidefinite certificate, each as a file that a short exact program checks.
The published records come from much larger semidefinite programs; the
section "What it would take" says how far away those are.

## The rule

An upper bound is claimed only for a **certificate that an exact verifier
accepts**. The verifiers use integers and fractions from the Python standard
library, nothing else, and share no code with the programs that search for
certificates. Floating point is used only to *find* candidate numbers.

```bash
python3 upper_bound/lp/verify_certificate.py upper_bound/certificates/lp/*.json
```

```bash
python3 upper_bound/sdp/verify_certificate.py upper_bound/certificates/sdp/*.json
```

Both finish in about a second.

## Results

"Best known" is from [Cohn's table](https://cohn.mit.edu/kissing-numbers)
(fetched 2026-11-09). "Verified here" is the best bound with a certificate in
this repository.

| dim | best known lower | best known upper | verified here | method here |
| ---: | ---: | ---: | ---: | --- |
| 3 | 12 | 12 | **12** | three-point SDP, degree 5 |
| 4 | 24 | 24 | 25 | LP |
| 5 | 40 | 43 | 46 | LP |
| 6 | 72 | 77 | 82 | LP |
| 7 | 126 | 132 | 140 | LP |
| 8 | 240 | 240 | **240** | LP, exact root-form certificate |
| 9 | 306 | 363 | 380 | LP |
| 10 | 510 | 553 | 595 | LP |
| 11 | 604 | 868 | 915 | LP |
| 12 | 841 | 1355 | 1416 | LP |
| 13 | 1154 | 2064 | 2233 | LP |
| 14 | 1932 | 3174 | 3492 | LP |
| 15 | 2564 | 4853 | 5431 | LP |
| 16 | 4320 | 7320 | 8313 | LP |
| 17 | 5730 | 10978 | 12218 | LP |
| 18 | 8358 | 16406 | 17877 | LP |
| 19 | 12460 | 24417 | 25900 | LP |
| 20 | 19448 | 36195 | 37974 | LP |
| 21 | 31560 | 53524 | 56851 | LP |
| 22 | 49896 | 80810 | 86537 | LP |
| 23 | 93150 | 122351 | 128095 | LP |
| 24 | 196560 | 196560 | **196560** | LP, exact root-form certificate |

Bold entries match the true kissing number. All three are classical theorems
(Schütte–van der Waerden 1953; Levenshtein and Odlyzko–Sloane 1979); the LP
column reproduces Odlyzko and Sloane's 1979 table. None of it is new.

## The two methods

### Linear programming (Delsarte) — `lp/`

If `F(t) = Σ f_k P_k(t)` is a combination of Gegenbauer polynomials with
`f_0 > 0`, all other `f_k ≥ 0`, and `F(t) ≤ 0` on `[-1, 1/2]`, then at most
`F(1)/f_0` unit vectors can have pairwise inner products at most 1/2. The
proof is three lines and is in the header of
[`lp/verify_certificate.py`](lp/verify_certificate.py).

A certificate is the list of rational `f_k`. The verifier expands `F`
exactly and proves it is negative on the whole interval with a Sturm
sequence (an exact count of real roots), so there is no sampling and no
tolerance anywhere. For dimensions 8 and 24 the optimal polynomial touches
zero inside the interval; those two certificates list the polynomial's
rational roots instead, and the verifier checks its sign between them.

[`lp/delsarte.py`](lp/delsarte.py) produces the certificates:

```bash
python3 upper_bound/lp/delsarte.py --all
```

### Three-point semidefinite programming (Bachoc–Vallentin) — `sdp/`

The LP only uses pairs of points. The three-point bound also uses triples,
and is strictly stronger: in dimension 3 the LP cannot do better than 13,
while the three-point bound gives 12.

A certificate consists of positive semidefinite matrices together with two
polynomial identities that express "this polynomial is nonnegative on that
region" as a sum of squares. The header of
[`sdp/verify_certificate.py`](sdp/verify_certificate.py) states exactly what
is assumed (two classical positivity theorems), derives the bound from them,
and lists what the program checks: positive definiteness of every matrix by
exact elimination, and both identities coefficient by coefficient.

| file | role | trusted? |
| --- | --- | --- |
| `sdp/verify_certificate.py` | exact verifier | yes — this is the proof checker |
| `sdp/make_certificate.py` | solves the SDP away from the boundary, rounds to rationals, repairs the identities exactly | no — it only has to find numbers the verifier accepts |
| `sdp/three_point.py` | plain floating-point solve, for exploring | no — numerical only |

Numerical optima from `three_point.py` (floating point, **not proofs**), to
show where the method stands at the degrees this machine can reach:

| dim | LP bound | three-point, degree 5 | three-point, degree 6 |
| ---: | ---: | ---: | ---: |
| 3 | 13.16 | 12.87 | – |
| 4 | 25.56 | 25.73 | 25.04 |

At low degree the three-point bound can be *worse* than a high-degree LP
bound, as in dimension 4 at degree 5. Bachoc and Vallentin's published
dimension-4 value, about 24.06, was computed at a higher degree than this
machine reaches with the unreduced model.

## What it would take to go further

The published records for dimensions 9–23 (Machado–de Oliveira Filho 2018,
Leijenhorst–de Laat 2024) are three-point bounds of this same kind at much
higher degree, solved in high-precision arithmetic with purpose-built
solvers and with the polynomial symmetry divided out. The pieces in this tree
are the same ones at toy scale. In order:

1. **Symmetry reduction.** The constraint polynomial is symmetric in its
   three variables, but the sum-of-squares blocks here do not use that. At
   degree 7 the unreduced model needs a 120×120 block and exhausted this
   machine's memory inside the modelling layer. Reducing by the symmetric
   group cuts the blocks roughly sixfold and is what makes degree 10+
   feasible. First target: an exact certificate for `k(4) ≤ 24` (needs
   degree ≥ 7).
2. **A better polynomial basis.** Monomials are badly conditioned beyond
   degree ~10; the literature uses orthogonal or sampled bases.
3. **A high-precision solver.** Double precision is enough to round a
   certificate at degree 5; it is not expected to be at the degrees the
   records use.
4. Only then does improving a record become a question of computing time
   rather than of missing machinery — and the existing records already used
   a great deal of it.

## Tests

```bash
python3 upper_bound/tests/test_lp_certificates.py
```

```bash
python3 upper_bound/tests/test_sdp_certificates.py
```

They check the building blocks against known closed forms and confirm that
altered certificates are rejected: a negative coefficient, a polynomial that
turns positive, a changed matrix entry, an indefinite or asymmetric block,
and a valid certificate relabelled with another dimension or angle. CI runs
them and re-verifies every committed certificate.
