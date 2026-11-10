# Kissing numbers

The **kissing number** k(n) is the largest number of unit spheres that can
touch one central unit sphere in R^n without overlapping; equivalently, the
most unit vectors with all pairwise inner products at most 1/2. It is known
exactly only for n = 1, 2, 3, 4, 8 and 24. Everywhere else there is a gap
between the best construction (a lower bound) and the best proof of
impossibility (an upper bound).

This repository works on both ends of that gap, in two separate trees:

| tree | question | what counts as a result |
| --- | --- | --- |
| [`lower_bound/`](lower_bound/README.md) | can we *build* more spheres? | an explicit configuration, checked in exact arithmetic |
| [`upper_bound/`](upper_bound/README.md) | can we *prove* fewer are possible? | a certificate that an exact verifier accepts |

**No record has been beaten in either direction.** For the short,
plain-English account of what has been done and what is still open, read
[`STATUS.md`](STATUS.md).

## Where things stand

Records are from [Henry Cohn's table](https://cohn.mit.edu/kissing-numbers), fetched 2026-11-09.

| dim | best known lower | best known upper | verified in this repository |
| ---: | ---: | ---: | --- |
| 3 | 12 | 12 | upper bound 12, exact three-point (semidefinite) certificate |
| 4 | 24 | 24 | upper bound 24, exact three-point (semidefinite) certificate |
| 8 | 240 | 240 | upper bound 240, exact LP certificate |
| 12 | 841 | 1355 | lower 841 (the published configuration, certified exactly here); upper 1416 (LP) |
| 13 | 1154 | 2064 | lower 1154 (reproduction, exact); upper 2233 (LP) |
| 14 | 1932 | 3174 | lower 1932 (reproduction, exact); upper 3492 (LP) |
| 24 | 196560 | 196560 | upper bound 196560, exact LP certificate |

Every bound in the right-hand column is a known result re-verified here, not
a new one. The upper bounds are the classical linear-programming bounds
(plus semidefinite certificates in dimensions 3 and 4); in dimensions 12–14 they
are weaker than the published semidefinite records. The full table is in
[`upper_bound/README.md`](upper_bound/README.md).

## Layout

```
lower_bound/      constructions and searches
  dim12/ dim13/ dim14/   per-dimension constructions, searches and results
  lib/                   verifiers, optimisers (C), search drivers
  campaigns/             records of large search runs on GitHub Actions
  experimental/          opt-in batched search and decimal certificates
  docs/                  optimiser calibration notes and reviews
  logs/progress.log      append-only log of every attempt
upper_bound/      proofs of impossibility
  lp/                    Delsarte LP bound: generator and exact verifier
  sdp/                   three-point semidefinite bound: solver, certificate maker, exact verifier
  certificates/          machine-checkable certificates
docs/HISTORY.md   what each old branch and pull request contained, and where it went
STATUS.md         plain-English status of both tracks
```

## Checking it yourself

```bash
python3 -m pip install numpy scipy scipy-openblas32 sympy
```

```bash
make -C lower_bound/lib all
```

Lower bounds (exact verification of the three record configurations):

```bash
python3 lower_bound/lib/verify_exact.py lower_bound/dim13/configs/ze99_1154_exact.json
```

```bash
python3 lower_bound/lib/certify_float.py lower_bound/lib/testdata/authors_841_coordinates.txt
```

Upper bounds (standard library only, nothing to install):

```bash
python3 upper_bound/lp/verify_certificate.py upper_bound/certificates/lp/*.json
```

```bash
python3 upper_bound/sdp/verify_certificate.py upper_bound/certificates/sdp/*.json
```

The full test suites and what they protect are described in
[`lower_bound/README.md`](lower_bound/README.md#running-the-checks) and
[`upper_bound/README.md`](upper_bound/README.md). CI
([`.github/workflows/checks.yml`](.github/workflows/checks.yml)) runs all of
them on every push to `main` and on every pull request.

## Ground rules

* A lower bound is only claimed for a configuration that passes an **exact**
  verifier (`verify_exact.py` for algebraic coordinates, `certify_float.py`
  for a float configuration with a strict margin). An optimiser's own report
  is never evidence; two false "feasible" reports have been produced by
  numerical bugs here and both were caught this way.
* An upper bound is only claimed for a certificate that
  one of the exact verifiers under `upper_bound/` accepts.
  A number printed by a semidefinite solver is not a proof and is labelled as
  numerical wherever it appears.
* Negative results are recorded with their reason, and timeouts are recorded
  as timeouts, not as exclusions.
