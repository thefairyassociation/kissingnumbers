# Structural audit: public 841-point configuration in dimension 12

This note records a reproducible numerical audit of the public successful
coordinates from Takhanov, Assylbekov, and Yun's
[`841_in_12D`](https://github.com/k-nic/841_in_12D) repository at
`eba37f0368f62828780d1f9d90315b367d2a612f`.  It uses the already attributed
fixture `lib/testdata/authors_841_coordinates.txt`; it is not an exact
certificate or a new-record claim.

Run the optional analysis with only NumPy:

```bash
python3 kissing/lib/analyze_published_841.py
# Optional full diagnostic; do not commit the generated JSON.
python3 kissing/lib/analyze_published_841.py --json-out /tmp/published841.json
```

The existing `python3 kissing/lib/test_faithful_841.py` regression already
covers direct coordinate normalization, the published maximum, serialization,
and the authors-style decimal-Gram/rank-12 handoff.  No duplicate test was
added here.

## Independent numerical checks

The normalized public coordinates have maximum off-diagonal inner product
`0.4999999377514321`, attained by rows `(413, 756)` under the published row
order.  The Gram matrix has numerical rank 12 at `1e-8`; its most negative
eigenvalue is `-3.58e-14`.  The existing faithful regression also verifies
that an authors-style `%.10f` decimal-Gram -> top-12-eigendirections -> normalization
handoff remains below `0.5`.  These are floating-point checks, not exact
arithmetic verification.

## Relation to the canonical 840 core

After one global orthogonal Procrustes fit of the first 840 public rows to the
canonical Clebsch/Leech--Sloane 840 configuration, the **natural-label** RMS
chordal displacement is `0.8725625` (maximum `1.8851624`).  The 840-by-840
Gram difference has Frobenius norm per row `0.3207033` and maximum entrywise
difference `1.4999991`.

The natural labels also are not close to the paper's local (O(4)) family.
Least-squares fits on the two 48-systems give residual RMS `0.9405858` and
`0.9583907`; their row-ℓ1 norms are `1.8452759` and `1.9503687`.  Theorem 2's
validity condition is at most (3/(2\sqrt2) \approx 1.0606602).  This is a
label-dependent numerical diagnostic, so it neither rules out a different
permutation/representation nor proves non-membership.  It does show that the
published endpoint should not be modelled as a *small*, naturally labelled
(O(4)) perturbation of the core.

## Fingerprints for future candidates

The threshold graph with an edge for IP at least `0.499999` has 7,359 edges,
degrees from 0 to 38, and 48 components (largest 794).  It is numerically
irregular rather than visibly highly symmetric; threshold graphs are not exact
contact graphs because the published margin is only about (6.22\times10^{-8}).

There is strong near-antipodal structure: 417 mutual-nearest-antipode pairs
cover 834 of 841 points; 512 rows have a nearest antipode below `-0.999` and
774 below `-0.99`.  The canonical core has partial antipodal structure too, so
this is a fingerprint for basin deduplication, **not** support for imposing a
hard antipodal constraint on the optimizer.

## Implications

* Treat the public witness as a regression gate and use its Gram/contact and
  antipodal statistics to distinguish saved basins.
* Keep theorem-valid near-identity (O(4)) core deformations as a separate
  breadth arm, but do not expect them to be a direct local interpolation to
  the published endpoint.
* Do not use a fixed-core local-polish failure as a test of whether the
  published basin exists; the numerical geometry indicates substantial
  collective motion.

## Batch-count terminology

The paper reports a search using **64 independent batches**.  Separately, the
public `search_841_riesz.py` code defaults to `B=512` candidates and
`--macro-repeats=100`, which would initialise 51,200 candidates if run with
those defaults.  These are different statements.  This repository must not
describe 51,200 as the paper-reported experimental scale without a source that
establishes that equivalence.
