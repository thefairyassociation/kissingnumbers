# Exact certificates for finite-decimal directions

`check.py` independently verifies a whitespace-separated coordinate file. Each
non-blank line is one vector; lines may contain `#` comments, including inline
comments. Coordinates must be finite decimal strings (scientific notation is
accepted). The checker parses them as exact `Decimal` values and `Fraction`s,
clears each row's denominators to an integer direction `v`, and computes
`q = v · v`.

For every pair, with `p = vi · vj`, it checks `p <= 0` or
`4*p*p <= qi*qj`. This is exactly the condition that the algebraically
normalized vectors `v/sqrt(q)` have inner product at most `1/2`, even when
their norms contain different square roots. `--strict` uses `<` for positive
`p`, which rejects an exact boundary pair. Every operation in the certificate
path uses exact integers; no floating-point estimate or tolerance is used.

The command exits 0 only for a certificate, and emits JSON. Use `--compact`
for a one-line result suitable for logs:

```bash
python3 kissing/experimental/decimal_certificate/check.py \
  kissing/lib/testdata/authors_841_coordinates.txt \
  --expected-dimension 12 --expected-count 841 --strict --compact \
  --label 'KNOWN published witness reproduction'
```

The result records the SHA256 of the original coordinate bytes, parsed source
count, checked pair count, exact arithmetic details, and pair counts. The
label is provenance metadata; it is never used to decide whether the rows
pass. Duplicate rows fail naturally as a pair violation. Zero rows,
non-finite or malformed coordinates, inconsistent dimensions, and wrong
counts are rejected before certification.

The checked 841-row file is a reproduction of the published dimension-12
witness. It is a known-witness verification fixture, not a new search result
or a recovery claim.

Run the focused tests with:

```bash
python3 -m unittest discover -s kissing/experimental/decimal_certificate -p 'test_*.py'
```
