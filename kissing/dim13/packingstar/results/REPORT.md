# PackingStar dimension-13 exact reconstruction

Source revision `50ea645a9805d4f29b96180550186d26a166c3be` from `/tmp/packingstar/input`.

This independently reconstructs exact rational Gram matrices from the public
PackingStar `.npy` files. A successful row proves symmetry, unit diagonal,
off-diagonal bound `<= 1/2`, positive semidefiniteness, rank 13, and distinctness
using integer/rational arithmetic. It also emits explicit exact coordinates.

| file | count | exact | max offdiag | rank | beats 1154 |
| --- | ---: | :---: | ---: | ---: | :---: |
| `13D-1146-cosmatrix.npy` | 1146 | yes | 1/2 | 13 | no |
| `13D-1146-cosmatrix_variant_1.npy` | 1146 | yes | 1/2 | 13 | no |

## Results

### 13D-1146-cosmatrix.npy

- Count: **1146**; current dimension-13 record: **1154**.
- Exact max off-diagonal: **1/2**.
- Exact rank: **13**; positive semidefinite: **yes**.
- Tight unordered pairs: **63270**.
- Exact antipodal pairs: **573**.
- Common Gram denominator: **4**.
- Reconstruction error from stored floats: `0.000e+00`.
- Exact coordinate certificate: `kissing/dim13/packingstar/results/configs/13D-1146-cosmatrix_exact.json`.

Raw exact-verifier result:

```json
{
  "status": "verified_exactly",
  "dimension": 13,
  "count": 1146,
  "rank": 13,
  "positive_semidefinite": true,
  "max_off_diagonal": "1/2",
  "max_off_diagonal_pair": [
    0,
    2
  ],
  "tight_pairs": 63270,
  "unique_rows": 1146,
  "beats_live_record": false,
  "live_record": 1154,
  "factorization_method": "numpy int64, proven bounds first=22464, final=1168128 < 2^62"
}
```

Single-point hole diagnostic (numerical only):

```json
{
  "available": false,
  "reason": "disabled for deterministic validation",
  "note": "The numerical hole search is not part of the exact certificate."
}
```

### 13D-1146-cosmatrix_variant_1.npy

- Count: **1146**; current dimension-13 record: **1154**.
- Exact max off-diagonal: **1/2**.
- Exact rank: **13**; positive semidefinite: **yes**.
- Tight unordered pairs: **63354**.
- Exact antipodal pairs: **69**.
- Common Gram denominator: **4**.
- Reconstruction error from stored floats: `0.000e+00`.
- Exact coordinate certificate: `kissing/dim13/packingstar/results/configs/13D-1146-cosmatrix_variant_1_exact.json`.

Raw exact-verifier result:

```json
{
  "status": "verified_exactly",
  "dimension": 13,
  "count": 1146,
  "rank": 13,
  "positive_semidefinite": true,
  "max_off_diagonal": "1/2",
  "max_off_diagonal_pair": [
    0,
    27
  ],
  "tight_pairs": 63354,
  "unique_rows": 1146,
  "beats_live_record": false,
  "live_record": 1154,
  "factorization_method": "numpy int64, proven bounds first=1340352, final=69698304 < 2^62"
}
```

Single-point hole diagnostic (numerical only):

```json
{
  "available": false,
  "reason": "disabled for deterministic validation",
  "note": "The numerical hole search is not part of the exact certificate."
}
```

## Interpretation

A verified 1,146-point rational configuration is a useful independent baseline,
but it does **not** improve the live 1,154-point Zinoviev–Ericson record. Any
record attempt starting from it needs a net gain of at least nine points.
