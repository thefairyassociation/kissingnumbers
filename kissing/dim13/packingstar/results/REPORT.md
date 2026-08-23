# PackingStar dimension-13 exact reconstruction

Generated 2026-08-23T06:50:27.738625+00:00 from `/tmp/packingstar`.

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
  "available": true,
  "success": true,
  "starts": 90,
  "successful_lp_solves": 331,
  "best_polytope_norm": 0.9354143466934854,
  "best_iterations": 2,
  "single_point_addition_numerically_possible": false,
  "normalized_candidate_max_inner": 0.5345224838248491,
  "normalized_candidate_conflicts": 77,
  "normalized_candidate_conflict_indices": [
    32,
    36,
    38,
    127,
    129,
    163,
    178,
    180,
    185,
    203,
    204,
    212,
    218,
    253,
    271,
    281,
    292,
    302,
    322,
    335,
    469,
    508,
    535,
    536,
    548,
    554,
    558,
    566,
    572,
    586,
    591,
    603,
    616,
    621,
    646,
    698,
    745,
    746,
    752,
    788,
    792,
    805,
    814,
    816,
    824,
    849,
    850,
    852,
    857,
    864,
    870,
    876,
    880,
    889,
    919,
    927,
    931,
    947,
    949,
    951,
    980,
    982,
    983,
    988,
    994,
    1082,
    1087,
    1107,
    1110,
    1117,
    1126,
    1128,
    1134,
    1136,
    1140,
    1141,
    1144
  ],
  "coordinate_norm_error": 4.440892098500626e-16,
  "sample_gram_error": 4.440892098500626e-16,
  "note": "Numerical diagnostic only; it is not an exact kissing certificate."
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
  "available": true,
  "success": true,
  "starts": 90,
  "successful_lp_solves": 336,
  "best_polytope_norm": 0.9354143466934866,
  "best_iterations": 4,
  "single_point_addition_numerically_possible": false,
  "normalized_candidate_max_inner": 0.5345224838248508,
  "normalized_candidate_conflicts": 75,
  "normalized_candidate_conflict_indices": [
    2,
    50,
    89,
    100,
    109,
    144,
    161,
    183,
    205,
    208,
    248,
    263,
    270,
    285,
    308,
    361,
    385,
    391,
    392,
    427,
    436,
    473,
    490,
    504,
    511,
    532,
    549,
    553,
    575,
    577,
    642,
    658,
    674,
    696,
    713,
    727,
    752,
    756,
    766,
    789,
    795,
    796,
    812,
    843,
    845,
    848,
    850,
    876,
    902,
    910,
    945,
    966,
    976,
    979,
    1013,
    1075,
    1081,
    1082,
    1083,
    1086,
    1088,
    1089,
    1090,
    1092,
    1102,
    1113,
    1115,
    1116,
    1117,
    1119,
    1120,
    1127,
    1128,
    1138,
    1140
  ],
  "coordinate_norm_error": 6.661338147750939e-16,
  "sample_gram_error": 4.440892098500626e-16,
  "note": "Numerical diagnostic only; it is not an exact kissing certificate."
}
```

## Interpretation

A verified 1,146-point rational configuration is a useful independent baseline,
but it does **not** improve the live 1,154-point Zinoviev–Ericson record. Any
record attempt starting from it needs a net gain of at least nine points.
