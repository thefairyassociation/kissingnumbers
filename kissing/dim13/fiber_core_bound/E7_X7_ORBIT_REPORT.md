# Exact Weyl-orbit classification of the E7 X7 complements

The deterministic integer computation in `e7_x7_orbit.py` strengthens the product-core analysis for PackingStar's dimension-13 family.

## Result

- The 126 E7 roots form 63 antipodal root lines.
- A maximum pairwise-nonpositive subset consists of seven mutually orthogonal root lines, hence 14 roots (`X7`).
- There are exactly **135** such `X7` subsets.
- Exact E7 root reflections generate an orbit of size **135** from one witness.
- Therefore all maximum `X7` subsets lie in one Weyl-group orbit.

Every coordinate, dot product, reflection, permutation, clique, and orbit comparison is integral. No floating point is used.

## Consequence for the 1,008-point product core

The saturated PackingStar variant uses 112 of the 126 E7 roots, omitting one `X7`. Because all 135 possible omitted `X7` subsets are Weyl-equivalent, merely choosing a different maximum 14-root omission cannot change the standalone rank-7 cap geometry up to isometry.

This rules out a previously open superficial variation of the fixed product-core construction. A record in this family would need one of the following:

1. a genuinely larger continuous cap for the same isometry class;
2. a different core/cap coupling that is not obtained by replacing one `X7` complement with another; or
3. departure from the saturated E6/E7 product-core family.

It does **not** prove that the 84-point rank-7 cap is globally maximal.

## Reproduce

```bash
python kissing/dim13/fiber_core_bound/e7_x7_orbit.py \
  --output kissing/dim13/fiber_core_bound/e7_x7_orbit.json
```

The exact output, witness indices, and witness root vectors are in `e7_x7_orbit.json`.
