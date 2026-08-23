# Exact Weyl-orbit classification of the E6 Y3 fibers

The deterministic integer computation in `e6_y3_orbit.py` classifies every maximum pairwise-nonpositive subset of the 72 E6 roots.

## Result

- A maximum pairwise-nonpositive subset of E6 has **9 roots**.
- Each maximum is three mutually orthogonal equilateral root triangles (`Y3`).
- There are exactly **320** maximum `Y3` subsets.
- The 36 distinct root reflections generate an orbit of size **320** from one witness.
- Therefore all maximum `Y3` subsets lie in one E6 Weyl-group orbit.

Every coordinate, dot product, reflection, permutation, clique, component classification, and orbit comparison is integral. No floating point is used.

## Consequence for the saturated E6/E7 product core

The 1,008-point product core saturates both local degree bounds: an E6 root can meet at most 14 E7 roots, while an E7 root can meet at most 9 E6 roots. The latter 9-root neighborhoods are `Y3` subsets. This computation proves that all standalone maximum `Y3` neighborhoods are locally isometric.

Changing one maximum E6 fiber to another therefore cannot introduce a new local fiber geometry. It does **not** prove global uniqueness of the 1,008-edge incidence structure: distinct couplings among many locally equivalent `Y3` and `X7` fibers may still exist.

Together with the exact `X7` orbit classification, this removes both obvious local-choice parameters. Any improvement within the product-core family must exploit globally different incidence coupling, a larger continuous cap, or a non-product construction.

## Reproduce

```bash
python kissing/dim13/fiber_core_bound/e6_y3_orbit.py \
  --output kissing/dim13/fiber_core_bound/e6_y3_orbit.json
```
