# Exact E6/E7 fiber-core ceiling in dimension 13

The deterministic integer computation in `fiber_core_bound.py` proves two finite root-system facts.

- Among the 126 normalized `E7` roots, a pairwise-nonpositive subset has size at most **14**. There are exactly **135** maximum subsets, and every one is an `X7`: seven antipodal pairs on seven mutually orthogonal lines.
- Among the 72 normalized `E6` roots, a pairwise-nonpositive subset has size at most **9**. There are exactly **320** maximum subsets, and every one is a `Y3`: three mutually orthogonal equilateral triangles.

All coordinates are integers of squared norm 8. Every comparison, clique search, witness check, and structural classification is exact; no floating point is used.

## Consequence for the PackingStar product core

A product point has the form `(a,b)/sqrt(2)`. For a fixed `E6` root `a`, its `E7` neighbors must be pairwise nonpositive, so it has at most 14 neighbors. Therefore the core has at most

```text
72 * 14 = 1008
```

points. If 1,008 is attained, a fixed `E7` root has at most 9 `E6` neighbors, so the core must use at least

```text
ceil(1008 / 9) = 112
```

distinct `E7` roots. At most 14 of the 126 `E7` roots can be omitted to relax the rank-7 cap.

The PackingStar non-antipodal variant attains both equalities: a 1,008-point core supported on 112 `E7` roots. Consequently, this route cannot improve by enlarging the core. A record in this family must instead enlarge the caps or change which 112-root support is used.

This does **not** prove that the existing 84-point rank-7 cap is globally maximal.

## Reproduce

```bash
python kissing/dim13/fiber_core_bound/fiber_core_bound.py \
  --output kissing/dim13/fiber_core_bound/results.json
```

The raw exact output and witnesses are in `results.json`.
