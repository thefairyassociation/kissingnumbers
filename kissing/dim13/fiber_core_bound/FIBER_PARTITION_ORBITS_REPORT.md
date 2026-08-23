# Exact partition orbits of the saturated E6/E7 product core

PackingStar's non-antipodal 1,008-point core is a disjoint union of eight complete `K_9,14` fibers. Exact local bounds force the two sides of every saturated product core to have the following form:

- the 72 E6 roots are partitioned into eight maximum 9-root `Y3` subsets;
- the 126 E7 roots are partitioned into nine maximum 14-root `X7` subsets, one omitted and eight used by the core.

`fiber_partition_orbits.py` exhaustively enumerates these partitions and classifies them under exact root reflections. It uses integer coordinates, bitset clique enumeration, exact-cover backtracking, and exact permutation orbits. No floating point is used.

## E7 side

- Maximum `X7` subsets: **135**.
- Partitions of all 63 antipodal root lines into nine `X7` blocks: **960**.
- Weyl orbits of partitions: **1**, of size **960**.
- Flagged partitions, in which one `X7` is distinguished as the omitted block: **8,640**.
- Weyl orbits of flagged partitions: **1**, of size **8,640**.

Thus both the complete E7 partition and the choice of omitted `X7` are unique up to E7 Weyl symmetry.

## E6 side

- Maximum `Y3` subsets: **320**.
- Partitions of all 72 roots into eight `Y3` blocks: **17,920**.
- Weyl orbits of partitions: **13**.
- Orbit sizes:

```text
40, 360, 480, 480, 480, 960,
1440, 1440, 1440, 2160, 2880, 2880, 2880
```

The JSON certificate records a deterministic witness and pair-profile invariant for every orbit.

## Consequence

The E7 half of the saturated product-core construction has no remaining discrete partition choice up to symmetry. The E6 half has exactly 13 global partition geometries. This sharply reduces the remaining structured search: after the local and E7 choices are quotiented out, the only discrete product-core alternatives are these 13 E6 partition orbits, together with their coupling to the caps.

This does not prove that the 1,008-point core is globally unique, nor that the 54- and 84-point caps are globally maximal. It provides a complete exact classification within the saturated disjoint-`K_9,14` product-fiber model.

## Reproduce

```bash
python kissing/dim13/fiber_core_bound/fiber_partition_orbits.py \
  --output kissing/dim13/fiber_core_bound/fiber_partition_orbits.json
```
