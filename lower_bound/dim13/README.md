# Dimension 13

Record: **1154** (Zinoviev–Ericson 1999). Upper bound: 2064. Target: an
explicit 1155-point configuration that passes the exact verifier.

**Nothing here beats 1154.** This directory holds an exact reproduction of the
record, a structural decomposition of it, and a series of searches for a
1155th point, each of which either came back negative or ran out of time.
The narrative version is in [`../README.md`](../README.md) (sections 1 and 3)
and in [`v2/NOTES.md`](v2/NOTES.md); this page is the index.

## The record configuration

In unit coordinates ZE99 has five heights along the 13th axis:

| layer | height | count | what it is |
| --- | --- | ---: | --- |
| poles | ±1 | 2 | ±e₁₃ |
| tetrads | 0 | 816 | 51 four-subsets × 16 signs, `(±1/2)⁴` |
| half-height layers | ±1/2 | 2 × 168 | 144 "diamonds" `(±1)¹²/√12` + 24 axes |

So `1154 = 2 + 816 + 168 + 168`. A layer at height ±1/2 must be a
**1/3-code** in R¹² that is compatible with the tetrads. Every search below
tries to enlarge one of these pieces, or to trade a few points of one for
more of another.

| file | purpose |
| --- | --- |
| `configs/ze99_1154_exact.json` | the 1154 vectors in exact Q(√3) coordinates |
| `constructions/ze99.py` | builds them |
| `verifier.py`, `_verifier_impl.py`, `test_verifier.py` | exact Q(√3) verifier and its tests |
| `run_baseline.py`, `verifier_output.txt`, `best.json` | reproduction run and its output |
| `data/ze99_structure.json` | the 51 tetrad supports, Steiner system and layer data |

## Searches, and what each one concluded

"Exhaustive" means the stated finite family was searched completely.
"Timed out" means the solver stopped without an answer — that is not an
exclusion.

| question | script | result |
| --- | --- | --- |
| Can one point be added to the 1154? | `../lib/holes.py` | **No.** LP hole search: largest admissible norm 0.9014 < 1. |
| Can any single point move? | `v2/jamming.py` | **No.** Every point is locally jammed. |
| Can a half-height layer have 169 points, from the 4248 primitive directions of magnitude ≤ 2? | [`prove_joint_layer_168.py`](prove_joint_layer_168.py) (proof), `search_ze99_joint_layer.py` (CP-SAT search) | **No — proved.** The maximum is exactly 168 and needs no triple directions. The CP-SAT search itself timed out; the proof replaces it and runs in one second. |
| Can the 24 irrational "auxiliary" directions per layer be 25, among ternary directions? | `search_ze99_aux.py` → [`results/ze99_aux/`](results/ze99_aux/REPORT.md) | **No.** Exhaustive: exactly 24 feasible directions exist. |
| …among primitive directions with entries up to 2? | `search_ze99_aux_b2.py`, `ze99_aux_bounded.py` → [`results/ze99_aux_b2/`](results/ze99_aux_b2/REPORT.md) | **No.** Exhaustive: again exactly 24. |
| …trading one axis for two new directions? | `search_ze99_aux_trade_scan.py` → [`results/ze99_aux_trade/`](results/ze99_aux_trade/REPORT.md) | None found. Numerical sampling only. |
| …anywhere on the sphere (continuous)? | `search_ze99_aux_continuous.py` | None found. The August CI jobs all failed within two minutes. Re-run 2026-11-09 with two seeds (44 and 46 continuous runs each before the session restarted): 24 again among sampled polytope vertices; best 25-point attempt violates a constraint by 0.204. Numerical only. |
| Does a Steiner-switch of the diamonds admit a 1155th integer vector? | `search_ze99_steiner_switch.py` → [`results/ze99_steiner_switch/`](results/ze99_steiner_switch/REPORT.md) | **No.** Exhaustive over the 8192-vector diamond shell. An earlier claim of a 2²⁶⁴ switch family was a duplicate-counting error. |
| Delete one tetrad support (−16) and fit 177 per layer (+18)? | `search_ze99_drop_tetrad_layer.py` → [`results/ze99_drop_tetrad/`](results/ze99_drop_tetrad/REPORT.md) | **Open.** All seven orbit representatives timed out (`UNKNOWN`) on the main case after 22 minutes. |
| Do the PackingStar 1146-point configurations help? | `packingstar/` → [`results/packingstar/`](results/packingstar/REPORT.md) | Both reconstructed and certified exactly (1146 < 1154). Their E6 × E7 fibre core is capped at 1008 (`fiber_core_bound/`). |
| Sections of the Leech lattice or of Ganzhinov's 1932? Integer shells? | `search_leech_sections.py`, `search_ganzhinov_sections.py`, `search_shells.py` | At most 984. |
| Global-linear-code constructions in the norm-8 shell of Z¹³? | `v2/search_globalcode.py`, `v2/search_gc2.py`, `v2/search_gc3.py` | Ruled out structurally: ceiling about 984, best built 760. |
| Group orbits of PSL(2,13) / PGL(2,13)? | `v2/orbits.py` | Did not reach max cosine ≤ 1/2. |
| Numerical optimisation of 1155 free points? | `../lib/fastriesz.c`, `../lib/campaign.py` | Best max inner product **0.506587** (needs ≤ 0.5). Not evidence either way — see the calibration discussion in `../README.md`. |

Other files: `v2/layers.py`, `v2/completeB.py`, `v2/greedyB.py` (layer
decomposition and attempts to enlarge a layer; they share scratch arrays through
`KISS_SCRATCH`), `v2/rigidity.py` and `rigidity/generate_ze99_rigidity.py`
(rigidity matrices; the generator's finite-field rank is not recorded in the
tree), `io_config.py` (config I/O).

## Running

```bash
python3 lower_bound/lib/verify_exact.py lower_bound/dim13/configs/ze99_1154_exact.json
```

```bash
python3 lower_bound/dim13/prove_joint_layer_168.py
```

```bash
python3 lower_bound/dim13/test_verifier.py
```

The CP-SAT searches need `pip install ortools` and take an explicit
`--output-dir`; write to `lower_bound/runs/…`, which is git-ignored, and copy
a finished result into `results/` by hand.
