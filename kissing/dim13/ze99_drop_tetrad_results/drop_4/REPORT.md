# Exact one-tetrad-deletion layer search

Generated `2026-08-23T20:36:12.255986+00:00`.

- Dropped support index: **4**.
- Dropped support: `[2, 3, 4, 7]`.
- Exact candidate directions: **4744**.
- Target half-height layer: **177**.
- Record witness found: **False**.
- Full dimension-13 count: **1154**.

- `with_sign` CP-SAT status: **UNKNOWN**.
- `signless` CP-SAT status: **INFEASIBLE**.

No lower-bound improvement is claimed by this run.

```json
{
  "baseline_exact_verification": {
    "all_offdiag_leq_bound": true,
    "count": 1154,
    "dim": 13,
    "distinct": true,
    "max_offdiag_pair": [
      0,
      1
    ],
    "max_offdiag_unit": "1/2",
    "max_offdiag_unnormalized": "8",
    "method": "Zinoviev-Ericson 1999 reproduction (exact Q(sqrt(3)))",
    "n_tight_pairs": 59640,
    "norm2": 16,
    "ok": true
  },
  "beats_record": false,
  "binary_distance_model": {
    "anticodes": 24576,
    "extraneous_compatible_pairs": 0,
    "forbidden_binary_pairs_covered": 610304,
    "missing_forbidden_pairs": 0,
    "vertices_per_anticode": 24
  },
  "candidate_count": 1154,
  "configuration": null,
  "dimension": 13,
  "drop_index": 4,
  "dropped_support": [
    2,
    3,
    4,
    7
  ],
  "dropped_support_mask": 156,
  "enumeration": {
    "arithmetic": "integer squared inequalities; no floating point",
    "binary_sign_directions": 4096,
    "candidate_count": 4744,
    "coordinate_bound": 2,
    "drop_index": 4,
    "magnitude_norm_squared_distribution": {
      "1": 12,
      "3": 20,
      "4": 1,
      "7": 4,
      "12": 2,
      "13": 4,
      "19": 2
    },
    "magnitude_patterns_after_retained_tetrads": 45,
    "magnitude_support_distribution": {
      "1": 12,
      "3": 20,
      "4": 9,
      "6": 1,
      "7": 2,
      "12": 1
    },
    "nonbinary_directions": 648,
    "primitive_magnitude_patterns_tested": 527345,
    "retained_tetrad_supports": 50,
    "signed_direction_norm_squared_distribution": {
      "1": 24,
      "12": 4160,
      "13": 64,
      "19": 256,
      "3": 160,
      "4": 16,
      "7": 64
    }
  },
  "exact_verification": null,
  "generated_at": "2026-08-23T20:36:12.255986+00:00",
  "method": "delete one 16-vector ZE99 tetrad support and exactly search a mixed primitive bound-2 half-height layer",
  "record_before": 1154,
  "selected_layer": null,
  "solver_cases": [
    {
      "case": "with_sign",
      "model": {
        "binary_sign_variables": 4096,
        "conditional_sign_terms": 245760,
        "edge_anticode_constraints": 24576,
        "known_binary_bound": "A_2(12,4)=144",
        "nonbinary_forbidden_sign_size_distribution": {
          "0": 24,
          "256": 80,
          "320": 64,
          "352": 256,
          "512": 224
        },
        "nonbinary_pair_conflicts": 37984,
        "nonbinary_variables": 648,
        "symmetry_break": "all-plus binary sign selected",
        "target_layer": 177
      },
      "num_branches": 93757,
      "num_conflicts": 3766,
      "response_stats": "CpSolverResponse summary:\nstatus: UNKNOWN\nobjective: 0\nbest_bound: 0\nintegers: 4460\nbooleans: 4371\nconflicts: 3766\nbranches: 93757\npropagations: 12852681\ninteger_propagations: 14837053\nrestarts: 6\nlp_iterations: 1898598\nwalltime: 1320.39\nusertime: 1320.39\ndeterministic_time: 3987.55\ngap_integral: 0\n",
      "solution": null,
      "status": "UNKNOWN",
      "time_limit_seconds": 1320.0,
      "user_time_seconds": 1320.393019197,
      "wall_time_seconds": 1320.393019117
    },
    {
      "case": "signless",
      "model": {
        "binary_sign_variables": 4096,
        "conditional_sign_terms": 245760,
        "edge_anticode_constraints": 24576,
        "known_binary_bound": "A_2(12,4)=144",
        "nonbinary_forbidden_sign_size_distribution": {
          "0": 24,
          "256": 80,
          "320": 64,
          "352": 256,
          "512": 224
        },
        "nonbinary_pair_conflicts": 37984,
        "nonbinary_variables": 648,
        "symmetry_break": "no binary signs",
        "target_layer": 177
      },
      "num_branches": 0,
      "num_conflicts": 0,
      "response_stats": "CpSolverResponse summary:\nstatus: INFEASIBLE\nobjective: NA\nbest_bound: NA\nintegers: 0\nbooleans: 0\nconflicts: 0\nbranches: 0\npropagations: 0\ninteger_propagations: 0\nrestarts: 0\nlp_iterations: 0\nwalltime: 0.360284\nusertime: 0.360284\ndeterministic_time: 0.0817187\ngap_integral: 0\n",
      "solution": null,
      "status": "INFEASIBLE",
      "time_limit_seconds": 120.0,
      "user_time_seconds": 0.36028442600000005,
      "wall_time_seconds": 0.36028434600000003
    }
  ],
  "status": "completed",
  "target": 1155
}
```
