# Exact one-tetrad-deletion layer search

Generated `2026-08-23T20:35:51.533845+00:00`.

- Dropped support index: **8**.
- Dropped support: `[0, 2, 3, 8]`.
- Exact candidate directions: **4520**.
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
  "drop_index": 8,
  "dropped_support": [
    0,
    2,
    3,
    8
  ],
  "dropped_support_mask": 269,
  "enumeration": {
    "arithmetic": "integer squared inequalities; no floating point",
    "binary_sign_directions": 4096,
    "candidate_count": 4520,
    "coordinate_bound": 2,
    "drop_index": 8,
    "magnitude_norm_squared_distribution": {
      "1": 12,
      "3": 20,
      "4": 1,
      "7": 6,
      "12": 2,
      "13": 4
    },
    "magnitude_patterns_after_retained_tetrads": 45,
    "magnitude_support_distribution": {
      "1": 12,
      "3": 20,
      "4": 11,
      "6": 1,
      "12": 1
    },
    "nonbinary_directions": 424,
    "primitive_magnitude_patterns_tested": 527345,
    "retained_tetrad_supports": 50,
    "signed_direction_norm_squared_distribution": {
      "1": 24,
      "12": 4160,
      "13": 64,
      "3": 160,
      "4": 16,
      "7": 96
    }
  },
  "exact_verification": null,
  "generated_at": "2026-08-23T20:35:51.533845+00:00",
  "method": "delete one 16-vector ZE99 tetrad support and exactly search a mixed primitive bound-2 half-height layer",
  "record_before": 1154,
  "selected_layer": null,
  "solver_cases": [
    {
      "case": "with_sign",
      "model": {
        "binary_sign_variables": 4096,
        "conditional_sign_terms": 163840,
        "edge_anticode_constraints": 24576,
        "known_binary_bound": "A_2(12,4)=144",
        "nonbinary_forbidden_sign_size_distribution": {
          "0": 24,
          "256": 112,
          "320": 64,
          "512": 224
        },
        "nonbinary_pair_conflicts": 13632,
        "nonbinary_variables": 424,
        "symmetry_break": "all-plus binary sign selected",
        "target_layer": 177
      },
      "num_branches": 66116,
      "num_conflicts": 1400,
      "response_stats": "CpSolverResponse summary:\nstatus: UNKNOWN\nobjective: 0\nbest_bound: 0\nintegers: 4256\nbooleans: 4171\nconflicts: 1400\nbranches: 66116\npropagations: 9718727\ninteger_propagations: 11151707\nrestarts: 5\nlp_iterations: 1377167\nwalltime: 1320.03\nusertime: 1320.03\ndeterministic_time: 4471.47\ngap_integral: 0\n",
      "solution": null,
      "status": "UNKNOWN",
      "time_limit_seconds": 1320.0,
      "user_time_seconds": 1320.033373021,
      "wall_time_seconds": 1320.0333729610002
    },
    {
      "case": "signless",
      "model": {
        "binary_sign_variables": 4096,
        "conditional_sign_terms": 163840,
        "edge_anticode_constraints": 24576,
        "known_binary_bound": "A_2(12,4)=144",
        "nonbinary_forbidden_sign_size_distribution": {
          "0": 24,
          "256": 112,
          "320": 64,
          "512": 224
        },
        "nonbinary_pair_conflicts": 13632,
        "nonbinary_variables": 424,
        "symmetry_break": "no binary signs",
        "target_layer": 177
      },
      "num_branches": 0,
      "num_conflicts": 0,
      "response_stats": "CpSolverResponse summary:\nstatus: INFEASIBLE\nobjective: NA\nbest_bound: NA\nintegers: 0\nbooleans: 0\nconflicts: 0\nbranches: 0\npropagations: 0\ninteger_propagations: 0\nrestarts: 0\nlp_iterations: 0\nwalltime: 0.247009\nusertime: 0.247009\ndeterministic_time: 0.0175868\ngap_integral: 0\n",
      "solution": null,
      "status": "INFEASIBLE",
      "time_limit_seconds": 120.0,
      "user_time_seconds": 0.24700864,
      "wall_time_seconds": 0.24700858
    }
  ],
  "status": "completed",
  "target": 1155
}
```
