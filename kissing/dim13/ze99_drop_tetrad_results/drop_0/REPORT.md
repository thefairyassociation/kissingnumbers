# Exact one-tetrad-deletion layer search

Generated `2026-08-23T20:36:10.906837+00:00`.

- Dropped support index: **0**.
- Dropped support: `[1, 2, 3, 5]`.
- Exact candidate directions: **4712**.
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
  "drop_index": 0,
  "dropped_support": [
    1,
    2,
    3,
    5
  ],
  "dropped_support_mask": 46,
  "enumeration": {
    "arithmetic": "integer squared inequalities; no floating point",
    "binary_sign_directions": 4096,
    "candidate_count": 4712,
    "coordinate_bound": 2,
    "drop_index": 0,
    "magnitude_norm_squared_distribution": {
      "1": 12,
      "3": 20,
      "4": 1,
      "7": 6,
      "12": 3,
      "13": 4,
      "19": 1
    },
    "magnitude_patterns_after_retained_tetrads": 47,
    "magnitude_support_distribution": {
      "1": 12,
      "3": 20,
      "4": 11,
      "6": 2,
      "7": 1,
      "12": 1
    },
    "nonbinary_directions": 616,
    "primitive_magnitude_patterns_tested": 527345,
    "retained_tetrad_supports": 50,
    "signed_direction_norm_squared_distribution": {
      "1": 24,
      "12": 4224,
      "13": 64,
      "19": 128,
      "3": 160,
      "4": 16,
      "7": 96
    }
  },
  "exact_verification": null,
  "generated_at": "2026-08-23T20:36:10.906837+00:00",
  "method": "delete one 16-vector ZE99 tetrad support and exactly search a mixed primitive bound-2 half-height layer",
  "record_before": 1154,
  "selected_layer": null,
  "solver_cases": [
    {
      "case": "with_sign",
      "model": {
        "binary_sign_variables": 4096,
        "conditional_sign_terms": 229376,
        "edge_anticode_constraints": 24576,
        "known_binary_bound": "A_2(12,4)=144",
        "nonbinary_forbidden_sign_size_distribution": {
          "0": 24,
          "256": 112,
          "320": 128,
          "352": 128,
          "512": 224
        },
        "nonbinary_pair_conflicts": 33664,
        "nonbinary_variables": 616,
        "symmetry_break": "all-plus binary sign selected",
        "target_layer": 177
      },
      "num_branches": 8758,
      "num_conflicts": 0,
      "response_stats": "CpSolverResponse summary:\nstatus: UNKNOWN\nobjective: 0\nbest_bound: 0\nintegers: 4438\nbooleans: 4349\nconflicts: 0\nbranches: 8758\npropagations: 1511301\ninteger_propagations: 1788183\nrestarts: 0\nlp_iterations: 61440\nwalltime: 1320.04\nusertime: 1320.04\ndeterministic_time: 5436.19\ngap_integral: 0\n",
      "solution": null,
      "status": "UNKNOWN",
      "time_limit_seconds": 1320.0,
      "user_time_seconds": 1320.043568329,
      "wall_time_seconds": 1320.043568283
    },
    {
      "case": "signless",
      "model": {
        "binary_sign_variables": 4096,
        "conditional_sign_terms": 229376,
        "edge_anticode_constraints": 24576,
        "known_binary_bound": "A_2(12,4)=144",
        "nonbinary_forbidden_sign_size_distribution": {
          "0": 24,
          "256": 112,
          "320": 128,
          "352": 128,
          "512": 224
        },
        "nonbinary_pair_conflicts": 33664,
        "nonbinary_variables": 616,
        "symmetry_break": "no binary signs",
        "target_layer": 177
      },
      "num_branches": 0,
      "num_conflicts": 0,
      "response_stats": "CpSolverResponse summary:\nstatus: INFEASIBLE\nobjective: NA\nbest_bound: NA\nintegers: 0\nbooleans: 0\nconflicts: 0\nbranches: 0\npropagations: 0\ninteger_propagations: 0\nrestarts: 0\nlp_iterations: 0\nwalltime: 0.284714\nusertime: 0.284714\ndeterministic_time: 0.0559738\ngap_integral: 0\n",
      "solution": null,
      "status": "INFEASIBLE",
      "time_limit_seconds": 120.0,
      "user_time_seconds": 0.284713973,
      "wall_time_seconds": 0.284713934
    }
  ],
  "status": "completed",
  "target": 1155
}
```
