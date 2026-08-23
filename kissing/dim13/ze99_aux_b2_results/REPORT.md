# ZE99 primitive bound-2 auxiliary search

- Exact feasible directions: **24**.
- Best auxiliary code: **24**.
- Full exact count: **1154**.
- Exact family optimum proved: **True**.
- Beats 1154: **False**.

```json
{
  "auxiliary_count": 24,
  "beats_record": false,
  "compatibility_graph": {
    "edges": 276,
    "maximum_degree": 23,
    "minimum_degree": 23,
    "vertices": 24
  },
  "constraint_rows": {
    "centrally_symmetric": true,
    "diamond_sign_rows": 144,
    "signed_tetrads": 816,
    "total": 960
  },
  "dimension": 13,
  "enumeration": {
    "bound": 2,
    "feasible_norm_squared_distribution": {
      "1": 24
    },
    "half_signings_tested_after_tetrads": 2124,
    "magnitude_patterns_after_tetrads": 29,
    "magnitude_patterns_total": 527345,
    "magnitude_support_distribution_after_tetrads": {
      "1": 12,
      "3": 16,
      "12": 1
    },
    "signed_candidates_feasible": 24
  },
  "exact_verification": {
    "all_off_diagonal_leq_8": true,
    "arithmetic": "integers, rational squares, and SymPy exact radicals; no floats",
    "auxiliary_pairs_checked_by_integer_squared_inequalities": 1128,
    "auxiliary_verification": {
      "count": 24,
      "fixed_base_inequalities_exact": true,
      "pairwise_inner_products_leq_one_third_exact": true,
      "tight_auxiliary_pairs": 0,
      "tight_fixed_constraints": 0
    },
    "base_pairs_checked_by_Q_sqrt3_verifier": 611065,
    "count": 1154,
    "diagonal_exactly_16": true,
    "dimension": 13,
    "distinct": true,
    "fixed_auxiliary_pairs_checked_by_integer_squared_inequalities": 53088,
    "fixed_base_verification": {
      "all_offdiag_leq_bound": true,
      "count": 1106,
      "dim": 13,
      "distinct": true,
      "max_offdiag_pair": [
        0,
        1
      ],
      "max_offdiag_unit": "1/2",
      "max_offdiag_unnormalized": "8",
      "method": "Zinoviev-Ericson 1999 reproduction (exact Q(sqrt(3)))",
      "n_tight_pairs": 59568,
      "norm2": 16,
      "ok": true
    },
    "max_off_diagonal_unit": "1/2",
    "max_off_diagonal_unnormalized": "8",
    "norm_squared": 16,
    "ok": true,
    "total_unordered_gram_pairs_checked": 665281
  },
  "fixed_base_count": 1106,
  "full_count": 1154,
  "generated_at": "2026-08-23T06:05:23.093548+00:00",
  "method": "primitive q/||q|| with q in {-2,-1,0,1,2}^12",
  "record": 1154,
  "search": {
    "algorithm": "exact branch-and-bound with greedy coloring and integer bitsets",
    "best": 24,
    "nodes": 1,
    "proven_optimal": true,
    "timed_out": false
  },
  "selected_norm_squared": {
    "1": 24
  },
  "status": "completed"
}
```
