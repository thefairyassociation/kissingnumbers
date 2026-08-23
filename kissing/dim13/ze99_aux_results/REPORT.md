# Exact ZE99 auxiliary-layer search

ZE99 was held fixed on 1,106 vectors. Each auxiliary unit vector is used
with both last-coordinate signs, so auxiliary size 25 would yield 1,156.

- Feasible ternary directions: **24**.
- Best auxiliary code: **24**.
- Full exactly verified count: **1154**.
- Exact max off-diagonal: **1/2**.
- Exact family optimum proved: **True**.
- Beats the record 1154: **False**.

```json
{
  "auxiliary_count": 24,
  "auxiliary_exact_verification": {
    "count": 24,
    "fixed_base_inequalities_exact": true,
    "pairwise_inner_products_leq_one_third_exact": true,
    "tight_auxiliary_pairs": 0,
    "tight_fixed_constraints": 0
  },
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
    "feasible": 24,
    "support_distribution_before": {
      "1": 24,
      "2": 264,
      "3": 1760,
      "4": 7920,
      "5": 25344,
      "6": 59136,
      "7": 101376,
      "8": 126720,
      "9": 112640,
      "10": 67584,
      "11": 24576,
      "12": 4096
    },
    "support_distribution_feasible": {
      "1": 24
    },
    "tested": 531440
  },
  "fixed_base_count": 1106,
  "full_count": 1154,
  "full_exact_verification": {
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
  "generated_at": "2026-08-23T05:31:17.798652+00:00",
  "method": "exhaustive q/sqrt(|supp q|), q in {0,+/-1}^12, exact clique search",
  "record": 1154,
  "search": {
    "algorithm": "exact branch-and-bound with greedy coloring and integer bitsets",
    "best": 24,
    "nodes": 1,
    "proven_optimal": true,
    "timed_out": false
  },
  "selected_support_distribution": {
    "1": 24
  },
  "status": "completed",
  "target_auxiliary_count": 25
}
```
