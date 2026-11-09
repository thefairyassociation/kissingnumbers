#!/usr/bin/env python3
"""Exhaust an exact ternary family for enlarging ZE99's 24-vector auxiliary code.

For unit y in R^12 and s in {+1,-1}, put u(y,s)=(2*sqrt(3)*y,2s).
ZE99 is its fixed first 1106 vectors plus u(+/-e_i,s).  Exact compatibility
requires a.y <= 2/sqrt(3) for 960 signed base rows and y.z <= 1/3 within
each last-sign layer.  Thus 25 feasible y's give 1106+2*25=1156 vectors.
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE / "constructions"))
from ze99 import ab_to_strings, generate_ab, verify_ab  # noqa: E402
from ze99_aux_core import (  # noqa: E402
    SearchError,
    baseline_clique,
    compatibility_graph,
    constraint_rows,
    enumerate_candidates,
    exact_maximum_clique,
    verify_selected,
)

RECORD = 1154
FIXED = 1106


def auxiliary_vector(q: np.ndarray, k: int, last: int) -> list[tuple[int, int]]:
    """Return Q(sqrt(3)) coordinates for 2*sqrt(3)*q/sqrt(k),2*last."""
    out = [(0, 0)] * 13
    if k == 1:
        for i, value in enumerate(q.tolist()):
            if value:
                out[i] = (0, 2 * int(value))
    elif k == 3:
        for i, value in enumerate(q.tolist()):
            if value:
                out[i] = (2 * int(value), 0)
    elif k == 12:
        for i, value in enumerate(q.tolist()):
            out[i] = (int(value), 0)
    else:
        raise SearchError(f"unexpected feasible support size {k}; exact field expanded")
    out[12] = (2 * last, 0)
    return out


def build_full(q: np.ndarray, k: np.ndarray) -> tuple[list, dict]:
    original = generate_ab()
    full = list(original[:FIXED])
    for last in (1, -1):
        for row, support in zip(q, k):
            full.append(auxiliary_vector(row, int(support), last))
    verification = verify_ab(full)
    if not verification.get("ok"):
        raise SearchError(f"full exact ZE99 verifier failed: {verification}")
    expected = FIXED + 2 * len(q)
    if len(full) != expected or verification.get("count") != expected:
        raise SearchError("full exact count mismatch")
    return full, verification


def write_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--time-limit", type=float, default=1200.0)
    parser.add_argument("--progress-log", type=Path)
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now(timezone.utc).isoformat()

    if args.progress_log:
        args.progress_log.parent.mkdir(parents=True, exist_ok=True)
        with args.progress_log.open("a", encoding="utf-8") as log:
            log.write(
                f"{stamp} method='ZE99 ternary auxiliary exact search' dimension=13 "
                "count=? result=started\n"
            )

    rows, row_info = constraint_rows(generate_ab)
    candidates, supports, enum_info = enumerate_candidates(rows)
    allowed = set(map(int, supports.tolist()))
    if not allowed <= {1, 3, 12}:
        raise SearchError(f"unexpected feasible support sizes {sorted(allowed)}")
    adjacency, order, graph_info = compatibility_graph(candidates, supports)
    baseline = baseline_clique(candidates, order)
    baseline_original = order[np.asarray(baseline, dtype=np.int32)]
    verify_selected(rows, candidates[baseline_original], supports[baseline_original])

    def checkpoint(clique: list[int], nodes: int) -> None:
        original = order[np.asarray(clique, dtype=np.int32)]
        write_json(
            args.output_dir / "checkpoint.json",
            {
                "generated_at": datetime.now(timezone.utc).isoformat(),
                "auxiliary_count": len(clique),
                "full_count": FIXED + 2 * len(clique),
                "nodes": nodes,
                "q": candidates[original].astype(int).tolist(),
                "support_sizes": supports[original].astype(int).tolist(),
            },
        )

    best_reordered, search_info = exact_maximum_clique(
        adjacency, baseline, args.time_limit, checkpoint
    )
    best_original = order[np.asarray(best_reordered, dtype=np.int32)]
    selected_q = candidates[best_original]
    selected_k = supports[best_original]
    auxiliary_check = verify_selected(rows, selected_q, selected_k)
    full, full_check = build_full(selected_q, selected_k)
    full_count = len(full)
    success = full_count > RECORD

    support_distribution = {
        str(int(value)): int(count)
        for value, count in zip(*np.unique(selected_k, return_counts=True))
    }
    analysis = {
        "generated_at": stamp,
        "status": "completed",
        "method": "exhaustive q/sqrt(|supp q|), q in {0,+/-1}^12, exact clique search",
        "dimension": 13,
        "record": RECORD,
        "fixed_base_count": FIXED,
        "target_auxiliary_count": 25,
        "constraint_rows": row_info,
        "enumeration": enum_info,
        "compatibility_graph": graph_info,
        "search": search_info,
        "auxiliary_count": len(selected_q),
        "selected_support_distribution": support_distribution,
        "auxiliary_exact_verification": auxiliary_check,
        "full_exact_verification": full_check,
        "full_count": full_count,
        "beats_record": success,
    }
    write_json(args.output_dir / "analysis.json", analysis)

    config_name = f"best_exact_{full_count}.json"
    write_json(
        args.output_dir / config_name,
        {
            "dimension": 13,
            "count": full_count,
            "coordinate_field": "Q(sqrt(3))",
            "norm_squared": 16,
            "unit_scaling": "divide coordinates by 4",
            "vectors": [ab_to_strings(v) for v in full],
            "max_off_diagonal_unit": full_check["max_offdiag_unit"],
            "construction": "ZE99 fixed 1106 plus two copies of the selected auxiliary code",
            "auxiliary_code": [
                {"q": q.astype(int).tolist(), "support_size": int(k)}
                for q, k in zip(selected_q, selected_k)
            ],
            "verification": full_check,
            "record": RECORD,
            "beats_record": success,
        },
    )
    write_json(
        args.output_dir / "best.json",
        {
            "dimension": 13,
            "count": full_count,
            "configuration": config_name,
            "max_off_diagonal_unit": full_check["max_offdiag_unit"],
            "record": RECORD,
            "beats_record": success,
            "family_proven_optimal": search_info["proven_optimal"],
        },
    )

    report = [
        "# Exact ZE99 auxiliary-layer search",
        "",
        "ZE99 was held fixed on 1,106 vectors. Each auxiliary unit vector is used",
        "with both last-coordinate signs, so auxiliary size 25 would yield 1,156.",
        "",
        f"- Feasible ternary directions: **{len(candidates)}**.",
        f"- Best auxiliary code: **{len(selected_q)}**.",
        f"- Full exactly verified count: **{full_count}**.",
        f"- Exact max off-diagonal: **{full_check['max_offdiag_unit']}**.",
        f"- Exact family optimum proved: **{search_info['proven_optimal']}**.",
        f"- Beats the record {RECORD}: **{success}**.",
        "",
        "```json",
        json.dumps(analysis, indent=2, sort_keys=True),
        "```",
        "",
    ]
    (args.output_dir / "REPORT.md").write_text("\n".join(report), encoding="utf-8")
    (args.output_dir / "raw_output.json").write_text(
        json.dumps(analysis, separators=(",", ":"), sort_keys=True) + "\n",
        encoding="utf-8",
    )

    if args.progress_log:
        with args.progress_log.open("a", encoding="utf-8") as log:
            outcome = "pass" if success else "fail"
            log.write(
                f"{datetime.now(timezone.utc).isoformat()} method='ZE99 ternary auxiliary "
                f"exact search' dimension=13 count={full_count} result={outcome} "
                f"family_optimal={search_info['proven_optimal']}\n"
            )

    print(json.dumps(analysis, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
