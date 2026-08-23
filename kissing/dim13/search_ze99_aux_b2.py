#!/usr/bin/env python3
"""Search primitive integer auxiliary directions q/||q|| with |q_i| <= 2."""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import sympy as sp

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE / "constructions"))
from ze99 import generate_ab, verify_ab  # noqa: E402
from ze99_aux_bounded import enumerate_bounded  # noqa: E402
from ze99_aux_core import (  # noqa: E402
    SearchError,
    baseline_clique,
    compatibility_graph,
    constraint_rows,
    exact_maximum_clique,
    verify_selected,
)

RECORD = 1154
FIXED_COUNT = 1106
GRAPH_LIMIT = 12000


def write_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def aux_expressions(q: np.ndarray, norm2: int, last: int) -> tuple[sp.Expr, ...]:
    scale = 2 * sp.sqrt(sp.Rational(3, norm2))
    return tuple(sp.simplify(int(value) * scale) for value in q) + (sp.Integer(2 * last),)


def fixed_expressions(v: list[tuple[int, int]]) -> tuple[sp.Expr, ...]:
    root3 = sp.sqrt(3)
    return tuple(sp.Integer(a) + sp.Integer(b) * root3 for a, b in v)


def exact_full_check(
    rows: np.ndarray, q: np.ndarray, norm2: np.ndarray
) -> tuple[list[tuple[sp.Expr, ...]], dict]:
    original = generate_ab()
    fixed = original[:FIXED_COUNT]
    base_check = verify_ab(fixed)
    if not base_check.get("ok"):
        raise SearchError(f"fixed ZE99 base failed exact verification: {base_check}")
    auxiliary_check = verify_selected(rows, q, norm2)

    vectors = [fixed_expressions(v) for v in fixed]
    for last in (1, -1):
        vectors.extend(
            aux_expressions(row, int(nn), last) for row, nn in zip(q, norm2)
        )
    expected = FIXED_COUNT + 2 * len(q)
    if len(vectors) != expected:
        raise SearchError("expanded vector count mismatch")
    if len(set(vectors)) != len(vectors):
        raise SearchError("expanded exact coordinates contain a duplicate")
    for i, vector in enumerate(vectors[FIXED_COUNT:]):
        if sp.simplify(sum(value * value for value in vector) - 16) != 0:
            raise SearchError(f"auxiliary vector {i} does not have exact norm squared 16")

    total_pairs = expected * (expected - 1) // 2
    base_pairs = FIXED_COUNT * (FIXED_COUNT - 1) // 2
    fixed_aux_pairs = FIXED_COUNT * (2 * len(q))
    auxiliary_pairs = (2 * len(q)) * (2 * len(q) - 1) // 2
    if base_pairs + fixed_aux_pairs + auxiliary_pairs != total_pairs:
        raise SearchError("pair-count accounting failed")

    return vectors, {
        "ok": True,
        "dimension": 13,
        "count": expected,
        "norm_squared": 16,
        "diagonal_exactly_16": True,
        "all_off_diagonal_leq_8": True,
        "max_off_diagonal_unnormalized": "8",
        "max_off_diagonal_unit": "1/2",
        "distinct": True,
        "total_unordered_gram_pairs_checked": total_pairs,
        "base_pairs_checked_by_Q_sqrt3_verifier": base_pairs,
        "fixed_auxiliary_pairs_checked_by_integer_squared_inequalities": fixed_aux_pairs,
        "auxiliary_pairs_checked_by_integer_squared_inequalities": auxiliary_pairs,
        "fixed_base_verification": base_check,
        "auxiliary_verification": auxiliary_check,
        "arithmetic": "integers, rational squares, and SymPy exact radicals; no floats",
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--time-limit", type=float, default=1500.0)
    parser.add_argument("--progress-log", type=Path)
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now(timezone.utc).isoformat()

    if args.progress_log:
        args.progress_log.parent.mkdir(parents=True, exist_ok=True)
        with args.progress_log.open("a", encoding="utf-8") as log:
            log.write(
                f"{stamp} method='ZE99 primitive bound-2 auxiliary search' "
                "dimension=13 count=? result=started\n"
            )

    rows, row_info = constraint_rows(generate_ab)
    q, norm2, enumeration = enumerate_bounded(rows, bound=2)
    if len(q) > GRAPH_LIMIT:
        analysis = {
            "generated_at": stamp,
            "status": "enumerated_graph_deferred",
            "constraint_rows": row_info,
            "enumeration": enumeration,
            "candidate_count": len(q),
            "graph_limit": GRAPH_LIMIT,
        }
        write_json(args.output_dir / "analysis.json", analysis)
        print(json.dumps(analysis, indent=2, sort_keys=True))
        return 0

    adjacency, order, graph_info = compatibility_graph(q, norm2)
    baseline = baseline_clique(q, order)

    def checkpoint(clique: list[int], nodes: int) -> None:
        original = order[np.asarray(clique, dtype=np.int32)]
        write_json(
            args.output_dir / "checkpoint.json",
            {
                "generated_at": datetime.now(timezone.utc).isoformat(),
                "auxiliary_count": len(clique),
                "full_count": FIXED_COUNT + 2 * len(clique),
                "nodes": nodes,
                "q": q[original].astype(int).tolist(),
                "norm_squared": norm2[original].astype(int).tolist(),
            },
        )

    best_reordered, search = exact_maximum_clique(
        adjacency, baseline, args.time_limit, checkpoint
    )
    best_original = order[np.asarray(best_reordered, dtype=np.int32)]
    selected_q = q[best_original]
    selected_norm2 = norm2[best_original]
    vectors, verification = exact_full_check(rows, selected_q, selected_norm2)
    full_count = len(vectors)
    success = full_count > RECORD

    analysis = {
        "generated_at": stamp,
        "status": "completed",
        "method": "primitive q/||q|| with q in {-2,-1,0,1,2}^12",
        "dimension": 13,
        "record": RECORD,
        "fixed_base_count": FIXED_COUNT,
        "constraint_rows": row_info,
        "enumeration": enumeration,
        "compatibility_graph": graph_info,
        "search": search,
        "auxiliary_count": len(selected_q),
        "selected_norm_squared": {
            str(int(value)): int(count)
            for value, count in zip(*np.unique(selected_norm2, return_counts=True))
        },
        "full_count": full_count,
        "beats_record": success,
        "exact_verification": verification,
    }
    write_json(args.output_dir / "analysis.json", analysis)
    config_name = f"best_exact_{full_count}.json"
    write_json(
        args.output_dir / config_name,
        {
            "dimension": 13,
            "count": full_count,
            "norm_squared": 16,
            "unit_scaling": "divide coordinates by 4",
            "coordinate_field": "exact real algebraic numbers",
            "vectors": [[str(value) for value in vector] for vector in vectors],
            "auxiliary_code": [
                {"q": row.astype(int).tolist(), "norm_squared": int(nn)}
                for row, nn in zip(selected_q, selected_norm2)
            ],
            "verification": verification,
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
            "max_off_diagonal_unit": "1/2",
            "record": RECORD,
            "beats_record": success,
            "family_proven_optimal": search["proven_optimal"],
        },
    )

    report = [
        "# ZE99 primitive bound-2 auxiliary search",
        "",
        f"- Exact feasible directions: **{len(q)}**.",
        f"- Best auxiliary code: **{len(selected_q)}**.",
        f"- Full exact count: **{full_count}**.",
        f"- Exact family optimum proved: **{search['proven_optimal']}**.",
        f"- Beats {RECORD}: **{success}**.",
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
            log.write(
                f"{datetime.now(timezone.utc).isoformat()} method='ZE99 primitive bound-2 "
                f"auxiliary search' dimension=13 count={full_count} "
                f"result={'pass' if success else 'fail'} "
                f"family_optimal={search['proven_optimal']}\n"
            )
    print(json.dumps(analysis, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
