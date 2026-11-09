#!/usr/bin/env python3
"""Canonical PackingStar analyzer with strict all-input verification.

The implementation is kept in ``_analyze_packingstar_impl.py`` so this small entry
point can enforce repository-wide invariants around it:

* every selected matrix must verify exactly;
* validation may disable the optional numerical hole diagnostic; and
* committed refresh outputs can be normalized to remove timestamps.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import os
import sys
from pathlib import Path
from typing import Any


_IMPL_PATH = Path(__file__).with_name("_analyze_packingstar_impl.py")
_SPEC = importlib.util.spec_from_file_location(
    "kissing_dim13_packingstar_analyze_impl", _IMPL_PATH
)
if _SPEC is None or _SPEC.loader is None:
    raise ImportError(f"cannot load PackingStar analyzer implementation at {_IMPL_PATH}")
_impl = importlib.util.module_from_spec(_SPEC)
sys.modules[_SPEC.name] = _impl
_SPEC.loader.exec_module(_impl)

# Preserve the module's existing public API for callers that import helper functions.
for _name in dir(_impl):
    if not _name.startswith("_"):
        globals()[_name] = getattr(_impl, _name)


def _wrapper_options(argv: list[str]) -> tuple[argparse.Namespace, list[str]]:
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--deterministic-output", action="store_true")
    parser.add_argument("--skip-hole-search", action="store_true")
    options, passthrough = parser.parse_known_args(argv)
    return options, passthrough


def _output_dir(argv: list[str]) -> Path:
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--output-dir", required=True, type=Path)
    options, _ = parser.parse_known_args(argv)
    return options.output_dir


def _normalize_outputs(output_dir: Path) -> None:
    """Remove wall-clock data from files intended for source control."""
    analysis_path = output_dir / "analysis.json"
    payload: dict[str, Any] = json.loads(analysis_path.read_text(encoding="utf-8"))
    payload.pop("generated_at", None)
    revision = os.environ.get("PACKINGSTAR_SOURCE_REVISION", "unknown")
    payload["source_revision"] = revision
    analysis_path.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )

    report_path = output_dir / "REPORT.md"
    lines = report_path.read_text(encoding="utf-8").splitlines()
    source_directory = payload.get("source_directory", "unknown")
    replacement = f"Source revision `{revision}` from `{source_directory}`."
    for index, line in enumerate(lines):
        if line.startswith("Generated "):
            lines[index] = replacement
            break
    report_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _load_records(output_dir: Path) -> list[dict[str, Any]]:
    payload = json.loads((output_dir / "analysis.json").read_text(encoding="utf-8"))
    records = payload.get("records")
    if not isinstance(records, list):
        raise ValueError("analysis.json is missing a records list")
    return records


def main(argv: list[str] | None = None) -> int:
    raw_argv = list(sys.argv[1:] if argv is None else argv)
    options, passthrough = _wrapper_options(raw_argv)
    output_dir = _output_dir(passthrough)

    original_hole_search = _impl.numerical_hole_search
    if options.skip_hole_search:
        _impl.numerical_hole_search = lambda _exact: {
            "available": False,
            "reason": "disabled for deterministic validation",
            "note": "The numerical hole search is not part of the exact certificate.",
        }
    try:
        implementation_status = int(_impl.main(passthrough))
    finally:
        _impl.numerical_hole_search = original_hole_search

    if implementation_status != 0:
        return implementation_status

    records = _load_records(output_dir)
    if not records:
        print("No selected PackingStar matrices were analyzed", file=sys.stderr)
        return 2
    failures = [
        record
        for record in records
        if record.get("status") != "verified_exactly"
    ]
    if failures:
        names = ", ".join(Path(str(record.get("source", "unknown"))).name for record in failures)
        print(
            f"Exact verification failed for {len(failures)} of {len(records)} selected "
            f"matrices: {names}",
            file=sys.stderr,
        )
        return 1

    if options.deterministic_output:
        _normalize_outputs(output_dir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
