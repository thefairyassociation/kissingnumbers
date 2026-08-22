#!/usr/bin/env python3
"""Self-tests for the repo-native O(4) breadth initializer."""

from __future__ import annotations

import argparse
import os
import subprocess
import tempfile
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent))
from o4_breadth import (  # noqa: E402
    DEFAULT_SAFETY_ROW_L1,
    THEOREM_ROW_L1,
    _draw_o4,
    audit_core,
    canonical_core,
    choose_extra,
    deformed_core,
    make_seed,
    write_seed,
)


def check_identity_equivalence() -> None:
    canonical = canonical_core()
    identity = deformed_core(np.eye(4), np.eye(4))
    if not np.array_equal(canonical, identity):
        raise AssertionError(
            f"deformed_core(I,I) differs from canonical; max diff={np.max(np.abs(canonical - identity))}"
        )
    print("identity equivalence: PASS")


def check_theorem_and_core_validity() -> None:
    for seed in range(4):
        rng = np.random.default_rng(seed)
        left, _ = _draw_o4(rng, 0.006, DEFAULT_SAFETY_ROW_L1)
        right, _ = _draw_o4(rng, 0.006, DEFAULT_SAFETY_ROW_L1)
        for label, matrix in (("left", left), ("right", right)):
            row_l1 = float(np.max(np.sum(np.abs(matrix), axis=1)))
            if row_l1 > DEFAULT_SAFETY_ROW_L1 + 1e-15:
                raise AssertionError(f"{label} row-l1 {row_l1} exceeds safety limit")
            if row_l1 > THEOREM_ROW_L1 + 1e-15:
                raise AssertionError(f"{label} row-l1 {row_l1} exceeds theorem bound")
        core = deformed_core(left, right)
        audit = audit_core(core, f"seed{seed}")
        if audit["fingerprint"]["max_ip"] > 0.5 + 2e-12:
            raise AssertionError("deformed core failed max-IP audit")
    print("theorem row-l1 and direct 840-core audits: PASS")


def check_extra_modes() -> None:
    core = canonical_core()
    extra, metadata = choose_extra(core, np.random.default_rng(7), "uniform-random")
    expected = 1.0 / np.sqrt(12.0)
    if metadata["extra_mode"] != "uniform-random":
        raise AssertionError("uniform mode was not recorded")
    if not np.all(np.isclose(np.abs(extra), expected)):
        raise AssertionError("uniform extra is not a hypercube vertex")
    if not 0 <= metadata["extra_hypercube_index"] < 4096:
        raise AssertionError("uniform extra index outside [0,4096)")
    _, legacy = choose_extra(core, np.random.default_rng(7), "deterministic-best")
    if "legacy diagnostic" not in legacy["extra_mode_label"]:
        raise AssertionError("deterministic-best is not explicitly labelled legacy")
    print("uniform-random and legacy deterministic-best extra modes: PASS")


def check_seed_roundtrip() -> None:
    with tempfile.TemporaryDirectory(prefix="o4-breadth-test-") as tmp:
        path = Path(tmp) / "seed841.txt"
        X, metadata = make_seed("deformed", 3, extra_mode="uniform-random")
        write_seed(path, X, metadata)
        loaded = np.loadtxt(path)
        if loaded.shape != (841, 12) or not np.isfinite(loaded).all():
            raise AssertionError("seed round-trip failed")
        if not path.with_suffix(path.suffix + ".json").exists():
            raise AssertionError("seed metadata sidecar missing")
    print("841 coordinate + metadata round-trip: PASS")


def check_source_contract() -> None:
    """Reuse the faithful-841 contract rather than keeping a second copy.

    This file used to duplicate the needle list, which meant it drifted from
    the optimizer independently and repeated the same non-unique needles.
    """
    sys.path.insert(0, str(ROOT / "kissing" / "lib"))
    from test_faithful_841 import check_source_contract as faithful_contract

    faithful_contract(ROOT / "kissing" / "lib" / "riesz.c")
    print("faithful C source contract: PASS")


def check_binary_smoke(binary: Path) -> None:
    with tempfile.TemporaryDirectory(prefix="o4-breadth-binary-") as tmp:
        directory = Path(tmp)
        seed_path = directory / "seed841.txt"
        X, metadata = make_seed("canonical", 19, extra_mode="uniform-random")
        write_seed(seed_path, X, metadata)
        env = os.environ.copy()
        env.update(
            {
                "KISS_FAITHFUL": "1",
                "KISS_SOLVER": "adam",
                "KISS_LOSS": "riesz",
                "KISS_POLISH": "0",
                "KISS_ADAM_BASE_END": "4",
                "KISS_THREADS": "1",
                "OMP_NUM_THREADS": "1",
                "OPENBLAS_NUM_THREADS": "1",
            }
        )
        proc = subprocess.run(
            [str(binary), "12", "841", "35000", "19", str(seed_path)],
            env=env,
            text=True,
            capture_output=True,
            check=False,
        )
        if proc.returncode != 0:
            raise AssertionError(f"faithful binary smoke failed:\n{proc.stderr}")
        if "uniform-random hypercube extra index=" not in proc.stderr:
            raise AssertionError("faithful binary did not record uniform extra index")
        output = Path(f"{seed_path}.riesz.s19.out")
        if not output.exists():
            raise AssertionError("faithful binary did not write candidate")
        # Find the faithful header by prefix rather than by line number: the
        # candidate now also carries a solver/loss provenance line.
        header = [
            line for line in output.read_text().splitlines()
            if line.startswith("#")
        ]
        faithful_lines = [line for line in header if line.startswith("# faithful=1")]
        if len(faithful_lines) != 1:
            raise AssertionError(
                f"expected exactly one faithful header line, got {len(faithful_lines)}"
            )
        metadata_line = faithful_lines[0]
        if "extra_mode=uniform-random" not in metadata_line:
            raise AssertionError("faithful output did not record extra mode")
        for token in (
            "search_stage_start=0",
            "search_stage_end=4",
            "search_updates=5000",
            "full_schedule=0",
            # a search-only run must say so, and must not claim the authors'
            # %.10f Gram handoff that only prepare_841_polish.py performs
            "polish=0",
            "gram_handoff=0",
        ):
            if token not in metadata_line:
                raise AssertionError(f"faithful partial-run metadata missing {token}")
        if not any(line.startswith("# solver=") for line in header):
            raise AssertionError("candidate is missing its solver provenance line")
    print("faithful binary stage-4 smoke: PASS")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--binary", type=Path, default=ROOT / "kissing" / "lib" / "riesz2")
    parser.add_argument("--binary-smoke", action="store_true")
    args = parser.parse_args()
    check_identity_equivalence()
    check_theorem_and_core_validity()
    check_extra_modes()
    check_seed_roundtrip()
    check_source_contract()
    if args.binary_smoke:
        if not args.binary.exists():
            raise SystemExit(f"binary not found: {args.binary}")
        check_binary_smoke(args.binary)
    print("PASS: O(4) breadth self-tests")


if __name__ == "__main__":
    main()
