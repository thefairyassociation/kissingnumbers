#!/usr/bin/env python3
"""Tests for strict PackingStar analysis and source locking."""

from __future__ import annotations

import hashlib
import json
import os
import tempfile
import unittest
import zipfile
from pathlib import Path
from unittest import mock

import analyze_packingstar
import fetch_source


class AnalyzePackingStarGuardTests(unittest.TestCase):
    def _fake_impl(self, records: list[dict[str, str]]):
        def run(argv: list[str]) -> int:
            output_index = argv.index("--output-dir") + 1
            output_dir = Path(argv[output_index])
            output_dir.mkdir(parents=True, exist_ok=True)
            (output_dir / "analysis.json").write_text(
                json.dumps(
                    {
                        "generated_at": "volatile",
                        "source_directory": "/tmp/input",
                        "records": records,
                    }
                ),
                encoding="utf-8",
            )
            (output_dir / "REPORT.md").write_text(
                "# Report\n\nGenerated volatile from `/tmp/input`.\n",
                encoding="utf-8",
            )
            return 0

        return run

    def test_partial_failure_returns_nonzero(self) -> None:
        records = [
            {"source": "/tmp/a.npy", "status": "verified_exactly"},
            {"source": "/tmp/b.npy", "status": "failed"},
        ]
        with tempfile.TemporaryDirectory() as directory, mock.patch.object(
            analyze_packingstar._impl, "main", self._fake_impl(records)
        ):
            status = analyze_packingstar.main(
                [
                    "--input-dir",
                    "/tmp/input",
                    "--output-dir",
                    directory,
                    "--progress-log",
                    "/tmp/progress.log",
                ]
            )
        self.assertEqual(status, 1)

    def test_deterministic_output_removes_timestamp(self) -> None:
        records = [
            {"source": "/tmp/a.npy", "status": "verified_exactly"},
            {"source": "/tmp/b.npy", "status": "verified_exactly"},
        ]
        with tempfile.TemporaryDirectory() as directory, mock.patch.object(
            analyze_packingstar._impl, "main", self._fake_impl(records)
        ), mock.patch.dict(os.environ, {"PACKINGSTAR_SOURCE_REVISION": "abc123"}):
            status = analyze_packingstar.main(
                [
                    "--input-dir",
                    "/tmp/input",
                    "--output-dir",
                    directory,
                    "--progress-log",
                    "/tmp/progress.log",
                    "--deterministic-output",
                ]
            )
            analysis = json.loads(
                (Path(directory) / "analysis.json").read_text(encoding="utf-8")
            )
            report = (Path(directory) / "REPORT.md").read_text(encoding="utf-8")
        self.assertEqual(status, 0)
        self.assertNotIn("generated_at", analysis)
        self.assertEqual(analysis["source_revision"], "abc123")
        self.assertNotIn("Generated ", report)


class SourceLockTests(unittest.TestCase):
    def test_offline_archive_hash_and_safe_extract(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            archive = root / "source.zip"
            with zipfile.ZipFile(archive, "w") as zipped:
                zipped.writestr("folder/matrix.npy", b"matrix")
            digest = hashlib.sha256(archive.read_bytes()).hexdigest()
            lock_path = root / "source_lock.json"
            lock_path.write_text(
                json.dumps(
                    {
                        "repository": "owner/repository",
                        "commit": "0" * 40,
                        "archive_path": "source.zip",
                        "archive_sha256": digest,
                    }
                ),
                encoding="utf-8",
            )
            extracted = root / "extracted"
            status = fetch_source.main(
                [
                    "--lock-file",
                    str(lock_path),
                    "--archive",
                    str(archive),
                    "--extract-dir",
                    str(extracted),
                    "--offline",
                ]
            )
            self.assertEqual(status, 0)
            self.assertEqual((extracted / "folder/matrix.npy").read_bytes(), b"matrix")

    def test_hash_mismatch_fails(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            archive = root / "source.zip"
            with zipfile.ZipFile(archive, "w") as zipped:
                zipped.writestr("matrix.npy", b"matrix")
            lock_path = root / "source_lock.json"
            lock_path.write_text(
                json.dumps(
                    {
                        "repository": "owner/repository",
                        "commit": "0" * 40,
                        "archive_path": "source.zip",
                        "archive_sha256": "1" * 64,
                    }
                ),
                encoding="utf-8",
            )
            status = fetch_source.main(
                [
                    "--lock-file",
                    str(lock_path),
                    "--archive",
                    str(archive),
                    "--extract-dir",
                    str(root / "extracted"),
                    "--offline",
                ]
            )
            self.assertEqual(status, 1)


if __name__ == "__main__":
    unittest.main()
