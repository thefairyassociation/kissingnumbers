#!/usr/bin/env python3
"""Regression tests for the exact verifier's supported coordinate schemas."""

from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

import verifier


class ExactVerifierTests(unittest.TestCase):
    def setUp(self) -> None:
        self.radicands = [
            {"num": 1, "den": 2},
            {"num": 1, "den": 2},
        ]
        self.square = [
            [{"num": 1, "den": 1}, {"num": 1, "den": 1}],
            [{"num": -1, "den": 1}, {"num": 1, "den": 1}],
            [{"num": -1, "den": 1}, {"num": -1, "den": 1}],
            [{"num": 1, "den": 1}, {"num": -1, "den": 1}],
        ]

    def test_radicands_are_applied_to_norms_and_inner_products(self) -> None:
        result = verifier.verify_unit_vectors(
            self.square,
            dim=2,
            coordinate_radicands=self.radicands,
        )
        self.assertTrue(result["ok"])
        self.assertTrue(result["coordinate_radicands_applied"])
        self.assertEqual(result["max_offdiag"], "0")

    def test_plain_algebraic_coordinates_still_verify(self) -> None:
        vectors = [
            ["sqrt(2)/2", "sqrt(2)/2"],
            ["-sqrt(2)/2", "sqrt(2)/2"],
            ["-sqrt(2)/2", "-sqrt(2)/2"],
            ["sqrt(2)/2", "-sqrt(2)/2"],
        ]
        result = verifier.verify_unit_vectors(vectors, dim=2)
        self.assertTrue(result["ok"])
        self.assertFalse(result["coordinate_radicands_applied"])

    def test_nonpositive_radicand_is_rejected(self) -> None:
        with self.assertRaisesRegex(ValueError, "must be positive"):
            verifier.verify_unit_vectors(
                self.square,
                dim=2,
                coordinate_radicands=[1, 0],
            )

    def test_config_file_uses_radicand_schema(self) -> None:
        payload = {
            "dimension": 2,
            "count": len(self.square),
            "unit": True,
            "coordinate_model": verifier.RADICAND_MODEL,
            "coordinate_radicands": self.radicands,
            "vectors": self.square,
        }
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "config.json"
            path.write_text(json.dumps(payload), encoding="utf-8")
            result = verifier.verify_config_file(str(path))
        self.assertTrue(result["ok"])
        self.assertTrue(result["coordinate_radicands_applied"])

    def test_declared_count_must_match_vectors(self) -> None:
        payload = {
            "dimension": 2,
            "count": 99,
            "unit": True,
            "coordinate_radicands": self.radicands,
            "vectors": self.square,
        }
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "config.json"
            path.write_text(json.dumps(payload), encoding="utf-8")
            result = verifier.verify_config_file(str(path))
        self.assertFalse(result["ok"])
        self.assertIn("declared count", result["reason"])


if __name__ == "__main__":
    unittest.main()
