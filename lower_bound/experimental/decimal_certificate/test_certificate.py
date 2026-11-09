import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

from certificate import CertificateError, certify


class DecimalCertificateTests(unittest.TestCase):
    def write_source(self, text: str) -> Path:
        handle = tempfile.NamedTemporaryFile("w", encoding="utf-8", delete=False)
        with handle:
            handle.write(text)
        self.addCleanup(lambda: Path(handle.name).unlink(missing_ok=True))
        return Path(handle.name)

    def test_boundary_is_certified_non_strict(self):
        path = self.write_source("1 1 0\n1 0 1\n")
        result = certify(path, expected_dimension=3, expected_count=2)
        self.assertTrue(result["ok"])
        self.assertEqual(result["pair_count"], 1)
        self.assertEqual(result["tight_pairs"], 1)

    def test_boundary_is_rejected_in_strict_mode(self):
        path = self.write_source("1 1 0\n1 0 1\n")
        with self.assertRaisesRegex(CertificateError, r"pair \(0,1\).*4\*p\^2"):
            certify(path, strict=True)

    def test_thirty_digit_boundary_perturbation_stays_exact(self):
        # A binary float rounds both z coordinates to 1.0.  Exact decimal
        # arithmetic distinguishes the failing side from the passing side.
        below = self.write_source(
            "1 1 0\n1 0 0.999999999999999999999999999999\n"
        )
        with self.assertRaisesRegex(CertificateError, r"pair \(0,1\)"):
            certify(below)
        above = self.write_source(
            "1 1 0\n1 0 1.000000000000000000000000000001\n"
        )
        self.assertTrue(certify(above, strict=True)["ok"])

    def test_above_half_is_rejected(self):
        path = self.write_source("1 1 0\n1 2 0\n")
        with self.assertRaisesRegex(CertificateError, r"pair \(0,1\)"):
            certify(path)

    def test_antipodal_and_scaled_directions(self):
        # The first pair is antipodal (p < 0); the second is exactly at 1/2
        # after normalization despite unequal coordinate scales.
        path = self.write_source("1 0 0 0 # inline comment\n-1 0 0 0\n0 2 2 0\n0 0 3 3\n")
        result = certify(path)
        self.assertEqual(result["nonpositive_pairs"], 5)
        self.assertEqual(result["tight_pairs"], 1)

    def test_zero_nonfinite_and_malformed_rows(self):
        for source, message in (
            ("0 0\n1 0\n", "zero direction"),
            ("nan 0\n1 0\n", "malformed coordinate"),
            ("inf 0\n1 0\n", "malformed coordinate"),
            ("1/2 0\n1 0\n", "malformed coordinate"),
            ("1 0\n1\n", "dimension"),
        ):
            with self.subTest(source=source):
                path = self.write_source(source)
                with self.assertRaisesRegex(CertificateError, message):
                    certify(path)

    def test_expected_shape_and_count_mismatches(self):
        path = self.write_source("1 0\n-1 0\n")
        with self.assertRaisesRegex(CertificateError, r"expected dimension 3"):
            certify(path, expected_dimension=3)
        with self.assertRaisesRegex(CertificateError, r"expected count 3"):
            certify(path, expected_count=3)

    def test_duplicate_rows_fail_through_pair_bound(self):
        path = self.write_source("0.5 0.25\n0.5 0.25\n")
        with self.assertRaisesRegex(CertificateError, r"pair \(0,1\)"):
            certify(path)

    def test_comments_scientific_notation_and_exact_hash(self):
        path = self.write_source("# comment\n1e-1 2.0e-1 # row\n\n-3e-1 4e-1\n")
        result = certify(path, expected_dimension=2, expected_count=2)
        self.assertEqual(result["source_count"], 2)
        self.assertEqual(len(result["source_sha256"]), 64)

    def test_cli_compact_success_and_failure_exit(self):
        path = self.write_source("1 1 0\n1 0 1\n")
        check = Path(__file__).with_name("check.py")
        good = subprocess.run(
            [sys.executable, str(check), str(path), "--compact"],
            check=False,
            capture_output=True,
            text=True,
        )
        self.assertEqual(good.returncode, 0)
        self.assertTrue(json.loads(good.stdout)["ok"])
        bad = subprocess.run(
            [sys.executable, str(check), str(path), "--strict", "--compact"],
            check=False,
            capture_output=True,
            text=True,
        )
        self.assertEqual(bad.returncode, 1)
        failure = json.loads(bad.stdout)
        self.assertFalse(failure["ok"])
        self.assertIn("pair (0,1)", failure["error"])


if __name__ == "__main__":
    unittest.main()
