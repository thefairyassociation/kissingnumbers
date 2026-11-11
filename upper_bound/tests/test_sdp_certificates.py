#!/usr/bin/env python3
"""Tests for the exact three-point certificate verifier (standard library only).

    python3 upper_bound/tests/test_sdp_certificates.py
"""
from __future__ import annotations

import copy
import json
import sys
import unittest
from fractions import Fraction as Fr
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "sdp"))
import verify_certificate as vc  # noqa: E402

CERT = ROOT / "certificates" / "sdp" / "three_point_n03.json"


class Pieces(unittest.TestCase):
    def test_positive_definite(self):
        self.assertTrue(vc.is_positive_definite([[2, 1], [1, 2]]))
        self.assertFalse(vc.is_positive_definite([[1, 2], [2, 1]]))
        self.assertFalse(vc.is_positive_definite([[1, 1], [1, 1]]))        # singular: rejected
        self.assertFalse(vc.is_positive_definite([[0]]))
        self.assertTrue(vc.is_positive_definite([[4, 2, 0], [2, 5, 3], [0, 3, 6]]))
        self.assertFalse(vc.is_positive_definite([[4, 2, 0], [2, 5, 3], [0, 3, 2]]))

    def test_Q_polynomials(self):
        # Q_0 = 1, Q_1 = t - uv; S_k(1,1,1) = 0 for k >= 1
        self.assertEqual(vc.Q_poly(3, 0), {(0, 0, 0): Fr(1)})
        self.assertEqual(vc.Q_poly(5, 1), {(0, 0, 1): Fr(1), (1, 1, 0): Fr(-1)})
        S = vc.S_matrices(4, 3)
        for k in range(1, 4):
            for row in S[k]:
                for p in row:
                    self.assertEqual(sum(p.values()), 0)
        self.assertTrue(all(sum(p.values()) == 1 for row in S[0] for p in row))

    def test_S_is_symmetric_in_the_variables(self):
        for row in vc.S_matrices(3, 3)[1]:
            for p in row:
                for (i, j, l), c in p.items():
                    self.assertEqual(p.get((j, i, l)), c)
                    self.assertEqual(p.get((l, j, i)), c)

    def test_Q2_at_special_points(self):
        # at u = v = 0, Q_k(0,0,t) = P_k^{n-1}(t)
        n = 5
        Pk = vc.gegenbauer(n - 1, 4)
        for k in range(5):
            q = vc.Q_poly(n, k)
            restricted = [Fr(0)] * (k + 1)
            for (i, j, l), c in q.items():
                if i == 0 and j == 0:
                    restricted[l] += c
            self.assertEqual(restricted, Pk[k])


class Certificate(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.cert = json.loads(CERT.read_text())

    def test_dimension_three(self):
        r = vc.verify(self.cert)
        self.assertEqual((r["n"], r["bound"]), (3, 12))

    def test_dimension_four(self):
        r = vc.verify(json.loads((CERT.parent / "three_point_n04.json").read_text()))
        self.assertEqual((r["n"], r["degree"], r["bound"]), (4, 7, 24))

    def _rejected(self, mutate):
        c = copy.deepcopy(self.cert)
        mutate(c)
        with self.assertRaises(ValueError):
            vc.verify(c)

    def test_rejects_changed_F(self):
        def m(c):
            c["F"][1]["rows"][0][0] += 1
        self._rejected(m)

    def test_rejects_changed_gram_matrix(self):
        def m(c):
            c["Q3"][2]["rows"][3][3] += 1
        self._rejected(m)

    def test_rejects_changed_a(self):
        def m(c):
            c["a"][2] = str(Fr(c["a"][2]) + Fr(1, 10 ** 9))
        self._rejected(m)

    def test_rejects_negative_a(self):
        def m(c):
            c["a"][0] = "-1/1000"
        self._rejected(m)

    def test_rejects_indefinite_block(self):
        def m(c):
            n = len(c["Q3"][4]["rows"])
            for i in range(n):
                c["Q3"][4]["rows"][i][i] = -abs(c["Q3"][4]["rows"][i][i])
        self._rejected(m)

    def test_rejects_asymmetric_block(self):
        def m(c):
            c["F"][0]["rows"][0][1] += 1
        self._rejected(m)

    def test_rejects_other_dimension_or_angle(self):
        # the same numbers are not a certificate for R^4, nor for a different angle
        def m4(c):
            c["n"] = 4
        def ms(c):
            c["s"] = "3/5"
        self._rejected(m4)
        self._rejected(ms)


if __name__ == "__main__":
    unittest.main()
