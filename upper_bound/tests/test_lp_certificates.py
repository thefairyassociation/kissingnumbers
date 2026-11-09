#!/usr/bin/env python3
"""Tests for the exact LP-certificate verifier (standard library only).

    python3 upper_bound/tests/test_lp_certificates.py
"""
from __future__ import annotations

import json
import sys
import unittest
from fractions import Fraction as Fr
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "lp"))
import verify_certificate as vc  # noqa: E402

CERTS = ROOT / "certificates" / "lp"

# Delsarte LP bounds for the kissing number, floor of the classical values
# (Odlyzko & Sloane 1979, Table 1).
EXPECTED = {3: 13, 4: 25, 5: 46, 6: 82, 7: 140, 8: 240, 9: 380, 10: 595, 11: 915, 12: 1416,
            13: 2233, 14: 3492, 15: 5431, 16: 8313, 17: 12218, 18: 17877, 19: 25900,
            20: 37974, 21: 56851, 22: 86537, 23: 128095}


def poly_from_roots(roots):
    p = [Fr(1)]
    for r in roots:
        nxt = [Fr(0)] * (len(p) + 1)
        for i, c in enumerate(p):
            nxt[i + 1] += c
            nxt[i] -= Fr(r) * c
        p = nxt
    return p


class Sturm(unittest.TestCase):
    def test_counts_distinct_roots(self):
        p = poly_from_roots([Fr(1, 3), Fr(-1, 2), Fr(-1, 2), 2])
        self.assertEqual(vc.roots_in_half_open(p, Fr(-1), Fr(1, 2)), 2)
        self.assertEqual(vc.roots_in_half_open(p, Fr(-1), Fr(3)), 3)
        self.assertEqual(vc.roots_in_half_open(p, Fr(0), Fr(1, 4)), 0)

    def test_no_real_roots(self):
        self.assertEqual(vc.roots_in_half_open([Fr(1), Fr(0), Fr(1)], Fr(-5), Fr(5)), 0)

    def test_close_roots(self):
        p = poly_from_roots([Fr(1, 1000), Fr(1, 1001), Fr(-1, 1000)])
        self.assertEqual(vc.roots_in_half_open(p, Fr(0), Fr(1)), 2)


class Gegenbauer(unittest.TestCase):
    def test_legendre_and_normalisation(self):
        P = vc.gegenbauer(3, 4)
        self.assertEqual(P[2], [Fr(-1, 2), 0, Fr(3, 2)])
        self.assertEqual(P[3], [0, Fr(-3, 2), 0, Fr(5, 2)])
        for n in (3, 4, 8, 24):
            for pk in vc.gegenbauer(n, 12):
                self.assertEqual(vc.evaluate(pk, Fr(1)), 1)
        # P_2 in R^n is (n t^2 - 1)/(n - 1)
        self.assertEqual(vc.gegenbauer(8, 2)[2], [Fr(-1, 7), 0, Fr(8, 7)])

    def test_roundtrip(self):
        F = [Fr(3), Fr(-1, 2), Fr(7, 5), Fr(0), Fr(2)]
        f = vc.to_gegenbauer(F, 5)
        back = [Fr(0)] * 5
        for fk, pk in zip(f, vc.gegenbauer(5, 4)):
            for i, c in enumerate(pk):
                back[i] += fk * c
        self.assertEqual(back, F)


class Certificates(unittest.TestCase):
    def test_committed_certificates(self):
        for n, bound in EXPECTED.items():
            r = vc.verify(json.loads((CERTS / f"delsarte_n{n:02d}.json").read_text()))
            self.assertEqual(r["n"], n)
            self.assertEqual(r["bound"], bound, f"n={n}")

    def test_sharp_cases_are_exact(self):
        r8 = vc.verify(json.loads((CERTS / "sharp_n08.json").read_text()))
        r24 = vc.verify(json.loads((CERTS / "sharp_n24.json").read_text()))
        self.assertEqual(r8["bound_exact"], "240")
        self.assertEqual(r24["bound_exact"], "196560")

    def test_rejects_negative_coefficient(self):
        cert = json.loads((CERTS / "delsarte_n05.json").read_text())
        cert["f"][3] = "-1/1000"
        with self.assertRaises(ValueError):
            vc.verify(cert)

    def test_rejects_polynomial_that_is_positive_somewhere(self):
        cert = json.loads((CERTS / "delsarte_n05.json").read_text())
        # raising f_0 lifts F by a constant and pushes its interior maxima above zero
        cert["f"][0] = str(Fr(cert["f"][0]) + Fr(1, 1000))
        with self.assertRaises(ValueError):
            vc.verify(cert)

    def test_rejects_bad_product_certificates(self):
        # (t+1)(t-1/2) is <= 0 on the interval, but its constant Gegenbauer coefficient is negative
        with self.assertRaises(ValueError):
            vc.verify({"n": 8, "s": "1/2", "roots": [["-1", 1], ["1/2", 1]]})
        # an odd root inside the interval makes the polynomial change sign there
        with self.assertRaises(ValueError):
            vc.verify({"n": 8, "s": "1/2", "roots": [["-1", 1], ["-1/2", 2], ["0", 1], ["1/2", 1]]})
        # the E8 polynomial is not a certificate in dimension 9 (a coefficient turns negative)
        # or, if it is, it cannot beat the known 306 points there
        e8 = [["-1", 1], ["-1/2", 2], ["0", 2], ["1/2", 1]]
        try:
            r = vc.verify({"n": 9, "s": "1/2", "roots": e8})
            self.assertGreaterEqual(r["bound"], 306)
        except ValueError:
            pass

    def test_bound_is_never_below_a_known_configuration(self):
        known = {3: 12, 4: 24, 5: 40, 6: 72, 7: 126, 8: 240, 9: 306, 10: 510, 11: 604, 12: 841,
                 13: 1154, 14: 1932, 15: 2564, 16: 4320, 24: 196560}
        for n, lower in known.items():
            name = "sharp_n24.json" if n == 24 else f"delsarte_n{n:02d}.json"
            r = vc.verify(json.loads((CERTS / name).read_text()))
            self.assertGreaterEqual(r["bound"], lower, f"n={n}")


if __name__ == "__main__":
    unittest.main()
