#!/usr/bin/env python3
"""Regression tests for fastriesz.c, hingepol.c and certify_float.py.

    python3 kissing/lib/test_fast_tools.py

Builds the two C tools if needed (fastriesz needs OpenBLAS; see the Makefile)
and checks, against independent NumPy / Fraction computations:

* certify_float: the published 841 witness is certified with margin >= 6e-8,
  the worst pair agrees with an exact Fraction recomputation, and a witness
  with one point moved onto a neighbour's cap is rejected;
* fastriesz: L = log sum r^-s and its gradient agree with a NumPy reference
  at s = 8, 64 (full evaluation), 1024 (both paths) and 40000 (neighbour list);
* hingepol: on the witness, E_{1/2} = 0 and the configuration is unchanged;
  at t = 0.4999999 (above the witness's own maximum minus its margin) the
  energy is positive, and polishing does not increase it.
"""
from __future__ import annotations

import subprocess
import sys
import tempfile
from fractions import Fraction
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
WITNESS = HERE / "testdata" / "authors_841_coordinates.txt"
sys.path.insert(0, str(HERE))
import certify_float  # noqa: E402


def unit(path) -> np.ndarray:
    X = np.loadtxt(path, comments="#", ndmin=2)
    return X / np.linalg.norm(X, axis=1, keepdims=True)


def build(target: str) -> Path:
    exe = HERE / target
    subprocess.run(["make", "-s", "-C", str(HERE), target], check=True)
    return exe


def test_certify() -> None:
    X = np.loadtxt(WITNESS, comments="#")
    V = certify_float.to_integer(X, 40)
    ok, (wq, (i, j)) = certify_float.certify(V, Fraction(6, 10 ** 8))
    assert ok, "witness must certify with margin 6e-8"
    d = sum(Fraction(a) * b for a, b in zip(V[i], V[j]))
    q = 4 * d * d / (sum(Fraction(a) ** 2 for a in V[i]) * sum(Fraction(b) ** 2 for b in V[j]))
    assert q == wq, "worst pair must agree with an exact Fraction recomputation"
    assert abs(float(q) ** 0.5 / 2 - 0.4999999377514) < 1e-12
    ok7, _ = certify_float.certify(V, Fraction(7, 10 ** 8))
    assert not ok7, "witness margin is below 7e-8, so that claim must fail"
    U = unit(WITNESS)
    G = U @ U.T
    np.fill_diagonal(G, -2)
    a = int(np.argmax(G.max(1)))
    b = int(np.argmax(G[a]))
    Y = U.copy()
    Y[a] = U[a] + 0.02 * (U[b] - U[a])  # push a towards b: cosine > 1/2
    okb, _ = certify_float.certify(certify_float.to_integer(Y, 40))
    assert not okb, "a perturbed witness must be rejected"
    # a spread-out configuration (max cosine well below 1/2) must never be
    # credited with a margin of 1/2 or more
    Q = np.vstack([np.eye(12), -np.eye(12)])
    Vq = certify_float.to_integer(Q, 40)
    okq, (wqq, _) = certify_float.certify(Vq)
    assert okq and certify_float.proven_margin(wqq) < Fraction(1, 2)
    try:
        certify_float.certify(Vq, Fraction(1, 2))
        raise AssertionError("margin 1/2 must be rejected")
    except ValueError:
        pass
    print("certify_float: PASS")


def ref_energy(U: np.ndarray, s: float):
    N = len(U)
    G = U @ U.T
    iu = np.triu_indices(N, 1)
    r2 = 2 - 2 * G[iu]
    lw = -0.5 * s * np.log(r2)
    M = lw.max()
    w = np.exp(lw - M)
    L = M + np.log(w.sum())
    C = np.zeros((N, N))
    C[iu] = s * (w / w.sum()) / r2
    C = C + C.T
    return L, C @ U - C.sum(1)[:, None] * U


def test_fastriesz() -> None:
    exe = build("fastriesz")
    U = unit(WITNESS)
    for s, list_s in ((8, 1e9), (64, 1e9), (1024, 1e9), (1024, 256), (40000, 256)):
        out = subprocess.run([str(exe), "12", "841", str(WITNESS), "/tmp", "--extra", "file",
                              "--dump-grad", str(s), "--list-s", str(list_s)],
                             capture_output=True, text=True, check=True).stdout.splitlines()
        L = float(out[0].split()[0][2:])
        g = np.array([[float(v) for v in ln.split()] for ln in out[1:] if ln.strip()])
        Lr, gr = ref_energy(U, s)
        rel = np.linalg.norm(g - gr) / np.linalg.norm(gr)
        assert abs(L - Lr) < 1e-9 * max(1.0, abs(Lr)), (s, L, Lr)
        assert rel < 1e-10, (s, list_s, rel)
    print("fastriesz: PASS")


def test_hingepol() -> None:
    exe = build("hingepol")
    with tempfile.TemporaryDirectory() as td:
        out = Path(td) / "h.txt"
        r = subprocess.run([str(exe), "12", "841", str(WITNESS), str(out), "--t", "0.5", "--quiet"],
                           capture_output=True, text=True, check=True).stdout
        d = dict(kv.split("=", 1) for kv in r.split())
        assert float(d["E"]) == 0.0 and int(d["iters"]) == 0, r
        assert np.abs(unit(out) - unit(WITNESS)).max() < 1e-15
        U = unit(WITNESS)
        G = U @ U.T
        np.fill_diagonal(G, -2)
        t = 0.4999999
        E0 = (np.maximum(G - t, 0) ** 2).sum() / 2
        r = subprocess.run([str(exe), "12", "841", str(WITNESS), str(out), "--t", str(t),
                            "--iters", "300", "--quiet"],
                           capture_output=True, text=True, check=True).stdout
        d = dict(kv.split("=", 1) for kv in r.split())
        V = unit(out)
        H = V @ V.T
        np.fill_diagonal(H, -2)
        E1 = (np.maximum(H - t, 0) ** 2).sum() / 2
        assert E0 > 0 and E1 <= E0 * (1 + 1e-9), (E0, E1)
        assert abs(E1 - float(d["E"])) <= 1e-6 * E1 + 1e-30, (E1, d["E"])
    print("hingepol: PASS")


if __name__ == "__main__":
    test_certify()
    test_fastriesz()
    test_hingepol()
    print("PASS: fast tools")
