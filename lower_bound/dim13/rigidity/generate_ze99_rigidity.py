#!/usr/bin/env python3
"""Generate an exact finite-field rigidity matrix for the ZE99 d13 code.

The 1,154 vectors live in Q(sqrt(3))^13 and have squared norm 16.  For a
prime p in which 3 has a square root r, reduction sqrt(3) -> r produces a
matrix over F_p.  Rows are:

* one tangent/norm row v_i . delta_i = 0 for every vector;
* one contact row v_j . delta_i + v_i . delta_j = 0 for every exact
  kissing contact <v_i,v_j> = 8.

If the reduced matrix has rank 13*1154 - binom(13,2) = 14924 for any such
prime, then the characteristic-zero rigidity matrix has the same rank:
a nonzero maximal minor survives reduction, while the 78 infinitesimal
rotations give the matching upper bound.  This proves infinitesimal
rigidity over Q(sqrt(3)).
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import sys
import tempfile
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from constructions.ze99 import generate_ab, inner_ab, verify_ab  # noqa: E402

DIMENSION = 13
COUNT = 1154
ROTATIONS = DIMENSION * (DIMENSION - 1) // 2
EXPECTED_RANK = DIMENSION * COUNT - ROTATIONS


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def centered(value: int, prime: int) -> int:
    value %= prime
    return value if value <= prime // 2 else value - prime


def field_vector(vector: list[tuple[int, int]], prime: int, root: int) -> list[int]:
    return [centered(a + root * b, prime) for a, b in vector]


def matrix_rank_mod(rows: list[list[int]], prime: int) -> int:
    matrix = [[value % prime for value in row] for row in rows]
    rank = 0
    columns = len(matrix[0]) if matrix else 0
    for column in range(columns):
        pivot = next((i for i in range(rank, len(matrix)) if matrix[i][column]), None)
        if pivot is None:
            continue
        matrix[rank], matrix[pivot] = matrix[pivot], matrix[rank]
        inverse = pow(matrix[rank][column], -1, prime)
        matrix[rank] = [(value * inverse) % prime for value in matrix[rank]]
        for i in range(len(matrix)):
            if i == rank or matrix[i][column] == 0:
                continue
            factor = matrix[i][column]
            matrix[i] = [
                (left - factor * right) % prime
                for left, right in zip(matrix[i], matrix[rank])
            ]
        rank += 1
        if rank == len(matrix):
            break
    return rank


def layer(index: int) -> str:
    if index < 816:
        return "tetrad"
    if index < 1104:
        return "diamond"
    if index < 1106:
        return "axial"
    return "irrational"


def iter_nonzero(values: Iterable[int]) -> Iterable[tuple[int, int]]:
    for column, value in enumerate(values):
        if value:
            yield column, value


def write_matrix_market(
    vectors_ab: list[list[tuple[int, int]]],
    prime: int,
    root: int,
    output: Path,
) -> dict:
    vectors = [field_vector(vector, prime, root) for vector in vectors_ab]
    if matrix_rank_mod(vectors, prime) != DIMENSION:
        raise RuntimeError("reduced vectors do not span F_p^13")

    contacts: list[tuple[int, int]] = []
    contact_profile: Counter[str] = Counter()
    for i in range(COUNT):
        for j in range(i + 1, COUNT):
            if inner_ab(vectors_ab[i], vectors_ab[j]) == (8, 0):
                contacts.append((i, j))
                contact_profile["/".join(sorted((layer(i), layer(j))))] += 1

    output.parent.mkdir(parents=True, exist_ok=True)
    body_fd, body_name = tempfile.mkstemp(prefix="ze99-rigidity-", suffix=".body")
    os.close(body_fd)
    body = Path(body_name)
    nnz = 0
    row = 0
    try:
        with body.open("w", encoding="ascii") as handle:
            for i, vector in enumerate(vectors):
                row += 1
                for coordinate, value in iter_nonzero(vector):
                    handle.write(f"{row} {i * DIMENSION + coordinate + 1} {value}\n")
                    nnz += 1

            for i, j in contacts:
                row += 1
                for coordinate, value in iter_nonzero(vectors[j]):
                    handle.write(f"{row} {i * DIMENSION + coordinate + 1} {value}\n")
                    nnz += 1
                for coordinate, value in iter_nonzero(vectors[i]):
                    handle.write(f"{row} {j * DIMENSION + coordinate + 1} {value}\n")
                    nnz += 1

        expected_rows = COUNT + len(contacts)
        if row != expected_rows:
            raise RuntimeError(f"row mismatch: {row} != {expected_rows}")
        with output.open("wb") as target:
            target.write(b"%%MatrixMarket matrix coordinate integer general\n")
            target.write(
                f"% ZE99 spherical-code rigidity over F_{prime}, sqrt(3)={root}\n".encode("ascii")
            )
            target.write(f"{expected_rows} {DIMENSION * COUNT} {nnz}\n".encode("ascii"))
            with body.open("rb") as source:
                shutil.copyfileobj(source, target, length=1 << 20)
    finally:
        body.unlink(missing_ok=True)

    return {
        "prime": prime,
        "sqrt3_root": root,
        "sqrt3_check": (root * root) % prime,
        "vector_count": COUNT,
        "dimension": DIMENSION,
        "columns": DIMENSION * COUNT,
        "norm_rows": COUNT,
        "contact_rows": len(contacts),
        "rows": COUNT + len(contacts),
        "nonzeros": nnz,
        "contact_profile": dict(sorted(contact_profile.items())),
        "coordinate_rank_mod_p": DIMENSION,
        "rotational_kernel_dimension": ROTATIONS,
        "maximum_possible_rank": EXPECTED_RANK,
        "matrix_market_path": str(output),
        "matrix_market_bytes": output.stat().st_size,
        "matrix_market_sha256": sha256_file(output),
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--prime", type=int, required=True)
    parser.add_argument("--sqrt3-root", type=int, required=True)
    parser.add_argument("--output-matrix", type=Path, required=True)
    parser.add_argument("--output-json", type=Path, required=True)
    args = parser.parse_args()

    if args.prime <= 2:
        raise SystemExit("prime must be odd")
    if args.sqrt3_root * args.sqrt3_root % args.prime != 3 % args.prime:
        raise SystemExit("supplied root does not square to 3 modulo prime")

    vectors = generate_ab()
    exact = verify_ab(vectors)
    if not exact.get("ok") or exact.get("count") != COUNT:
        raise RuntimeError(f"exact ZE99 verification failed: {exact}")

    matrix = write_matrix_market(vectors, args.prime, args.sqrt3_root, args.output_matrix)
    payload = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "construction": "Zinoviev-Ericson 1999 exact reproduction",
        "exact_characteristic_zero_verification": exact,
        "finite_field_matrix": matrix,
        "proof_rule": (
            "If exact finite-field rank equals 14924, a 14924x14924 minor is nonzero "
            "after reduction and hence nonzero over Z[sqrt(3)]. The 78 independent "
            "infinitesimal rotations give rank <=14924, so equality proves "
            "characteristic-zero infinitesimal rigidity."
        ),
        "exact": True,
    }
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    print(json.dumps(payload, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
