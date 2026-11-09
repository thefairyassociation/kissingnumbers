"""Independent exact checker for finite-decimal direction files.

The input format is deliberately small: every non-blank, non-comment line is
one row of whitespace-separated finite decimal coordinates.  A ``#`` starts a
comment, including an inline comment.  Coordinates are parsed as exact
``Decimal`` values and immediately converted to ``Fraction``; no binary float
is used anywhere in the certificate path.

For a row ``x``, the checker clears its exact coordinate denominators to get
an integer direction ``v`` and sets ``q = v . v``.  The unit vector represented
by the row is ``v / sqrt(q)``.  For a pair of rows with ``p = vi . vj``,

    p <= 0                         implies the bound directly;
    p > 0 and 4*p*p <= qi*qj       is equivalent to IP <= 1/2.

The optional strict mode replaces ``<=`` by ``<`` for positive ``p``.  This
is an algebraic comparison of integers and does not estimate square roots or
use a floating-point tolerance.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import sys
from decimal import Decimal, InvalidOperation
from fractions import Fraction
from functools import reduce
from math import gcd, lcm
from pathlib import Path
from typing import Iterable, Sequence


class CertificateError(ValueError):
    """A source file is malformed or fails the exact spherical-code bound."""


# This accepts ordinary decimal notation and scientific notation, but not
# fractions, hexadecimal numbers, NaN, infinities, or other Decimal syntax.
_FINITE_DECIMAL = re.compile(
    r"[+-]?(?:(?:[0-9]+(?:\.[0-9]*)?)|(?:\.[0-9]+))(?:[eE][+-]?[0-9]+)?\Z"
)


def _parse_coordinate(token: str, line_number: int) -> Fraction:
    if _FINITE_DECIMAL.fullmatch(token) is None:
        raise CertificateError(
            f"line {line_number}: malformed coordinate {token!r}; "
            "expected a finite decimal"
        )
    try:
        decimal_value = Decimal(token)
        if not decimal_value.is_finite():
            raise CertificateError(
                f"line {line_number}: non-finite coordinate {token!r}"
            )
        # Fraction(Decimal) is an exact conversion.  In particular, this is
        # intentionally not Fraction(float(token)).
        return Fraction(decimal_value)
    except CertificateError:
        raise
    except (InvalidOperation, ValueError, OverflowError, MemoryError) as exc:
        raise CertificateError(
            f"line {line_number}: cannot parse coordinate {token!r} exactly"
        ) from exc


def parse_coordinate_file(path: str | Path) -> tuple[list[tuple[Fraction, ...]], str]:
    """Read ``path`` and return exact rows plus the SHA256 of source bytes.

    The hash is computed over the original bytes, including comments and line
    endings, so the certificate identifies precisely the file that was read.
    """

    source_path = Path(path)
    try:
        source_bytes = source_path.read_bytes()
    except OSError as exc:
        raise CertificateError(f"cannot read source {source_path}: {exc}") from exc
    source_hash = hashlib.sha256(source_bytes).hexdigest()
    try:
        source_text = source_bytes.decode("utf-8")
    except UnicodeDecodeError as exc:
        raise CertificateError(f"source {source_path} is not valid UTF-8") from exc

    rows: list[tuple[Fraction, ...]] = []
    dimension: int | None = None
    for line_number, raw_line in enumerate(source_text.splitlines(), 1):
        body = raw_line.split("#", 1)[0].strip()
        if not body:
            continue
        tokens = body.split()
        row = tuple(_parse_coordinate(token, line_number) for token in tokens)
        if dimension is None:
            dimension = len(row)
            if dimension == 0:  # Defensive; split() already makes this moot.
                raise CertificateError(f"line {line_number}: empty coordinate row")
        elif len(row) != dimension:
            raise CertificateError(
                f"line {line_number}: dimension {len(row)} != {dimension}"
            )
        rows.append(row)

    if not rows:
        raise CertificateError("source contains no coordinate rows")
    return rows, source_hash


def _integer_direction(row: Sequence[Fraction], row_number: int) -> tuple[int, ...]:
    """Clear denominators and make one exact primitive integer direction."""

    denominator = 1
    for coordinate in row:
        denominator = lcm(denominator, coordinate.denominator)
    values = [coordinate.numerator * (denominator // coordinate.denominator) for coordinate in row]
    if not any(values):
        raise CertificateError(f"row {row_number} is the zero direction (q=0)")
    common = reduce(gcd, (abs(value) for value in values))
    return tuple(value // common for value in values)


def _bits(value: int) -> int:
    return abs(value).bit_length()


def certify(
    path: str | Path,
    *,
    expected_dimension: int | None = None,
    expected_count: int | None = None,
    strict: bool = False,
    label: str | None = None,
) -> dict[str, object]:
    """Certify all rows in a finite-decimal direction file.

    ``CertificateError`` is raised on malformed input, inconsistent shape,
    zero rows, or the first violating pair.  A successful return is JSON
    serializable and contains only exact integer/string metadata.
    """

    rows, source_hash = parse_coordinate_file(path)
    dimension = len(rows[0])
    count = len(rows)
    if expected_dimension is not None and dimension != expected_dimension:
        raise CertificateError(
            f"dimension {dimension} != expected dimension {expected_dimension}"
        )
    if expected_count is not None and count != expected_count:
        raise CertificateError(f"source count {count} != expected count {expected_count}")

    directions = [_integer_direction(row, i) for i, row in enumerate(rows)]
    norms = [sum(coordinate * coordinate for coordinate in direction) for direction in directions]
    pair_count = count * (count - 1) // 2
    positive_pairs = 0
    nonpositive_pairs = 0
    tight_pairs = 0
    max_direction_bits = max(_bits(value) for row in directions for value in row)
    max_norm_bits = max(_bits(value) for value in norms)

    checked_pairs = 0
    for i in range(count):
        vi = directions[i]
        qi = norms[i]
        for j in range(i):
            checked_pairs += 1
            p = sum(a * b for a, b in zip(vi, directions[j]))
            if p <= 0:
                nonpositive_pairs += 1
                continue
            positive_pairs += 1
            left = 4 * p * p
            right = qi * norms[j]
            if left == right:
                tight_pairs += 1
            fails = left >= right if strict else left > right
            if fails:
                comparator = "<" if strict else "<="
                raise CertificateError(
                    f"pair ({j},{i}) violates normalized IP {'<' if strict else '<='} 1/2: "
                    f"p={p}, 4*p^2={left} {comparator} q_i*q_j={right} is false"
                )

    result: dict[str, object] = {
        "ok": True,
        "status": "CERTIFIED",
        "label": label or "UNLABELED DECIMAL DIRECTIONS",
        "source": str(path),
        "source_sha256": source_hash,
        "dimension": dimension,
        "source_count": count,
        "pair_count": pair_count,
        "checked_pairs": checked_pairs,
        "positive_pairs": positive_pairs,
        "nonpositive_pairs": nonpositive_pairs,
        "tight_pairs": tight_pairs,
        "strict": strict,
        "arithmetic": {
            "coordinates": "finite decimal strings -> Decimal -> Fraction; no float intermediate",
            "directions": "clear each row denominator, then divide by the integer gcd",
            "normalization": "v_i / sqrt(q_i), q_i = v_i dot v_i > 0",
            "pair_test": "p <= 0, or 4*p^2 <= q_i*q_j for p > 0",
            "strict_pair_test": "p <= 0, or 4*p^2 < q_i*q_j for p > 0 when strict=true",
            "comparison": "exact Python integers only",
            "max_direction_coordinate_bits": max_direction_bits,
            "max_norm_bits": max_norm_bits,
        },
    }
    if expected_dimension is not None:
        result["expected_dimension"] = expected_dimension
    if expected_count is not None:
        result["expected_count"] = expected_count
    return result


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Certify finite-decimal spherical-code directions with exact integer arithmetic."
    )
    parser.add_argument("source", type=Path, help="whitespace-separated coordinate rows")
    parser.add_argument("--expected-dimension", type=int, metavar="N")
    parser.add_argument("--expected-count", type=int, metavar="COUNT")
    parser.add_argument(
        "--strict",
        action="store_true",
        help="require normalized inner products to be strictly below 1/2",
    )
    parser.add_argument("--label", help="provenance label copied to the result metadata")
    parser.add_argument(
        "--compact",
        action="store_true",
        help="write one-line JSON instead of indented JSON",
    )
    return parser


def main(argv: Iterable[str] | None = None) -> int:
    args = _parser().parse_args(list(argv) if argv is not None else None)
    try:
        result = certify(
            args.source,
            expected_dimension=args.expected_dimension,
            expected_count=args.expected_count,
            strict=args.strict,
            label=args.label,
        )
        exit_code = 0
    except CertificateError as exc:
        result = {
            "ok": False,
            "status": "REJECTED",
            "source": str(args.source),
            "error": str(exc),
            "arithmetic": "exact Decimal/Fraction parsing and integer pair comparison",
        }
        exit_code = 1
    print(json.dumps(result, indent=None if args.compact else 2, sort_keys=True, separators=(",", ":") if args.compact else None))
    return exit_code


if __name__ == "__main__":
    raise SystemExit(main())
