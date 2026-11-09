#!/usr/bin/env python3
"""Canonical exact verifier with coefficient/radicand schema support.

The original symbolic verifier lives in ``_verifier_impl.py``.  This entry point
preserves that API and adds the exact schema emitted by the PackingStar
reconstruction:

``x[i,j] = coefficient[i,j] * sqrt(coordinate_radicands[j])``.

Because each radicand is shared by an entire coordinate column, every norm and
inner product is a rational weighted dot product.  The fast path below evaluates
those products with ``fractions.Fraction`` and never uses floating point.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import sys
from fractions import Fraction
from pathlib import Path
from typing import Any, Iterable, Sequence


_IMPL_PATH = Path(__file__).with_name("_verifier_impl.py")
_SPEC = importlib.util.spec_from_file_location("kissing_dim13_verifier_impl", _IMPL_PATH)
if _SPEC is None or _SPEC.loader is None:
    raise ImportError(f"cannot load exact verifier implementation at {_IMPL_PATH}")
_impl = importlib.util.module_from_spec(_SPEC)
sys.modules[_SPEC.name] = _impl
_SPEC.loader.exec_module(_impl)

for _name in dir(_impl):
    if not _name.startswith("_"):
        globals()[_name] = getattr(_impl, _name)

RADICAND_MODEL = "x[i,j] = coefficient[i,j] * sqrt(radicand[j])"


def parse_fraction(value: Any, *, label: str = "value") -> Fraction:
    """Parse an exact rational and reject floating-point input."""
    if isinstance(value, float):
        raise ValueError(f"float {label}s are not allowed in the exact verifier")
    if isinstance(value, Fraction):
        return value
    if isinstance(value, int) and not isinstance(value, bool):
        return Fraction(value, 1)
    if isinstance(value, dict) and "num" in value and "den" in value:
        numerator = value["num"]
        denominator = value["den"]
        if not isinstance(numerator, int) or isinstance(numerator, bool):
            raise TypeError(f"{label} numerator must be an integer: {numerator!r}")
        if not isinstance(denominator, int) or isinstance(denominator, bool):
            raise TypeError(f"{label} denominator must be an integer: {denominator!r}")
        return Fraction(numerator, denominator)
    if isinstance(value, str):
        try:
            return Fraction(value.strip())
        except (ValueError, ZeroDivisionError):
            expression = _impl.parse_coord(value)
            if expression.is_rational:
                return Fraction(int(expression.p), int(expression.q))
            raise ValueError(f"{label} must be rational, got {value!r}")
    # SymPy Rational/Integer objects expose exact p/q attributes.
    if getattr(value, "is_rational", False) and hasattr(value, "p") and hasattr(value, "q"):
        return Fraction(int(value.p), int(value.q))
    raise TypeError(f"unsupported rational {label} type {type(value)!r}: {value!r}")


def _prepare_weighted_rationals(
    vectors: Sequence[Sequence[Any]],
    coordinate_radicands: Sequence[Any],
    *,
    dim: int | None,
) -> tuple[list[tuple[Fraction, ...]], list[tuple[Fraction, ...]], int]:
    if not vectors:
        raise ValueError("empty configuration")
    points = [
        tuple(parse_fraction(value, label="coordinate coefficient") for value in row)
        for row in vectors
    ]
    dimension = len(points[0])
    if dim is not None and dimension != dim:
        raise ValueError(f"expected dimension {dim}, got {dimension}")
    if any(len(row) != dimension for row in points):
        raise ValueError("ragged vectors")
    if len(coordinate_radicands) != dimension:
        raise ValueError(
            f"expected {dimension} coordinate radicands, got {len(coordinate_radicands)}"
        )
    weights = tuple(
        parse_fraction(value, label="coordinate radicand")
        for value in coordinate_radicands
    )
    for index, weight in enumerate(weights):
        if weight <= 0:
            raise ValueError(
                f"coordinate radicand {index} must be positive, got {weight}"
            )
    weighted_points = [
        tuple(coefficient * weight for coefficient, weight in zip(row, weights))
        for row in points
    ]
    return points, weighted_points, dimension


def _weighted_inner(
    weighted_left: Sequence[Fraction], right: Sequence[Fraction]
) -> Fraction:
    total = Fraction(0)
    for left, right_coordinate in zip(weighted_left, right):
        total += left * right_coordinate
    return total


def _verify_weighted_unit(
    vectors: Sequence[Sequence[Any]],
    coordinate_radicands: Sequence[Any],
    *,
    dim: int | None,
    max_inner: Any,
) -> dict[str, Any]:
    points, weighted_points, dimension = _prepare_weighted_rationals(
        vectors, coordinate_radicands, dim=dim
    )
    count = len(points)
    bound = parse_fraction(max_inner, label="inner-product bound")

    for index, (point, weighted_point) in enumerate(zip(points, weighted_points)):
        squared_norm = _weighted_inner(weighted_point, point)
        if squared_norm != 1:
            return {
                "ok": False,
                "reason": f"vector {index} has norm^2 = {squared_norm} != 1",
                "count": count,
                "dim": dimension,
                "coordinate_radicands_applied": True,
            }

    seen: dict[tuple[Fraction, ...], int] = {}
    duplicate_count = 0
    for index, point in enumerate(points):
        if point in seen:
            duplicate_count += 1
        else:
            seen[point] = index

    tight_pairs = 0
    worst: Fraction | None = None
    worst_pair: tuple[int, int] | None = None
    for left_index, weighted_left in enumerate(weighted_points):
        for right_index in range(left_index + 1, count):
            inner_product = _weighted_inner(weighted_left, points[right_index])
            if inner_product > bound:
                return {
                    "ok": False,
                    "reason": (
                        f"inner product vectors {left_index},{right_index} = "
                        f"{inner_product} > {bound}"
                    ),
                    "count": count,
                    "dim": dimension,
                    "violating_inner": str(inner_product),
                    "coordinate_radicands_applied": True,
                }
            if inner_product == bound:
                tight_pairs += 1
            if worst is None or inner_product > worst:
                worst = inner_product
                worst_pair = (left_index, right_index)

    distinct = duplicate_count == 0
    return {
        "ok": distinct,
        "count": count,
        "dim": dimension,
        "distinct": distinct,
        "n_duplicate_pairs": duplicate_count,
        "n_tight_pairs": tight_pairs,
        "max_offdiag": str(worst) if worst is not None else None,
        "max_offdiag_pair": worst_pair,
        "bound": str(bound),
        "all_unit": True,
        "all_offdiag_leq_bound": True,
        "coordinate_radicands_applied": True,
        "coordinate_model": RADICAND_MODEL,
    }


def _verify_weighted_equal_norm(
    vectors: Sequence[Sequence[Any]],
    coordinate_radicands: Sequence[Any],
    *,
    dim: int | None,
) -> dict[str, Any]:
    points, weighted_points, dimension = _prepare_weighted_rationals(
        vectors, coordinate_radicands, dim=dim
    )
    count = len(points)
    norms = [
        _weighted_inner(weighted, point)
        for weighted, point in zip(weighted_points, points)
    ]
    norm = norms[0]
    if norm == 0:
        return {"ok": False, "reason": "zero vector", "count": count, "dim": dimension}
    for index, squared_norm in enumerate(norms):
        if squared_norm != norm:
            return {
                "ok": False,
                "reason": f"unequal norms: ||v0||^2={norm}, ||v{index}||^2={squared_norm}",
                "count": count,
                "dim": dimension,
                "coordinate_radicands_applied": True,
            }

    seen: dict[tuple[Fraction, ...], int] = {}
    for index, point in enumerate(points):
        if point in seen:
            return {
                "ok": False,
                "reason": f"duplicate vectors {seen[point]} and {index}",
                "count": count,
                "dim": dimension,
                "coordinate_radicands_applied": True,
            }
        seen[point] = index

    bound = norm / 2
    tight_pairs = 0
    worst: Fraction | None = None
    worst_pair: tuple[int, int] | None = None
    for left_index, weighted_left in enumerate(weighted_points):
        for right_index in range(left_index + 1, count):
            inner_product = _weighted_inner(weighted_left, points[right_index])
            if inner_product > bound:
                return {
                    "ok": False,
                    "reason": (
                        f"inner product vectors {left_index},{right_index} = "
                        f"{inner_product} > half-norm {bound}"
                    ),
                    "count": count,
                    "dim": dimension,
                    "violating_inner": str(inner_product),
                    "norm2": str(norm),
                    "coordinate_radicands_applied": True,
                }
            if inner_product == bound:
                tight_pairs += 1
            if worst is None or inner_product > worst:
                worst = inner_product
                worst_pair = (left_index, right_index)

    return {
        "ok": True,
        "count": count,
        "dim": dimension,
        "distinct": True,
        "n_tight_pairs": tight_pairs,
        "norm2": str(norm),
        "max_offdiag_unnormalized": str(worst),
        "max_offdiag_unit": str(worst / norm) if worst is not None else None,
        "max_offdiag_pair": worst_pair,
        "bound_unnormalized": str(bound),
        "all_unit_after_scale": True,
        "all_offdiag_leq_bound": True,
        "coordinate_radicands_applied": True,
        "coordinate_model": RADICAND_MODEL,
    }


def verify_unit_vectors(
    vectors: Sequence[Sequence[Any]],
    *,
    dim: int | None = None,
    max_inner: Any = _impl.HALF,
    coordinate_radicands: Sequence[Any] | None = None,
) -> dict[str, Any]:
    if coordinate_radicands is None:
        result = _impl.verify_unit_vectors(vectors, dim=dim, max_inner=max_inner)
        result["coordinate_radicands_applied"] = False
        return result
    return _verify_weighted_unit(
        vectors,
        coordinate_radicands,
        dim=dim,
        max_inner=max_inner,
    )


def verify_equal_norm(
    vectors: Sequence[Sequence[Any]],
    *,
    dim: int | None = None,
    coordinate_radicands: Sequence[Any] | None = None,
) -> dict[str, Any]:
    if coordinate_radicands is None:
        result = _impl.verify_equal_norm(vectors, dim=dim)
        result["coordinate_radicands_applied"] = False
        return result
    return _verify_weighted_equal_norm(vectors, coordinate_radicands, dim=dim)


def verify_config_file(path: str) -> dict[str, Any]:
    config = _impl.load_config(path)
    if "vectors" not in config:
        raise ValueError("configuration is missing 'vectors'")
    vectors = config["vectors"]
    if "count" in config and config["count"] != len(vectors):
        return {
            "ok": False,
            "reason": (
                f"declared count {config['count']} does not match {len(vectors)} vectors"
            ),
            "count": len(vectors),
            "dim": config.get("dimension"),
            "path": path,
            "method": config.get("method"),
        }

    radicands = config.get("coordinate_radicands")
    if radicands is not None:
        model = config.get("coordinate_model")
        if model not in (None, RADICAND_MODEL):
            raise ValueError(f"unsupported coordinate_model for radicands: {model!r}")
    if config.get("unit"):
        result = verify_unit_vectors(
            vectors,
            dim=config.get("dimension"),
            coordinate_radicands=radicands,
        )
    else:
        result = verify_equal_norm(
            vectors,
            dim=config.get("dimension"),
            coordinate_radicands=radicands,
        )
    result["path"] = path
    result["method"] = config.get("method")
    return result


def main(argv: Iterable[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Exact kissing-number verifier")
    parser.add_argument("config", help="JSON configuration with exact coordinates")
    args = parser.parse_args(list(argv) if argv is not None else None)
    try:
        result = verify_config_file(args.config)
    except Exception as exc:
        result = {
            "ok": False,
            "path": args.config,
            "error_type": type(exc).__name__,
            "reason": str(exc),
        }
    print(json.dumps(result, indent=2, default=str))
    return 0 if result.get("ok") else 1


if __name__ == "__main__":
    raise SystemExit(main())
