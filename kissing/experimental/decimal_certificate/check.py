#!/usr/bin/env python3
"""Command-line entry point for the decimal direction certificate checker."""

try:  # Works both as ``python check.py`` and ``python -m ...check``.
    from .certificate import main
except ImportError:  # pragma: no cover - exercised by direct script execution
    from certificate import main


if __name__ == "__main__":
    raise SystemExit(main())
