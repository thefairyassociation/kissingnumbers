#!/usr/bin/env python3
"""One list of the environment flags ``riesz.c`` reads.

Drivers that launch the optimizer must not inherit these from whatever shell
they were started in: a leftover ``KISS_FAITHFUL=1`` makes a 120000-step worker
fail the exact-35000-step guard, ``KISS_LOSS=ip`` silently changes the
objective, and ``KISS_FAITHFUL_EXTRA`` pins every run to one hypercube extra.
Build the child environment with :func:`clean_optimizer_env` and set the flags
the driver actually intends.

Keep this list in step with the ``KISS_*`` names in ``riesz.c``; it is shared so
the drivers cannot drift apart from each other.
"""

from __future__ import annotations

import os

OPTIMIZER_FLAGS: tuple[str, ...] = (
    "KISS_FAITHFUL",
    "KISS_FAITHFUL_EXTRA",
    "KISS_LOSS",
    "KISS_SOLVER",
    "KISS_JIT",
    "KISS_S0",
    "KISS_SMUL",
    "KISS_SMAX",
    "KISS_M",
    "KISS_INNER",
    "KISS_POLISH",
    "KISS_ADAM_POLISH",
    "KISS_ADAM_POLISH_ONLY",
    "KISS_ADAM_POLISH_STEPS",
    "KISS_ADAM_POLISH_STAGES",
    "KISS_ADAM_POLISH_LR_SCALE",
    "KISS_ADAM_RAW",
    "KISS_ADAM_EPS",
    "KISS_ADAM_BASE_START",
    "KISS_ADAM_BASE_END",
    "KISS_PENALTY_ONLY",
    "KISS_PENALTY_TARGET",
    "KISS_PROFILE",
    "KISS_SELFTEST",
    "KISS_THREADS",
    "OMP_NUM_THREADS",
    "OPENBLAS_NUM_THREADS",
)


def clean_optimizer_env(**settings: str) -> dict[str, str]:
    """The current environment with every optimizer flag stripped, then
    ``settings`` applied.  Values must already be strings."""
    env = {k: v for k, v in os.environ.items() if k not in OPTIMIZER_FLAGS}
    unknown = sorted(set(settings) - set(OPTIMIZER_FLAGS))
    if unknown:
        raise ValueError(f"not optimizer flags: {', '.join(unknown)}")
    env.update(settings)
    return env
