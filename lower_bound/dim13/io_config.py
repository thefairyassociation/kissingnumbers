"""Save and load exact kissing configurations as JSON."""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

_DIM13 = Path(__file__).resolve().parent
CONFIG_DIR = _DIM13 / "configs"
BEST_PATH = _DIM13 / "best.json"
PROGRESS = _DIM13 / "progress.log"


def utc_now() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def save_config(
    *,
    dimension: int,
    count: int,
    vectors: list[list[str]],
    max_off_diagonal: str,
    method: str,
    unit: bool = False,
    extra: dict[str, Any] | None = None,
    filename: str | None = None,
    verified: bool = False,
    verifier: dict[str, Any] | None = None,
) -> Path:
    CONFIG_DIR.mkdir(parents=True, exist_ok=True)
    ts = utc_now()
    payload = {
        "dimension": dimension,
        "count": count,
        "unit": unit,
        "vectors": vectors,
        "max_off_diagonal": max_off_diagonal,
        "method": method,
        "timestamp": ts,
        "verified": verified,
        "coordinates": "exact (strings of rationals / algebraic numbers)",
    }
    if extra:
        payload["extra"] = extra
    if verifier:
        payload["verifier"] = verifier
    if filename is None:
        filename = f"n{dimension}_m{count}_{ts.replace(':','')}.json"
    path = CONFIG_DIR / filename
    path.write_text(json.dumps(payload, indent=2) + "\n")
    return path


def maybe_update_best(cfg_path: Path) -> None:
    cfg = json.loads(cfg_path.read_text())
    if not cfg.get("verified"):
        return
    best = None
    if BEST_PATH.exists():
        best = json.loads(BEST_PATH.read_text())
    if best is None or cfg["count"] > best.get("count", -1):
        BEST_PATH.write_text(
            json.dumps(
                {
                    "dimension": cfg["dimension"],
                    "count": cfg["count"],
                    "path": str(cfg_path),
                    "method": cfg["method"],
                    "max_off_diagonal": cfg["max_off_diagonal"],
                    "timestamp": cfg["timestamp"],
                    "verified": True,
                    "beats_live_record": cfg["count"] > 1154,
                    "live_record": {
                        "dimension": 13,
                        "lower": 1154,
                        "citation": "Zinoviev–Ericson 1999",
                        "source": "https://cohn.mit.edu/kissing-numbers",
                        "fetched": "2026-08-21",
                    },
                },
                indent=2,
            )
            + "\n"
        )


def log_progress(method: str, dimension: int, count: int, status: str, notes: str) -> None:
    PROGRESS.parent.mkdir(parents=True, exist_ok=True)
    line = f"{utc_now()}  method={method}  dim={dimension}  count={count}  {status}  {notes}\n"
    with PROGRESS.open("a", encoding="utf-8") as f:
        f.write(line)
