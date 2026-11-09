#!/usr/bin/env python3
"""Opt-in batched N=841 Riesz search driver.

This driver is deliberately separate from the historical search scripts.  It
implements the public search protocol around :mod:`kernel`: 840 fixed seed
rows, one independent hypercube point per candidate, raw Adam updates, and the
published 13-stage schedule.  It does not polish a candidate or assert an
exact kissing configuration.

A full invocation is explicit (``--full``); small ``--max-updates`` runs are
useful for smoke tests and checkpoint/resume checks.  A checkpoint is one
atomically replaced ``.npz`` containing numeric arrays and a UTF-8 JSON
metadata payload; loading never enables pickle.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import sys
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping

import numpy as np

try:  # torch is intentionally lazy at import time for source/document checks.
    import torch
except Exception:  # pragma: no cover - exercised on installations without torch
    torch = None  # type: ignore[assignment]


# ``run.py`` is normally imported as an experimental package, but it is also a
# useful executable.  The kernel is supplied by the sibling implementation.
try:  # pragma: no cover - import branch depends on invocation style
    from . import kernel  # type: ignore[import-not-found]
except ImportError:  # pragma: no cover
    _HERE = Path(__file__).resolve().parent
    if str(_HERE) not in sys.path:
        sys.path.insert(0, str(_HERE))
    import kernel  # type: ignore[no-redef]


N_POINTS = 841
DIMENSION = 12
N_FIXED = 840
FULL_SCHEDULE_UPDATES = 35_000
SCHEMA_VERSION = 1

# Keep this duplicate as a validation tripwire.  The source of truth used for
# execution is kernel.SEARCH_SCHEDULE; changing either must be deliberate.
_EXPECTED_SCHEDULE: tuple[tuple[float, int, float], ...] = (
    (8.0, 1000, 0.005),
    (16.0, 1000, 0.003),
    (32.0, 1000, 0.002),
    (64.0, 2000, 0.001),
    (128.0, 2000, 0.0005),
    (256.0, 2000, 0.0002),
    (512.0, 2000, 0.0001),
    (1024.0, 4000, 0.00005),
    (2048.0, 4000, 0.00001),
    (4096.0, 4000, 0.00001),
    (10000.0, 4000, 0.000005),
    (20000.0, 4000, 0.000001),
    (40000.0, 4000, 0.000001),
)


class RunnerError(RuntimeError):
    """A user-facing configuration, checkpoint, or numerical failure."""


class CheckpointError(RunnerError):
    """A checkpoint cannot safely be resumed."""


@dataclass(frozen=True)
class RunConfig:
    """Immutable search choices plus output/checkpoint locations.

    ``max_updates`` is a stopping bound rather than part of the optimizer
    configuration.  It may be raised when resuming a bounded diagnostic.
    """

    seed: int = 0
    seed_file: Path | None = None
    batch_size: int = 4
    macro_repeats: int = 1
    device: str = "cpu"
    dtype: str = "float64"
    max_updates: int | None = None
    checkpoint: Path | None = None
    output: Path | None = None
    resume: bool = False
    checkpoint_every: int = 0
    threads: int = 1

    @property
    def total_updates(self) -> int:
        return self.macro_repeats * FULL_SCHEDULE_UPDATES

    @property
    def immutable(self) -> dict[str, Any]:
        return {
            "seed": self.seed,
            "batch_size": self.batch_size,
            "macro_repeats": self.macro_repeats,
            "device": self.device,
            "dtype": self.dtype,
            "threads": self.threads,
            "torch_version": str(_require_torch().__version__),
            "kernel_sha256": hashlib.sha256(Path(kernel.__file__).read_bytes()).hexdigest(),
            "dimension": DIMENSION,
            "points": N_POINTS,
            "fixed_seed_rows": N_FIXED,
            "schedule": [list(row) for row in normalized_schedule()],
        }


def normalized_schedule() -> tuple[tuple[float, int, float], ...]:
    """Validate and return the sibling kernel's exact published schedule."""
    try:
        raw_schedule: Iterable[Iterable[Any]] = kernel.SEARCH_SCHEDULE
    except AttributeError as exc:  # pragma: no cover - catches a broken kernel
        raise RunnerError("kernel.SEARCH_SCHEDULE is required") from exc
    try:
        schedule = tuple((float(row[0]), int(row[1]), float(row[2])) for row in raw_schedule)
    except (TypeError, IndexError, ValueError) as exc:
        raise RunnerError("kernel.SEARCH_SCHEDULE must contain (s, steps, lr) rows") from exc
    if schedule != _EXPECTED_SCHEDULE:
        raise RunnerError(
            "kernel.SEARCH_SCHEDULE differs from the published 35,000-update "
            f"schedule: {schedule!r}"
        )
    if sum(row[1] for row in schedule) != FULL_SCHEDULE_UPDATES:
        raise RunnerError("published schedule must contain exactly 35000 updates")
    for s, steps, lr in schedule:
        if not (math.isfinite(s) and s > 0 and steps > 0 and math.isfinite(lr) and lr > 0):
            raise RunnerError("search schedule contains a non-finite or non-positive value")
    return schedule


def _require_torch() -> Any:
    if torch is None:
        raise RunnerError("batched841 requires PyTorch; install the configured runtime first")
    return torch


def _validate_config(config: RunConfig) -> None:
    normalized_schedule()
    destinations = []
    if config.output is not None:
        destinations.extend([config.output.resolve(), config.output.with_suffix(".json").resolve()])
    if config.checkpoint is not None:
        destinations.append(_checkpoint_paths(config.checkpoint)[0].resolve())
    if len(destinations) != len(set(destinations)):
        raise RunnerError("candidate, JSON sidecar, and checkpoint must use different paths")
    if config.seed_file is not None and config.seed_file.resolve() in destinations:
        raise RunnerError("output or checkpoint must not overwrite the input seed")
    if config.seed < 0 or config.seed >= 2**63:
        raise RunnerError("--seed must be in [0, 2**63)")
    if config.batch_size < 1:
        raise RunnerError("--batch-size must be positive")
    if config.macro_repeats < 1:
        raise RunnerError("--macro-repeats must be positive")
    if config.device not in {"cpu", "cuda"}:
        raise RunnerError("--device must be cpu or cuda")
    if config.dtype != "float64":
        raise RunnerError("batched841 only permits --dtype float64")
    if config.max_updates is not None:
        if config.max_updates < 0:
            raise RunnerError("--max-updates must be non-negative")
        if config.max_updates > config.total_updates:
            raise RunnerError(
                f"--max-updates cannot exceed {config.total_updates} for "
                f"{config.macro_repeats} macro repeat(s)"
            )
    if config.checkpoint_every < 0:
        raise RunnerError("--checkpoint-every must be non-negative")
    if config.threads < 1:
        raise RunnerError("--threads must be positive")
    if config.resume and config.checkpoint is None:
        raise RunnerError("--resume requires --checkpoint")
    if config.device == "cuda" and not _require_torch().cuda.is_available():
        raise RunnerError("CUDA was requested but torch.cuda.is_available() is false")


def _sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _load_core(seed_file: Path | None) -> tuple[np.ndarray, str, str | None]:
    """Read only the fixed 840 rows and return their content digest.

    ``seed841.py`` historically writes an 841st row.  That row is deliberately
    ignored here because every batched candidate gets its own fresh random
    hypercube point.
    """
    if seed_file is None:
        try:
            root = Path(__file__).resolve().parents[3]
            dim12 = root / "kissing" / "dim12"
            if str(dim12) not in sys.path:
                sys.path.insert(0, str(dim12))
            from constructions.clebsch840 import construction_840, vectors_as_float

            core = vectors_as_float(construction_840())
        except Exception as exc:  # pragma: no cover - repository corruption
            raise RunnerError("could not construct the default 840-point seed") from exc
        source = None
        digest = _sha256_bytes(np.ascontiguousarray(core, dtype=np.float64).tobytes())
    else:
        seed_file = Path(seed_file)
        try:
            raw = np.loadtxt(seed_file, dtype=np.float64)
        except Exception as exc:
            raise RunnerError(f"could not read seed file {seed_file}: {exc}") from exc
        if raw.ndim != 2 or raw.shape[1] != DIMENSION or raw.shape[0] < N_FIXED:
            raise RunnerError(
                f"seed file must contain at least {N_FIXED} rows x {DIMENSION} columns; "
                f"got {raw.shape}"
            )
        core = np.ascontiguousarray(raw[:N_FIXED], dtype=np.float64)
        # Hash the rows that affect this protocol.  A historical seed841 file
        # also contains an ignored 841st row; changing that row must not make
        # a checkpoint incompatible because this runner redraws every extra.
        digest = _sha256_bytes(np.ascontiguousarray(core, dtype=np.float64).tobytes())
        source = str(seed_file.resolve())
    if core.shape != (N_FIXED, DIMENSION):
        raise RunnerError(f"fixed seed has shape {core.shape}, expected {(N_FIXED, DIMENSION)}")
    if not np.isfinite(core).all():
        raise RunnerError("fixed seed contains non-finite coordinates")
    norms = np.linalg.norm(core, axis=1)
    if not np.isfinite(norms).all() or np.any(norms <= 1e-15):
        raise RunnerError("fixed seed contains a non-finite or zero row")
    return core, digest, source


def _torch_dtype() -> Any:
    return _require_torch().float64


def _make_generator(device: str, seed: int) -> Any:
    t = _require_torch()
    try:
        generator = t.Generator(device=device)
    except Exception as exc:  # pragma: no cover - old torch builds
        raise RunnerError(f"could not create a {device} torch generator") from exc
    generator.manual_seed(seed)
    return generator


def _generator_state(generator: Any) -> np.ndarray:
    state = generator.get_state()
    return np.ascontiguousarray(state.detach().cpu().numpy(), dtype=np.uint8)


def _restore_generator(device: str, state: np.ndarray) -> Any:
    t = _require_torch()
    if state.ndim != 1 or state.dtype != np.uint8 or state.size == 0:
        raise CheckpointError("checkpoint RNG state is not a non-empty uint8 vector")
    generator = _make_generator(device, 0)
    state_tensor = t.from_numpy(np.ascontiguousarray(state, dtype=np.uint8)).to(device="cpu")
    try:
        generator.set_state(state_tensor)
    except Exception as exc:
        raise CheckpointError("checkpoint RNG state cannot be restored on this device") from exc
    return generator


def _to_numpy(value: Any, name: str) -> np.ndarray:
    if torch is not None and isinstance(value, torch.Tensor):
        value = value.detach().to(device="cpu").numpy()
    arr = np.asarray(value)
    if arr.dtype == object:
        raise RunnerError(f"kernel state {name} has an object dtype")
    # np.ascontiguousarray turns a zero-dimensional scalar into shape (1,)
    # (notably for the Python integer state['step']).  Preserve scalar shape.
    if arr.ndim == 0:
        return np.array(arr, copy=True)
    return np.ascontiguousarray(arr)


def _scalar_int(value: Any, name: str) -> int:
    arr = _to_numpy(value, name)
    if arr.shape != () or arr.dtype.kind not in "iu":
        raise RunnerError(f"kernel state {name} must be an integer scalar")
    number = int(arr.item())
    if number < 0:
        raise RunnerError(f"kernel state {name} is negative")
    return number


def _assert_finite_state(state: Mapping[str, Any], where: str, expected_shape: tuple[int, ...]) -> None:
    for key in ("raw", "m", "v"):
        if key not in state:
            raise RunnerError(f"kernel state is missing {key} at {where}")
        value = state[key]
        if torch is not None and isinstance(value, torch.Tensor):
            shape = tuple(value.shape)
            if shape != expected_shape:
                raise RunnerError(
                    f"kernel state {key} has shape {shape} at {where}; expected {expected_shape}"
                )
            if not bool(torch.isfinite(value).all().item()):
                raise RunnerError(f"non-finite {key} at {where}")
            if key == "v" and not bool((value >= 0).all().item()):
                raise RunnerError(f"negative second moment v at {where}")
            if key == "raw":
                norms = torch.linalg.vector_norm(value, dim=-1)
                if not bool(torch.isfinite(norms).all().item()) or not bool(
                    (norms > 1.0e-12).all().item()
                ):
                    raise RunnerError(f"raw state contains an invalid row at {where}")
            continue
        arr = _to_numpy(value, key)
        if arr.shape != expected_shape:
            raise RunnerError(
                f"kernel state {key} has shape {arr.shape} at {where}; expected {expected_shape}"
            )
        if not np.isfinite(arr).all():
            raise RunnerError(f"non-finite {key} at {where}")
        if key == "v" and np.any(arr < 0):
            raise RunnerError(f"negative second moment v at {where}")
        if key == "raw":
            norms = np.linalg.norm(arr, axis=-1)
            if not np.isfinite(norms).all() or np.any(norms <= 1.0e-12):
                raise RunnerError(f"raw state contains an invalid row at {where}")
    _scalar_int(state.get("step"), "step")


def _new_batch(core: np.ndarray, batch_size: int, device: str, generator: Any) -> Any:
    """Build B copies of the exact core and B torch-uniform hypercube extras."""
    t = _require_torch()
    core_t = t.from_numpy(np.ascontiguousarray(core, dtype=np.float64)).to(
        device=device, dtype=_torch_dtype()
    )
    # Draw on the selected generator/device.  The generator state is the only
    # random state this runner owns, and is stored in every checkpoint.
    bits = t.randint(
        0,
        2,
        (batch_size, DIMENSION),
        generator=generator,
        device=device,
        dtype=t.int64,
    )
    extras = (bits.to(dtype=_torch_dtype()) * 2.0 - 1.0) / math.sqrt(DIMENSION)
    fixed = core_t.unsqueeze(0).expand(batch_size, -1, -1)
    return t.cat((fixed, extras.unsqueeze(1)), dim=1)


def _new_state(raw: Any) -> dict[str, Any]:
    try:
        state = kernel.initial_state(raw)
    except Exception as exc:
        raise RunnerError(f"kernel.initial_state failed: {exc}") from exc
    if not isinstance(state, dict):
        raise RunnerError("kernel.initial_state must return a dict")
    return state


def _step(state: dict[str, Any], s: float, lr: float, expected_shape: tuple[int, ...]) -> dict[str, Any]:
    before = _scalar_int(state.get("step"), "step")
    try:
        returned = kernel.adam_step(state, s, lr)
    except Exception as exc:
        raise RunnerError(f"kernel.adam_step failed at s={s:g}, lr={lr:g}: {exc}") from exc
    if returned is not None:
        if not isinstance(returned, dict):
            raise RunnerError("kernel.adam_step returned a non-dict state")
        state = returned
    after = _scalar_int(state.get("step"), "step")
    if after != before + 1:
        raise RunnerError(
            f"kernel.adam_step must increment state['step'] by one (was {before}, now {after})"
        )
    _assert_finite_state(state, "after Adam update", expected_shape)
    return state


def _normalised_numpy(raw: Any) -> np.ndarray:
    """Read back one raw candidate through the kernel view, then verify in NumPy."""
    t = _require_torch()
    if isinstance(raw, np.ndarray):
        raw = t.from_numpy(np.ascontiguousarray(raw, dtype=np.float64))
    if not isinstance(raw, t.Tensor):
        raise RunnerError("candidate raw state is not a torch tensor or NumPy array")
    if raw.ndim == 2:
        raw = raw.unsqueeze(0)
    if raw.ndim != 3 or raw.shape[0] != 1:
        raise RunnerError(f"candidate raw state must have shape (1, N, d), got {tuple(raw.shape)}")
    try:
        view = kernel.normalized_view(raw)
    except Exception as exc:
        raise RunnerError(f"kernel.normalized_view failed: {exc}") from exc
    arr = _to_numpy(view[0], "normalized_view")
    if arr.shape != (N_POINTS, DIMENSION):
        raise RunnerError(
            f"normalized candidate has shape {arr.shape}; expected {(N_POINTS, DIMENSION)}"
        )
    if not np.isfinite(arr).all():
        raise RunnerError("normalized candidate contains non-finite coordinates")
    norms = np.linalg.norm(arr, axis=1)
    if not np.isfinite(norms).all() or np.any(norms <= 1e-15):
        raise RunnerError("normalized candidate contains an invalid row")
    # Independent normalization is intentional: do not trust the kernel view's
    # row norms or any optimizer-reported objective.
    normalized = arr / norms[:, None]
    gram = normalized @ normalized.T
    upper = gram[np.triu_indices(N_POINTS, 1)]
    maximum = float(np.max(upper))
    if not math.isfinite(maximum):
        raise RunnerError("independent NumPy Gram recomputation is non-finite")
    return normalized


def _metrics(candidate: np.ndarray) -> dict[str, Any]:
    if candidate.shape != (N_POINTS, DIMENSION) or not np.isfinite(candidate).all():
        raise RunnerError("cannot measure an invalid candidate")
    norms = np.linalg.norm(candidate, axis=1)
    if not np.isfinite(norms).all() or np.any(norms <= 1.0e-15):
        raise RunnerError("cannot measure a candidate with an invalid row")
    gram = candidate @ candidate.T
    upper = gram[np.triu_indices(N_POINTS, 1)]
    maximum = float(np.max(upper))
    return {
        "numpy_max_inner_product": maximum,
        "numpy_max_norm_error": float(np.max(np.abs(norms - 1.0))),
        "numpy_finite": True,
        "numpy_rows": N_POINTS,
        "numpy_dimension": DIMENSION,
    }


def _checkpoint_paths(path: Path) -> tuple[Path, Path]:
    path = Path(path)
    if path.suffix == ".npz":
        return path, path.with_suffix(".json")
    if path.suffix == ".json":
        return path.with_suffix(".npz"), path
    return Path(str(path) + ".npz"), Path(str(path) + ".json")


def _atomic_replace_bytes(path: Path, data: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, name = tempfile.mkstemp(prefix=f".{path.name}.", dir=str(path.parent))
    try:
        with os.fdopen(fd, "wb") as stream:
            stream.write(data)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(name, path)
    except Exception:
        try:
            os.unlink(name)
        except OSError:
            pass
        raise


def _atomic_npz(path: Path, arrays: Mapping[str, np.ndarray]) -> None:
    """Write one durable NPZ payload and atomically replace the old one."""
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, name = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".npz", dir=str(path.parent))
    try:
        with os.fdopen(fd, "wb") as stream:
            np.savez_compressed(stream, **arrays)
            stream.flush()
            os.fsync(stream.fileno())
    except Exception:
        try:
            os.unlink(name)
        except OSError:
            pass
        raise
    try:
        os.replace(name, path)
    except Exception:
        try:
            os.unlink(name)
        except OSError:
            pass
        raise


def _json_safe(value: Any) -> Any:
    if isinstance(value, (str, int, bool)) or value is None:
        return value
    if isinstance(value, float):
        if not math.isfinite(value):
            raise RunnerError("checkpoint metadata contains a non-finite float")
        return value
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, np.generic):
        return _json_safe(value.item())
    if isinstance(value, Mapping):
        return {str(k): _json_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(v) for v in value]
    raise RunnerError(f"checkpoint metadata value is not JSON-safe: {type(value).__name__}")


def _checkpoint_arrays(
    state: Mapping[str, Any], generator: Any, best: np.ndarray | None, best_metric: float
) -> dict[str, np.ndarray]:
    arrays = {
        "raw": _to_numpy(state["raw"], "raw").astype(np.float64, copy=False),
        "m": _to_numpy(state["m"], "m").astype(np.float64, copy=False),
        "v": _to_numpy(state["v"], "v").astype(np.float64, copy=False),
        "step": np.asarray(_scalar_int(state["step"], "step"), dtype=np.int64),
        "rng_state": _generator_state(generator),
        "best_valid": np.asarray(1 if best is not None else 0, dtype=np.int8),
        "best_raw": np.zeros((N_POINTS, DIMENSION), dtype=np.float64)
        if best is None
        else np.ascontiguousarray(best, dtype=np.float64),
        "best_max_inner_product": np.asarray(best_metric, dtype=np.float64),
    }
    for key, value in arrays.items():
        if value.dtype == object:
            raise RunnerError(f"checkpoint array {key} unexpectedly has object dtype")
    return arrays


def _write_checkpoint(
    path: Path,
    *,
    config: RunConfig,
    seed_digest: str,
    seed_source: str | None,
    state: Mapping[str, Any],
    generator: Any,
    macro: int,
    stage: int,
    in_stage: int,
    updates_total: int,
    best: np.ndarray | None,
    best_metric: float,
) -> None:
    npz_path, _ = _checkpoint_paths(path)
    arrays = _checkpoint_arrays(state, generator, best, best_metric)
    # Validate before publishing the payload.
    expected_shape = (config.batch_size, N_POINTS, DIMENSION)
    _assert_finite_state(state, "checkpoint", expected_shape)
    metadata = {
        "schema_version": SCHEMA_VERSION,
        "kind": "batched841_checkpoint",
        "array_file": npz_path.name,
        "config": config.immutable,
        "seed_digest": seed_digest,
        "seed_source": seed_source,
        "cursor": {
            "macro": macro,
            "stage": stage,
            "in_stage": in_stage,
            "updates_total": updates_total,
        },
        "step": _scalar_int(state["step"], "step"),
        "rng": {"kind": "torch.Generator", "device": config.device, "array": "rng_state"},
        "state_shapes": {
            "raw": list(arrays["raw"].shape),
            "m": list(arrays["m"].shape),
            "v": list(arrays["v"].shape),
        },
    }
    # Metadata is JSON encoded inside the same atomic NPZ as the state arrays;
    # there is no second file that can get out of sync with a replacement.
    arrays["metadata_json"] = np.frombuffer(
        (json.dumps(_json_safe(metadata), sort_keys=True) + "\n").encode("utf-8"),
        dtype=np.uint8,
    ).copy()
    _atomic_npz(npz_path, arrays)


def _read_checkpoint(
    path: Path, config: RunConfig, seed_digest: str
) -> tuple[dict[str, Any], Any, dict[str, Any], np.ndarray | None, float]:
    npz_path, _ = _checkpoint_paths(path)
    if not npz_path.exists():
        raise CheckpointError(f"resume requires checkpoint NPZ {npz_path}")
    try:
        with np.load(npz_path, allow_pickle=False) as loaded:
            # Load one atomic snapshot, including metadata and all arrays.
            arrays = {key: np.array(loaded[key], copy=True) for key in loaded.files}
            if "metadata_json" not in loaded.files:
                raise CheckpointError("checkpoint metadata_json payload is missing")
            metadata_bytes = arrays["metadata_json"]
            if metadata_bytes.ndim != 1 or metadata_bytes.dtype != np.uint8:
                raise CheckpointError("checkpoint metadata_json payload is malformed")
            metadata = json.loads(metadata_bytes.tobytes().decode("utf-8"))
    except CheckpointError:
        raise
    except Exception as exc:
        raise CheckpointError(f"could not read checkpoint metadata: {exc}") from exc
    if not isinstance(metadata, dict) or metadata.get("schema_version") != SCHEMA_VERSION:
        raise CheckpointError("unsupported checkpoint schema")
    if metadata.get("kind") != "batched841_checkpoint":
        raise CheckpointError("checkpoint kind is not batched841")
    if metadata.get("seed_digest") != seed_digest:
        raise CheckpointError("seed digest differs; refusing to resume with another seed")
    if metadata.get("config") != config.immutable:
        raise CheckpointError("checkpoint configuration differs; refusing incompatible resume")
    required = {
        "raw", "m", "v", "step", "rng_state", "best_valid", "best_raw",
        "best_max_inner_product", "metadata_json",
    }
    if set(arrays) != required:
        missing = sorted(required - set(arrays))
        extra = sorted(set(arrays) - required)
        raise CheckpointError(f"checkpoint arrays differ (missing={missing}, extra={extra})")
    shape = (config.batch_size, N_POINTS, DIMENSION)
    for key in ("raw", "m", "v"):
        if arrays[key].shape != shape or arrays[key].dtype != np.float64:
            raise CheckpointError(f"checkpoint {key} has invalid shape or dtype")
        if not np.isfinite(arrays[key]).all():
            raise CheckpointError(f"checkpoint {key} contains non-finite values")
    if arrays["step"].shape != () or arrays["step"].dtype not in (np.int64, np.int32):
        raise CheckpointError("checkpoint step is not a scalar integer")
    if arrays["rng_state"].ndim != 1 or arrays["rng_state"].dtype != np.uint8:
        raise CheckpointError("checkpoint RNG state has invalid shape or dtype")
    if arrays["metadata_json"].ndim != 1 or arrays["metadata_json"].dtype != np.uint8:
        raise CheckpointError("checkpoint metadata_json payload has invalid shape or dtype")
    cursor = metadata.get("cursor")
    if not isinstance(cursor, dict):
        raise CheckpointError("checkpoint cursor is missing")
    def strict_int(value: Any, name: str) -> int:
        if isinstance(value, bool) or not isinstance(value, int):
            raise CheckpointError(f"checkpoint cursor {name} is not an integer")
        return value

    try:
        macro = strict_int(cursor["macro"], "macro")
        stage = strict_int(cursor["stage"], "stage")
        in_stage = strict_int(cursor["in_stage"], "in_stage")
        updates_total = strict_int(cursor["updates_total"], "updates_total")
    except KeyError as exc:
        raise CheckpointError("checkpoint cursor is malformed") from exc
    schedule = normalized_schedule()
    if not (0 <= macro <= config.macro_repeats and 0 <= stage <= len(schedule)):
        raise CheckpointError("checkpoint macro/stage cursor is out of bounds")
    if not (0 <= in_stage <= (schedule[stage][1] if stage < len(schedule) else 0)):
        raise CheckpointError("checkpoint in-stage cursor is out of bounds")
    if updates_total < 0 or updates_total > config.total_updates:
        raise CheckpointError("checkpoint total update cursor is out of bounds")
    expected_step = int(arrays["step"].item())
    if metadata.get("step") != expected_step:
        raise CheckpointError("checkpoint metadata step differs from NPZ step")
    if macro == config.macro_repeats:
        if stage != 0 or in_stage != 0 or updates_total != config.total_updates:
            raise CheckpointError("completed checkpoint has an invalid terminal cursor")
        # The final completed macro retains its last state for output, so its
        # Adam step is the final stage count even though the canonical cursor
        # has advanced to macro_repeats/stage 0.
        if expected_step != FULL_SCHEDULE_UPDATES:
            raise CheckpointError("completed checkpoint has an invalid Adam step")
    else:
        if stage == len(schedule):
            local_step = FULL_SCHEDULE_UPDATES
        else:
            local_step = sum(schedule[i][1] for i in range(stage)) + in_stage
        expected_state_step = macro * FULL_SCHEDULE_UPDATES + local_step
        if updates_total != expected_state_step:
            raise CheckpointError("checkpoint total updates do not match macro/stage cursor")
        if expected_step != local_step:
            raise CheckpointError("checkpoint Adam step does not match stage cursor")
    rng = metadata.get("rng")
    if not isinstance(rng, dict) or rng.get("device") != config.device or rng.get("array") != "rng_state":
        raise CheckpointError("checkpoint RNG metadata is incompatible")
    state = {
        "raw": _require_torch().from_numpy(arrays["raw"]).to(device=config.device, dtype=_torch_dtype()),
        "m": _require_torch().from_numpy(arrays["m"]).to(device=config.device, dtype=_torch_dtype()),
        "v": _require_torch().from_numpy(arrays["v"]).to(device=config.device, dtype=_torch_dtype()),
        "step": expected_step,
    }
    _assert_finite_state(state, "resumed state", shape)
    generator = _restore_generator(config.device, arrays["rng_state"])
    best_valid = arrays["best_valid"]
    if best_valid.shape != () or best_valid.dtype not in (np.int8, np.int64, np.int32) or int(best_valid.item()) not in (0, 1):
        raise CheckpointError("checkpoint best_valid marker is malformed")
    best: np.ndarray | None
    if int(best_valid.item()):
        if arrays["best_raw"].shape != (N_POINTS, DIMENSION) or not np.isfinite(arrays["best_raw"]).all():
            raise CheckpointError("checkpoint best candidate is malformed")
        best = np.ascontiguousarray(arrays["best_raw"], dtype=np.float64)
        if not np.allclose(np.linalg.norm(best, axis=1), 1.0, rtol=0.0, atol=1e-12):
            raise CheckpointError("checkpoint best candidate has non-unit rows")
        best_metric = float(arrays["best_max_inner_product"].item())
        if not math.isfinite(best_metric):
            raise CheckpointError("checkpoint best metric is non-finite")
        recomputed = _metrics(best)["numpy_max_inner_product"]
        if not math.isclose(recomputed, best_metric, rel_tol=0.0, abs_tol=2e-14):
            raise CheckpointError("checkpoint best metric does not match its candidate")
    else:
        best = None
        best_metric = math.inf
    return state, generator, {"macro": macro, "stage": stage, "in_stage": in_stage, "updates_total": updates_total}, best, best_metric


def _atomic_text(path: Path, text: str) -> None:
    _atomic_replace_bytes(path, text.encode())


def _write_candidate(path: Path, candidate: np.ndarray, metadata: dict[str, Any]) -> dict[str, Any]:
    path.parent.mkdir(parents=True, exist_ok=True)
    # Use a temporary text file in the destination directory so a killed run
    # cannot leave a half-written candidate that looks complete.
    fd, name = tempfile.mkstemp(prefix=f".{path.name}.", dir=str(path.parent), text=True)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as stream:
            np.savetxt(stream, candidate, fmt="%.17g")
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(name, path)
    except Exception:
        try:
            os.unlink(name)
        except OSError:
            pass
        raise
    # Re-open the exact text that was published and independently normalise
    # and score it.  The sidecar reports this readback, rather than the in
    # memory tensor that happened to be passed to np.savetxt.
    try:
        written = np.loadtxt(path, dtype=np.float64)
    except Exception as exc:
        raise RunnerError(f"could not read written candidate {path}: {exc}") from exc
    if written.shape != (N_POINTS, DIMENSION) or not np.isfinite(written).all():
        raise RunnerError("written candidate failed shape or finite-value validation")
    norms = np.linalg.norm(written, axis=1)
    if not np.isfinite(norms).all() or np.any(norms <= 1.0e-15):
        raise RunnerError("written candidate contains an invalid row")
    written_normalized = written / norms[:, None]
    readback = _metrics(written_normalized)
    published = dict(metadata)
    published["numpy_verification"] = readback
    published["written_readback"] = True
    sidecar = path.with_suffix(".json")
    _atomic_text(sidecar, json.dumps(_json_safe(published), indent=2, sort_keys=True) + "\n")
    return published


def _best_from_batch(state: Mapping[str, Any]) -> tuple[np.ndarray, float, list[dict[str, Any]]]:
    raw_np = _to_numpy(state["raw"], "raw")
    if raw_np.ndim != 3 or raw_np.shape[1:] != (N_POINTS, DIMENSION):
        raise RunnerError("kernel raw state has an invalid batch shape")
    candidates: list[np.ndarray] = []
    batch_metrics: list[dict[str, Any]] = []
    for row in raw_np:
        candidate = _normalised_numpy(row)
        metrics = _metrics(candidate)
        candidates.append(candidate)
        batch_metrics.append(metrics)
    idx = int(np.argmin([m["numpy_max_inner_product"] for m in batch_metrics]))
    return candidates[idx], float(batch_metrics[idx]["numpy_max_inner_product"]), batch_metrics


def run_search(config: RunConfig) -> dict[str, Any]:
    """Run a bounded or complete batched search and return output metadata.

    The return value is JSON-serializable.  If ``config.output`` is provided,
    the selected normalized candidate and a sidecar JSON are written.
    """
    _validate_config(config)
    t = _require_torch()
    if config.output is not None and not config.resume:
        output_sidecar = config.output.with_suffix(".json")
        if config.output.exists() or output_sidecar.exists():
            raise RunnerError(
                f"output already exists: {config.output}; choose a new path or pass --resume"
            )
    if config.checkpoint is not None and not config.resume:
        checkpoint_npz, checkpoint_json = _checkpoint_paths(config.checkpoint)
        if checkpoint_npz.exists() or checkpoint_json.exists():
            raise RunnerError(
                f"checkpoint already exists: {checkpoint_npz}; choose a new path or pass --resume"
            )
    if config.device == "cpu":
        t.set_num_threads(config.threads)
    core, seed_digest, seed_source = _load_core(config.seed_file)
    schedule = normalized_schedule()
    requested_updates = config.max_updates
    if requested_updates is None:
        # Programmatic callers can request a full schedule explicitly by
        # leaving the stopping bound unset; the CLI requires --full or a bound.
        requested_updates = config.total_updates
    if config.resume:
        state, generator, cursor, best, best_metric = _read_checkpoint(
            config.checkpoint, config, seed_digest  # type: ignore[arg-type]
        )
        macro = cursor["macro"]
        stage = cursor["stage"]
        in_stage = cursor["in_stage"]
        updates_total = cursor["updates_total"]
        if requested_updates < updates_total:
            raise CheckpointError(
                f"--max-updates={requested_updates} is below checkpoint progress "
                f"{updates_total}; refusing to rewind"
            )
    else:
        generator = _make_generator(config.device, config.seed)
        raw = _new_batch(core, config.batch_size, config.device, generator)
        state = _new_state(raw)
        expected_shape = (config.batch_size, N_POINTS, DIMENSION)
        _assert_finite_state(state, "initial state", expected_shape)
        macro, stage, in_stage, updates_total = 0, 0, 0, 0
        best = None
        best_metric = math.inf
    expected_shape = (config.batch_size, N_POINTS, DIMENSION)
    last_batch_metrics: list[dict[str, Any]] = []
    current_batch_metrics: list[dict[str, Any]] = []
    while (updates_total < requested_updates or stage >= len(schedule)) and macro < config.macro_repeats:
        if stage >= len(schedule):
            # Accept an intermediate stage-13 cursor (for example, a process
            # stopped after writing that boundary) and canonicalise it before
            # continuing.  Normal writes below already use the canonical form.
            macro += 1
            stage = 0
            in_stage = 0
            if macro >= config.macro_repeats:
                break
            raw = _new_batch(core, config.batch_size, config.device, generator)
            state = _new_state(raw)
            _assert_finite_state(state, "new macro state", expected_shape)
            if config.checkpoint is not None:
                _write_checkpoint(
                    config.checkpoint,
                    config=config,
                    seed_digest=seed_digest,
                    seed_source=seed_source,
                    state=state,
                    generator=generator,
                    macro=macro,
                    stage=stage,
                    in_stage=in_stage,
                    updates_total=updates_total,
                    best=best,
                    best_metric=best_metric,
                )
            continue
        s, stage_steps, lr = schedule[stage]
        remaining_bound = requested_updates - updates_total
        remaining_stage = stage_steps - in_stage
        number = min(remaining_bound, remaining_stage)
        for _ in range(number):
            state = _step(state, s, lr, expected_shape)
            updates_total += 1
            in_stage += 1
            if config.checkpoint is not None and config.checkpoint_every > 0 and updates_total % config.checkpoint_every == 0:
                _write_checkpoint(
                    config.checkpoint,
                    config=config,
                    seed_digest=seed_digest,
                    seed_source=seed_source,
                    state=state,
                    generator=generator,
                    macro=macro,
                    stage=stage,
                    in_stage=in_stage,
                    updates_total=updates_total,
                    best=best,
                    best_metric=best_metric,
                )
        if in_stage == stage_steps:
            stage += 1
            in_stage = 0
            if stage == len(schedule):
                candidate, metric, last_batch_metrics = _best_from_batch(state)
                if metric < best_metric:
                    best, best_metric = candidate.copy(), metric
                # A macro always owns a fresh raw batch and fresh moments.  Do
                # this at the boundary even when max_updates stops exactly
                # here, so a later resume has the same RNG consumption as an
                # uninterrupted run with a larger stopping bound.
                macro += 1
                stage = 0
                in_stage = 0
                if macro < config.macro_repeats:
                    raw = _new_batch(core, config.batch_size, config.device, generator)
                    state = _new_state(raw)
                    _assert_finite_state(state, "new macro state", expected_shape)
            if config.checkpoint is not None:
                _write_checkpoint(
                    config.checkpoint,
                    config=config,
                    seed_digest=seed_digest,
                    seed_source=seed_source,
                    state=state,
                    generator=generator,
                    macro=macro,
                    stage=stage,
                    in_stage=in_stage,
                    updates_total=updates_total,
                    best=best,
                    best_metric=best_metric,
                )

    # A bounded run may stop in the middle of a macro; retain its best current
    # candidate for a useful diagnostic while keeping the checkpoint cursor.
    if macro >= config.macro_repeats and best is not None:
        candidate = best.copy()
        metric = best_metric
    else:
        current, current_metric, current_batch_metrics = _best_from_batch(state)
        if best is None or current_metric < best_metric:
            candidate, metric = current, current_metric
        else:
            candidate, metric = best.copy(), best_metric
    completed = updates_total == config.total_updates and macro >= config.macro_repeats
    full_schedule = completed
    metadata: dict[str, Any] = {
        "kind": "batched841_candidate",
        "schema_version": SCHEMA_VERSION,
        "status": "full" if full_schedule else "diagnostic",
        "diagnostic": not full_schedule,
        "full_schedule": full_schedule,
        "updates": updates_total,
        "requested_updates": requested_updates,
        "macro_repeats": config.macro_repeats,
        "batch_size": config.batch_size,
        "seed": config.seed,
        "seed_digest": seed_digest,
        "seed_source": seed_source,
        "device": config.device,
        "dtype": config.dtype,
        "cursor": {"macro": macro, "stage": stage, "in_stage": in_stage},
        "extra": "one independent uniform hypercube row per candidate; 840 seed rows fixed",
        "numpy_verification": _metrics(candidate),
        "claim": "numeric search candidate only; not an exact certificate or a record claim",
        "handoff": "For polish, use prepare_841_polish.py decimal Gram handoff first; this runner does not polish.",
    }
    if last_batch_metrics:
        metadata["batch_metrics_at_last_completed_macro"] = last_batch_metrics
    if current_batch_metrics and not last_batch_metrics:
        metadata["batch_metrics_at_current_cursor"] = current_batch_metrics
    if config.output is not None:
        metadata = _write_candidate(config.output, candidate, metadata)
    if config.checkpoint is not None:
        # Publish the stopping point even when max_updates=0 or when no stage
        # boundary was crossed; resume then starts from exactly this state.
        _write_checkpoint(
            config.checkpoint,
            config=config,
            seed_digest=seed_digest,
            seed_source=seed_source,
            state=state,
            generator=generator,
            macro=macro,
            stage=stage,
            in_stage=in_stage,
            updates_total=updates_total,
            best=best,
            best_metric=best_metric,
        )
    return metadata


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed-file", type=Path, help="seed text file; first 840 rows are fixed")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--macro-repeats", type=int, default=1)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    parser.add_argument("--dtype", choices=("float64",), default="float64")
    parser.add_argument("--max-updates", type=int, help="bounded diagnostic update count")
    parser.add_argument("--full", action="store_true", help="run the complete 35,000-update schedule")
    parser.add_argument("--checkpoint", type=Path, help="checkpoint prefix or .npz path")
    parser.add_argument("--resume", action="store_true", help="resume the checkpoint named by --checkpoint")
    parser.add_argument("--checkpoint-every", type=int, default=0, help="also checkpoint every N updates")
    parser.add_argument("--output", type=Path, help="normalized candidate text output")
    parser.add_argument("--threads", type=int, default=1)
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = _build_parser()
    args = parser.parse_args(argv)
    if args.full and args.max_updates is not None:
        parser.error("choose --full or --max-updates, not both")
    if not args.full and args.max_updates is None:
        parser.error("refusing an unbounded large job: pass --max-updates for a diagnostic or --full")
    if args.resume and args.checkpoint is None:
        parser.error("--resume requires --checkpoint")
    max_updates = FULL_SCHEDULE_UPDATES * args.macro_repeats if args.full else args.max_updates
    config = RunConfig(
        seed=args.seed,
        seed_file=args.seed_file,
        batch_size=args.batch_size,
        macro_repeats=args.macro_repeats,
        device=args.device,
        dtype=args.dtype,
        max_updates=max_updates,
        checkpoint=args.checkpoint,
        output=args.output,
        resume=args.resume,
        checkpoint_every=args.checkpoint_every,
        threads=args.threads,
    )
    try:
        metadata = run_search(config)
    except RunnerError as exc:
        parser.error(str(exc))
    print(json.dumps(metadata, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
