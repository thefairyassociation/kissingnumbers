#!/usr/bin/env python3
"""Parity and bounded CPU checks for the batched N=841 Adam kernel.

The test intentionally imports the *new* kernel from a path supplied on the
command line (or the repository's ``kernel.py``), then compares it with a
temporary executable that includes the committed ``kissing/lib/riesz.c``
source.  Including the C source gives this test access to the existing static
Riesz oracle without changing that source or committing a generated binary.

The default probes are tiny (N=7) so they are suitable for a regression check.
They still use three separated schedule stages, including ``s=40000``, and
carry Adam moments and the bias-correction step across stage boundaries.  The
N=841 benchmark is opt-in and intentionally bounded; it is a CPU timing probe,
not a search run or a recovery of the published result.

The C oracle uses the source's faithful convention: pair terms are counted
once (i < j), all terms are retained, the normalized-view chain rule is applied
to raw coordinates, and the squared-distance floor is ``1e-12``.  At the floor
the C code retains the derivative of the clamped value.  ``--clamp-probe``
exercises that edge explicitly, because ``torch.clamp_min`` alone has a zero
derivative below the floor and would silently create a source-level mismatch.
"""

from __future__ import annotations

import argparse
import importlib.util
import os
import subprocess
import sys
import tempfile
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

import numpy as np


ROOT = Path(__file__).resolve().parents[3]
ORACLE_SOURCE = ROOT / "kissing" / "experimental" / "batched841" / "oracle_raw_adam.c"
C_SOURCE = ROOT / "kissing" / "lib" / "riesz.c"


EXPECTED_SCHEDULE = (
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


@dataclass
class OracleRecord:
    global_step: int
    stage: int
    iteration: int
    s: float
    loss: float
    max_ip: float
    grad: np.ndarray
    raw: np.ndarray
    m: np.ndarray
    v: np.ndarray


@dataclass
class PythonRecord:
    global_step: int
    stage: int
    iteration: int
    s: float
    loss: float
    grad: np.ndarray
    raw: np.ndarray
    m: np.ndarray
    v: np.ndarray


def _find_kernel(explicit: str | None) -> Path:
    if explicit:
        path = Path(explicit).expanduser().resolve()
        if not path.exists():
            raise SystemExit(f"kernel module does not exist: {path}")
        return path
    d = ROOT / "kissing" / "experimental" / "batched841"
    candidates = [d / "kernel.py"]
    for path in candidates:
        if path.exists():
            return path.resolve()
    searched = "\n  ".join(str(p) for p in candidates)
    raise SystemExit(
        "could not find the new batched kernel; pass --kernel PATH. Searched:\n  "
        + searched
    )


def _load_kernel(path: Path) -> Any:
    # The experimental module may import sibling helper modules, so expose its
    # directory on sys.path exactly as a normal script invocation would.
    sys.path.insert(0, str(path.parent))
    spec = importlib.util.spec_from_file_location("luna_batched841_kernel", path)
    if spec is None or spec.loader is None:
        raise SystemExit(f"cannot import kernel module: {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    try:
        spec.loader.exec_module(module)
    except Exception as exc:  # pragma: no cover - error context is the point
        raise SystemExit(f"failed importing kernel {path}: {type(exc).__name__}: {exc}") from exc
    return module


def _load_runner(path: Path, kernel_module: Any) -> Any:
    """Load an optional runner and force it to use the kernel under test."""
    sys.path.insert(0, str(path.parent))
    spec = importlib.util.spec_from_file_location("luna_batched841_runner", path)
    if spec is None or spec.loader is None:
        raise SystemExit(f"cannot import runner module: {path}")
    runner = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = runner
    try:
        spec.loader.exec_module(runner)
    except Exception as exc:  # pragma: no cover - error context is the point
        raise SystemExit(f"failed importing runner {path}: {type(exc).__name__}: {exc}") from exc
    runner.kernel = kernel_module
    return runner


def _require_torch() -> Any:
    try:
        import torch
    except Exception as exc:  # pragma: no cover - exercised on minimal installs
        raise SystemExit(
            "PyTorch is required for this parity harness; install the CPU wheel "
            "or pass through the project's runtime"
        ) from exc
    torch.set_num_threads(1)
    if hasattr(torch, "set_num_interop_threads"):
        try:
            torch.set_num_interop_threads(1)
        except RuntimeError:
            pass
    return torch


def _as_numpy(value: Any) -> np.ndarray:
    if hasattr(value, "detach"):
        value = value.detach().cpu().numpy()
    return np.asarray(value)


def _as_scalar(value: Any) -> float:
    array = _as_numpy(value)
    if array.size != 1:
        raise AssertionError(f"expected scalar, got shape {array.shape}")
    result = float(array.reshape(-1)[0])
    if not np.isfinite(result):
        raise AssertionError(f"non-finite scalar {result}")
    return result


def _tensor_like(torch: Any, value: Any, *, requires_grad: bool = False, batch: bool = False) -> Any:
    if isinstance(value, torch.Tensor):
        result = value.detach().clone().to(dtype=torch.float64, device="cpu")
    else:
        result = torch.as_tensor(np.asarray(value), dtype=torch.float64, device="cpu").clone()
    if batch and result.ndim == 2:
        result = result.unsqueeze(0)
    return result.requires_grad_(requires_grad)


def normalized_view(module: Any, raw: Any, torch: Any) -> Any:
    fn = getattr(module, "normalized_view", None)
    if fn is None:
        raise AssertionError("kernel API is missing normalized_view(raw)")
    value = _tensor_like(torch, raw, batch=True)
    result = fn(value)
    if not hasattr(result, "shape") or tuple(result.shape) != tuple(value.shape):
        raise AssertionError(f"normalized_view shape {getattr(result, 'shape', None)}, expected {tuple(value.shape)}")
    return result


def loss_and_grad(module: Any, raw: Any, s: float, torch: Any) -> tuple[float, Any]:
    fn = getattr(module, "loss_and_grad", None)
    if fn is None:
        raise AssertionError("kernel API is missing loss_and_grad(raw, s)")
    value = _tensor_like(torch, raw, requires_grad=True, batch=True)
    result = fn(value, float(s))
    if not isinstance(result, (tuple, list)) or len(result) != 2:
        raise AssertionError("loss_and_grad must return (loss, grad) or a dict with loss/grad")
    loss, grad = result
    loss_value = _as_scalar(loss)
    grad_value = _tensor_like(torch, grad)
    if tuple(grad_value.shape) != tuple(value.shape):
        raise AssertionError(f"gradient shape {tuple(grad_value.shape)}, expected {tuple(value.shape)}")
    if not bool(torch.isfinite(grad_value).all()):
        raise AssertionError("loss_and_grad returned a non-finite gradient")
    return loss_value, grad_value


def initial_state(module: Any, raw: Any, torch: Any) -> dict[str, Any]:
    fn = getattr(module, "initial_state", None)
    if fn is None:
        raise AssertionError("kernel API is missing initial_state(raw)")
    value = _tensor_like(torch, raw, batch=True)
    state = fn(value)
    if not isinstance(state, dict):
        raise AssertionError(f"initial_state must return a dict, got {type(state).__name__}")
    missing = {"raw", "m", "v", "step"} - set(state)
    if missing:
        raise AssertionError(f"initial_state missing keys: {sorted(missing)}")
    return state


def _step_call(module: Any, state: dict[str, Any], s: float, lr: float) -> dict[str, Any]:
    """Apply the exact public ``adam_step(state, s, lr)`` API in place."""
    fn = getattr(module, "adam_step", None)
    if fn is None:
        raise AssertionError("kernel API is missing adam_step(state, s, lr)")
    before_raw = state["raw"].detach().clone()
    before_m = state["m"].detach().clone()
    before_v = state["v"].detach().clone()
    before_step = state["step"]
    result = fn(state, float(s), float(lr))
    if result is not None and not isinstance(result, dict):
        raise AssertionError(f"adam_step returned {type(result).__name__}, expected state dict or None")
    if not isinstance(result, dict):
        result = state
    if result is not state:
        raise AssertionError("adam_step must mutate and return the input state")
    if (
        torch_equal(before_raw, state["raw"])
        and torch_equal(before_m, state["m"])
        and torch_equal(before_v, state["v"])
        and _as_scalar(before_step) == _as_scalar(state["step"])
    ):
        raise AssertionError("adam_step did not mutate its state")
    return state


def torch_equal(left: Any, right: Any) -> bool:
    try:
        import torch
        return bool(torch.equal(left, right))
    except Exception:
        return bool(np.array_equal(_as_numpy(left), _as_numpy(right)))


def _state_array(state: dict[str, Any], key: str) -> np.ndarray:
    if key not in state:
        raise AssertionError(f"state lost required key {key!r}")
    return _as_numpy(state[key]).astype(np.float64, copy=True)


def _state_step(state: dict[str, Any]) -> int:
    value = _as_scalar(state["step"])
    if value != int(value):
        raise AssertionError(f"Adam step is not integral: {value}")
    return int(value)


def _validate_schedule(module: Any) -> list[tuple[float, int, float]]:
    schedule = getattr(module, "SEARCH_SCHEDULE", None)
    if schedule is None:
        raise AssertionError("kernel API is missing SEARCH_SCHEDULE")
    parsed: list[tuple[float, int, float]] = []
    for index, item in enumerate(schedule):
        if not isinstance(item, (tuple, list)) or len(item) != 3:
            raise AssertionError(f"SEARCH_SCHEDULE[{index}] is not an (s, steps, lr) tuple")
        s, steps, lr = float(item[0]), int(item[1]), float(item[2])
        if not (np.isfinite(s) and s > 0 and steps > 0 and np.isfinite(lr) and lr > 0):
            raise AssertionError(f"invalid SEARCH_SCHEDULE[{index}]={item!r}")
        parsed.append((s, steps, lr))
    if tuple(parsed) != EXPECTED_SCHEDULE:
        raise AssertionError(
            "published SEARCH_SCHEDULE changed:\n"
            f"  got      {tuple(parsed)!r}\n"
            f"  expected {EXPECTED_SCHEDULE!r}"
        )
    return parsed


def _python_trace(module: Any, raw_np: np.ndarray, schedule: Iterable[tuple[float, int, float]], torch: Any) -> tuple[list[PythonRecord], dict[str, Any]]:
    state = initial_state(module, raw_np, torch)
    raw_batch = raw_np if raw_np.ndim == 3 else raw_np[None, ...]
    if not torch_equal(_state_array(state, "m"), np.zeros_like(raw_batch)):
        raise AssertionError("initial_state m is not zero")
    if not torch_equal(_state_array(state, "v"), np.zeros_like(raw_batch)):
        raise AssertionError("initial_state v is not zero")
    if _state_step(state) != 0:
        raise AssertionError("initial_state step is not zero")
    records: list[PythonRecord] = []
    for stage, (s, steps, lr) in enumerate(schedule):
        for iteration in range(steps):
            loss, grad = loss_and_grad(module, state["raw"], s, torch)
            grad_copy = grad.detach().clone()
            _step_call(module, state, s, lr)
            step = _state_step(state)
            if step != len(records) + 1:
                raise AssertionError(f"Adam step did not advance globally: {step}")
            records.append(
                PythonRecord(
                    global_step=step,
                    stage=stage,
                    iteration=iteration,
                    s=s,
                    loss=loss,
                    grad=_as_numpy(grad_copy),
                    raw=_state_array(state, "raw"),
                    m=_state_array(state, "m"),
                    v=_state_array(state, "v"),
                )
            )
    return records, state


def _parse_values(line: str, label: str) -> np.ndarray:
    prefix = label + " "
    if not line.startswith(prefix):
        raise AssertionError(f"expected {label!r} line, got {line[:80]!r}")
    values = np.fromstring(line[len(prefix) :], sep=" ", dtype=np.float64)
    if values.size == 0:
        raise AssertionError(f"empty {label} vector")
    return values


def _parse_oracle(path: Path, count: int) -> list[OracleRecord]:
    lines = path.read_text().splitlines()
    records: list[OracleRecord] = []
    index = 1  # header
    while index < len(lines):
        if lines[index].startswith("stage "):
            index += 1
            continue
        fields = lines[index].split()
        if len(fields) != 7 or fields[0] != "record":
            raise AssertionError(f"malformed oracle record line: {lines[index]!r}")
        record = dict(
            global_step=int(fields[1]),
            stage=int(fields[2]),
            iteration=int(fields[3]),
            s=float(fields[4]),
            loss=float(fields[5]),
            max_ip=float(fields[6]),
        )
        vectors: dict[str, np.ndarray] = {}
        for label in ("grad", "raw", "m", "v"):
            index += 1
            values = _parse_values(lines[index], label)
            if values.size != count:
                raise AssertionError(f"oracle {label} has {values.size} values, expected {count}")
            vectors[label] = values
        records.append(OracleRecord(**record, **vectors))
        index += 1
    return records


def _scipy_blas_flags() -> list[str]:
    """Return compiler/linker flags used by kissing/lib/Makefile."""
    try:
        import scipy_openblas32 as scipy_openblas
    except Exception:
        # A system OpenBLAS installation is enough for the helper.  The common
        # fallback is scipy-openblas32, handled above when available.
        return ["-fopenmp", "-lopenblas", "-lm"]
    include = scipy_openblas.get_include_dir()
    lib = scipy_openblas.get_lib_dir()
    return [
        "-fopenmp",
        f"-I{include}",
        "-include",
        "cblas.h",
        "-Dcblas_dgemm=scipy_cblas_dgemm",
        "-Dcblas_dscal=scipy_cblas_dscal",
        "-Dcblas_ddot=scipy_cblas_ddot",
        "-Dcblas_daxpy=scipy_cblas_daxpy",
        "-Dcblas_dcopy=scipy_cblas_dcopy",
        "-Dcblas_dnrm2=scipy_cblas_dnrm2",
        "-Dopenblas_set_num_threads=scipy_openblas_set_num_threads",
        f"-L{lib}",
        f"-Wl,-rpath,{lib}",
        "-lscipy_openblas",
        "-lm",
    ]


def _compile_oracle(directory: Path, source: Path = ORACLE_SOURCE) -> Path:
    if not source.exists():
        raise AssertionError(f"oracle harness source does not exist: {source}")
    binary = directory / "oracle_raw_adam"
    # Keep the source before the BLAS/math libraries: GNU ld resolves symbols
    # left-to-right, and putting -lscipy_openblas before this translation unit
    # produces a misleading set of undefined cblas/sqrt references.
    command = ["gcc", "-std=c11", "-O2", "-o", str(binary), str(source), *_scipy_blas_flags()]
    result = subprocess.run(command, text=True, capture_output=True, check=False)
    if result.returncode:
        raise AssertionError(
            "could not compile C oracle:\n"
            + " ".join(command)
            + "\n"
            + result.stderr
        )
    return binary


def _run_oracle(binary: Path, raw: np.ndarray, schedule: list[tuple[float, int, float]], directory: Path, eps: float = 1e-8) -> list[OracleRecord]:
    raw_path = directory / "raw.bin"
    schedule_path = directory / "schedule.txt"
    output_path = directory / "oracle.txt"
    raw.astype(np.float64, copy=False).tofile(raw_path)
    schedule_path.write_text("\n".join(f"{s:.17g} {steps} {lr:.17g}" for s, steps, lr in schedule) + "\n")
    env = os.environ.copy()
    env.update({"OMP_NUM_THREADS": "1", "OPENBLAS_NUM_THREADS": "1"})
    result = subprocess.run(
        [str(binary), str(raw.shape[-1]), str(raw.shape[-2]), str(raw_path), str(schedule_path), str(output_path), repr(eps)],
        text=True,
        capture_output=True,
        env=env,
        check=False,
    )
    if result.returncode:
        raise AssertionError(f"C oracle failed ({result.returncode}):\n{result.stderr}")
    return _parse_oracle(output_path, int(raw.size))


def _assert_close(name: str, expected: Any, actual: Any, *, rtol: float = 2e-10, atol: float = 2e-11) -> None:
    left = np.asarray(expected, dtype=np.float64)
    right = np.asarray(actual, dtype=np.float64)
    if left.shape != right.shape:
        if left.size == right.size:
            left = left.reshape(-1)
            right = right.reshape(-1)
        else:
            raise AssertionError(f"{name}: shape {left.shape} != {right.shape}")
    if not np.isfinite(left).all() or not np.isfinite(right).all():
        raise AssertionError(f"{name}: non-finite value in comparison")
    diff = np.abs(left - right)
    scale = np.maximum(np.abs(left), np.abs(right))
    allowed = atol + rtol * scale
    if np.any(diff > allowed):
        where = np.unravel_index(int(np.argmax(diff - allowed)), diff.shape)
        raise AssertionError(
            f"{name}: max abs diff={float(diff.max()):.3g} at {where}; "
            f"expected={left[where]:.17g} actual={right[where]:.17g} "
            f"tol={float(allowed[where]):.3g}"
        )


def _compare_traces(oracle: list[OracleRecord], python_records: list[PythonRecord], *, label: str) -> None:
    if len(oracle) != len(python_records):
        raise AssertionError(f"{label}: C emitted {len(oracle)} records, Python emitted {len(python_records)}")
    for c, p in zip(oracle, python_records):
        if (c.global_step, c.stage, c.iteration) != (p.global_step, p.stage, p.iteration):
            raise AssertionError(f"{label}: step identity mismatch C={c} Python={p}")
        _assert_close(f"{label} step {c.global_step} loss", c.loss, p.loss, rtol=5e-11, atol=5e-11)
        _assert_close(f"{label} step {c.global_step} gradient", c.grad, p.grad)
        _assert_close(f"{label} step {c.global_step} raw", c.raw, p.raw)
        _assert_close(f"{label} step {c.global_step} m", c.m, p.m)
        _assert_close(f"{label} step {c.global_step} v", c.v, p.v)


def _expect_rejected(call: Any, label: str) -> None:
    try:
        call()
    except (AssertionError, FloatingPointError, RuntimeError, ValueError, TypeError, OverflowError):
        return
    raise AssertionError(f"invalid state was accepted: {label}")


def run_b1_parity(module: Any, torch: Any, binary: Path, directory: Path, *, clamp_probe: bool = True) -> None:
    rng = np.random.default_rng(841001)
    raw = rng.normal(size=(7, 5)).astype(np.float64)
    schedule_all = _validate_schedule(module)
    indices = (0, len(schedule_all) // 2, len(schedule_all) - 1)
    schedule = [(schedule_all[i][0], 2, schedule_all[i][2]) for i in indices]
    (directory / "b1").mkdir(exist_ok=True)
    oracle = _run_oracle(binary, raw, schedule, directory / "b1")
    python_records, _state = _python_trace(module, raw, schedule, torch)
    _compare_traces(oracle, python_records, label="B1 raw Adam")
    print(f"B1 parity: {len(python_records)} raw Adam updates across s={[s for s, _, _ in schedule]}")

    if clamp_probe:
        # Rows 0 and 1 are distinct but their normalized squared distance is
        # below C's 1e-12 floor.  This catches the common torch.clamp_min
        # derivative mismatch while leaving the state valid (no zero row).
        near = np.array(
            [[1.0, 0.0, 0.0, 0.0], [1.0, 1e-8, 0.0, 0.0], [0.0, 0.0, 1.0, 0.0]],
            dtype=np.float64,
        )
        clamp_schedule = [(8.0, 1, 1e-3)]
        clamp_dir = directory / "clamp"
        clamp_dir.mkdir(exist_ok=True)
        ctrace = _run_oracle(binary, near, clamp_schedule, clamp_dir)
        ptrace, _ = _python_trace(module, near, clamp_schedule, torch)
        _compare_traces(ctrace, ptrace, label="C floor derivative probe")
        print("C floor derivative probe: passed (clamped-distance gradient conventions match)")


def run_n841_probe(module: Any, torch: Any, binary: Path, directory: Path) -> None:
    """Compare two real N=841 updates through the unchanged C Adam stage."""
    fixture = ROOT / "kissing" / "lib" / "testdata" / "authors_841_coordinates.txt"
    if not fixture.exists():
        raise AssertionError(f"N=841 parity fixture is missing: {fixture}")
    raw = np.loadtxt(fixture, dtype=np.float64)
    if raw.shape != (841, 12) or not np.isfinite(raw).all():
        raise AssertionError(f"N=841 parity fixture has invalid shape or values: {raw.shape}")
    schedule = [(8.0, 2, 0.005)]
    probe_dir = directory / "n841"
    probe_dir.mkdir(exist_ok=True)
    oracle = _run_oracle(binary, raw, schedule, probe_dir)
    python_records, _ = _python_trace(module, raw, schedule, torch)
    _compare_traces(oracle, python_records, label="N=841 B1 raw Adam")
    print("N=841 B1 parity: 2 updates through unchanged C adam_raw_stage passed")


def run_runner_crosscheck(module: Any, torch: Any, runner: Any) -> None:
    """Cross-check runner schedule and normalized coordinates with the kernel."""
    schedule = tuple(_validate_schedule(module))
    if tuple(runner.normalized_schedule()) != schedule:
        raise AssertionError("runner normalized_schedule differs from kernel SEARCH_SCHEDULE")
    fixture = ROOT / "kissing" / "lib" / "testdata" / "authors_841_coordinates.txt"
    raw = np.loadtxt(fixture, dtype=np.float64)
    state = initial_state(module, raw, torch)
    view = normalized_view(module, state["raw"], torch)
    runner_view = runner._normalised_numpy(state["raw"])
    _assert_close("runner normalized candidate", _as_numpy(view[0]), runner_view, rtol=1e-12, atol=1e-12)
    metrics = runner._metrics(runner_view)
    if metrics.get("numpy_finite") is not True or metrics.get("numpy_rows") != 841:
        raise AssertionError(f"runner metrics rejected kernel coordinates: {metrics}")
    print("runner cross-check: schedule and normalized N=841 coordinates match kernel")


def run_b2_batch(module: Any, torch: Any, binary: Path, directory: Path) -> None:
    rng = np.random.default_rng(841002)
    raw = rng.normal(size=(2, 7, 5)).astype(np.float64)
    s, steps, lr = 8.0, 1, 1e-3
    schedule = [(s, steps, lr)]
    b1_oracles = []
    for index in range(2):
        one_dir = directory / f"b2_c{index}"
        one_dir.mkdir(exist_ok=True)
        b1_oracles.append(_run_oracle(binary, raw[index], schedule, one_dir)[0])

    # The C oracle is B=1, so this also verifies the individual Python losses
    # and gradients before checking the batch reduction.
    b1_losses = []
    b1_grads = []
    for index in range(2):
        loss, grad = loss_and_grad(module, raw[index], s, torch)
        b1_losses.append(loss)
        b1_grads.append(_as_numpy(grad))
        _assert_close(f"B2 candidate {index} loss vs C", b1_oracles[index].loss, loss)
        _assert_close(f"B2 candidate {index} gradient vs C", b1_oracles[index].grad, grad.detach().cpu().numpy())

    batch_loss, batch_grad = loss_and_grad(module, raw, s, torch)
    batch_loss_np = _as_numpy(batch_loss)
    if batch_loss_np.size != 1:
        # A per-candidate loss is useful diagnostically, but the public
        # contract promises a batch mean; accepting it here would hide a
        # scaling bug in Adam.
        raise AssertionError(f"B>1 loss is not a scalar batch mean: shape {batch_loss_np.shape}")
    _assert_close("B2 true mean loss", np.mean(b1_losses), float(batch_loss_np.reshape(-1)[0]), rtol=1e-10, atol=1e-11)
    expected_batch_grad = np.concatenate(b1_grads, axis=0) / 2.0
    _assert_close("B2 true mean gradient", expected_batch_grad, _as_numpy(batch_grad), rtol=2e-10, atol=2e-11)

    # Explicitly test the published epsilon denominator and the fact that
    # Adam is raw: at step 1, bias correction makes m_hat=g and v_hat=g^2.
    eps = 1.0e-8
    state = initial_state(module, raw, torch)
    _step_call(module, state, s, lr)
    expected_raw = raw - lr * _as_numpy(batch_grad) / (np.abs(_as_numpy(batch_grad)) + eps)
    _assert_close("B2 epsilon raw update", expected_raw, _state_array(state, "raw"), rtol=3e-10, atol=3e-11)
    _assert_close("B2 epsilon first moment", 0.1 * _as_numpy(batch_grad), _state_array(state, "m"), rtol=3e-10, atol=3e-11)
    _assert_close("B2 epsilon second moment", 0.001 * _as_numpy(batch_grad) ** 2, _state_array(state, "v"), rtol=3e-10, atol=3e-11)
    if _state_step(state) != 1:
        raise AssertionError("B2 epsilon Adam step did not advance to one")
    print("B2 parity: true mean loss/gradient and explicit epsilon denominator passed")


def run_invalid_states(module: Any, torch: Any) -> None:
    raw = np.ones((4, 5), dtype=np.float64)
    raw[0] = 0
    _expect_rejected(lambda: normalized_view(module, raw, torch), "zero row in normalized_view")
    _expect_rejected(lambda: initial_state(module, raw, torch), "zero row in initial_state")
    nan_raw = np.ones((4, 5), dtype=np.float64)
    nan_raw[1, 2] = np.nan
    _expect_rejected(lambda: normalized_view(module, nan_raw, torch), "NaN row in normalized_view")
    _expect_rejected(lambda: initial_state(module, nan_raw, torch), "NaN row in initial_state")

    good = np.random.default_rng(841003).normal(size=(4, 5))
    state = initial_state(module, good, torch)
    state["m"][0, 0, 0] = float("nan")
    _expect_rejected(
        lambda: _step_call(module, state, 8.0, 1e-3),
        "NaN Adam moment",
    )
    print("invalid states: zero rows, non-finite input, and non-finite moments rejected")


def _benchmark_once(module: Any, torch: Any, raw: np.ndarray, batch_size: int, steps: int) -> tuple[float, float]:
    value = np.repeat(raw[None, ...], batch_size, axis=0)
    if batch_size == 1:
        value = raw
    state = initial_state(module, value, torch)
    start = time.perf_counter()
    for _ in range(steps):
        _step_call(module, state, 8.0, 1e-5)
    elapsed = time.perf_counter() - start
    loss, _ = loss_and_grad(module, state["raw"], 8.0, torch)
    return elapsed, float(loss)


def run_benchmark(module: Any, torch: Any, *, steps: int = 2) -> None:
    if steps < 1 or steps > 20:
        raise SystemExit("--benchmark-steps must be between 1 and 20")
    fixture = ROOT / "kissing" / "lib" / "testdata" / "authors_841_coordinates.txt"
    if fixture.exists():
        raw = np.loadtxt(fixture, dtype=np.float64)
    else:
        raw = np.random.default_rng(841004).normal(size=(841, 12))
    if raw.shape != (841, 12):
        raise AssertionError(f"benchmark state is {raw.shape}, expected (841, 12)")
    print(
        f"CPU benchmark: N=841 d=12; {steps} Adam updates at s=8; "
        f"torch={torch.__version__}; device=cpu; GPU available={bool(torch.cuda.is_available())}"
    )
    for batch_size in (1, 2):
        elapsed, loss = _benchmark_once(module, torch, raw, batch_size, steps)
        print(
            f"  B={batch_size}: elapsed={elapsed:.6f}s total; "
            f"{elapsed / steps:.6f}s/update; final_loss={loss:.9g}"
        )
    print("Bound: this is a short CPU timing probe, with no GPU timing, long search, or record claim.")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--kernel", help="path to the new kernel Python module")
    parser.add_argument("--runner", help="optional path to experimental batched841/run.py for coordinate cross-check")
    parser.add_argument("--benchmark", action="store_true", help="run bounded N=841 B1/B2 CPU timings")
    parser.add_argument("--benchmark-steps", type=int, default=2)
    parser.add_argument("--no-clamp-probe", action="store_true", help="skip the explicit 1e-12 floor derivative probe")
    parser.add_argument("--keep-temp", action="store_true", help="keep the temporary C oracle build and traces")
    args = parser.parse_args(argv)

    torch = _require_torch()
    kernel_path = _find_kernel(args.kernel)
    module = _load_kernel(kernel_path)
    default_runner = ROOT / "kissing" / "experimental" / "batched841" / "run.py"
    runner_path = Path(args.runner).expanduser().resolve() if args.runner else default_runner
    if args.runner and not runner_path.exists():
        raise SystemExit(f"runner module does not exist: {runner_path}")
    runner = _load_runner(runner_path, module) if runner_path.exists() else None
    if not C_SOURCE.exists():
        raise SystemExit(f"existing C oracle source does not exist: {C_SOURCE}")
    temp_context = tempfile.TemporaryDirectory(prefix="luna-parity-")
    directory = Path(temp_context.name)
    try:
        binary = _compile_oracle(directory)
        run_b1_parity(module, torch, binary, directory, clamp_probe=not args.no_clamp_probe)
        run_n841_probe(module, torch, binary, directory)
        if runner is not None:
            run_runner_crosscheck(module, torch, runner)
        run_b2_batch(module, torch, binary, directory)
        run_invalid_states(module, torch)
        if args.benchmark:
            run_benchmark(module, torch, steps=args.benchmark_steps)
    finally:
        if args.keep_temp:
            print(f"kept parity temporary files at {directory}")
            temp_context.cleanup = lambda: None  # type: ignore[method-assign]
        temp_context.cleanup()
    print(f"parity passed for kernel {kernel_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
