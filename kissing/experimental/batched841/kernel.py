"""Batched float64 raw-Adam Riesz kernel for the faithful 841 protocol.

The functions in this module are intentionally small building blocks rather
than a search driver.  A raw candidate tensor has shape ``(B, N, d)``.  Each
candidate is normalised row-wise only to form the loss view; Adam updates the
raw tensor and never retracts it.  This follows the faithful path documented
in ``kissing/lib/FAITHFUL_841.md`` and the corresponding ``adam_raw_stage`` in
``kissing/lib/riesz.c``.

The Riesz loss for candidate ``b`` is

    log(sum_{i<j} ||z_i-z_j||**(-s)),

where ``z = normalized_view(raw)``.  Pairwise squared distances are clamped
below at ``1e-12``, as in the C implementation.  Its hand-coded derivative
also follows C: a pair below the clamp still contributes ``s * weight / r2``
to the derivative.  This differs from ordinary autograd through
``torch.clamp`` (which gives zero derivative in the clamped region), and is
intentional for source parity.

All returned values are float64 tensors on the input device.  The returned
loss and gradient are detached: callers using a manual optimizer should use
the gradient returned here, while ``normalized_view`` itself remains an
ordinary differentiable tensor operation.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import torch


# The published faithful search stages from riesz.c.  Keeping the values here
# as one immutable tuple makes it harder for a batched caller to accidentally
# alter the continuation protocol.
SEARCH_SCHEDULE: tuple[tuple[int, int, float], ...] = (
    (8, 1000, 0.005),
    (16, 1000, 0.003),
    (32, 1000, 0.002),
    (64, 2000, 0.001),
    (128, 2000, 0.0005),
    (256, 2000, 0.0002),
    (512, 2000, 0.0001),
    (1024, 4000, 0.00005),
    (2048, 4000, 0.00001),
    (4096, 4000, 0.00001),
    (10000, 4000, 0.000005),
    (20000, 4000, 0.000001),
    (40000, 4000, 0.000001),
)

PAIR_DISTANCE_FLOOR = 1.0e-12
BETA1 = 0.9
BETA2 = 0.999
ADAM_EPS = 1.0e-8


def _require_raw(raw: torch.Tensor) -> tuple[int, int, int, torch.Tensor]:
    """Validate a raw batch and return dimensions plus row norms.

    Validation is deliberately explicit because a non-finite or zero row can
    otherwise produce a finite looking normalised tensor and poison Adam's
    persistent moments later.  The check accepts any device, but requires
    float64 to keep the faithful CPU arithmetic and to make the CUDA choice an
    explicit caller decision.
    """

    if not isinstance(raw, torch.Tensor):
        raise TypeError(f"raw must be a torch.Tensor, got {type(raw).__name__}")
    if raw.ndim != 3:
        raise ValueError(f"raw must have shape (B, N, d), got {tuple(raw.shape)}")
    if raw.dtype != torch.float64:
        raise TypeError(f"raw must have dtype torch.float64, got {raw.dtype}")
    if raw.shape[0] < 1:
        raise ValueError("raw batch must contain at least one candidate")
    if raw.shape[1] < 2:
        raise ValueError("raw candidates must contain at least two rows")
    if raw.shape[2] < 1:
        raise ValueError("raw candidates must have positive dimension")

    # .item() is used only for validation, keeping the actual kernel free of
    # host synchronisation beyond these requested input guards.
    if not bool(torch.isfinite(raw).all().item()):
        raise ValueError("raw contains NaN or infinity")
    squared_norms = torch.sum(raw * raw, dim=-1)
    norms = torch.sqrt(squared_norms)
    if not bool(torch.isfinite(norms).all().item()):
        raise ValueError("raw contains a row whose norm is not finite")
    if not bool((norms > PAIR_DISTANCE_FLOOR).all().item()):
        raise ValueError(
            "raw contains a zero or too-small row (norm must exceed 1e-12)"
        )
    return raw.shape[0], raw.shape[1], raw.shape[2], norms


def _validate_exponent(s: float) -> float:
    try:
        exponent = float(s)
    except (TypeError, ValueError) as exc:
        raise TypeError(f"s must be a finite positive scalar, got {s!r}") from exc
    if not torch.isfinite(torch.tensor(exponent, dtype=torch.float64)) or exponent <= 0:
        raise ValueError(f"s must be a finite positive scalar, got {s!r}")
    return exponent


def _validate_learning_rate(lr: float) -> float:
    try:
        rate = float(lr)
    except (TypeError, ValueError) as exc:
        raise TypeError(f"lr must be a finite positive scalar, got {lr!r}") from exc
    if not torch.isfinite(torch.tensor(rate, dtype=torch.float64)) or rate <= 0:
        raise ValueError(f"lr must be a finite positive scalar, got {lr!r}")
    return rate


def normalized_view(raw: torch.Tensor) -> torch.Tensor:
    """Return the row-normalised loss view of raw ``(B, N, d)`` candidates.

    The input is not changed.  The operation remains connected to autograd,
    which is useful for inspecting the ordinary smooth loss away from the
    distance clamp.  ``loss_and_grad`` uses its own explicit derivative to
    preserve the faithful C clamp convention.
    """

    _, _, _, norms = _require_raw(raw)
    return raw / norms.unsqueeze(-1)


def loss_and_grad(raw: torch.Tensor, s: float) -> tuple[torch.Tensor, torch.Tensor]:
    """Compute the batch-mean faithful Riesz log loss and raw gradient.

    Parameters
    ----------
    raw:
        Float64 tensor with shape ``(B, N, d)``.  Each row must be finite and
        norm greater than ``1e-12``.  Candidates may be on CPU or CUDA, chosen by
    the caller.
    s:
        Positive Riesz exponent.

    Returns
    -------
    loss, grad:
        A detached scalar loss equal to the mean of the B candidate losses,
        and a detached tensor with shape ``(B, N, d)``.  The gradient includes
        the row-normalisation chain rule and therefore is the gradient with
        respect to raw coordinates.  It is divided by B exactly once, as a
        true batch mean.
    """

    exponent = _validate_exponent(s)
    batch, n_rows, _, norms = _require_raw(raw)
    z = raw / norms.unsqueeze(-1)

    # The upper triangle is exactly the i<j set used by riesz.c.  Building one
    # shared index pair list avoids forming or summing the diagonal.
    pair_i, pair_j = torch.triu_indices(
        n_rows, n_rows, offset=1, device=raw.device
    )
    gram = torch.bmm(z, z.transpose(1, 2))
    inner = gram[:, pair_i, pair_j]
    distance2 = torch.clamp(2.0 - 2.0 * inner, min=PAIR_DISTANCE_FLOOR)

    # logsumexp retains every pair term, matching faithful C mode's stable
    # torch.logsumexp semantics.  Computing the weights from the same shifted
    # values gives a stable explicit derivative.
    log_terms = -(0.5 * exponent) * torch.log(distance2)
    candidate_loss = torch.logsumexp(log_terms, dim=1)
    loss = candidate_loss.mean()
    weights = torch.softmax(log_terms, dim=1)

    # This is the Euclidean derivative used by C after its distance clamp:
    # c_ij = s * weight_ij / r2_ij.  In particular, it deliberately does not
    # zero c_ij when the clamp is active.  Scatter the pair contributions to
    # both endpoints, then apply C's tangent projection.
    coeff = exponent * weights / distance2
    # Keep the memory layout close to C: one symmetric B*N*N coefficient
    # matrix and one batched matrix multiply.  Materialising pair_delta with
    # shape B*P*d is substantially larger for the intended B=512,N=841,d=12
    # workload.  The diagonal is represented by subtracting each row sum.
    coeff_matrix = torch.zeros(
        (batch, n_rows, n_rows), dtype=raw.dtype, device=raw.device
    )
    coeff_matrix[:, pair_i, pair_j] = coeff
    coeff_matrix[:, pair_j, pair_i] = coeff
    row_sum = torch.sum(coeff_matrix, dim=-1, keepdim=True)
    grad_z = torch.bmm(coeff_matrix, z) - row_sum * z
    radial = torch.sum(grad_z * z, dim=-1, keepdim=True)
    grad_z = grad_z - radial * z

    # z_i = raw_i / ||raw_i||.  Since the C gradient above has already been
    # tangent projected, the Jacobian contributes only one division by ||raw||.
    grad_raw = grad_z / norms.unsqueeze(-1)
    grad_raw = grad_raw / batch
    if not bool(torch.isfinite(loss).item()) or not bool(torch.isfinite(grad_raw).all().item()):
        raise FloatingPointError("Riesz loss or gradient is non-finite")
    return loss.detach(), grad_raw.detach()


def initial_state(raw: torch.Tensor) -> dict[str, Any]:
    """Create detached raw-Adam state with empty persistent moments.

    The returned ``raw`` is a clone, so later ``adam_step`` calls do not alter
    the caller's input tensor.  State tensors retain the caller's device and
    float64 dtype.
    """

    _require_raw(raw)
    raw_state = raw.detach().clone()
    return {
        "raw": raw_state,
        "m": torch.zeros_like(raw_state),
        "v": torch.zeros_like(raw_state),
        "step": 0,
    }


def _validate_state(state: Mapping[str, Any]) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, int]:
    if not isinstance(state, dict):
        raise TypeError("state must be the dictionary returned by initial_state")
    missing = {"raw", "m", "v", "step"}.difference(state)
    if missing:
        raise ValueError(f"state is missing keys: {sorted(missing)}")
    raw = state["raw"]
    m = state["m"]
    v = state["v"]
    if not isinstance(m, torch.Tensor) or not isinstance(v, torch.Tensor):
        raise TypeError("state moments must be torch tensors")
    _require_raw(raw)
    if m.shape != raw.shape or v.shape != raw.shape:
        raise ValueError("state moments must have the same shape as state['raw']")
    if m.dtype != raw.dtype or v.dtype != raw.dtype:
        raise TypeError("state moments must have the same dtype as state['raw']")
    if m.device != raw.device or v.device != raw.device:
        raise ValueError("state moments must be on the same device as state['raw']")
    if not bool(torch.isfinite(m).all().item()) or not bool(torch.isfinite(v).all().item()):
        raise ValueError("state moments contain NaN or infinity")
    if not bool((v >= 0).all().item()):
        raise ValueError("state second moment must be nonnegative")
    step = state["step"]
    if isinstance(step, bool) or not isinstance(step, int) or step < 0:
        raise ValueError("state step must be a nonnegative Python integer")
    return raw, m, v, step


def adam_step(state: dict[str, Any], s: float, lr: float) -> dict[str, Any]:
    """Apply one in-place faithful raw-Adam update and return ``state``.

    Moments persist across calls and therefore across continuation stages.
    The loss is evaluated on the normalised view, but ``state['raw']`` is
    updated directly; there is no normalisation or retraction after the step.
    """

    rate = _validate_learning_rate(lr)
    raw, m, v, step = _validate_state(state)
    loss, grad = loss_and_grad(raw, s)

    next_step = step + 1
    # Build and validate the complete next state before touching the existing
    # tensors.  A failed finite/row guard therefore leaves the caller's state
    # recoverable rather than committing only updated moments.
    with torch.no_grad():
        next_m = BETA1 * m + (1.0 - BETA1) * grad
        next_v = BETA2 * v + (1.0 - BETA2) * grad.square()
        bias1 = 1.0 - BETA1**next_step
        bias2 = 1.0 - BETA2**next_step
        m_hat = next_m / bias1
        v_hat = next_v / bias2
        next_raw = raw - rate * m_hat / (torch.sqrt(v_hat) + ADAM_EPS)
        finite = (
            torch.isfinite(next_raw).all()
            and torch.isfinite(next_m).all()
            and torch.isfinite(next_v).all()
        )
        if not bool(finite.item()):
            raise FloatingPointError("Adam update produced NaN or infinity")
        if not bool((next_v >= 0).all().item()):
            raise FloatingPointError("Adam update produced a negative second moment")
        try:
            _require_raw(next_raw)
        except (ValueError, TypeError) as exc:
            raise FloatingPointError(f"Adam update produced an invalid raw row: {exc}") from exc
        raw.copy_(next_raw)
        m.copy_(next_m)
        v.copy_(next_v)
    state["step"] = next_step
    return state
