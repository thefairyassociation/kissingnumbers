"""Regression checks for the batched faithful 841 kernel.

Run from the repository root with ``python -m unittest
kissing.experimental.batched841.test_kernel``.  The independent oracle below
follows the scalar loops in ``kissing/lib/riesz.c`` so these checks do not just
restate the vectorised implementation.
"""

from __future__ import annotations

import math
import unittest

import torch

from .kernel import (
    ADAM_EPS,
    BETA1,
    BETA2,
    PAIR_DISTANCE_FLOOR,
    SEARCH_SCHEDULE,
    adam_step,
    initial_state,
    loss_and_grad,
    normalized_view,
)


def scalar_c_oracle(raw: torch.Tensor, s: float) -> tuple[float, torch.Tensor]:
    """Independent B=1 scalar-loop oracle for C's value and gradient."""

    if raw.shape[0] != 1:
        raise ValueError("oracle only accepts B=1")
    x = raw[0].detach().cpu().tolist()
    norms = [math.sqrt(sum(value * value for value in row)) for row in x]
    z = [[value / norm for value in row] for row, norm in zip(x, norms)]
    n_rows = len(z)
    pairs: list[tuple[int, int, float]] = []
    r2min = math.inf
    for i in range(n_rows):
        for j in range(i + 1, n_rows):
            inner = sum(z[i][k] * z[j][k] for k in range(len(z[i])))
            r2 = max(2.0 - 2.0 * inner, PAIR_DISTANCE_FLOOR)
            pairs.append((i, j, r2))
            r2min = min(r2min, r2)
    e_terms = [math.exp(-0.5 * s * math.log(r2 / r2min)) for _, _, r2 in pairs]
    energy = sum(e_terms)
    loss = math.log(energy) - 0.5 * s * math.log(r2min)
    grad = [[0.0] * len(z[0]) for _ in z]
    for (i, j, r2), e in zip(pairs, e_terms):
        c = s * e / (r2 * energy)
        for k in range(len(z[i])):
            delta = z[j][k] - z[i][k]
            grad[i][k] += c * delta
            grad[j][k] -= c * delta
    # C first forms the Euclidean pair gradient and then project_tangent().
    for i in range(n_rows):
        radial = sum(grad[i][k] * z[i][k] for k in range(len(z[i])))
        for k in range(len(z[i])):
            grad[i][k] = (grad[i][k] - radial * z[i][k]) / norms[i]
    output = torch.tensor(grad, dtype=torch.float64).unsqueeze(0)
    return loss, output


class Batched841KernelTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        torch.set_num_threads(1)
        torch.manual_seed(841)

    def test_schedule_is_published_35000_step_schedule(self) -> None:
        self.assertEqual(len(SEARCH_SCHEDULE), 13)
        self.assertEqual(sum(stage[1] for stage in SEARCH_SCHEDULE), 35000)
        self.assertEqual(
            [(stage[0], stage[1]) for stage in SEARCH_SCHEDULE],
            [
                (8, 1000),
                (16, 1000),
                (32, 1000),
                (64, 2000),
                (128, 2000),
                (256, 2000),
                (512, 2000),
                (1024, 4000),
                (2048, 4000),
                (4096, 4000),
                (10000, 4000),
                (20000, 4000),
                (40000, 4000),
            ],
        )

    def test_b1_matches_independent_c_scalar_oracle(self) -> None:
        raw = torch.randn(1, 6, 4, dtype=torch.float64)
        got_loss, got_grad = loss_and_grad(raw, 13.0)
        want_loss, want_grad = scalar_c_oracle(raw, 13.0)
        self.assertAlmostEqual(got_loss.item(), want_loss, places=11)
        torch.testing.assert_close(got_grad, want_grad, rtol=2e-11, atol=2e-11)

    def test_gradient_finite_difference_away_from_clamp(self) -> None:
        # Scaling and pair separation keep all r2 values well above 1e-12.
        raw = torch.randn(1, 5, 3, dtype=torch.float64)
        loss, grad = loss_and_grad(raw, 8.0)
        self.assertTrue(math.isfinite(loss.item()))
        for index in ((0, 0, 0), (0, 2, 1), (0, 4, 2)):
            h = 1e-6
            plus = raw.clone()
            minus = raw.clone()
            plus[index] += h
            minus[index] -= h
            plus_loss, _ = loss_and_grad(plus, 8.0)
            minus_loss, _ = loss_and_grad(minus, 8.0)
            numerical = (plus_loss - minus_loss).item() / (2.0 * h)
            self.assertAlmostEqual(grad[index].item(), numerical, delta=4e-7)

    def test_batch_mean_scales_each_candidate_gradient(self) -> None:
        one = torch.randn(1, 5, 3, dtype=torch.float64)
        one_loss, one_grad = loss_and_grad(one, 32.0)
        batch = one.expand(4, -1, -1).clone()
        batch_loss, batch_grad = loss_and_grad(batch, 32.0)
        self.assertAlmostEqual(batch_loss.item(), one_loss.item(), places=12)
        torch.testing.assert_close(batch_grad[0], one_grad[0] / 4.0, rtol=1e-12, atol=1e-12)
        torch.testing.assert_close(batch_grad[3], one_grad[0] / 4.0, rtol=1e-12, atol=1e-12)

    def test_active_clamp_uses_c_derivative_convention(self) -> None:
        # The second row is distinct but closer than sqrt(1e-12).  Autograd
        # through torch.clamp would zero this pair's derivative; C retains it.
        raw = torch.tensor(
            [[[1.0, 0.0], [1.0, 1.0e-8], [0.0, 1.0]]], dtype=torch.float64
        )
        got_loss, got_grad = loss_and_grad(raw, 8.0)
        want_loss, want_grad = scalar_c_oracle(raw, 8.0)
        self.assertTrue(math.isfinite(got_loss.item()))
        torch.testing.assert_close(got_grad, want_grad, rtol=2e-11, atol=2e-11)
        self.assertGreater(abs(got_grad[0, 0, 1].item()), 1.0e-3)

    def test_adam_step_updates_raw_and_persists_moments(self) -> None:
        raw = torch.randn(2, 5, 3, dtype=torch.float64)
        state = initial_state(raw)
        before = state["raw"].clone()
        _, grad = loss_and_grad(before, 8.0)
        expected_m = (1.0 - BETA1) * grad
        expected_v = (1.0 - BETA2) * grad.square()
        expected_raw = before - 0.005 * (expected_m / (1.0 - BETA1)) / (
            (expected_v / (1.0 - BETA2)).sqrt() + ADAM_EPS
        )
        adam_step(state, 8.0, 0.005)
        self.assertEqual(state["step"], 1)
        torch.testing.assert_close(state["m"], expected_m, rtol=1e-13, atol=1e-13)
        torch.testing.assert_close(state["v"], expected_v, rtol=1e-13, atol=1e-13)
        torch.testing.assert_close(state["raw"], expected_raw, rtol=1e-13, atol=1e-13)
        # Raw Adam intentionally leaves row norms unconstrained.
        self.assertGreater(float(torch.max(torch.abs(state["raw"].norm(dim=-1) - 1.0))), 1e-8)

    def test_rejects_nonfinite_and_zero_rows(self) -> None:
        good = torch.ones(1, 3, 2, dtype=torch.float64)
        for bad in (
            good.clone().fill_(float("nan")),
            good.clone().fill_(float("inf")),
            torch.cat((good[:, :2], torch.zeros(1, 1, 2, dtype=torch.float64)), dim=1),
            torch.cat((good[:, :2], torch.full((1, 1, 2), 1.0e-13, dtype=torch.float64)), dim=1),
        ):
            with self.assertRaises(ValueError):
                normalized_view(bad)
        with self.assertRaises(TypeError):
            normalized_view(good.float())


if __name__ == "__main__":
    unittest.main()
