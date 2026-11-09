"""Bounded smoke and checkpoint tests for the batched 841 runner.

These tests intentionally use four updates rather than launching the complete
35,000-update search.  The full protocol remains an explicit user choice.
"""
from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

import numpy as np
import torch

from . import run


class Batched841RunnerTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        torch.set_num_threads(1)

    def _seed_file(self, directory: Path) -> Path:
        rng = np.random.default_rng(841)
        core = rng.normal(size=(run.N_FIXED, run.DIMENSION))
        core /= np.linalg.norm(core, axis=1, keepdims=True)
        path = directory / "seed.txt"
        np.savetxt(path, np.vstack([core, np.zeros((1, run.DIMENSION))]), fmt="%.17g")
        return path

    def _config(self, seed: Path, **kwargs: object) -> run.RunConfig:
        values: dict[str, object] = {
            "seed_file": seed,
            "seed": 123,
            "batch_size": 1,
            "macro_repeats": 1,
            "device": "cpu",
            "dtype": "float64",
            "threads": 1,
        }
        values.update(kwargs)
        return run.RunConfig(**values)  # type: ignore[arg-type]

    def test_hypercube_extra_and_fixed_core(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            directory = Path(tmp)
            path = self._seed_file(directory)
            core, _, _ = run._load_core(path)
            generator = run._make_generator("cpu", 123)
            raw = run._new_batch(core, 4, "cpu", generator).cpu().numpy()
            np.testing.assert_array_equal(
                raw[:, : run.N_FIXED], np.broadcast_to(core[None, :, :], (4, run.N_FIXED, run.DIMENSION))
            )
            extra = raw[:, -1]
            np.testing.assert_allclose(np.abs(extra), 1.0 / np.sqrt(run.DIMENSION))
            self.assertTrue(np.all(np.isin(np.sign(extra), (-1.0, 1.0))))

    def test_resume_matches_uninterrupted_and_readback_is_independent(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            directory = Path(tmp)
            seed = self._seed_file(directory)
            checkpoint = directory / "resume"
            first = self._config(
                seed,
                max_updates=2,
                checkpoint=checkpoint,
                checkpoint_every=1,
                output=directory / "first.txt",
            )
            run.run_search(first)
            resumed = self._config(
                seed,
                max_updates=4,
                checkpoint=checkpoint,
                resume=True,
                checkpoint_every=1,
                output=directory / "resumed.txt",
            )
            run.run_search(resumed)
            uninterrupted = self._config(
                seed,
                max_updates=4,
                checkpoint=directory / "whole",
                checkpoint_every=1,
                output=directory / "whole.txt",
            )
            run.run_search(uninterrupted)
            np.testing.assert_array_equal(
                np.loadtxt(directory / "resumed.txt"),
                np.loadtxt(directory / "whole.txt"),
            )
            metadata = json.loads((directory / "resumed.json").read_text())
            self.assertTrue(metadata["written_readback"])
            self.assertTrue(metadata["diagnostic"])
            self.assertEqual(metadata["updates"], 4)

    def test_resume_rejects_rewind_and_incompatible_batch(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            directory = Path(tmp)
            seed = self._seed_file(directory)
            checkpoint = directory / "resume"
            run.run_search(
                self._config(
                    seed,
                    max_updates=2,
                    checkpoint=checkpoint,
                    output=directory / "first.txt",
                )
            )
            with self.assertRaises(run.CheckpointError):
                run.run_search(
                    self._config(
                        seed,
                        max_updates=1,
                        checkpoint=checkpoint,
                        resume=True,
                        output=directory / "rewind.txt",
                    )
                )
            with self.assertRaises(run.CheckpointError):
                run.run_search(
                    self._config(
                        seed,
                        max_updates=3,
                        batch_size=2,
                        checkpoint=checkpoint,
                        resume=True,
                        output=directory / "wrong-batch.txt",
                    )
                )


if __name__ == "__main__":
    unittest.main()
