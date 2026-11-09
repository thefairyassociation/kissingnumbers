"""Checkpoint lifecycle checks using a tiny test-only schedule and geometry.

The separate C parity suite enforces the real 35,000-update schedule. These
tests shorten only the fixture to exercise stage/macro/terminal boundaries
and interruption recovery quickly; they are not full-search evidence.
"""
from contextlib import ExitStack
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np

from . import run


class RunnerIntegrityTests(unittest.TestCase):
    def setUp(self):
        self.stack = ExitStack()
        self.addCleanup(self.stack.close)
        self.directory = Path(self.stack.enter_context(tempfile.TemporaryDirectory()))
        schedule = ((8, 2, .005), (16, 2, .003))
        self.stack.enter_context(patch.multiple(
            run, N_POINTS=7, N_FIXED=6, DIMENSION=3,
            FULL_SCHEDULE_UPDATES=4, _EXPECTED_SCHEDULE=schedule,
        ))
        self.stack.enter_context(patch.object(run.kernel, "SEARCH_SCHEDULE", schedule))
        self.seed = self.directory / "seed.txt"
        core = np.random.default_rng(54).normal(size=(6, 3))
        core /= np.linalg.norm(core, axis=1, keepdims=True)
        np.savetxt(self.seed, core, fmt="%.17g")

    def config(self, name="split", **overrides):
        settings = dict(seed=17, seed_file=self.seed, batch_size=2,
                        macro_repeats=2, checkpoint=self.directory / name,
                        max_updates=8, threads=1)
        settings.update(overrides)
        return run.RunConfig(**settings)

    def snapshot(self, name="split"):
        with np.load(self.directory / (name + ".npz"), allow_pickle=False) as data:
            arrays = {key: np.array(data[key], copy=True) for key in data.files}
        metadata = json.loads(arrays["metadata_json"].tobytes())
        return arrays, metadata

    def test_stage_macro_and_terminal_resume_preserve_all_state(self):
        whole = run.run_search(self.config("whole"))
        self.assertTrue(whole["full_schedule"])
        run.run_search(self.config(max_updates=2))
        arrays, metadata = self.snapshot()
        self.assertEqual(metadata["cursor"], dict(macro=0, stage=1, in_stage=0, updates_total=2))
        self.assertTrue(np.any(arrays["m"] != 0))
        run.run_search(self.config(max_updates=4, resume=True))
        arrays, metadata = self.snapshot()
        self.assertEqual(metadata["cursor"], dict(macro=1, stage=0, in_stage=0, updates_total=4))
        self.assertEqual(int(arrays["step"]), 0)
        self.assertTrue(np.all(arrays["m"] == 0))
        self.assertTrue(np.all(arrays["v"] == 0))
        finished = run.run_search(self.config(resume=True))
        self.assertTrue(finished["full_schedule"])
        self.assertEqual(finished["cursor"], dict(macro=2, stage=0, in_stage=0))
        actual, _ = self.snapshot()
        expected, _ = self.snapshot("whole")
        for key in actual.keys() - {"metadata_json"}:
            np.testing.assert_array_equal(actual[key], expected[key], err_msg=key)
        # A terminal resume must perform no additional update or random draw.
        run.run_search(self.config(resume=True))
        terminal, _ = self.snapshot()
        for key in actual.keys() - {"metadata_json"}:
            np.testing.assert_array_equal(terminal[key], actual[key], err_msg=key)

    def test_interrupted_atomic_replace_preserves_previous_checkpoint(self):
        run.run_search(self.config(max_updates=1))
        path = self.directory / "split.npz"
        previous = path.read_bytes()
        with patch.object(run.os, "replace", side_effect=OSError("simulated interruption")):
            with self.assertRaisesRegex(OSError, "simulated interruption"):
                run.run_search(self.config(max_updates=2, resume=True))
        self.assertEqual(path.read_bytes(), previous)
        # The old snapshot is still loadable and produces the intended finish.
        result = run.run_search(self.config(resume=True))
        self.assertTrue(result["full_schedule"])

    def test_corrupt_cursor_and_nonfinite_or_invalid_state_are_rejected(self):
        run.run_search(self.config(max_updates=1))
        original, original_metadata = self.snapshot()
        for fault in ("wrong_total", "boolean_cursor", "nan", "negative_v", "zero_row"):
            with self.subTest(fault=fault):
                arrays = {key: value.copy() for key, value in original.items()}
                metadata = json.loads(json.dumps(original_metadata))
                if fault == "wrong_total":
                    metadata["cursor"]["updates_total"] += 1
                elif fault == "boolean_cursor":
                    metadata["cursor"]["macro"] = False
                elif fault == "nan":
                    arrays["raw"][0, 0, 0] = np.nan
                elif fault == "negative_v":
                    arrays["v"][0, 0, 0] = -1
                else:
                    arrays["raw"][0, 0, :] = 0
                arrays["metadata_json"] = np.frombuffer(json.dumps(metadata).encode(), dtype=np.uint8)
                np.savez(self.directory / "split.npz", **arrays)
                with self.assertRaises(run.RunnerError):
                    run.run_search(self.config(max_updates=2, resume=True))

    def test_seed_and_runtime_changes_are_rejected(self):
        run.run_search(self.config(max_updates=1))
        with self.assertRaises(run.CheckpointError):
            run.run_search(self.config(max_updates=2, threads=2, resume=True))
        changed = np.loadtxt(self.seed)
        changed[0, 0] += .001
        np.savetxt(self.seed, changed, fmt="%.17g")
        with self.assertRaises(run.CheckpointError):
            run.run_search(self.config(max_updates=2, resume=True))

    def test_fresh_run_cannot_replace_checkpoint_or_alias_outputs(self):
        run.run_search(self.config(max_updates=0))
        previous = (self.directory / "split.npz").read_bytes()
        with self.assertRaises(run.RunnerError):
            run.run_search(self.config(max_updates=1))
        self.assertEqual((self.directory / "split.npz").read_bytes(), previous)
        for output in (self.seed, self.directory / "candidate.json", self.directory / "split.npz"):
            with self.subTest(output=output):
                with self.assertRaises(run.RunnerError):
                    run.run_search(self.config(max_updates=0, resume=True, output=output))


if __name__ == "__main__":
    unittest.main()
