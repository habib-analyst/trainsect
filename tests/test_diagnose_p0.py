"""Integration: diagnose the P0 validation run with the P1 package.

Must reproduce the P0 result: trigger step 40, shard 2, necessity +
sufficiency true, CONFIRMED. Slow (~20 s of replays) -- this is the
regression test that the P1 rewrite did not change the proven algorithms.
"""
import glob
import os
import subprocess
import sys
import unittest

from trainsect.diagnose import diagnose

RUN_DIR = os.path.join(
    os.path.dirname(os.path.abspath(__file__)), "..", "validation", "run"
)


def _ensure_fixture():
    """Regenerate validation/run if its checkpoints are absent.

    Checkpoint .pt fixtures are regenerable and not stored in git, so a
    fresh clone must rebuild them before diagnosing.
    """
    if not glob.glob(os.path.join(RUN_DIR, "ckpt_*.pt")):
        gen = os.path.join(
            os.path.dirname(os.path.abspath(__file__)), "..",
            "validation", "make_run.py",
        )
        subprocess.run([sys.executable, gen], check=True)


class TestDiagnoseP0Run(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        _ensure_fixture()
    def test_p0_run_reproduced(self):
        report = diagnose(RUN_DIR, verbose=False)
        self.assertTrue(report.fidelity_bit_exact)
        self.assertTrue(report.confirmed_causal)
        self.assertEqual(report.verdict, "CONFIRMED (causal)")
        self.assertEqual(report.bracket, [40, 80])
        self.assertEqual(report.trigger_step, 40)
        self.assertEqual(report.shard, 2)
        self.assertEqual(report.batch_fingerprint, "e3768bc902f668d7")
        self.assertTrue(report.confirmation["necessity"])
        self.assertTrue(report.confirmation["sufficiency"])


if __name__ == "__main__":
    unittest.main()
