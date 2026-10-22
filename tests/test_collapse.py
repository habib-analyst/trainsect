"""Eval predicate: no train-spike misfires (the P0 bug).

A benign transient train-loss spike on a healthy model must NOT count as a
collapse. Only eval-loss threshold crossings (or non-finite eval) count.
"""
import math
import unittest

from trainsect.collapse import (
    collapse_threshold,
    detect_change_point,
    has_collapse,
    has_train_spike,
    is_healthy,
    train_spike_threshold,
)


class TestEvalPredicate(unittest.TestCase):
    def test_benign_train_spike_is_not_collapse(self):
        # P0 Run 2: benign train spike 9.18 at step 120, eval stayed healthy.
        train_losses = {120: 9.18, 121: 0.5}
        spike, at = has_train_spike(train_losses, 5.0)
        self.assertTrue(spike)
        self.assertEqual(at, 120)
        # ...but the eval predicate says NO collapse:
        evals = {120: 0.69, 130: 0.71}
        collapse, _ = has_collapse(evals, 3.0)
        self.assertFalse(collapse)

    def test_eval_crossing_is_collapse(self):
        evals = {0: 0.5, 10: 0.6, 20: 231.0}
        collapse, at = has_collapse(evals, 3.0)
        self.assertTrue(collapse)
        self.assertEqual(at, 20)

    def test_nonfinite_eval_is_collapse(self):
        collapse, _ = has_collapse({50: math.nan}, 3.0)
        self.assertTrue(collapse)
        collapse, _ = has_collapse({50: math.inf}, 3.0)
        self.assertTrue(collapse)

    def test_healthy_drift_below_threshold(self):
        # Healthy training drifting to eval 3.1 must NOT cross the threshold
        # derived from a 0.69 baseline (P0 bug #3 calibration).
        thresh, baseline, _ = collapse_threshold([[0, 0.69], [10, 0.70], [20, 0.69]])
        self.assertAlmostEqual(baseline, 0.69, places=2)
        self.assertGreater(thresh, 3.1)  # 10x baseline = 6.9
        self.assertFalse(has_collapse({80: 3.097}, thresh)[0])
        self.assertTrue(has_collapse({50: 231.0}, thresh)[0])

    def test_is_healthy_rejects_already_collapsed(self):
        self.assertTrue(is_healthy(0.5, 3.0))
        self.assertFalse(is_healthy(231.0, 3.0))
        self.assertFalse(is_healthy(math.nan, 3.0))

    def test_change_point_requires_sustained(self):
        log = [[0, 0.5], [10, 0.6], [20, 100.0], [30, 0.6], [40, 0.7]]
        res = detect_change_point(log, threshold=3.0, sustained=2)
        self.assertIsNone(res["change_point_step"])  # single blip, not sustained
        log2 = [[0, 0.5], [10, 0.6], [20, 100.0], [30, 200.0]]
        res2 = detect_change_point(log2, threshold=3.0, sustained=2)
        self.assertEqual(res2["change_point_step"], 30)

    def test_train_spike_threshold(self):
        sidecar = [{"train_loss": 0.5 + 0.01 * i} for i in range(20)]
        # lowest quartile of early losses: 0.50..0.54, median 0.52 -> 5.2
        self.assertAlmostEqual(train_spike_threshold(sidecar), 5.2, places=2)

    def test_train_spike_threshold_ignores_warmup(self):
        # Warmup-contaminated early losses must not inflate the threshold
        # above the real trigger spike (12.46 in the demo).
        losses = [4.38, 3.5, 2.8, 2.0, 1.5, 1.0, 0.7, 0.5, 0.3, 0.2,
                  0.15, 0.12, 0.1, 0.08, 0.05, 0.03, 0.02, 0.01, 0.005, 0.002]
        sidecar = [{"train_loss": v} for v in losses]
        thresh = train_spike_threshold(sidecar)
        self.assertLess(thresh, 12.46)
        self.assertGreaterEqual(thresh, 2.0)

    def test_warmup_transient_is_not_collapse(self):
        # Demo shape: untrained eval 4.3 at step 0, healthy 0.002 by step 40,
        # collapse to ~3.9 at step 70. Warmup must not flag; collapse must.
        log = [[0, 4.3], [10, 1.75], [20, 0.07], [30, 0.006], [40, 0.002],
               [50, 0.002], [60, 0.002], [70, 3.87], [80, 4.08]]
        res = detect_change_point(log)
        self.assertEqual(res["change_point_step"], 80)  # sustained: 70 + 80
        self.assertAlmostEqual(res["baseline"], 0.002, places=3)

    def test_baseline_from_lowest_quartile_not_first_points(self):
        # First points are warmup (4.3); baseline must come from healthy tail.
        thresh, baseline, _ = collapse_threshold(
            [[0, 4.3], [10, 1.75], [20, 0.07], [30, 0.006], [40, 0.002],
             [50, 0.002], [60, 0.002], [70, 3.87]]
        )
        self.assertAlmostEqual(baseline, 0.002, places=3)
        self.assertEqual(thresh, 3.0)  # floor dominates for a near-zero baseline


if __name__ == "__main__":
    unittest.main()
