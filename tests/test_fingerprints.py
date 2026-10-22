"""Fingerprint stability: same content -> same fingerprint, any change -> different."""
import unittest

import torch

from trainsect.fingerprints import fingerprint_batch, fingerprint_params


class TestFingerprints(unittest.TestCase):
    def test_stable_across_calls(self):
        x = torch.randn(8, 4)
        y = torch.randn(8, 1)
        self.assertEqual(fingerprint_batch(x, y), fingerprint_batch(x, y))

    def test_equal_values_same_fingerprint(self):
        x1 = torch.tensor([[1.0, 2.0], [3.0, 4.0]])
        x2 = torch.tensor([[1.0, 2.0], [3.0, 4.0]])
        y = torch.zeros(2, 1)
        self.assertEqual(fingerprint_batch(x1, y), fingerprint_batch(x2, y))

    def test_row_order_matters(self):
        x1 = torch.tensor([[1.0, 2.0], [3.0, 4.0]])
        x2 = torch.tensor([[3.0, 4.0], [1.0, 2.0]])  # same rows, swapped order
        y = torch.zeros(2, 1)
        self.assertNotEqual(fingerprint_batch(x1, y), fingerprint_batch(x2, y))

    def test_single_value_change_matters(self):
        x1 = torch.zeros(4, 4)
        x2 = torch.zeros(4, 4)
        x2[0, 0] = 1e-9
        y = torch.zeros(4, 1)
        self.assertNotEqual(fingerprint_batch(x1, y), fingerprint_batch(x2, y))

    def test_x_y_swap_matters(self):
        a = torch.randn(4, 4)
        b = torch.randn(4, 4)
        self.assertNotEqual(fingerprint_batch(a, b), fingerprint_batch(b, a))

    def test_shape_matters(self):
        a = torch.arange(4, dtype=torch.float32).reshape(2, 2)
        b = torch.arange(4, dtype=torch.float32).reshape(4, 1)
        y = torch.zeros(2, 1)
        # Same bytes, different shape -> different fingerprint.
        self.assertNotEqual(fingerprint_batch(a, y), fingerprint_batch(b, y))

    def test_dtype_matters(self):
        a32 = torch.ones(2, 2, dtype=torch.float32)
        a64 = torch.ones(2, 2, dtype=torch.float64)
        y = torch.zeros(2, 1)
        self.assertNotEqual(fingerprint_batch(a32, y), fingerprint_batch(a64, y))

    def test_param_hash_changes_after_update(self):
        from trainsect.models import MLP

        m = MLP()
        h1 = fingerprint_params(m)
        with torch.no_grad():
            for p in m.parameters():
                p.add_(1.0)
        self.assertNotEqual(h1, fingerprint_params(m))


if __name__ == "__main__":
    unittest.main()
