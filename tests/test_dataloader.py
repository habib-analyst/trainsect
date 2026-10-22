"""Indexed dataloader: shard mapping, fingerprint reproduction, substitution."""
import unittest

import torch

from trainsect.dataloader import IndexedDataLoader


def _raw(n_shards=3, per_shard=4):
    raw = []
    for shard in range(n_shards):
        for i in range(per_shard):
            x = torch.full((2, 2), float(shard * 10 + i))
            y = torch.full((2, 1), float(shard))
            raw.append((x, y, shard))
    return raw


class TestIndexedDataLoader(unittest.TestCase):
    def setUp(self):
        self.loader = IndexedDataLoader.from_raw(_raw())

    def test_shard_ids_and_mapping(self):
        self.assertEqual(self.loader.shard_ids(), [0, 1, 2])
        self.assertEqual(self.loader.steps_for_shard(1), [4, 5, 6, 7])
        self.assertEqual(self.loader.shard_of(9), 2)

    def test_batches_in_range(self):
        steps = [b.step for b in self.loader.batches_in_range(4, 8)]
        self.assertEqual(steps, [4, 5, 6, 7])
        self.assertEqual(self.loader.shards_in_range(4, 8), [1])

    def test_reproduce_by_fingerprint(self):
        target = self.loader[5]
        found = self.loader.reproduce(target.fingerprint)
        self.assertIsNotNone(found)
        self.assertEqual(found.step, 5)
        self.assertTrue(torch.equal(found.x, target.x))
        self.assertTrue(self.loader.verify_fingerprint(target.fingerprint))
        self.assertIsNone(self.loader.reproduce("deadbeefdeadbeef"))
        self.assertFalse(self.loader.verify_fingerprint("deadbeefdeadbeef"))

    def test_fingerprints_stable_across_rebuilds(self):
        rebuilt = IndexedDataLoader.from_raw(_raw())
        for a, b in zip(self.loader.batches, rebuilt.batches):
            self.assertEqual(a.fingerprint, b.fingerprint)

    def test_substitute_keeps_and_replaces(self):
        benign = self.loader.benign_batches(0)
        # keep shard 1, replace everything else with benign (shard-0) batches
        x, _ = self.loader.substitute(5, lambda s, sh: sh == 1, benign)
        self.assertTrue(torch.equal(x, self.loader[5].x))
        x2, _ = self.loader.substitute(8, lambda s, sh: sh == 1, benign)
        # step 8 (shard 2) replaced by benign[8 % 4] = benign[0], shard-0 content
        self.assertEqual(x2[0, 0].item(), 0.0)  # shard-0 content

    def test_benign_batches_empty_shard_raises(self):
        with self.assertRaises(ValueError):
            self.loader.benign_batches(99)

    def test_sidecar_mismatch_raises(self):
        sidecar = [
            {"step": i, "batch_fp": b.fingerprint} for i, b in enumerate(self.loader.batches)
        ]
        sidecar[3]["batch_fp"] = "0000000000000000"
        with self.assertRaises(ValueError):
            IndexedDataLoader.from_sidecar(_raw(), sidecar)

    def test_sidecar_ok(self):
        sidecar = [
            {"step": i, "batch_fp": b.fingerprint} for i, b in enumerate(self.loader.batches)
        ]
        loader = IndexedDataLoader.from_sidecar(_raw(), sidecar)
        self.assertEqual(len(loader), 12)


if __name__ == "__main__":
    unittest.main()
