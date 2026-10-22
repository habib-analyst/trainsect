"""Bisect correctness + necessity/sufficiency logic.

Checkpoint binary search is tested against scripted predicates (pure logic),
and shard bisect / confirm against a fake replay engine whose collapse is
caused by exactly one poison shard -- the same causal structure as the real
pipeline, without the cost of real replays.
"""
import unittest

import torch

from trainsect.bisect import bisect_checkpoints, confirm, shard_bisect
from trainsect.dataloader import IndexedDataLoader


def _predicate_from_transition(transition_step):
    """P(c) = True for c <= transition, False after."""
    def p(c):
        return (c <= transition_step, None)
    return p


class TestCheckpointBisect(unittest.TestCase):
    def test_finds_transition(self):
        ckpts = [0, 20, 40, 60, 80, 100]
        res = bisect_checkpoints(ckpts, _predicate_from_transition(40), 120)
        self.assertTrue(res["ok"])
        self.assertEqual(res["bracket"], (40, 60))

    def test_transition_at_first_interval(self):
        ckpts = [0, 20, 40, 60]
        res = bisect_checkpoints(ckpts, _predicate_from_transition(0), 80)
        self.assertEqual(res["bracket"], (0, 20))

    def test_transition_at_last_interval(self):
        ckpts = [0, 20, 40, 60]
        res = bisect_checkpoints(ckpts, _predicate_from_transition(40), 80)
        self.assertEqual(res["bracket"], (40, 60))

    def test_all_true_bracket_is_final_interval(self):
        ckpts = [0, 20, 40]
        res = bisect_checkpoints(ckpts, _predicate_from_transition(10_000), 80)
        self.assertTrue(res["ok"])
        self.assertEqual(res["bracket"], (40, 80))

    def test_no_reproduce_from_earliest_aborts(self):
        ckpts = [0, 20, 40]
        res = bisect_checkpoints(ckpts, _predicate_from_transition(-1), 80)
        self.assertFalse(res["ok"])
        self.assertIn("does NOT reproduce", res["reason"])


class FakeEngine:
    """Collapse iff at least one KEPT (non-substituted) poison-shard batch
    remains in the replayed range. Substitution is detected by object
    identity: replaced batches are different tensor objects."""

    def __init__(self, loader, poison_shard):
        self.loader = loader
        self.poison_shard = poison_shard

    def replay(self, lo, hi, replace=None):
        collapse = False
        for step in range(lo, hi):
            batch = self.loader[step]
            x, _y = batch.x, batch.y
            if replace is not None:
                x, _y = replace(step, x, _y, batch.shard_id)
            kept_original = x is batch.x
            if kept_original and batch.shard_id == self.poison_shard:
                collapse = True
        evals = {hi: 100.0 if collapse else 0.1}
        return {}, evals, "fakehash"


def _loader_with_poison(poison_shard=2, n_shards=4, per_shard=5):
    raw = []
    for shard in range(n_shards):
        for i in range(per_shard):
            raw.append((torch.full((2, 2), float(shard)), torch.zeros(2, 1), shard))
    return IndexedDataLoader.from_raw(raw)


class TestShardBisectAndConfirm(unittest.TestCase):
    def test_shard_bisect_finds_poison(self):
        loader = _loader_with_poison(poison_shard=2)
        engine = FakeEngine(loader, poison_shard=2)
        found = shard_bisect(engine, loader, 0, 20, eval_threshold=3.0)
        self.assertEqual(found, 2)

    def test_shard_bisect_finds_first_shard_poison(self):
        loader = _loader_with_poison(poison_shard=0)
        engine = FakeEngine(loader, poison_shard=0)
        # clean_shard must differ from poison for substitutes to be benign:
        # use shard 1 as the benign source via a loader whose shard 0 is
        # poison -- benign_batches(1) still works since substitutes are
        # recognised by identity, not content.
        found = shard_bisect(engine, loader, 0, 20, eval_threshold=3.0,
                             clean_shard=1)
        self.assertEqual(found, 0)

    def test_confirm_necessity_and_sufficiency(self):
        loader = _loader_with_poison(poison_shard=2)
        engine = FakeEngine(loader, poison_shard=2)
        conf = confirm(engine, loader, 0, 20, 2, eval_threshold=3.0)
        self.assertTrue(conf["necessity"])
        self.assertTrue(conf["sufficiency"])

    def test_confirm_fails_for_innocent_shard(self):
        loader = _loader_with_poison(poison_shard=2)
        engine = FakeEngine(loader, poison_shard=2)
        conf = confirm(engine, loader, 0, 20, 1, eval_threshold=3.0)
        # Excluding innocent shard 1 leaves poison -> collapse persists.
        self.assertFalse(conf["necessity"])
        # Keeping only innocent shard 1 removes poison -> no collapse.
        self.assertFalse(conf["sufficiency"])


class FakePredicateEngine:
    """checkpoint_eval_loss from a dict; replay always reproduces collapse."""

    def __init__(self, health_by_step):
        self.health_by_step = health_by_step

    def checkpoint_eval_loss(self, step):
        return self.health_by_step[step]

    def replay(self, lo, hi, replace=None):
        return {}, {hi: 100.0}, "fakehash"


class TestPredicateHealth(unittest.TestCase):
    def test_untrained_checkpoint_is_not_already_collapsed(self):
        # ckpt 0 eval 4.3 > threshold 3.0, but the model was never healthy
        # before it (no prior log points) -> NOT already collapsed, and
        # replay reproduces the collapse -> P(0) is True.
        from trainsect.bisect import make_predicate

        engine = FakePredicateEngine({0: 4.3})
        pred = make_predicate(engine, 120, 3.0,
                              loss_log=[[0, 4.3]], baseline=0.002)
        self.assertTrue(pred(0)[0])

    def test_previously_healthy_then_collapsed_is_already_collapsed(self):
        from trainsect.bisect import make_predicate

        engine = FakePredicateEngine({90: 4.08})
        pred = make_predicate(engine, 120, 3.0,
                              loss_log=[[0, 4.3], [30, 0.006], [60, 0.002],
                                        [90, 4.08]],
                              baseline=0.002)
        self.assertFalse(pred(90)[0])


if __name__ == "__main__":
    unittest.main()
