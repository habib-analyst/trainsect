"""Evidence report: repro commands + JSON/text round-trip."""
import unittest

from trainsect.evidence import EvidenceReport


def _report(**over):
    base = dict(
        run_dir="/tmp/run",
        verdict="CONFIRMED (causal)",
        confirmed_causal=True,
        fidelity_bit_exact=True,
        bracket=[40, 80],
        trigger_step=40,
        shard=2,
        batch_fingerprint="e3768bc902f668d7",
        confirmation={"necessity": True, "sufficiency": True},
    )
    base.update(over)
    return EvidenceReport(**base)


class TestEvidence(unittest.TestCase):
    def test_repro_commands_name_shard_and_batch(self):
        cmds = _report().repro_commands()
        self.assertEqual(len(cmds), 3)
        self.assertIn("--exclude-shard 2", cmds[0])
        self.assertIn("collapse vanishes", cmds[0])
        self.assertIn("--only-shard 2", cmds[1])
        self.assertIn("collapse reproduces", cmds[1])
        self.assertIn("--only-batch e3768bc902f668d7", cmds[2])

    def test_no_repro_without_bracket(self):
        self.assertEqual(_report(bracket=[], shard=None).repro_commands(), [])

    def test_text_report_is_actionable(self):
        text = _report().to_text()
        for needle in ("CONFIRMED", "shard-2", "e3768bc902f668d7",
                       "exclude-shard", "only-shard", "necessity", "sufficiency"):
            self.assertIn(needle, text)

    def test_json_round_trip(self):
        import json

        d = json.loads(_report().to_json())
        self.assertEqual(d["shard"], 2)
        self.assertEqual(d["batch_fingerprint"], "e3768bc902f668d7")
        self.assertTrue(d["confirmed_causal"])


if __name__ == "__main__":
    unittest.main()
