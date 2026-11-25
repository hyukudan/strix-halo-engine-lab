"""Host checks for the native route gate's evidence comparison."""

import unittest

from route_gate import ARMS, compare, overlap, signature


class ReceiptTest(unittest.TestCase):
    def setUp(self):
        self.phase = dict(exl3_identity="g4-exl3-v1", exl3_identity_selection="auto", exl3_max_streams=8,
                          finish_reason="length", dflash_requested="off", prefill_identity="prefill-exact",
                          logits_rows=2, logits_fnv64="123456", state_digest="0a" * 32, frontier=520,
                          model_hash="model", rope_hash="rope", tokenizer_hash="tokenizer",
                          decode_identity="g4-v1", sampling_identity="sampling-v1")

    def record(self, ids=(100, 101), **changes):
        return signature(ids, self.phase | changes, "auto", "v1", 8, 2)

    def test_m5_cores_run_as_explicit_values(self):
        # Automatic M5 follows the session's streams, so CLI-versus-server evidence uses explicit values only.
        self.assertEqual(ARMS["M5"], (("forced", "v1"), ("forced", "v3")))
        self.assertEqual(ARMS["M4"][0], ("auto", "v3"))

    def test_complete_evidence(self):
        reference = self.record()
        compare("same request", reference, self.record())
        for field, changed in (("ids", {"ids": (100, 102)}), ("logits_fnv64", {"logits_fnv64": "654321"}),
                               ("state_digest", {"state_digest": "b0" * 32}), ("frontier", {"frontier": 521})):
            with self.subTest(field=field), self.assertRaisesRegex(ValueError, field):
                compare("different request", reference, self.record(**changed))

    def test_incomplete_or_historical_identity_is_refused(self):
        for changes in ({"ids": (100,)}, {"logits_rows": 1}, {"logits_fnv64": ""}, {"state_digest": ""},
                        {"exl3_identity": "g4-exl3-v3"}, {"exl3_identity_selection": "forced"},
                        {"exl3_max_streams": 1}):
            with self.subTest(changes=changes), self.assertRaises(ValueError):
                self.record(**changes)


class OverlapTest(unittest.TestCase):
    def test_scheduled_counters_alone_are_refused(self):
        scheduled = [dict(max_batch=8, steps_sharing_prefill=3)] * 8
        with self.assertRaisesRegex(ValueError, "execution receipts"):
            overlap(scheduled)

    def test_execution_receipts(self):
        row = dict(max_batch=8, steps_sharing_prefill=3, executed_max_streams=8, executed_beside_prefill_calls=1)
        overlap([row, row | dict(executed_beside_prefill_calls=0)])
        with self.assertRaisesRegex(ValueError, "8-stream"):
            overlap([row | dict(executed_max_streams=7)])
        with self.assertRaisesRegex(ValueError, "beside a sliced prefill"):
            overlap([row | dict(executed_beside_prefill_calls=0)])


if __name__ == "__main__":
    unittest.main()
