import copy
import json
from pathlib import Path
import tempfile
import unittest

from check_promotion_manifest import GATE_FIELDS, header, seal, validate, validate_legacy_limits, validate_sources


class PromotionTests(unittest.TestCase):
    def setUp(self):
        self.draft = {
            "id": "fixture-prefill", "source": "include/strix/profiles/fixture.hpp",
            "tier": "fast", "phase": "prefill", "device": "gfx1151", "lever": "rows",
            "value": "512", "identity": "v2", "contexts": [1024, 2048], "lanes": [1, 2], "noise": 0.02,
            "revision": "a" * 40,
            "reference": "ladder-arm/v4;exec/v1 backend=hip device=gfx1151|Device A|cu40|mem64|clk1000|hip7|drv7 "
            "prompt=1024 context=8192 lanes=1 chunk=512 prefill_rows=512 decode_rows=1 verify_rows=0 generated=64 "
            f"settings=pack={'b' * 64}|route=cli|ranks=1;other={{v1}},rows=64{{v1}};candidate=rows=512{{v2}};tier=fast",
        }
        self.proof = dict.fromkeys(GATE_FIELDS, "b" * 64)
        self.gate = {"schema": "strix-fast-gate-v2/v1", "pooled": {"state": "PASS", "pass": True,
                     "frozen_qualification": {"sha256": self.proof["plan_sha256"]}},
                     "promotion": {"schema": "strix-gate-promotion/v1", **self.proof}}
        self.timing = {"schema": "strix-profile-abba/v1", "candidate_pack_sha256": "b" * 64,
                       "binary_sha256": "b" * 64, "reference_config_sha256": "b" * 64,
                       "candidate_config_sha256": "b" * 64, "reference": self.draft["reference"], "revision": "a" * 40,
                       "samples": [{"prompt": p, "lanes": lanes, "abba_ms": [100, 80, 85, 101]}
                                   for p in (1024, 2048) for lanes in (1, 2)]}

    def test_seals_and_renders_a_measured_row(self):
        with tempfile.TemporaryDirectory() as d:
            gate, timing = (Path(d) / path for path in ("gate.json", "timing.json"))
            timing.write_text(json.dumps(self.timing))
            gate.write_text(json.dumps(self.gate))
            entry = seal(self.draft, gate, timing)
            self.assertIn('.promotion = id', header([entry]))
            validate_sources([entry], {entry["id"]: entry["source"]}, {}, {})
            with self.assertRaisesRegex(ValueError, "bindings"):
                validate_sources([entry], {}, {}, {})
            for field in ("promotion", "pooled"):
                with self.subTest(field=field):
                    changed = {**self.gate, field: {}}
                    gate.write_text(json.dumps(changed))
                    with self.assertRaisesRegex(ValueError, "promotion PASS"):
                        seal(self.draft, gate, timing)
            for field, value in (("reference", "wrong"), ("binary_sha256", "c" * 64),
                                 ("candidate_config_sha256", "c" * 64)):
                with self.subTest(field=field):
                    gate.write_text(json.dumps(self.gate))
                    timing.write_text(json.dumps({**self.timing, field: value}))
                    with self.assertRaisesRegex(ValueError, "timing capture differs"):
                        seal(self.draft, gate, timing)

    def test_rejects_poor_coverage_and_scope(self):
        entry = {**self.draft, "gate": {**self.proof, "receipt_sha256": "c" * 64},
                 "timing": {"samples": self.timing["samples"], "receipt_sha256": "d" * 64}}
        self.assertGreater(validate(entry), 1.02)
        for old, new, error in (("|ranks=1", "", "rank count"), ("|cu40", "", "full device"),
                                ("candidate=rows=512{v2}", "candidate=rows=512{v1}", "candidate"),
                                ("pack=" + "b" * 64, "pack=" + "a" * 64, "gate pack")):
            with self.subTest(old=old):
                with self.assertRaisesRegex(ValueError, error):
                    validate({**entry, "reference": entry["reference"].replace(old, new)})
        for samples in (entry["timing"]["samples"][:-1], [{**s, "abba_ms": [1, 1, 1, 1]} for s in self.timing["samples"]]):
            broken = copy.deepcopy(entry)
            broken["timing"]["samples"] = samples
            with self.assertRaisesRegex(ValueError, "gain beyond noise"):
                validate(broken)
        with self.assertRaisesRegex(ValueError, "new incumbent receipt"):
            validate_sources([], {}, {"fixture": {"incumbent": 1}}, {})

    def test_historical_allowances_only_shrink(self):
        validate_legacy_limits({"file": {"band": 1}}, {"file": {"band": 2}})
        with self.assertRaisesRegex(ValueError, "only shrink"):
            validate_legacy_limits({"file": {"band": 2}}, {"file": {"band": 1}})


if __name__ == "__main__":
    unittest.main()
