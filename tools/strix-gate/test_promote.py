"""promote.py: plan before candidate, witness + gate v2 promotion receipt, ABBA coverage, draft from the engine's
promotion key and the sealed manifest entry (check_promotion_manifest)."""
import contextlib
import hashlib
import io
import json
import pathlib
import sys
import tempfile
import unittest

import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
import promote  # noqa: E402
import check_promotion_manifest as cpm  # noqa: E402

V, N, R = 32, 80, 64
PRODUCT = "gfx1151|Radeon 8060S|cu40|mem131072|clk2900000|hip60342|drv1"


def promotion_reference(pack, prompt=1024):
    # The spelling profile::LadderArms::promotion_over() emits (tests/profile_ladder_tests.cpp promotion_key()).
    return (f"ladder-arm/v4;exec/v1 backend=hip device={PRODUCT} prompt={prompt} context=8192 lanes=1 chunk=512 "
            f"prefill_rows=0 decode_rows=0 verify_rows=0 generated=0 settings=pack={pack}|route=cli|ranks=1|tp2;"
            "fixture-w4a4@1=off{v1};candidate=fixture-w4a4@1=a8{v1};tier=fast")


class Promote(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.d = pathlib.Path(self.tmp.name)
        rng = np.random.default_rng(11)
        self.cases = [f"w{i:02d}" for i in range(13)]
        man = []
        self.cand = {}
        for c in self.cases:
            ids = rng.integers(0, V, N)
            (self.d / f"{c}.ids").write_text(" ".join(map(str, ids)), encoding="utf-8")
            man.append(f"{c}\t{N}\t{c}.ids")
            ref = 3.0 * rng.normal(size=(R, V))
            for arm, s in (("exact", 0.0), ("n0", 0.03)):
                (self.d / f"{c}-{arm}.f32").write_bytes((ref + s * rng.normal(size=ref.shape)).astype("<f4").tobytes())
            self.cand[c] = (ref + 0.02 * rng.normal(size=ref.shape)).astype("<f4").tobytes()
        (self.d / "m.tsv").write_text("\n".join(man) + "\n", encoding="utf-8")
        self.pack = self.d / "pack"
        (self.pack / "sub").mkdir(parents=True)
        (self.pack / ".cache").mkdir()
        (self.pack / "model.safetensors").write_bytes(b"weights")
        (self.pack / "sub" / "config.json").write_bytes(b"{}")
        (self.pack / ".cache" / "download.lock").write_bytes(b"noise")
        self.spec = {
            "schema": "strix-promotion-band/v1", "id": "fixture-band", "source": "include/strix/profiles/fixture.hpp",
            "lever": "fixture-w4a4@1", "tier": "fast", "phase": "prefill", "device": "gfx1151",
            "contexts": [1024, 2048], "lanes": [1, 1], "noise": 0.02, "revision": "a" * 40, "out": ".",
            "capture": {"format": "head", "manifest": "m.tsv", "head_rows": R, "cases": self.cases,
                        "classes": {c: "prose" for c in self.cases},
                        "arms": {"ref": "exact", "cand": "fast", "null": ["n0"]}},
            "execution": {"binary_sha256": "c" * 64, "pack": "pack",
                          "configs": {"reference": {"profile": "exact"}, "candidate": {"profile": "fast"}},
                          "controls": {"n0": {"order": "test-order@1", "applications": R}}},
        }
        self.spec_path = self.d / "band.json"
        self.spec_path.write_text(json.dumps(self.spec), encoding="utf-8")

    def tearDown(self):
        self.tmp.cleanup()

    def band(self):
        return promote.Band(self.spec_path)

    def capture_candidate(self):
        for c in self.cases:
            (self.d / f"{c}-fast.f32").write_bytes(self.cand[c])

    def test_pack_identity_matches_the_engine(self):
        # Known answers of tests/model_content_hash_tests.cpp (strix::pack_content_identity).
        self.assertEqual(promote.pack_identity(self.pack),
                         "d90fb2f77c1417081f16b123f054938d0597156d879eb65ed8450dbfbdce95a6")
        self.assertEqual(promote.pack_identity(self.pack / "model.safetensors"),
                         "ae0cc1e3f634297485d68cbbd048bfdcbb16ffa889cc441c44f12009c4d6d5ab")

    def test_plan_precedes_the_candidate(self):
        band = self.band()
        sha = promote.plan(band)
        doc = json.loads(band.plan_path.read_text(encoding="utf-8"))
        self.assertEqual(sha, hashlib.sha256(band.plan_path.read_bytes()).hexdigest())
        self.assertEqual(set(doc["windows"]), set(self.cases))
        self.assertEqual(len(doc["reference_nll"]["prose"]), len(self.cases))
        self.assertGreater(doc["null_envelope"]["max_kl"], 0.0)
        self.capture_candidate()
        with self.assertRaisesRegex(promote.PromotionError, "before the plan"):
            promote.plan(band)

    def test_band_seals_a_manifest_entry(self):
        band = self.band()
        promote.plan(band)
        self.capture_candidate()
        promote.witness(band)
        with contextlib.redirect_stdout(io.StringIO()):
            rc, state = promote.gate(band)
        self.assertEqual((rc, state), (0, "PASS"))
        gate = json.loads(band.gate_path.read_text(encoding="utf-8"))
        pack = promote.pack_identity(self.pack)
        self.assertEqual(gate["promotion"]["candidate_pack_sha256"], pack)
        receipt = self.d / "run.log"
        profile = {"profile": "fast", "levers": {"fixture-w4a4@1": {"promotion_reference": promotion_reference(pack)}}}
        startup = {"profile": "fast", "levers": {"fixture-w4a4@1": {"value": "a8"}}}  # no prompt yet: no key
        receipt.write_text(f"profile {json.dumps(startup)}\nloaded\nprofile {json.dumps(profile)}\n", encoding="utf-8")
        draft = promote.draft(band, receipt)
        self.assertEqual((draft["value"], draft["identity"]), ("a8", "v1"))
        samples = self.d / "samples.json"
        corners = [(1024, 1), (2048, 1)]
        samples.write_text(json.dumps([{"prompt": p, "lanes": n, "abba_ms": [100, 80, 82, 101]} for p, n in corners[:1]]))
        with self.assertRaisesRegex(promote.PromotionError, "miss"):
            promote.abba(band, samples)
        samples.write_text(json.dumps([{"prompt": p, "lanes": n, "abba_ms": [100, 80, 82, 101]} for p, n in corners]))
        ratios = promote.abba(band, samples)
        self.assertGreater(min(ratios.values()), 1.2)
        manifest = self.d / "promotion-receipts.json"
        manifest.write_text(json.dumps({"schema": "strix-profile-promotion-manifest/v1", "promotions": []}))
        entry = promote.seal(band, manifest)
        self.assertEqual(json.loads(manifest.read_text(encoding="utf-8"))["promotions"], [entry])
        self.assertIn('.promotion = id', cpm.header([entry]))
        with self.assertRaisesRegex(promote.PromotionError, "already sealed"):
            promote.seal(band, manifest)

    def test_draft_refuses_a_run_without_the_key_or_another_pack(self):
        band = self.band()
        receipt = self.d / "run.json"
        receipt.write_text(json.dumps({"profile": "fast", "levers": {"fixture-w4a4@1": {}}}), encoding="utf-8")
        with self.assertRaisesRegex(promote.PromotionError, "no promotion_reference"):
            promote.draft(band, receipt)
        other = {"profile": "fast", "levers": {"fixture-w4a4@1": {"promotion_reference": promotion_reference("e" * 64)}}}
        receipt.write_text(json.dumps(other), encoding="utf-8")
        with self.assertRaisesRegex(promote.PromotionError, "pack identity differs"):
            promote.draft(band, receipt)

    def test_witness_binds_controls_and_configs(self):
        band = self.band()
        promote.plan(band)
        self.capture_candidate()
        promote.witness(band)
        doc = json.loads(band.witness_path.read_text(encoding="utf-8"))
        record = doc["cases"][self.cases[0]]
        self.assertIsNone(record["exact"]["control"])
        self.assertEqual(record["n0"]["control"], {"order": "test-order@1", "applications": R})
        self.assertNotEqual(record["fast"]["numerical_config_sha256"], record["exact"]["numerical_config_sha256"])
        self.assertEqual(record["n0"]["numerical_config_sha256"], record["exact"]["numerical_config_sha256"])
        self.spec["execution"]["controls"] = {}
        self.spec_path.write_text(json.dumps(self.spec), encoding="utf-8")
        with self.assertRaisesRegex(promote.PromotionError, "null control"):
            promote.witness(self.band())

    def test_route_nudge_null_promotes(self):
        # The MoE route-file null (one FP32 ulp on layer L) is recorded as its receipt reports it, not as an order.
        nudge = {"layer": 3, "seed": 1, "nudged": 262144}
        self.spec["execution"]["controls"] = {"n0": nudge}
        self.spec_path.write_text(json.dumps(self.spec), encoding="utf-8")
        band = self.band()
        promote.plan(band)
        self.capture_candidate()
        promote.witness(band)
        with contextlib.redirect_stdout(io.StringIO()):
            self.assertEqual(promote.gate(band), (0, "PASS"))
        accepted = promote.v2.witnessed_null_control
        self.assertTrue(accepted({"order": "rms-swap@1", "applications": 2}) and accepted(nudge))
        for bad in (None, {}, {"order": " ", "applications": 1}, {"order": "x", "applications": 0},
                    {**nudge, "nudged": 0}, {**nudge, "layer": -1}, {**nudge, "extra": 1}, {"layer": 3, "seed": 1}):
            self.assertFalse(accepted(bad), bad)


if __name__ == "__main__":
    unittest.main()
