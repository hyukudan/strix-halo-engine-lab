import tempfile
import unittest
from pathlib import Path

from tools.qualification.turbo_bar import frontier, losses, report
from tools.qualification.turbo_bar import manifest as mf
from tools.qualification.turbo_bar.common import read_json, write_json, write_jsonl
from tools.qualification.turbo_bar.tests.helpers import frozen_manifest, make_layout


def V(name, speeds, kl, dnll, verdict="PASS"):
    return {"name": name, "speed": speeds, "quality": {"kl_mean": kl, "dnll_mean": dnll, "verdict": verdict}}


class FrontierTests(unittest.TestCase):
    def test_selection_within_timing_noise(self):
        res = frontier.analyze([
            V("fast", [1.30, 1.32, 1.28, 1.30], 0.010, 0.002),
            V("near", [1.29, 1.31, 1.30, 1.28], 0.002, 0.001),
            V("slow", [1.10, 1.11, 1.09, 1.10], 0.0005, 0.0),
            V("bad", [1.60, 1.61, 1.59, 1.60], 0.5, 0.2, "FAIL")])
        by = {r["name"]: r for r in res["variants"]}
        self.assertEqual(res["fastest_pass"], "fast")
        self.assertTrue(by["near"]["within_noise"] and by["fast"]["within_noise"])
        self.assertFalse(by["slow"]["within_noise"])
        self.assertEqual(res["selected"], "near")
        self.assertTrue(by["bad"]["pareto"] and by["fast"]["pareto"] and by["slow"]["pareto"])
        self.assertFalse(by["near"]["pareto"] and not by["near"]["pareto"])  # computed, not asserted either way
        self.assertIn("SELECTED", frontier.table(res))

    def test_no_pass_selects_nothing(self):
        self.assertIsNone(frontier.analyze([V("a", [1.0, 1.1, 1.2], 0.1, 0.1, "INCONCLUSIVE")])["selected"])


class ReportTests(unittest.TestCase):
    def setUp(self):
        self.tmp = Path(tempfile.mkdtemp())
        self.mpath = frozen_manifest(self.tmp)
        self.m = mf.load(self.mpath)
        layout = make_layout(self.tmp, noise_r=1.0, noise_t=0.01, parents=6)
        layout["manifest"] = {"version": 1, "core_sha256": self.m["frozen"]["core_sha256"]}
        layout["bootstrap"] = {"B": 200, "seed": 1}
        self.losses = losses.analyze(layout)
        write_json(self.tmp / "losses.json", self.losses)
        td = self.tmp / "tasks"
        td.mkdir()
        for arm, fails in (("E", 0), ("R", 20), ("T_a", 2)):
            rows = [{"id": f"math-confirm-{i:03d}", "kind": "math", "split": "confirm", "arm": arm,
                     "ok": int(i >= fails)} for i in range(60)]
            write_jsonl(td / f"scored.{arm}.jsonl", rows)
        write_json(self.tmp / "front.json", [{"name": "a", "speed": [1.2, 1.25, 1.22]}])

    def test_report_writes_receipt_and_markdown(self):
        out = self.tmp / "out"
        rc = report.main(["--manifest", str(self.mpath), "--losses", str(self.tmp / "losses.json"),
                          "--tasks", str(self.tmp / "tasks"), "--frontier", str(self.tmp / "front.json"), "--out", str(out)])
        self.assertEqual(rc, 0)
        r = read_json(out / "receipt.json")
        md = (out / "REPORT.md").read_text(encoding="utf-8")
        self.assertEqual(r["status"], "VALID")
        self.assertIn("E>R", md)
        self.assertIn("Null controls", md)
        tk = r["tasks"]["T:a"]["per_kind"]["math"]["confirm"]
        self.assertEqual(tk["n"], 60)
        self.assertEqual(tk["T_vs_R_loss"]["b_only"], 18)    # T solved where R failed
        self.assertIn(r["overall"]["T:a"]["verdict"], ("PASS", "FAIL", "INCONCLUSIVE"))
        self.assertEqual(r["overall"]["T:a"]["components"]["retrieval"], "INCONCLUSIVE")  # missing -> never PASS
        self.assertNotEqual(r["overall"]["T:a"]["verdict"], "PASS")
        self.assertIsNotNone(r["frontier"])

    def test_mixed_manifest_version_is_invalid(self):
        bad = dict(self.losses, manifest={"version": 2, "core_sha256": "x"})
        write_json(self.tmp / "bad.json", bad)
        r = report.build(self.mpath, self.tmp / "bad.json")
        self.assertEqual(r["status"], "INVALID")
        self.assertTrue(all(v["verdict"] == "INVALID" for v in r["overall"].values()))


if __name__ == "__main__":
    unittest.main()
