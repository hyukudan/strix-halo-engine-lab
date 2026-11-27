#!/usr/bin/env python3
"""Host tests for the best-on-record gate."""

from copy import deepcopy
from decimal import Decimal
import json
from pathlib import Path
import subprocess
import sys
from tempfile import TemporaryDirectory
import unittest

import check_best_on_record as gate


EXACT = "qwen3.8/one/exact/cli/prefill-8k"
FAST = EXACT.replace("/exact/", "/fast/")
TURBO = EXACT.replace("/exact/", "/turbo/")
SERVED = EXACT.replace("/cli/", "/served/")
EVIDENCE = dict(commit="ca2de6ae8", date="2026-11-25", receipt="README.md#results")


def measurement(value, prompt=8192, *, clock="prefill"):
    return dict(**EVIDENCE, value=value, workload=dict(prompt_tokens=prompt, generated_tokens=128,
                                                      requests=1, timing=clock))


def ledger():
    return dict(version=1, records={EXACT: measurement(100)}, acceptances=[])


def campaign(rows):
    return dict(version=1, runs={"default": rows})


class Comparisons(unittest.TestCase):
    def test_noise_edges_and_threshold(self):
        for value, verdict in ((96.99, "BELOW RECORD"), (97, "WITHIN NOISE"), (100, "WITHIN NOISE"),
                               (103, "WITHIN NOISE"), (103.01, "ABOVE RECORD")):
            with self.subTest(value=value):
                results, _ = gate.compare(ledger(), campaign({EXACT: measurement(value)}), Decimal(3), set())
                self.assertEqual(results[0]["status"], verdict)

    def test_missing_no_record_and_failed(self):
        result, _ = gate.compare(ledger(), campaign({FAST: measurement(120), TURBO: dict(**EVIDENCE, failure="timeout")}),
                                 Decimal(3), {EXACT})
        self.assertEqual({r["row"]: r["status"] for r in result},
                         {EXACT: "MISSING", FAST: "NO RECORD", TURBO: "FAILED MEASUREMENT"})

    def test_comparable_tiers_and_routes(self):
        rows = {EXACT: measurement(100), FAST: measurement(99), TURBO: measurement(98, 9000),
                SERVED: measurement(97.9, 7950)}
        _, diagnostics = gate.compare(ledger(), campaign(rows), Decimal(3), set())
        self.assertEqual({x["row"]: x["status"] for x in diagnostics},
                         {FAST: "tier slower than the tier below", SERVED: "served more than 2% below the CLI"})
        rows[SERVED]["value"] = 98
        _, diagnostics = gate.compare(ledger(), campaign(rows), Decimal(3), set())
        self.assertEqual(len(diagnostics), 1)

    def test_mismatched_clock_cannot_replace_record(self):
        data = ledger()
        results, _ = gate.compare(data, campaign({EXACT: measurement(150, clock="ttft")}), Decimal(3), set())
        self.assertTrue(results[0]["timing_differs"])
        gate.propose(data, results, {})
        self.assertEqual(data, ledger())

    def test_raises_and_acceptance_history(self):
        data = ledger()
        results, _ = gate.compare(data, campaign({EXACT: measurement(110), FAST: measurement(120)}), Decimal(3), set())
        gate.propose(data, results, {})
        self.assertEqual(data["records"][EXACT]["value"], 110)
        self.assertEqual(data["records"][FAST]["value"], 120)
        previous = deepcopy(data)
        results, _ = gate.compare(data, campaign({EXACT: measurement(90)}), Decimal(3), set())
        gate.propose(data, results, {EXACT: "The new route restores deterministic greedy IDs."})
        self.assertEqual(results[0]["status"], "ACCEPTED REGRESSION")
        self.assertEqual(data["acceptances"][0]["previous"]["value"], 110)
        self.assertEqual(data["acceptances"][0]["replacement"]["value"], 90)
        gate.validate_ledger(data)
        gate.check_history(data, previous)
        data["acceptances"].clear()
        with self.assertRaisesRegex(ValueError, "lowered without"):
            gate.check_history(data, previous)
        data["records"].pop(EXACT)
        with self.assertRaisesRegex(ValueError, "record removed"):
            gate.check_history(data, previous)


class Validation(unittest.TestCase):
    def test_invalid_measurements(self):
        for row in (measurement(True), measurement(float("nan")), measurement(0), measurement(100, 2048)):
            with self.subTest(row=row), self.assertRaises(ValueError):
                gate.validate_campaign(campaign({EXACT: row}))
        with self.assertRaisesRegex(ValueError, "duplicate JSON key"):
            gate.read_json('{"version": 1, "version": 2}')
        with self.assertRaisesRegex(ValueError, "mixed revisions"):
            gate.validate_campaign(campaign({EXACT: measurement(100),
                                             FAST: {**measurement(110), "commit": "123abcd"}}))

    def test_receipt_must_be_public_and_resolve(self):
        for receipt in ("../README.md", "README.md#nonexistent", "docs/not-present.md"):
            data = ledger()
            data["records"][EXACT]["receipt"] = receipt
            with self.subTest(receipt=receipt), self.assertRaises(ValueError):
                gate.validate_ledger(data)

    def test_catalog_claim_must_link_to_own_engine(self):
        row = dict(name="Measured route", engine="qwen", default="auto", gain="8K prefill 95 -> 100 tok/s.")
        records = ledger()["records"]
        with self.assertRaisesRegex(ValueError, "classify"):
            gate.check_catalog([row], records)
        with self.assertRaisesRegex(ValueError, "unknown record"):
            gate.check_catalog([{**row, "record_rows": [FAST]}], records)
        with self.assertRaisesRegex(ValueError, "different engine"):
            gate.check_catalog([{**row, "engine": "minimax", "record_rows": [EXACT]}], records)
        gate.check_catalog([{**row, "record_rows": [EXACT]}], records)
        gate.check_catalog([{**row, "default": "OFF"}], records)
        with self.assertRaisesRegex(ValueError, "removed record coverage"):
            gate.check_catalog([{**row, "gain": "No throughput claim."}], records,
                               [{**row, "record_rows": [EXACT]}])

    def test_shipped_ledger_catalog_and_view(self):
        data = gate.load(gate.RECORDS)
        gate.validate_ledger(data)
        gate.check_catalog(gate.load(gate.CATALOG), data["records"])
        self.assertEqual(gate.VIEW.read_text(encoding="utf-8"), gate.view(data))
        gate.validate_campaign(gate.load(gate.RECORDS.parent / "example-campaign.json"))


class CommandLine(unittest.TestCase):
    def test_write_is_blocked_until_each_loss_is_accepted(self):
        with TemporaryDirectory() as temporary:
            path = Path(temporary)
            record_file = path / "records.json"
            campaign_file = path / "campaign.json"
            original = json.dumps(ledger())
            record_file.write_text(original, encoding="utf-8")
            campaign_file.write_text(json.dumps(campaign({EXACT: measurement(90), FAST: measurement(110)})),
                                     encoding="utf-8")
            command = [sys.executable, "-B", gate.__file__, str(campaign_file), "--ledger", str(record_file), "--write"]
            result = subprocess.run(command, capture_output=True, text=True)
            self.assertEqual(result.returncode, 1, result.stderr)
            self.assertIn("BELOW RECORD", result.stdout)
            self.assertEqual(record_file.read_text(encoding="utf-8"), original)
            result = subprocess.run([*command, "--accept-regression", EXACT, "Restores the model's token identity."],
                                    capture_output=True, text=True)
            self.assertEqual(result.returncode, 0, result.stderr)
            saved = gate.load(record_file)
            self.assertEqual(saved["records"][EXACT]["value"], 90)
            self.assertEqual(saved["records"][FAST]["value"], 110)
            self.assertEqual(len(saved["acceptances"]), 1)
            self.assertEqual(record_file.with_suffix(".md").read_text(encoding="utf-8"), gate.view(saved))

    def test_concurrent_writers_cannot_read_the_same_floor(self):
        with TemporaryDirectory() as temporary:
            path = Path(temporary) / "ledger.json"
            with gate.write_lock(path):
                with self.assertRaises(FileExistsError), gate.write_lock(path):
                    self.fail("two writers hold the ledger lock")
            self.assertFalse(path.with_name("ledger.json.lock").exists())


if __name__ == "__main__":
    unittest.main()
