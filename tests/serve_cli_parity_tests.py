#!/usr/bin/env python3
"""Host tests of tools/serve_cli_parity.py: every resolver print format flattens to the same key space, an
unexplained server/CLI difference fails, a pinned explanation stops applying when a value moves, the shipped
explained table is well formed, and the ratio report fails below the bound (no engine, no GPU)."""
import contextlib
import io
import json
import os
import sys
import tempfile
import unittest

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "tools"))
import serve_cli_parity as sp  # noqa: E402

GLM_CLI = """device: 0 of 1
effective prefill=gemv (auto)
effective max-rows=auto-8192 (default)
effective env STRIX_GLM_PF_EXACT_X=1 (default, item pf_exact_x)
prefill_chunk tokens=0 rows=0
"""
GLM_SERVE = GLM_CLI.replace("auto-8192 (default)", "4096 (auto-4096-serve)") + "serve slots=8 prefill-chunk=512 kmax=4\n"
DS_CLI = "app rank\nenv STRIX_DS41_X=1\nunset STRIX_OLD\narg --tp-attention\narg --rank\narg 0\nenv STRIX_DIST_T 600 (default)\n"
DS_SERVE = "app serve\nenv STRIX_DS41_X=1\nunset STRIX_OLD\narg --slots\narg 8\narg --rank\narg 0\nenv STRIX_DIST_T 600 (default)\n"
JSON_CLI = ('{"route":"cli","admitted":{"tile_rows":2048,"pipe":255},"argv":["m.gguf","1"]}\n'
            'strix-route-receipt {"schema":"strix-route-receipt-v1","levers":'
            '[{"name":"qwen.prefill_auto","effective":"1","reason":"selected"}]}\n')
JSON_SERVE = JSON_CLI.replace('"cli"', '"http-single"').replace('"pipe":255', '"pipe":18')


def table(rows, engine="e"):
    return {"schema": sp.SCHEMA, "engines": {"*": [{"key": "app", "reason": "names the entry point itself"},
                                                   {"key": "route", "reason": "names the entry point itself"}],
                                             engine: rows}}


class Parse(unittest.TestCase):
    def test_effective_lines(self):
        c = sp.parse_config(GLM_SERVE)
        self.assertEqual(c["prefill"], "gemv")
        self.assertEqual(c["max-rows"], "4096")
        self.assertEqual(c["env.STRIX_GLM_PF_EXACT_X"], "1")
        self.assertEqual(c["serve.slots"], "8")
        self.assertEqual(c["prefill_chunk.tokens"], "0")
        self.assertFalse(any(key.startswith("device") for key in c))

    def test_overlay_lines(self):
        c = sp.parse_config(DS_SERVE)
        self.assertEqual(c["app"], "serve")
        self.assertEqual(c["env.STRIX_OLD"], sp.UNSET)
        self.assertEqual(c["arg.--slots"], "8")
        self.assertEqual(c["arg.--rank"], "0")
        self.assertEqual(c["env.STRIX_DIST_T"], "600")

    def test_json_and_route_receipt(self):
        c = sp.parse_config(JSON_CLI)
        self.assertEqual(c["admitted.pipe"], "255")
        self.assertEqual(c["argv"], "m.gguf 1")
        self.assertEqual(c["lever.qwen.prefill_auto"], "1")
        self.assertEqual(c["lever.qwen.prefill_auto.reason"], "selected")

    def test_empty_is_an_error(self):
        with self.assertRaises(ValueError):
            sp.parse_config("device: 0 of 1\n")


class Gate(unittest.TestCase):
    def test_same_request_same_config_passes(self):
        r = sp.check_config("e", GLM_CLI, GLM_CLI, table([]))
        self.assertEqual((r["differences"], r["unexplained"]), ([], []))

    def test_unexplained_difference_fails(self):
        rows = [{"key": "serve.*", "reason": "serving-only scheduler state"}]
        r = sp.check_config("e", GLM_CLI, GLM_SERVE, table(rows))
        self.assertEqual([d[0] for d in r["unexplained"]], ["max-rows"])
        r = sp.check_config("e", JSON_CLI, JSON_SERVE, table([]))
        self.assertEqual([d[0] for d in r["unexplained"]], ["admitted.pipe"])  # route is entry identity

    def test_absent_key_is_a_difference(self):
        r = sp.check_config("e", DS_CLI, DS_SERVE, table([]))
        self.assertEqual(sorted(d[0] for d in r["unexplained"]), ["arg.--slots", "arg.--tp-attention"])
        self.assertIn(("arg.--tp-attention", "", sp.ABSENT, None), r["unexplained"])

    def test_pinned_explanation_stops_applying_when_a_value_moves(self):
        rows = [{"key": "serve.*", "reason": "serving-only scheduler state"},
                {"key": "max-rows", "cli": "auto-8192", "serve": "4096", "reason": "eight slots share the row budget"}]
        self.assertEqual(sp.check_config("e", GLM_CLI, GLM_SERVE, table(rows))["unexplained"], [])
        moved = GLM_SERVE.replace("4096 (auto-4096-serve)", "2048 (auto)")
        self.assertEqual([d[0] for d in sp.check_config("e", GLM_CLI, moved, table(rows))["unexplained"]], ["max-rows"])

    def test_stale_explanations_are_reported(self):
        r = sp.check_config("e", GLM_CLI, GLM_CLI, table([{"key": "max-rows", "reason": "no longer differs at all"}]))
        self.assertIn("max-rows", r["stale"])

    def test_main_exit_codes(self):
        with tempfile.TemporaryDirectory() as d:
            paths = {}
            for name, text in (("cli", GLM_CLI), ("serve", GLM_SERVE), ("t", json.dumps(table([], "glm")))):
                paths[name] = os.path.join(d, name)
                with open(paths[name], "w", encoding="utf-8") as f:
                    f.write(text)
            common = ["config", "--engine", "glm", "--explained", paths["t"], "--json"]
            with contextlib.redirect_stdout(io.StringIO()):
                self.assertEqual(sp.main(common + ["--cli", paths["cli"], "--serve", paths["serve"]]), 1)
                self.assertEqual(sp.main(common + ["--cli", paths["cli"], "--serve", paths["cli"]]), 0)

    def test_shipped_table(self):
        t = sp.load_table()
        for engine in ("*", "qwen", "glm", "minimax", "ds41", "gemma"):
            self.assertIn(engine, t["engines"])
        with tempfile.TemporaryDirectory() as d:
            path = os.path.join(d, "t.json")
            with open(path, "w", encoding="utf-8") as f:
                json.dump(table([{"key": "x", "reason": "todo"}]), f)
            with self.assertRaises(ValueError):
                sp.load_table(path)


class Ratio(unittest.TestCase):
    GLM_RUN = "rank 0 decode: prompt 8192 tokens prefill 8.00 s, 64 tokens, 20.00 ms/token (50.00 tok/s), x\n"
    MM_RUN = ("rank 0 usb4: prompt 8192 tokens, TTFT 6.400 s, prefill 1280.0 tok/s (a)\n"
              "rank 0 decode 64 steps at ctx 8256: median(steps 16..) 30.00 ms/token = 33.33 tok/s; x\n")

    @staticmethod
    def serve(prefill_ms, decode_ms, n=64, prompt=8192):
        last = {"usage": {"prompt_tokens": prompt, "completion_tokens": n},
                "strix": {"prefill_ms": prefill_ms, "decode_ms": decode_ms}}
        return 'data: {"choices":[]}\n\ndata: ' + json.dumps(last) + "\n\ndata: [DONE]\n"

    def test_equal_speed_passes(self):
        r = sp.ratio_report("glm", [self.GLM_RUN] * 3, [self.serve(8000.0, 1260.0)] * 3)
        self.assertEqual(r["failures"], [])
        self.assertAlmostEqual(r["prefill_tps"]["ratio"], 1.0)
        self.assertAlmostEqual(r["decode_tps"]["serve"], 50.0)

    def test_slower_server_fails(self):
        r = sp.ratio_report("glm", [self.GLM_RUN], [self.serve(8800.0, 1260.0)])
        self.assertTrue(any(f.startswith("prefill_tps") for f in r["failures"]))
        self.assertAlmostEqual(r["prefill_tps"]["ratio"], 8000.0 / 8800.0)

    def test_token_count_mismatch_fails(self):
        r = sp.ratio_report("glm", [self.GLM_RUN], [self.serve(8000.0, 1240.0, n=63)])
        self.assertTrue(any(f.startswith("output_tokens") for f in r["failures"]))

    def test_timings_envelope_and_minimax_lines(self):
        body = json.dumps({"usage": {"prompt_tokens": 8192, "completion_tokens": 64},
                           "timings": {"prefill_ms": 6400.0, "decode_ms": 1890.0, "decode_tok_s": 33.33,
                                       "completion_tokens": 64, "ttft_ms": 6400.0}})
        r = sp.ratio_report("minimax", [self.MM_RUN], [body])
        self.assertEqual(r["failures"], [])
        self.assertAlmostEqual(r["prefill_tps"]["cli"], 1280.0)
        self.assertEqual(r["ttft_ms"], 6400.0)

    def test_missing_measurement_fails(self):
        r = sp.ratio_report("glm", ["nothing"], [self.serve(8000.0, 1260.0)])
        self.assertEqual(len(r["failures"]), 2)


if __name__ == "__main__":
    unittest.main()
