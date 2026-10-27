import json
import math
from pathlib import Path
import tempfile
import unittest
from tools.qualification.longctx_fidelity.pcg32 import PCG32
from tools.qualification.longctx_fidelity import tasks
from tools.qualification.longctx_fidelity.common import canonical, sha, ids_bytes, read_ids
from tools.qualification.longctx_fidelity.build import build_case, write_case
from tools.qualification.longctx_fidelity.tok import FakeTokenizer, render, PREFIX, SUFFIX, validate_symbols


class CoreTests(unittest.TestCase):
    def test_pcg_published_sequence(self):
        rng = PCG32(42, 54)
        self.assertEqual([rng.next() for _ in range(6)],
                         [0xa15c02b7, 0x7b47f409, 0xba1d3330, 0x83d2f293, 0xbfa4784b, 0xcbed606e])

    def test_pcg_bounds(self):
        r = PCG32(99)
        for b in (1, 3, 97, 2**32):
            self.assertTrue(all(0 <= r.below(b) < b for _ in range(100)))
        for b in (0, -1, 2**32+1):
            with self.assertRaises(ValueError):
                r.below(b)

    def test_canonical(self):
        self.assertEqual(canonical({"z": 1, "a": "x"}), b'{"a":"x","z":1}\n')
        with self.assertRaises(ValueError):
            canonical({"bad": math.nan})

    def test_chat_template(self):
        self.assertEqual(render(" hello "), PREFIX + "hello" + SUFFIX)
        self.assertTrue(render("x").endswith("<think>\n\n</think>\n\n"))
        self.assertNotIn("system", render("x"))

    def test_fake_utf8(self):
        tok = FakeTokenizer()
        self.assertEqual(tok.decode(tok.encode("cafe \u00e9")), "cafe \u00e9")

    def test_latest_obsolete_and_prefix(self):
        self.assertEqual(tasks.latest([("k", 9, "new"), ("k0", 12, "other"), ("k", 2, "old")]),
                         {"k": "new", "k0": "other"})
        with self.assertRaises(ValueError):
            tasks.latest([("k", 9, "x"), ("k", 9, "y")])

    def test_track_order_swap_self_negative_wrap(self):
        initial = list(range(16))
        events = [(5, "inc", 0, -100), (2, "swap", 0, 1), (1, "copy", 0, 2),
                  (4, "swap", 1, 1), (3, "inc", 0, 96)]
        out = tasks.track(initial, events)
        self.assertEqual(out[:2], [94, 2])
        self.assertEqual(initial, list(range(16)))
        with self.assertRaises(ValueError):
            tasks.track(initial, [(1, "inc", 0, 1), (1, "inc", 0, 2)])

    def test_walk_namespace_cycle_ambiguity(self):
        edges = [("a", "x", "y"), ("a", "y", "z"), ("b", "x", "bad")]
        self.assertEqual(tasks.walk(edges, "a", "x", 2), "z")
        with self.assertRaises(ValueError):
            tasks.walk(edges + [("a", "x", "q")], "a", "x", 2)
        with self.assertRaises(ValueError):
            tasks.walk([("a", "x", "y"), ("a", "y", "x")], "a", "x", 2)
        with self.assertRaises(KeyError):
            tasks.walk(edges, "a", "missing", 1)

    def test_aggregate_signed_filtered_duplicates_empty(self):
        events = [("g0", "keep", -7), ("g0", "keep", -7), ("g0", "keep", 3),
                  ("g0", "drop", -99), ("g1", "keep", 99)]
        self.assertEqual([tasks.aggregate(events, "g0", op) for op in ("sum", "count", "min")], [-11, 3, -7])
        self.assertEqual([tasks.aggregate([], "g0", op) for op in ("sum", "count", "min")], [0, 0, 0])

    def test_every_family_oracle(self):
        families = (*tasks.FAMILIES, "multikey32", "vartrack256", "multihop2", "multihop8", "aggregate128")
        for family in families:
            for seed in (104729, 130363, 155921):
                with self.subTest(family=family, seed=seed):
                    t = tasks.generate(family, 32768, .5, seed)
                    d = t["oracle_data"]
                    if family == "niah_single":
                        answer = d["value"]
                        self.assertEqual(len(d["key"]), 32)
                    elif family.startswith("multikey"):
                        values = tasks.latest(d["records"])
                        answer = [[k, values[k]] for k in d["keys"]]
                        self.assertIn(len(d["keys"]), (8, 32))
                        self.assertTrue(3 <= len(t["facts"]) <= 8)
                    elif family.startswith("vartrack"):
                        answer = str(tasks.track(d["initial"], d["events"])[d["target"]])
                    elif family.startswith("multihop"):
                        answer = tasks.walk(d["edges"], "primary", d["start"], d["hops"])
                    else:
                        answer = str(tasks.aggregate(d["events"], d["group"], d["operation"]))
                    self.assertEqual(answer, t["answer"])
                    self.assertEqual(t, tasks.generate(family, 32768, .5, seed))

    def test_filler_strata_not_tiled(self):
        for stratum in ("prose", "records", "similar-records"):
            a, b = tasks.filler_units(42, stratum), tasks.filler_units(42, stratum)
            rows = [next(a) for _ in range(100)]
            self.assertEqual(rows, [next(b) for _ in range(100)])
            self.assertEqual(len(set(rows)), 100)
            self.assertFalse(any("Return" in row or "Ignore" in row for row in rows))

    def test_golden_id_hashes(self):
        golden = {"niah_single": "c5a7f7037673ae1d026767555e4a4e119db2c642c65493f76bd804ee5ad6249a",
                  "aggregate": "e5d1e04cb0982e462e583695f611e5312baac8278fdf7fd1b386d1a6aacec7ae",
                  "multihop": "b9c08275c8bdf112b1550f664ecd2676a436327a980e3110c648733371473f6e"}
        for family, expected in golden.items():
            a = build_case(family, 8192, .5, 104729, FakeTokenizer())
            b = build_case(family, 8192, .5, 104729, FakeTokenizer())
            self.assertEqual(canonical(a[2]), canonical(b[2]))
            self.assertEqual(sha(ids_bytes(a[1])), expected)
            self.assertEqual(a[0], b[0])

    def test_exact_lengths_and_span_recovery(self):
        tok = FakeTokenizer()
        for family in tasks.FAMILIES:
            for position in (.05, .5, .95):
                with self.subTest(family=family, position=position):
                    text, ids, m = build_case(family, 32768, position, 104729, tok)
                    self.assertEqual(len(ids), 32768)
                    self.assertEqual(tok.decode(ids), text)
                    self.assertEqual(text, render(text[len(PREFIX):-len(SUFFIX)]))
                    for span in m["spans"]:
                        needle = text[span["char_start"]:span["char_end"]]
                        self.assertIn(needle, tok.decode(ids[span["start"]:span["end"]]))
                        self.assertEqual(sha(needle.encode()), span["sha256"])
                        self.assertLessEqual(abs(span["target_start"] - span["start"]), 8)

    def test_cap_and_edge(self):
        for depth, g, variant in ((262144, 1, "normal"), (262143, 32, "normal"), (262143, 1, "normal")):
            with self.assertRaises(ValueError):
                build_case("niah_single", depth, .5, 1, FakeTokenizer(), g, variant=variant)
        text, ids, m = build_case("niah_single", 262143, .95, 104729, FakeTokenizer(), 1, variant="symbol")
        self.assertEqual(len(ids), 262143)
        self.assertEqual(m["first_output_position"], 262143)
        self.assertEqual(m["last_input_position"], 262142)
        self.assertEqual(len(m["acceptable_token_ids"]), 1)
        self.assertIn(m["ground_truth"], tasks.SYMBOLS)

    def test_symbol_all_families(self):
        for family in tasks.FAMILIES:
            t = tasks.generate(family, 8192, .5, 104729, "symbol")
            self.assertIn(t["answer"], tasks.SYMBOLS)
        class BadTokenizer(FakeTokenizer):
            def encode(self, text):
                return super().encode(text) + [0]
        with self.assertRaises(ValueError):
            validate_symbols(BadTokenizer(), "context")

    def test_impossible_depth(self):
        with self.assertRaises(ValueError):
            build_case("vartrack", 100, .5, 42, FakeTokenizer())

    def test_write_and_u32(self):
        with tempfile.TemporaryDirectory() as tmp:
            text, ids, m = build_case("niah_single", 2048, .5, 42, FakeTokenizer())
            path = write_case(tmp, text, ids, m)
            self.assertEqual(path.read_bytes(), canonical(m))
            self.assertEqual(read_ids(Path(tmp) / (m["case_id"] + ".u32")), ids)
            self.assertEqual((Path(tmp) / (m["case_id"] + ".txt")).read_bytes(), text.encode())
            (Path(tmp) / "bad.u32").write_bytes(b"x")
            with self.assertRaises(ValueError):
                read_ids(Path(tmp) / "bad.u32")

    def test_tokenizer_merges_and_covering_spans(self):
        class PairTokenizer(FakeTokenizer):
            # Fixed 2-byte tokens exercise boundaries crossing fact delimiters.
            def encode(self, text):
                data = text.encode()
                return [256 + data[i] * 256 + data[i+1] if i+1 < len(data) else data[i]
                        for i in range(0, len(data), 2)]
            def decode(self, ids):
                return b"".join(bytes(divmod(i-256, 256)) if i >= 256 else bytes([i]) for i in ids).decode()
            def boundaries(self, ids, offsets):
                return [p // 2 for p in offsets]
        tok = PairTokenizer()
        for family in ("niah_single", "multihop", "vartrack"):
            text, ids, m = build_case(family, 8192, .5, 104729, tok)
            self.assertEqual(len(ids), 8192)
            self.assertEqual(tok.decode(ids), text)
            for s in m["spans"]:
                self.assertIn(text[s["char_start"]:s["char_end"]], tok.decode(ids[s["start"]:s["end"]]))

    def test_roundtrip_drift_refused(self):
        class DriftTokenizer(FakeTokenizer):
            def decode(self, ids):
                return super().decode(ids).replace("vault", "fault")
        with self.assertRaisesRegex(ValueError, "roundtrip drift"):
            build_case("niah_single", 2048, .5, 42, DriftTokenizer())

    def test_manifest_only_writes_no_prompts(self):
        with tempfile.TemporaryDirectory() as tmp:
            result = build_case("niah_single", 2048, .5, 42, FakeTokenizer())
            write_case(tmp, *result, manifest_only=True)
            self.assertEqual(len(list(Path(tmp).iterdir())), 1)
