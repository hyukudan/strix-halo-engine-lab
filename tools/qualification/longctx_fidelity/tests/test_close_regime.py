"""CPU construction fixtures for newline merges and unstable word cuts."""
import bisect
import inspect
import re
import sys
import unittest

from tools.qualification.longctx_fidelity import tasks
from tools.qualification.longctx_fidelity.build import build_case
from tools.qualification.longctx_fidelity.tok import FakeTokenizer


class CutTokenizer(FakeTokenizer):
    """Reversible lexical merges, including a two-token jump at abc -> abcd.

    Odd-length words merge pairs; even-length words retain character tokens.
    This deliberately frustrates proportional adjustment near a word cut.
    IDs encode literal one/two-character pieces without a mutable vocabulary.
    """
    def encode(self, text):
        pieces = []
        for match in re.finditer(r"\n\n| x|[A-Za-z]+|.", text, re.DOTALL):
            word = match.group()
            if word in ("\n\n", " x"):
                pieces.append(word)
            else:
                step = 2 if word.isalpha() and len(word) % 2 else 1
                pieces.extend(word[i:i + step] for i in range(0, len(word), step))
        return [ord(p) if len(p) == 1 else 256 + ord(p[0]) * 128 + ord(p[1])
                for p in pieces]

    def decode(self, ids):
        return "".join(chr(i) if i < 256 else "".join(map(chr, divmod(i - 256, 128)))
                       for i in ids)

    def boundaries(self, ids, offsets):
        ends = [0]
        for token in ids:
            ends.append(ends[-1] + len(self.decode([token])))
        return [bisect.bisect_right(ends, offset) - 1 for offset in offsets]


def traced_build(tokenizer, depth, family="niah_single", position=.5, states=None):
    """Record actual loop decisions without changing the construction algorithm."""
    states = [] if states is None else states
    source, first = inspect.getsourcelines(build_case)
    decision = first + next(i for i, line in enumerate(source) if line.strip() == "if error == 0:")
    def trace(frame, event, arg):
        if frame.f_code is not build_case.__code__:
            return None
        if event == "line":
            local = frame.f_locals
            # error is assigned immediately before this branch on every attempt.
            if frame.f_lineno == decision:
                states.append({k: local[k] for k in
                               ("attempt", "error", "snap", "pad", "last")})
        return trace
    previous = sys.gettrace()
    sys.settrace(trace)
    try:
        return build_case(family, depth, position, 104729, tokenizer), states
    finally:
        sys.settrace(previous)


class CloseRegimeTests(unittest.TestCase):
    def assert_case(self, result, tokenizer, depth, family="niah_single"):
        text, ids, manifest = result
        self.assertEqual(len(ids), depth)
        self.assertEqual(tokenizer.encode(text), ids)
        self.assertEqual(tokenizer.decode(ids), text)
        facts = tasks.generate(family, depth, .5, 104729)["facts"]
        self.assertEqual(len(manifest["spans"]), len(facts))
        self.assertEqual(sorted(s["fact_index"] for s in manifest["spans"]), list(range(len(facts))))
        for span in manifest["spans"]:
            fact = facts[span["fact_index"]]
            self.assertEqual(text[span["char_start"]:span["char_end"]], fact)
            self.assertIn(fact, tokenizer.decode(ids[span["start"]:span["end"]]))
            self.assertLessEqual(abs(span["start"] - span["target_start"]), 8)

    def test_fixture_merge_contract(self):
        tok = CutTokenizer()
        for text in ("\n\n", "\n x", "\n x x", "abc", "abcd"):
            self.assertEqual(tok.decode(tok.encode(text)), text)
        self.assertEqual(tok.count("\n\n"), 1)
        self.assertEqual(tok.count("\n x\n"), tok.count("\n\n") + 2)
        self.assertEqual(tok.count("\n x x\n"), tok.count("\n x\n") + 1)
        self.assertEqual(tok.count("abcd\nQuestion:") - tok.count("abc\nQuestion:"), 2)

    def test_close_regime_error_of_one_and_persistent_pad(self):
        tok = CutTokenizer()
        result, states = traced_build(tok, 2160)
        self.assert_case(result, tok, 2160)
        entry = next(s for s in states if s["attempt"] >= 12 and abs(s["error"]) <= 256)
        self.assertFalse(entry["snap"])
        one = next(i for i, s in enumerate(states) if s["snap"] and s["error"] == 1)
        self.assertEqual(states[one]["pad"], " x")
        self.assertEqual(states[one + 1]["pad"], " x x")
        self.assertEqual(states[one + 1]["error"], 0)
        self.assertTrue(all(s["pad"].startswith(" x") for s in states if s["snap"]))

    def test_pad_underflow_drops_a_whole_line(self):
        tok = CutTokenizer()
        result, states = traced_build(tok, 2035)
        self.assert_case(result, tok, 2035)
        i = next(i for i, s in enumerate(states) if s["snap"] and s["error"] < 0
                 and len(s["pad"]) - 2 < -2 * s["error"])
        before, after = states[i:i + 2]
        self.assertEqual(before["pad"], after["pad"], " x")
        self.assertTrue(before["last"].endswith("\n"))
        self.assertEqual(after["last"], before["last"][:before["last"].rfind("\n", 0, -1) + 1])
        self.assertGreater(after["error"], 0)
        self.assertEqual(states[i + 2]["error"], 0)

    def test_close_regime_all_facts_and_placement_correction(self):
        tok = CutTokenizer()
        result, states = traced_build(tok, 8192, "vartrack")
        self.assert_case(result, tok, 8192, "vartrack")
        self.assertGreater(len(result[2]["spans"]), 1)
        self.assertTrue(any(s["attempt"] >= 12 and s["snap"] for s in states))
        # A zero-error snapped candidate is followed by another attempt: placement
        # correction must run, reset snapping, and still recover every fact.
        i = next(i for i, s in enumerate(states[:-1]) if s["snap"] and s["error"] == 0)
        self.assertFalse(states[i + 1]["snap"])

    def test_unreachable_count_refuses_at_attempt_cap(self):
        class EvenTokenizer(CutTokenizer):
            # Pairs are inseparable code units, so an odd target cannot exist.
            def encode(self, text):
                return [part for token in super().encode(text) for part in (token, token)]

            def decode(self, ids):
                return super().decode(ids[::2])

        tok = EvenTokenizer()
        self.assertEqual(tok.decode(tok.encode("abc\n\n x")), "abc\n\n x")
        states = []
        with self.assertRaisesRegex(ValueError, "token budget drift: no exact construction"):
            traced_build(tok, 4097, states=states)
        self.assertEqual([s["attempt"] for s in states], list(range(96)))
        self.assertTrue(any(s["snap"] for s in states))
        self.assertTrue(all(s["error"] % 2 for s in states))
