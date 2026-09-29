#!/usr/bin/env python3

from __future__ import annotations

import json
import math
import pathlib
import unittest

from qwen4_exp_oracle import (
    apply_partial_rope,
    gated_residual,
    ngram_ids,
    qsa_select_tokens_no_ties,
    recurrent_gated_delta_rule,
    rms_norm,
    stable_top_k,
)


ROOT = pathlib.Path(__file__).resolve().parent
FIXTURE = json.loads((ROOT / "fixtures" / "qwen4_exp_small.json").read_text())


def assert_nested_close(test: unittest.TestCase, actual, expected, *, tol: float = 1e-12) -> None:
    if isinstance(expected, list):
        test.assertIsInstance(actual, list)
        test.assertEqual(len(actual), len(expected))
        for left, right in zip(actual, expected):
            assert_nested_close(test, left, right, tol=tol)
    else:
        test.assertTrue(
            math.isclose(float(actual), float(expected), rel_tol=tol, abs_tol=tol),
            msg=f"{actual!r} != {expected!r}",
        )


class OracleTests(unittest.TestCase):
    def test_rms_norm_fixture(self) -> None:
        fixture = FIXTURE["rms_norm"]
        actual = rms_norm(
            fixture["values"], fixture["weight"], eps=fixture["eps"]
        )
        assert_nested_close(self, actual, fixture["expected"])

    def test_partial_rope_fixture(self) -> None:
        fixture = FIXTURE["rope"]
        self.assertEqual(
            apply_partial_rope(fixture["values"], fixture["cos"], fixture["sin"]),
            fixture["expected"],
        )

    def test_gdn_fixture_and_resume_equivalence(self) -> None:
        fixture = FIXTURE["gdn"]
        output, state = recurrent_gated_delta_rule(
            fixture["query"],
            fixture["key"],
            fixture["value"],
            fixture["decay_log"],
            fixture["beta"],
        )
        assert_nested_close(self, output, fixture["expected_output"])
        assert_nested_close(self, state, fixture["expected_state"])

        first_output, first_state = recurrent_gated_delta_rule(
            fixture["query"][:1],
            fixture["key"][:1],
            fixture["value"][:1],
            fixture["decay_log"][:1],
            fixture["beta"][:1],
        )
        second_output, resumed_state = recurrent_gated_delta_rule(
            fixture["query"][1:],
            fixture["key"][1:],
            fixture["value"][1:],
            fixture["decay_log"][1:],
            fixture["beta"][1:],
            initial_state=first_state,
        )
        assert_nested_close(self, first_output + second_output, output)
        assert_nested_close(self, resumed_state, state)

    def test_ngram_fixture_and_resume_equivalence(self) -> None:
        fixture = FIXTURE["ngram"]
        kwargs = {
            key: fixture[key]
            for key in (
                "eos_token_id",
                "unigram_vocab_size",
                "ngram_size",
                "heads_per_ngram",
                "ngram_vocab_size_base",
                "ple_layer_index",
                "seed",
            )
        }
        rows, context = ngram_ids(fixture["input_ids"], **kwargs)
        self.assertEqual(rows, fixture["expected"])
        self.assertEqual(context, fixture["expected_context"])

        split = 3
        first_rows, first_context = ngram_ids(
            fixture["input_ids"][:split], **kwargs
        )
        second_rows, second_context = ngram_ids(
            fixture["input_ids"][split:],
            previous_context=first_context,
            **kwargs,
        )
        self.assertEqual(first_rows + second_rows, rows)
        self.assertEqual(second_context, context)


    def test_gated_residual_fixture(self) -> None:
        fixture = FIXTURE["gated_residual"]
        mixed, injection = gated_residual(
            fixture["hyper_input"],
            hc_count=fixture["hc_count"],
            hidden_size=fixture["hidden_size"],
            norm_weight=fixture["norm_weight"],
            down_weight=fixture["down_weight"],
            up_weight=fixture["up_weight"],
            inject_weight=fixture["inject_weight"],
            eps=fixture["eps"],
        )
        assert_nested_close(self, mixed, fixture["expected_mixed"])
        assert_nested_close(self, injection, fixture["expected_injection"])

    def test_qsa_no_tie_fixture(self) -> None:
        fixture = FIXTURE["qsa_no_ties"]
        selected = qsa_select_tokens_no_ties(
            fixture["query_heads"],
            fixture["raw_keys"],
            fixture["visible_indices"],
            compress_ratio=fixture["compress_ratio"],
            token_budget=fixture["token_budget"],
            key_norm_weight=fixture["key_norm_weight"],
            eps=fixture["eps"],
        )
        self.assertEqual(selected, fixture["expected"])

    def test_stable_top_k_policy(self) -> None:
        fixture = FIXTURE["top_k"]
        self.assertEqual(
            stable_top_k(fixture["values"], fixture["k"]),
            fixture["expected"],
        )


if __name__ == "__main__":
    unittest.main()
