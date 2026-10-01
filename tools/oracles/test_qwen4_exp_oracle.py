#!/usr/bin/env python3

from __future__ import annotations

import json
import math
import pathlib
import unittest

from qwen4_exp_oracle import (
    apply_partial_rope,
    gated_residual,
    moe_route_deterministic,
    ngram_ids,
    qsa_select_tokens_deterministic,
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

    def test_qsa_deterministic_fixture(self) -> None:
        fixture = FIXTURE["qsa_no_ties"]
        selected = qsa_select_tokens_deterministic(
            fixture["query_heads"],
            fixture["raw_keys"],
            fixture["visible_indices"],
            compress_ratio=fixture["compress_ratio"],
            token_budget=fixture["token_budget"],
            key_norm_weight=fixture["key_norm_weight"],
            block_rope_cos=fixture["block_rope_cos"],
            block_rope_sin=fixture["block_rope_sin"],
            eps=fixture["eps"],
        )
        self.assertEqual(selected, fixture["expected"])


    def test_qsa_cutoff_tie_uses_lower_block_index(self) -> None:
        fixture = FIXTURE["qsa_tie"]
        selected = qsa_select_tokens_deterministic(
            fixture["query_heads"],
            fixture["raw_keys"],
            fixture["visible_indices"],
            compress_ratio=fixture["compress_ratio"],
            token_budget=fixture["token_budget"],
            key_norm_weight=fixture["key_norm_weight"],
            block_rope_cos=fixture["block_rope_cos"],
            block_rope_sin=fixture["block_rope_sin"],
            eps=fixture["eps"],
        )
        self.assertEqual(selected, fixture["expected"])

    def test_moe_route_deterministic_fixture(self) -> None:
        fixture = FIXTURE["moe_route_no_ties"]
        indices, weights = moe_route_deterministic(
            fixture["logits"],
            top_k=fixture["top_k"],
            normalize_selected=fixture["normalize_selected"],
        )
        self.assertEqual(indices, fixture["expected_indices"])
        assert_nested_close(self, weights, fixture["expected_weights"])

    def test_moe_cutoff_tie_uses_lower_expert_index(self) -> None:
        indices, weights = moe_route_deterministic(
            [3.0, 2.0, 2.0, 1.0],
            top_k=2,
            normalize_selected=True,
        )
        self.assertEqual(indices, [0, 1])
        self.assertTrue(math.isclose(sum(weights), 1.0, rel_tol=0.0, abs_tol=1e-15))

    def test_stable_top_k_policy(self) -> None:
        fixture = FIXTURE["top_k"]
        self.assertEqual(
            stable_top_k(fixture["values"], fixture["k"]),
            fixture["expected"],
        )


if __name__ == "__main__":
    unittest.main()
