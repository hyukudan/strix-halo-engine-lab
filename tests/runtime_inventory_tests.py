"""Portable evidence/contradiction tests; no executable or GPU is invoked."""
import contextlib
import copy
import hashlib
import importlib.util
import io
import json
from pathlib import Path
import tempfile
import unittest
from unittest import mock

ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location("runtime_inventory", ROOT / "tools/qualification/runtime_inventory.py")
INVENTORY = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(INVENTORY)


class InventoryTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)

    def text(self, name, value):
        path = self.root / name
        path.write_text(value, encoding="utf-8")
        return path

    def evidence(self, name, value):
        return self.text(name, json.dumps(value))

    def cache(self, hip="ON", a8="OFF"):
        return self.text("CMakeCache.txt", f"//fixture\nSTRIX_ENABLE_HIP_RUNTIME:BOOL={hip}\n"
                         f"STRIX_ENABLE_EXPERIMENTAL_Q4K_A8_RUNTIME:BOOL={a8}\n"
                         "CMAKE_BUILD_TYPE:STRING=Release\nCMAKE_HIP_FLAGS:STRING=-fno-fast-math\n")

    def receipt(self, applied=False, reason="a8-runtime-not-built"):
        return {"optimization_profile_requested": "auto",
                "optimization_profile_selected": "canonical-a8-wave-qsa-hc-shared-deferred-v3" if applied else "manual",
                "optimization_profile_reason": reason,
                "optimization_profile_automatic_applied": applied}

    def collect(self, **kwargs):
        return INVENTORY.collect(self.root, **kwargs)

    def selected_inventory(self):
        return {"schema_version": 1, "evidence_kind": "compiled-selected-launchers",
                "phase": "decode", "selected_logical_rows": 1, "selected_request_concurrency": 1,
                "verify_admitted_by_this_receipt": False, "prefill_admitted_by_this_receipt": False,
                "execution_identity": 101, "kernel_plan_identity": 102, "launch_plan_identity": 103,
                "native_execution_observed": None, "qualification": None,
                "whole_plan_graph_safe_declared": True,
                "steps": [{"step_id": 0, "operator": "lm_head", "layer": None,
                           "runtime_step": False, "launcher_bound": True,
                           "launcher_implementation_identity": 104, "kernel_id": "selected.head",
                           "declared_variant": {
                               "id": "selected.head", "phase": "decode", "family": 17,
                               "hidden_size": 2560, "auxiliary_size": 248320, "wave_size": 32, "logical_rows": 1,
                               "request_concurrency": 1, "graph_safe": True,
                               "qualification": "oracle_matched", "input_layout": "rows",
                               "output_layout": "logits", "compiler_identity": "host-fixture",
                               "numerics": {"input_precision": "bf16", "accumulator_precision": "fp32",
                                            "output_precision": "fp32", "row_invariant": True, "deterministic": True}}}]}

    def test_auto_base_profile_with_overlays_is_consistent_but_not_combination_qualified(self):
        modes = {"eager_submission_mode": "prepared-static-plan-experimental-v1",
                 "moe_shared_overlap_mode": "eager-sibling-streams-experimental-v1",
                 "generation_output_mode": "finite-firstmax-token-v1", "batched_moe_activation": True,
                 "prefetch_q8_lm_head": True, "wave8_q8_lm_head": True, "wave8_q8_gdn_input": True}
        for field, value in modes.items():
            with self.subTest(field=field):
                receipt = self.receipt(True, "qualified-checkpoint")
                receipt[field] = value
                result = self.collect(cache_path=self.cache(a8="ON"),
                                      receipt_path=self.evidence("overlay.json", receipt))
                self.assertTrue(result["consistent"])
                self.assertEqual(result["conflicts"], [])
                self.assertEqual(result["effective_profile"]["selected"], receipt["optimization_profile_selected"])
                self.assertEqual(result["reported_opt_in_overlays"][field], value)
                self.assertEqual(result["profile_overlay_qualification"], "not_assessed")
                self.assertTrue(any("combined configuration" in text for text in result["limitations"]))
        receipt = self.receipt(True, "qualified-checkpoint")
        receipt.update(modes)
        result = self.collect(cache_path=self.cache(a8="ON"), receipt_path=self.evidence("all-overlays.json", receipt))
        self.assertEqual(result["reported_opt_in_overlays"], modes)

    def test_overlay_absence_false_unknown_and_invalid_types(self):
        inactive = {key: False for key in ("batched_moe_activation", "prefetch_q8_lm_head",
                                          "wave8_q8_lm_head", "wave8_q8_gdn_input")}
        result = self.collect(receipt_path=self.evidence("no-overlays.json", inactive))
        for key in inactive:
            self.assertIs(result["reported_opt_in_overlays"][key], False)
        self.assertIsNone(result["profile_overlay_qualification"])
        self.assertIsNone(result["reported_opt_in_overlays"]["generation_output_mode"])
        result = self.collect(receipt_path=self.evidence("future-overlay.json", {"eager_submission_mode": "future-v2"}))
        self.assertEqual(result["reported_opt_in_overlays"]["eager_submission_mode"], "future-v2")
        self.assertTrue(result["consistent"])
        for field in ("eager_submission_mode", "moe_shared_overlap_mode", "generation_output_mode"):
            for value in (True, 1, {}, [] , ""):
                with self.subTest(field=field, value=value), self.assertRaises(ValueError):
                    self.collect(receipt_path=self.evidence("invalid-overlay.json", {field: value}))
        for field in ("batched_moe_activation", "prefetch_q8_lm_head", "wave8_q8_lm_head", "wave8_q8_gdn_input"):
            for value in (1, 0, 0.0, "true", [], {}):
                with self.subTest(field=field, value=value), self.assertRaises(ValueError):
                    self.collect(receipt_path=self.evidence("invalid-bool-overlay.json", {field: value}))
            unknown = self.collect(receipt_path=self.evidence("null-overlay.json", {field: None}))
            self.assertIsNone(unknown["reported_opt_in_overlays"][field])
            self.assertIsNone(unknown["profile_overlay_qualification"])

    def test_auto_fallback_is_effective_manual_not_v3(self):
        receipt = self.receipt()
        receipt.update(context_tokens=8192, prefix_tokens=[3, 4, 5], next_position=19)
        result = self.collect(cache_path=self.cache(), receipt_path=self.evidence("receipt.json", receipt))
        self.assertTrue(result["consistent"])
        self.assertEqual(result["effective_profile"]["selected"], "manual")
        self.assertEqual(result["effective_profile"]["reason"], "a8-runtime-not-built")
        self.assertEqual(result["context_capacity_tokens"], 8192)
        self.assertEqual(result["occupied_prefix_tokens"], 3)
        self.assertEqual(result["final_position"], 19)

    def test_cache_alone_does_not_resolve_or_qualify(self):
        result = self.collect(cache_path=self.cache(a8="ON"))
        self.assertEqual(result["configured_build"]["CMAKE_HIP_FLAGS"], "-fno-fast-math")
        self.assertIsNone(result["effective_profile"]["selected"])
        self.assertIsNone(result["build_completed"])
        self.assertIsNone(result["native_tests_executed"])
        self.assertEqual(result["qualification_state"], "not_assessed")

    def test_auto_receipt_and_disabled_build_are_contradictory(self):
        result = self.collect(cache_path=self.cache(), receipt_path=self.evidence(
            "receipt.json", self.receipt(True, "qualified-checkpoint")))
        self.assertFalse(result["consistent"])
        self.assertIn("profile/build", [item["field"] for item in result["conflicts"]])

    def test_impossible_cmake_capability_configuration(self):
        self.assertFalse(self.collect(cache_path=self.cache("OFF", "ON"))["consistent"])

    def test_manual_receipt_does_not_gain_auto_from_build(self):
        receipt = self.receipt(reason="explicit-manual")
        receipt["optimization_profile_requested"] = "manual"
        result = self.collect(cache_path=self.cache(a8="ON"), receipt_path=self.evidence("receipt.json", receipt))
        self.assertTrue(result["consistent"])
        self.assertFalse(result["effective_profile"]["automatic_applied"])

    def test_receipt_profile_internal_contradiction(self):
        receipt = self.receipt()
        receipt["optimization_profile_automatic_applied"] = True
        self.assertFalse(self.collect(receipt_path=self.evidence("receipt.json", receipt))["consistent"])

    def test_partial_profile_and_boolean_integer_fail_closed(self):
        with self.assertRaises(ValueError):
            self.collect(receipt_path=self.evidence("partial.json", {"optimization_profile_selected": "manual"}))
        receipt = self.receipt()
        receipt["context_tokens"] = True
        with self.assertRaises(ValueError):
            self.collect(receipt_path=self.evidence("bad.json", receipt))

    def test_json_duplicates_nonfinite_and_cache_duplicates_rejected(self):
        for value in ('{"a":1,"a":2}', '{"a":NaN}', '[]'):
            with self.assertRaises(ValueError):
                INVENTORY.load_json(self.text("invalid.json", value))
        with self.assertRaises(ValueError):
            INVENTORY.cmake_cache(self.text("bad-cache", "BUILD_TESTING:BOOL=ON\nBUILD_TESTING:BOOL=OFF\n"))

    def test_model_hash_is_content_based_and_aggregate_is_distinct(self):
        model = self.text("looks-canonical.gguf", "abc")
        result = self.collect(model_path=model)
        self.assertEqual(result["model"]["sha256"],
                         "ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad")
        # Explicit serialized V1 fixture: domain, logical-name length/name,
        # content size and raw file SHA bytes. Little-endian u64 lengths.
        expected = hashlib.sha256(b"STRIX-MODEL-CONTENT-V1\x05\0\0\0\0\0\0\0model"
                                  b"\x03\0\0\0\0\0\0\0" + bytes.fromhex(result["model"]["sha256"])).hexdigest()
        self.assertEqual(result["model"]["content_identity"], expected)
        self.assertIsNone(result["effective_profile"]["selected"])
        model.write_text("different contents", encoding="utf-8")
        self.assertNotEqual(self.collect(model_path=model)["model"]["content_identity"], expected)

    def test_file_sha_and_model_identity_contradictions_are_separate(self):
        model = self.text("model.gguf", "abc")
        receipt = {"model_sha256": "0" * 64, "model_content_identity": "1" * 64}
        result = self.collect(model_path=model, receipt_path=self.evidence("receipt.json", receipt))
        self.assertEqual({item["field"] for item in result["conflicts"]},
                         {"model_sha256", "model_content_identity"})

    def test_binary_hash_contradiction_and_unbound_source_remain_explicit(self):
        binary = self.text("decoder-not-run.bin", "not an executable")
        receipt = self.evidence("receipt.json", {"binary_sha256": "0" * 64})
        result = self.collect(binary_path=binary, receipt_path=receipt)
        self.assertFalse(result["consistent"])
        self.assertEqual(result["conflicts"][0]["field"], "binary_sha256")
        self.assertIsNone(result["source"]["revision"])
        self.assertIsNone(result["native_tests_executed"])

    def test_tensor_admission_and_formats_never_imply_executable_support(self):
        audit = {"manifest_layout_identity": 7, "compiled": True,
                 "requirements": [{"op": "lm_head", "status": "satisfied", "tensor_indices": [0]}]}
        manifest = {"layout_identity": 7, "tensors": [{"ggml_type": 8}]}
        result = self.collect(audit_path=self.evidence("audit.json", audit),
                              manifest_path=self.evidence("manifest.json", manifest))
        self.assertEqual(len(result["operators"]), 3)
        self.assertEqual({entry["phase"] for entry in result["operators"]}, {"decode", "verify", "prefill"})
        for entry in result["operators"]:
            self.assertEqual(entry["tensor_formats_ggml"], [8])
            for key in ("compiled_support", "executable_support", "allowed_logical_rows",
                        "allowed_request_concurrency", "qualification", "graph_safe"):
                self.assertIsNone(entry[key])
        manifest["layout_identity"] = 8
        self.assertFalse(self.collect(audit_path=self.root / "audit.json",
                                     manifest_path=self.evidence("manifest.json", manifest))["consistent"])

    def test_missing_tensor_formats_remain_unknown_and_bad_indices_rejected(self):
        audit = {"requirements": [{"op": "gated_delta_net", "tensor_indices": [True]}]}
        audit_path = self.evidence("audit.json", audit)
        self.assertIsNone(self.collect(audit_path=audit_path)["operators"][0]["tensor_formats_ggml"])
        with self.assertRaises(ValueError):
            self.collect(audit_path=audit_path, manifest_path=self.evidence("manifest.json", {"tensors": [{"ggml_type": 8}]}))

    def test_context_conflicts_do_not_silently_rewrite_occupied_or_capacity(self):
        receipt = self.evidence("receipt.json", {"context_tokens": 128, "prefix_tokens": [1, 2]})
        result = self.collect(receipt_path=receipt, capacity=8192, occupied=3)
        self.assertFalse(result["consistent"])
        self.assertEqual((result["context_capacity_tokens"], result["occupied_prefix_tokens"]), (8192, 3))
        self.assertFalse(self.collect(capacity=1, occupied=2)["consistent"])
        self.assertIsNone(self.collect(capacity=8192)["occupied_prefix_tokens"])
        self.assertEqual(result["context_evidence"]["capacity"], "caller-declaration")

    def test_external_layout_identities_cannot_exploit_boolean_integer_equality(self):
        with self.assertRaises(ValueError):
            self.collect(audit_path=self.evidence("bool-layout-audit.json", {"manifest_layout_identity": True}),
                         manifest_path=self.evidence("numeric-layout.json", {"layout_identity": 1}))
        result = self.collect(audit_path=self.evidence("null-layout-audit.json", {"manifest_layout_identity": None}),
                              manifest_path=self.evidence("null-layout.json", {"layout_identity": None}))
        self.assertTrue(result["consistent"])

    def test_cli_writes_conflict_json_and_nonzero_exit_without_execution(self):
        receipt = self.evidence("receipt.json", self.receipt(True, "qualified-checkpoint"))
        output = self.root / "inventory.json"
        code = INVENTORY.main(["--repo", str(self.root), "--cmake-cache", str(self.cache()),
                               "--receipt", str(receipt), "--output", str(output)])
        self.assertEqual(code, 3)
        self.assertFalse(json.loads(output.read_text(encoding="utf-8"))["consistent"])
        before = receipt.read_bytes()
        with contextlib.redirect_stderr(io.StringIO()), self.assertRaises(SystemExit):
            INVENTORY.main(["--receipt", str(receipt), "--output", str(receipt)])
        self.assertEqual(receipt.read_bytes(), before)

    def test_selected_export_retains_declared_rows_and_never_promotes_qualification(self):
        receipt = self.evidence("receipt.json", {"execution_inventory": self.selected_inventory()})
        result = self.collect(receipt_path=receipt)
        self.assertEqual(len(result["operators"]), 3)
        decode, verify, prefill = result["operators"]
        self.assertEqual(decode["selected_logical_rows"], 1)
        self.assertTrue(decode["launcher_bound_declared"])
        self.assertEqual(decode["declared_variant"]["qualification"], "oracle_matched")
        for record in result["operators"]:
            self.assertIsNone(record["qualification"])
            self.assertIsNone(record["executable_support"])
            self.assertIsNone(record["allowed_logical_rows"])
        for record in (verify, prefill):
            self.assertFalse(record["admitted_by_this_receipt"])
            self.assertIsNone(record["selected_logical_rows"])
            self.assertIsNone(record["declared_variant"])

    def test_selected_export_binding_mutants_rejected(self):
        original = self.selected_inventory()
        mutations = [lambda value: value.update(native_execution_observed=True),
                     lambda value: value.update(verify_admitted_by_this_receipt=True),
                     lambda value: value.update(selected_logical_rows=True),
                     lambda value: value["steps"][0].update(step_id=1),
                     lambda value: value["steps"][0].update(launcher_bound=False),
                     lambda value: value["steps"][0].update(launcher_implementation_identity=0),
                     lambda value: value["steps"][0].update(kernel_id="different"),
                     lambda value: value["steps"][0].update(runtime_step=True),
                     lambda value: value["steps"][0]["declared_variant"].update(logical_rows=4),
                     lambda value: value["steps"][0]["declared_variant"].update(phase="prefill")]
        for mutate in mutations:
            value = copy.deepcopy(original)
            mutate(value)
            with self.subTest(mutate=mutate), self.assertRaises(ValueError):
                self.collect(receipt_path=self.evidence("receipt.json", {"execution_inventory": value}))

    def test_selected_export_identity_disagrees_with_outer_receipt(self):
        result = self.collect(receipt_path=self.evidence("receipt.json", {
            "execution_identity": 999, "execution_inventory": self.selected_inventory()}))
        self.assertFalse(result["consistent"])
        self.assertEqual(result["conflicts"][0]["field"], "execution_identity")

    def test_cmake_cache_change_during_parse_is_rejected(self):
        path = self.cache()
        original = INVENTORY.cmake_cache

        def mutate(cache_path):
            parsed = original(cache_path)
            cache_path.write_text(cache_path.read_text(encoding="utf-8") + "BUILD_TESTING:BOOL=ON\n", encoding="utf-8")
            return parsed

        with mock.patch.object(INVENTORY, "cmake_cache", side_effect=mutate), self.assertRaises(ValueError):
            self.collect(cache_path=path)

    def test_missing_git_reports_unknown_without_fabricating_clean_source(self):
        with mock.patch.object(INVENTORY.subprocess, "run", side_effect=FileNotFoundError("git absent")):
            result = self.collect()
        self.assertIsNone(result["source"]["revision"])
        self.assertIsNone(result["source"]["dirty"])
        self.assertIsNone(result["source"]["status_porcelain"])
        self.assertTrue(result["consistent"])

    def test_explicit_A8_flags_are_strict_and_checked_without_auto_profile(self):
        for flag in ("q4k_a8g32_gate_up", "grouped_exact_q51_down"):
            with self.subTest(flag=flag):
                result = self.collect(cache_path=self.cache(a8="OFF"),
                    receipt_path=self.evidence("explicit.json", {flag: True}))
                self.assertFalse(result["consistent"])
                self.assertIn("build/" + flag, {x["field"] for x in result["conflicts"]})
                self.assertTrue(self.collect(cache_path=self.cache(a8="ON"),
                    receipt_path=self.evidence("enabled.json", {flag: True}))["consistent"])
                self.assertTrue(self.collect(receipt_path=self.evidence("unknown.json", {flag: True}))["consistent"])
                for value in (0, 1, "false", None, [], {}):
                    with self.subTest(value=value), self.assertRaises(ValueError):
                        self.collect(receipt_path=self.evidence("badflag.json", {flag: value}))

    def test_overflowing_JSON_floats_rejected_even_in_unconsumed_nested_metadata(self):
        for text in ('{"extra":{"value":1e999}}', '{"extra":[-1e999]}', '{"value":NaN}'):
            with self.subTest(text=text), self.assertRaises(ValueError):
                self.collect(receipt_path=self.text("nonfinite.json", text))
        self.assertTrue(self.collect(receipt_path=self.text("finite.json", '{"extra":{"value":1e300}}'))["consistent"])

    def test_outer_u64_identities_and_positions_reject_boolean_and_overflow(self):
        fields = ("execution_identity", "kernel_plan_identity", "launch_plan_identity",
                  "runtime_compatibility_identity", "final_position", "next_position",
                  "committed_tokens", "steps_executed")
        for key in fields:
            for value in (True, False, -1, 1 << 64, 1.0, "1"):
                with self.subTest(key=key, value=value), self.assertRaises(ValueError):
                    self.collect(receipt_path=self.evidence("bad-u64.json", {key: value}))
            self.assertTrue(self.collect(receipt_path=self.evidence("null-u64.json", {key: None}))["consistent"])
            self.assertTrue(self.collect(receipt_path=self.evidence("max-u64.json", {key: (1 << 64) - 1}))["consistent"])
        receipt = {"execution_identity": True, "execution_inventory": self.selected_inventory()}
        receipt["execution_inventory"]["execution_identity"] = 1
        with self.assertRaises(ValueError):
            self.collect(receipt_path=self.evidence("bool-one.json", receipt))

    def test_outer_digest_and_git_identity_fields_are_validated(self):
        for key in ("model_sha256", "model_content_identity", "binary_sha256"):
            for value in (True, 1, "f" * 63, "g" * 64, " f" * 32, ""):
                with self.subTest(key=key, value=value), self.assertRaises(ValueError):
                    self.collect(receipt_path=self.evidence("bad-hex.json", {key: value}))
            self.assertTrue(self.collect(receipt_path=self.evidence("valid-hex.json", {key: "a" * 64}))["consistent"])
        with self.assertRaises(ValueError):
            self.collect(receipt_path=self.evidence("bad-git.json", {"source_revision": "not-git"}))
        for width in (40, 64):
            with mock.patch.object(INVENTORY.subprocess, "run", side_effect=OSError("git unavailable")):
                result = self.collect(receipt_path=self.evidence("git.json", {"source_revision": "a" * width}))
            self.assertTrue(result["consistent"])
            self.assertIsNone(result["source"]["revision"])

    def test_own_exporter_schemas_are_closed_but_external_receipt_extras_are_open(self):
        base = self.selected_inventory()
        for target in ("inventory", "step", "variant", "numerics"):
            value = copy.deepcopy(base)
            obj = value if target == "inventory" else value["steps"][0]
            if target in ("variant", "numerics"):
                obj = obj["declared_variant"]
            if target == "numerics":
                obj = obj["numerics"]
            obj["unrecognized_native_pass"] = True
            with self.subTest(target=target), self.assertRaises(ValueError):
                self.collect(receipt_path=self.evidence("unknown-owned.json", {"execution_inventory": value}))
        receipt = {"execution_inventory": base, "future_receipt_metadata": {"nested": [1, 2, "valid"]}}
        result = self.collect(receipt_path=self.evidence("real-shaped.json", receipt),
            audit_path=self.evidence("audit-extra.json", {"extra": {"metadata": True}}),
            manifest_path=self.evidence("manifest-extra.json", {"extra": [3]}),
            platform_path=self.evidence("platform-extra.json", {"nested": {"new": "field"}}))
        self.assertTrue(result["consistent"])
        self.assertIsNone(result["native_tests_executed"])
        self.assertEqual(result["operators"][0]["declared_variant"]["numerics"]["row_invariant"], True)

    def test_exporter_optional_declarations_validated_without_rejecting_zero_shapes(self):
        for field in ("whole_plan_graph_safe_declared", "family", "hidden_size", "auxiliary_size", "wave_size",
                      "row_invariant", "deterministic"):
            value = self.selected_inventory()
            if field == "whole_plan_graph_safe_declared":
                value[field] = "true"
            elif field in ("row_invariant", "deterministic"):
                value["steps"][0]["declared_variant"]["numerics"][field] = 1
            else:
                value["steps"][0]["declared_variant"][field] = True
            with self.subTest(field=field), self.assertRaises(ValueError):
                self.collect(receipt_path=self.evidence("bad-declaration.json", {"execution_inventory": value}))
        value = self.selected_inventory()
        value["steps"][0]["declared_variant"]["auxiliary_size"] = 0
        value["steps"][0]["declared_variant"]["qualification"] = "rejected"
        result = self.collect(receipt_path=self.evidence("declared-rejected.json", {"execution_inventory": value}))
        self.assertTrue(result["consistent"])
        self.assertIsNone(result["operators"][0]["qualification"])

    def test_CMake_filters_unrelated_entries_before_parsing_full_boolean_table(self):
        for text in ("ON", "TRUE", "YES", "Y", "1", "2", "-1", "0.25", "1e3"):
            with self.subTest(value=text):
                cache = INVENTORY.cmake_cache(self.text("truth-cache", f"STRIX_ENABLE_HIP_RUNTIME:BOOL={text}\nUNRELATED:BOOL=arbitrary\n"))
                self.assertIs(cache["STRIX_ENABLE_HIP_RUNTIME"], True)
                self.assertNotIn("UNRELATED", cache)
        for text in ("OFF", "FALSE", "NO", "N", "0", "", "IGNORE", "NOTFOUND", "unused-NOTFOUND", "0.0", "-0", "0e99"):
            with self.subTest(value=text):
                self.assertIs(INVENTORY.cmake_cache(self.text("false-cache", f"BUILD_TESTING:BOOL={text}\n"))["BUILD_TESTING"], False)
        with self.assertRaises(ValueError):
            self.collect(cache_path=self.text("not-BOOL-cache", "STRIX_ENABLE_HIP_RUNTIME:STRING=OFF\n"))

    def test_CLI_invalid_evidence_exit_2_distinct_from_consistent_conflict_exit_3(self):
        path = self.text("invalid-CLI.json", '{"unexpected":1e999}')
        with contextlib.redirect_stderr(io.StringIO()), self.assertRaises(SystemExit) as caught:
            INVENTORY.main(["--repo", str(self.root), "--receipt", str(path)])
        self.assertEqual(caught.exception.code, 2)


    def test_hex_identity_normalizes_case(self):
        self.assertEqual(INVENTORY.hex_identity("AB" * 32, "digest"), "ab" * 32)
        path = self.text("model.bin", "payload")
        digest = hashlib.sha256(b"payload").hexdigest()
        result = self.collect(model_path=path,
            receipt_path=self.evidence("uppercase-receipt.json", {"model_sha256": digest.upper()}))
        self.assertTrue(result["consistent"])

    def test_external_audit_status_must_be_string(self):
        for status in (None, [], {}, 1, True):
            with self.subTest(status=status), self.assertRaises(ValueError):
                INVENTORY.operator_records({"requirements": [{"op": "lm_head", "status": status}]}, {})


    def test_reported_environment_never_proves_effective_queues_or_overlap(self):
        value = {"source": "decoder-process", "requested": {key: None for key in INVENTORY.RUNTIME_ENVIRONMENT_KEYS},
                 "effective_hw_queues": None, "concurrent_execution_observed": None}
        value["requested"]["GPU_MAX_HW_QUEUES"] = "1"
        result = self.collect(receipt_path=self.evidence("environment.json", {"hip_runtime_environment": value}))
        self.assertEqual(result["reported_hip_runtime_environment"], value)
        self.assertTrue(result["consistent"])
        for field, setting in (("effective_hw_queues", 1), ("concurrent_execution_observed", True), ("source", "collector-process")):
            bad = copy.deepcopy(value)
            bad[field] = setting
            with self.subTest(field=field), self.assertRaises(ValueError):
                INVENTORY.validate_runtime_environment(bad)
        for setting in (False, 1, []):
            bad = copy.deepcopy(value)
            bad["requested"]["GPU_MAX_HW_QUEUES"] = setting
            with self.subTest(setting=setting), self.assertRaises(ValueError):
                INVENTORY.validate_runtime_environment(bad)



if __name__ == "__main__":
    unittest.main()
