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

    def test_reported_uma_observation_is_retained_without_capacity_qualification(self):
        sample = {"kind": "reported-linux-host-counters-not-physical-fit", "supported": True,
            "observation_identity": 1, "scope": "reported-system-and-self-process-memory",
            "sampling": "bounded-two-file-read-interval-not-atomic", "begin_steady_ns": 10, "end_steady_ns": 20,
            "meminfo_status": "read", "process_status": "read", "mem_total_bytes": 131072,
            "mem_available_bytes": 65536, "process_rss_bytes": 1024, "process_peak_bytes": 2048,
            "process_peak_kind": "VmHWM-resident-high-water-mark", "gtt_reported_bytes": None, "rss_plus_gtt_computed": False}
        receipt = {"uma_setup_observation_boundary": "after-base-runtime-D1-catalog-before-optional-request-setup",
                   "uma_setup_observation": sample}
        observed = INVENTORY.reported_uma_setup_observation(receipt)
        self.assertEqual(observed["reported"], sample)
        self.assertIsNone(observed["independently_verified"])
        self.assertIsNone(observed["capacity_qualification"])
        output = self.collect(receipt_path=self.evidence("uma.json", receipt))
        self.assertEqual(output["reported_uma_setup_observation"], observed)
        self.assertIsNone(INVENTORY.reported_uma_setup_observation({}))
        for field, value in (("mem_available_bytes", 131073), ("process_rss_bytes", 2049),
                             ("mem_total_bytes", True), ("end_steady_ns", 9), ("observation_identity", 0),
                             ("rss_plus_gtt_computed", True), ("gtt_reported_bytes", 1),
                             ("meminfo_status", "malformed"), ("supported", False)):
            bad = copy.deepcopy(receipt);bad["uma_setup_observation"][field] = value
            with self.subTest(field=field), self.assertRaises(ValueError):
                INVENTORY.reported_uma_setup_observation(bad)
        bad = copy.deepcopy(receipt);del bad["uma_setup_observation_boundary"]
        with self.assertRaises(ValueError): INVENTORY.reported_uma_setup_observation(bad)
        unknown = copy.deepcopy(receipt)
        for field in ("mem_total_bytes", "mem_available_bytes", "process_rss_bytes", "process_peak_bytes"):
            unknown["uma_setup_observation"][field] = None
        unknown["uma_setup_observation"].update(supported=False, observation_identity=0,
            begin_steady_ns=0, end_steady_ns=0, meminfo_status="unsupported", process_status="unsupported")
        self.assertIsNone(INVENTORY.reported_uma_setup_observation(unknown)["reported"]["mem_total_bytes"])

    def test_serving_prefill_source_declarations_do_not_qualify_combined_profile(self):
        declaration = {"rows": 16, "qualification": "not_assessed", "execution": "known-input-native-chunks-with-R1-tail",
            "workspace_payload_bytes": 4096, "known_plan_metadata_payload_bytes": 128,
            "runtime_payload_reservation_bytes": 8192, "per_request_payload_reservation_bytes": 16384,
            "scope": "declared-payload-plus-explicit-serving-headroom-not-RSS-or-fit"}
        for width in (2, 4, 8, 16, 17, 2048):
            value = dict(declaration, rows=width)
            output = self.collect(receipt_path=self.evidence("serving-prefill.json", {"serving_prefill": value}))
            self.assertEqual(output["reported_opt_in_overlays"]["serving_prefill"], width)
            self.assertEqual(output["reported_serving_prefill"], value)
            self.assertEqual(output["profile_overlay_qualification"], "not_assessed")
            self.assertIsNone(output["native_tests_executed"])
        for field, value in (("rows", 3), ("rows", 15), ("rows", 2049), ("rows", True),
            ("qualification", "passed"), ("workspace_payload_bytes", 0),
            ("runtime_payload_reservation_bytes", False), ("scope", "physical-fit")):
            bad = dict(declaration);bad[field] = value
            with self.subTest(field=field, value=value), self.assertRaises(ValueError):
                INVENTORY.validate_serving_prefill(bad)
        self.assertIsNone(self.collect()["reported_opt_in_overlays"]["serving_prefill"])

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

    def test_tuning_exporter_fields_roundtrip_without_inventing_qualification(self):
        for reason in (None, "qualified_record", "invalid_key", "ambiguous_records",
                       "no_qualified_record", "selection_policy_mismatch"):
            with self.subTest(reason=reason):
                inventory = self.selected_inventory()
                step = inventory["steps"][0]
                step["offline_tuning_selection"] = reason
                step["declared_variant"]["numerical_contract_identity"] = 91 if reason == "qualified_record" else None
                result = self.collect(receipt_path=self.evidence("tuning.json", {"execution_inventory": inventory}))
                self.assertTrue(result["consistent"])
                decode, verify, prefill = result["operators"]
                self.assertEqual(decode["offline_tuning_selection"], reason)
                self.assertIsNone(decode["qualification"])
                self.assertIsNone(decode["executable_support"])
                self.assertIsNone(verify["offline_tuning_selection"])
                self.assertIsNone(prefill["offline_tuning_selection"])
                self.assertEqual(result["qualification_state"], "not_assessed")

    def test_tuning_exporter_rejects_unknown_tags_and_forged_numerical_evidence(self):
        for reason in (True, 1, [], {}, "", "fastest"):
            inventory = self.selected_inventory()
            inventory["steps"][0]["offline_tuning_selection"] = reason
            with self.subTest(reason=reason), self.assertRaises(ValueError):
                self.collect(receipt_path=self.evidence("bad-tuning.json", {"execution_inventory": inventory}))
        for identity in (True, False, 0, -1, 1 << 64, 1.0, "91"):
            inventory = self.selected_inventory()
            inventory["steps"][0]["declared_variant"]["numerical_contract_identity"] = identity
            with self.subTest(identity=identity), self.assertRaises(ValueError):
                self.collect(receipt_path=self.evidence("bad-contract.json", {"execution_inventory": inventory}))
        inventory = self.selected_inventory()
        inventory["steps"][0]["offline_tuning_selection"] = "qualified_record"
        with self.assertRaises(ValueError):
            self.collect(receipt_path=self.evidence("missing-contract.json", {"execution_inventory": inventory}))
        inventory["steps"][0].update(runtime_step=True, kernel_id=None, declared_variant=None)
        with self.assertRaises(ValueError):
            self.collect(receipt_path=self.evidence("runtime-tuning.json", {"execution_inventory": inventory}))

    def test_auto_base_profile_with_overlays_is_consistent_but_not_combination_qualified(self):
        modes = {"eager_submission_mode": "prepared-static-plan-experimental-v1",
                 "moe_shared_overlap_mode": "eager-sibling-streams-experimental-v1",
                 "generation_output_mode": "finite-firstmax-token-v1", "batched_moe_activation": True,
                 "prefetch_q8_lm_head": True, "wave8_q8_lm_head": True, "wave8_q8_gdn_input": True,
                 "wave8_q8_gdn_output": True, "wave8_q8_qsa_output": True,
                 "prefetch_hc_projection_next_tile": True, "q51_slot_parallel_down": True}
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
        self.assertEqual({key: value for key, value in result["reported_opt_in_overlays"].items()
                          if value is not None}, modes)
        self.assertIsNone(result["reported_opt_in_overlays"]["qsa_hierarchical_selection"])

    def test_new_runtime_consumers_are_reported_without_combination_qualification(self):
        overlays = {"known_prefix_window": True, "known_prefix_expert_union_requested": True,
                    "prompt_lookup": True, "serving_prefix_cache_requested": True,
                    "known_prefix_output_policy": "state-only", "gdn_recurrent_storage": "value-key",
                    "gdn_recurrent_layout": ".lds33-v1", "requested_logprobs_scope": "full-vocabulary",
                    "ple_row_cache_requested_rows": 64, "ple_cpu_helper_budget_requested": 2,
                    "offline_tuning_cache_sha256": "a" * 64,
                    "prompt_lookup_adaptive": {"enabled": True, "completed_cycles": 9},
                    "device_queue_trace": {"profiling_only": True, "kernel_busy_ns": None}}
        for field, declaration in overlays.items():
            receipt = self.receipt(True, "qualified-checkpoint")
            receipt[field] = declaration
            result = self.collect(cache_path=self.cache(a8="ON"), receipt_path=self.evidence("new-overlay.json", receipt))
            with self.subTest(field=field):
                self.assertTrue(result["consistent"])
                self.assertEqual(result["profile_overlay_qualification"], "not_assessed")
                expected = declaration if not isinstance(declaration, dict) else True
                self.assertEqual(result["reported_opt_in_overlays"][field], expected)
                self.assertIsNone(result["native_tests_executed"])

    def test_new_overlay_neutral_null_and_malformed_declarations(self):
        neutral = {"known_prefix_window": False, "known_prefix_expert_union_requested": False,
                   "prompt_lookup": False, "serving_prefix_cache_requested": False,
                   "gdn_recurrent_storage": "key-value", "ple_row_cache_requested_rows": 0,
                   "ple_cpu_helper_budget_requested": 0, "prompt_lookup_adaptive": {"enabled": False},
                   "device_queue_trace": {"profiling_only": False}}
        result = self.collect(receipt_path=self.evidence("neutral-new.json", neutral))
        self.assertIsNone(result["profile_overlay_qualification"])
        for field in neutral:
            self.assertIs(result["reported_opt_in_overlays"][field], False)
        for field in ("known_prefix_window", "known_prefix_expert_union_requested", "prompt_lookup",
                      "serving_prefix_cache_requested", "known_prefix_output_policy", "gdn_recurrent_storage",
                      "ple_row_cache_requested_rows", "ple_cpu_helper_budget_requested", "offline_tuning_cache_sha256",
                      "prompt_lookup_adaptive", "device_queue_trace"):
            unknown = self.collect(receipt_path=self.evidence("unknown-new.json", {field: None}))
            self.assertIsNone(unknown["reported_opt_in_overlays"][field])
            self.assertIsNone(unknown["profile_overlay_qualification"])
        bad = [{"known_prefix_window": 1}, {"known_prefix_expert_union_requested": "true"},
               {"serving_prefix_cache_requested": {}}, {"known_prefix_output_policy": "partial"},
               {"gdn_recurrent_storage": "transposed"}, {"ple_row_cache_requested_rows": True},
               {"ple_cpu_helper_budget_requested": 9}, {"offline_tuning_cache_sha256": "short"},
               {"prompt_lookup_adaptive": True}, {"prompt_lookup_adaptive": {"enabled": 1}},
               {"device_queue_trace": {"profiling_only": "true"}}]
        for declaration in bad:
            with self.subTest(declaration=declaration), self.assertRaises(ValueError):
                self.collect(receipt_path=self.evidence("invalid-new.json", declaration))

    def test_tiled_prefill_and_real_serving_cohort_declarations(self):
        for width in (2, 4):
            declaration = {"known_prefix_tiled_prefill": True,
                           "serving_tensor_cohort": {"requested_rows": width,
                                                    "qualification": "not_assessed",
                                                    "cohorts": 7}}
            result = self.collect(receipt_path=self.evidence("tiled-cohort.json", declaration))
            self.assertTrue(result["reported_opt_in_overlays"]["known_prefix_tiled_prefill"])
            self.assertEqual(result["reported_opt_in_overlays"]["serving_tensor_cohort"], width)
            self.assertIsNone(result["profile_overlay_qualification"])
        neutral = self.collect(receipt_path=self.evidence("neutral-cohort.json", {
            "known_prefix_tiled_prefill": False, "serving_tensor_cohort": {"requested_rows": 1}}))
        self.assertIs(neutral["reported_opt_in_overlays"]["serving_tensor_cohort"], False)
        self.assertIs(neutral["reported_opt_in_overlays"]["known_prefix_tiled_prefill"], False)
        for field in ("known_prefix_tiled_prefill", "serving_tensor_cohort"):
            unknown = self.collect(receipt_path=self.evidence("unknown-cohort.json", {field: None}))
            self.assertIsNone(unknown["reported_opt_in_overlays"][field])
        for declaration in ({"known_prefix_tiled_prefill": 1}, {"serving_tensor_cohort": True},
                            {"serving_tensor_cohort": {}}, {"serving_tensor_cohort": {"requested_rows": True}},
                            {"serving_tensor_cohort": {"requested_rows": 3}},
                            {"serving_tensor_cohort": {"requested_rows": "4"}}):
            with self.subTest(declaration=declaration), self.assertRaises(ValueError):
                self.collect(receipt_path=self.evidence("invalid-cohort.json", declaration))

    def test_cost_policy_and_snapshot_format_are_only_declarations(self):
        active = self.collect(receipt_path=self.evidence("cost-compact.json", {
            "moe_token_cost_policy": {"enabled": True, "qualification": "not_assessed"},
            "prompt_lookup_snapshot": {"policy": "occupied-kv-v1", "max_payload_bytes": 8192}}))
        self.assertIs(active["reported_opt_in_overlays"]["moe_token_cost_policy"], True)
        self.assertEqual(active["reported_opt_in_overlays"]["prompt_lookup_snapshot"], "occupied-kv-v1")
        self.assertIsNone(active["profile_overlay_qualification"])
        neutral = self.collect(receipt_path=self.evidence("cost-full.json", {
            "moe_token_cost_policy": {"enabled": False},
            "prompt_lookup_snapshot": {"policy": "full-components", "max_payload_bytes": 0}}))
        self.assertIs(neutral["reported_opt_in_overlays"]["moe_token_cost_policy"], False)
        self.assertIs(neutral["reported_opt_in_overlays"]["prompt_lookup_snapshot"], False)
        for field in ("moe_token_cost_policy", "prompt_lookup_snapshot"):
            result = self.collect(receipt_path=self.evidence("absent-cost.json", {field: None}))
            self.assertIsNone(result["reported_opt_in_overlays"][field])
        for value in (True, {}, {"policy": "occupied-kv-v1"},
                      {"policy": "occupied-kv-v1", "max_payload_bytes": 0},
                      {"policy": "occupied-kv-v1", "max_payload_bytes": True},
                      {"policy": "future", "max_payload_bytes": 8192}):
            with self.subTest(value=value), self.assertRaises(ValueError):
                self.collect(receipt_path=self.evidence("bad-snapshot.json", {"prompt_lookup_snapshot": value}))
        with self.assertRaises(ValueError):
            self.collect(receipt_path=self.evidence("bad-cost.json", {"moe_token_cost_policy": {"enabled": 1}}))

    def test_overlay_absence_false_unknown_and_invalid_types(self):
        inactive = {key: False for key in ("batched_moe_activation", "prefetch_q8_lm_head",
                                          "wave8_q8_lm_head", "wave8_q8_gdn_input",
                                          "wave8_q8_gdn_output", "wave8_q8_qsa_output", "prefetch_hc_projection_next_tile", "q51_slot_parallel_down", "qsa_hierarchical_selection")}
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
        for field in ("batched_moe_activation", "prefetch_q8_lm_head", "wave8_q8_lm_head", "wave8_q8_gdn_input",
                      "wave8_q8_gdn_output", "wave8_q8_qsa_output", "prefetch_hc_projection_next_tile", "q51_slot_parallel_down", "qsa_hierarchical_selection"):
            for value in (1, 0, 0.0, "true", [], {}):
                with self.subTest(field=field, value=value), self.assertRaises(ValueError):
                    self.collect(receipt_path=self.evidence("invalid-bool-overlay.json", {field: value}))
            unknown = self.collect(receipt_path=self.evidence("null-overlay.json", {field: None}))
            self.assertIsNone(unknown["reported_opt_in_overlays"][field])
            self.assertIsNone(unknown["profile_overlay_qualification"])

    def test_slot_parallel_counter_is_a_typed_observation_not_a_qualification(self):
        result = self.collect(receipt_path=self.evidence("slot-count.json", {
            "q51_slot_parallel_down": True, "slot_parallel_q51_submissions": 43,
            "q51_slot_parallel_down_qualification": "not_assessed"}))
        self.assertTrue(result["reported_opt_in_overlays"]["q51_slot_parallel_down"])
        self.assertEqual(result["profile_overlay_qualification"], "not_assessed")
        for value in (-1, True, 1.5, "43", 2**64):
            with self.subTest(value=value), self.assertRaises(ValueError):
                self.collect(receipt_path=self.evidence("bad-slot-count.json", {
                    "slot_parallel_q51_submissions": value}))

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
        for flag in ("q4k_a8g32_gate_up", "grouped_exact_q51_down", "q51_slot_parallel_down"):
            with self.subTest(flag=flag):
                result = self.collect(cache_path=self.cache(a8="OFF"),
                    receipt_path=self.evidence("explicit.json", {flag: True}))
                self.assertFalse(result["consistent"])
                self.assertIn("build/" + flag, {x["field"] for x in result["conflicts"]})
                self.assertTrue(self.collect(cache_path=self.cache(a8="ON"),
                    receipt_path=self.evidence("enabled.json", {flag: True}))["consistent"])
                self.assertTrue(self.collect(receipt_path=self.evidence("unknown.json", {flag: True}))["consistent"])
                for value in (0, 1, "false", None, [], {}):
                    if flag == "q51_slot_parallel_down" and value is None:
                        continue  # Nullable overlay declaration remains unknown, not false.
                    with self.subTest(value=value), self.assertRaises(ValueError):
                        self.collect(receipt_path=self.evidence("badflag.json", {flag: value}))

    def test_slot_parallel_declared_qualification_and_prerequisite_conflicts(self):
        for value in ("qualified", "", None, False, 1, [], {}):
            with self.subTest(value=value), self.assertRaises(ValueError):
                self.collect(receipt_path=self.evidence("bad-slot-qualification.json", {
                    "q51_slot_parallel_down_qualification": value}))
        for prerequisite in ("grouped_exact_q51_down", "q4k_a8g32_gate_up"):
            receipt = {"q51_slot_parallel_down": True, prerequisite: False}
            result = self.collect(receipt_path=self.evidence("bad-slot-prerequisite.json", receipt))
            self.assertFalse(result["consistent"])
            expected = "overlay/q51-grouped" if prerequisite == "grouped_exact_q51_down" else "overlay/q51-A8"
            self.assertIn(expected, {item["field"] for item in result["conflicts"]})
        # The roundtrip intentionally emits no grouped/A8 declarations. Absence
        # stays unknown, and an actual build capability is checked separately.
        receipt = {"q51_slot_parallel_down": True,
                   "q51_slot_parallel_down_qualification": "not_assessed",
                   "slot_parallel_q51_submissions": 43}
        for cache in (None, self.cache(a8="ON")):
            self.assertTrue(self.collect(cache_path=cache,
                receipt_path=self.evidence("valid-slot-roundtrip.json", receipt))["consistent"])
        for receipt in ({"slot_parallel_q51_submissions": 1},
                        {"q51_slot_parallel_down": True, "slot_parallel_q51_submissions": 43}):
            result = self.collect(cache_path=self.cache(a8="OFF"),
                receipt_path=self.evidence("slot-count-no-A8.json", receipt))
            self.assertFalse(result["consistent"])
            self.assertIn("build/slot_parallel_q51_submissions", {item["field"] for item in result["conflicts"]})
        self.assertTrue(self.collect(cache_path=self.cache(a8="OFF"),
            receipt_path=self.evidence("slot-count-zero.json", {
                "q51_slot_parallel_down": False, "slot_parallel_q51_submissions": 0}))["consistent"])

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


    def test_grouped_attention_and_recurrent_layout_contract(self):
        for heads in (1, 2, 4):
            INVENTORY.validate_outer_receipt({"qsa_gqa_heads_per_block": heads, "qsa_gqa_qualification": "not_assessed"})
        for heads in (True, 0, 3, "2", None):
            with self.subTest(heads=heads), self.assertRaises(ValueError):
                INVENTORY.validate_outer_receipt({"qsa_gqa_heads_per_block": heads})
        for layout in (".lds32-v1", ".lds33-v1", ".lds-transposed-v1"):
            INVENTORY.validate_outer_receipt({"gdn_recurrent_layout": layout, "gdn_recurrent_layout_qualification": "not_assessed"})
        with self.assertRaises(ValueError):
            INVENTORY.validate_outer_receipt({"gdn_recurrent_layout": "unknown"})
        with self.assertRaises(ValueError):
            INVENTORY.validate_outer_receipt({"qsa_gqa_submissions": 1, "qsa_gqa_layer_submissions": [0] * 48})

    def test_hierarchical_qsa_observation_contract(self):
        counts = [0] * 48
        counts[3] = 2
        good = {"qsa_hierarchical_selection": True,
                "qsa_hierarchical_selection_qualification": "not_assessed",
                "qsa_hierarchical_selection_submissions": 2,
                "qsa_hierarchical_selection_layer_submissions": counts}
        INVENTORY.validate_outer_receipt(good)
        for key, value in (("qsa_hierarchical_selection", 1),
                           ("qsa_hierarchical_selection_qualification", "passed"),
                           ("qsa_hierarchical_selection_submissions", 3),
                           ("qsa_hierarchical_selection_layer_submissions", counts[:-1])):
            bad = dict(good); bad[key] = value
            with self.subTest(key=key), self.assertRaises(ValueError):
                INVENTORY.validate_outer_receipt(bad)
        bad = dict(good); bad["qsa_hierarchical_selection"] = False
        with self.assertRaises(ValueError):
            INVENTORY.validate_outer_receipt(bad)

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
