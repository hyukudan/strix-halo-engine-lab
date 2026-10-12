from __future__ import annotations

import copy
import importlib.util
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest import mock

sys.dont_write_bytecode = True

SPEC = importlib.util.spec_from_file_location("numerical_campaign", Path(__file__).with_name("campaign.py"))
c = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(c)


class Fixture:
    def __init__(self, root):
        self.root = Path(root); self.serial = 0
        self.model = self.raw(b"GGUF\x03\x00\x00\x00" + bytes(56), "model.gguf")
        self.exe = self.raw(b"synthetic-test-binary", "runtime")
        source = self.raw(b"synthetic-test-source", "source.cpp")
        self.sources = self.json({"schema":"strix-source-manifest-v1", "files":[source]}, "sources.json")
        tensor = {"name":"output.weight", "role":"output_head", "residency":"resident", "placement":{"kind":"local","node":0},
                  "layer":None, "ggml_type":2, "byte_size":54, "absolute_offset":8, "dimensions":[32,3]}
        manifest = {"architecture":"synthetic-test-only", "layout_identity":17, "total_tensor_bytes":54,
                    "file_backed_candidate_bytes":0, "bounded_cache_candidate_bytes":0, "unknown_tensor_count":0, "tensors":[tensor]}
        audit = {"manifest_layout_identity":17, "requirements":[{"step_id":0,"op":"lm_head","slot":0,"cardinality":"exactly_one",
                 "role":"output_head","layer":None,"name_contains":"output","match_count":1,"status":"satisfied","tensor_indices":[0]}],
                 "satisfied":1,"missing":0,"ambiguous":0,"compiled":True,"tensor_access_plan_identity":19}
        self.receipt = {"schema":c.SCHEMA,"kind":"host_tensor_audit","model":self.model,"tools":[self.exe,self.exe],
                        "manifest":self.json(manifest,"manifest.json"),"tensor_audit":self.json(audit,"audit.json"),"native_execution":False}
        self.receipt_pin = self.json(self.receipt, "receipt.json")
        prompts = c.load(Path(__file__).with_name("workloads.json"))
        for case in prompts["cases"]:
            case["context_min_tokens"] = 2
        self.prompts = self.json(prompts, "prompts.json")
        self.plan = c.prepare_plan(self.receipt_pin["path"], self.prompts["path"])
        self.plan["runtime"] = {"status":"recorded","executable":self.exe,"source_manifest":self.sources,"mode":"eager_integer_lm_head",
                                "candidate_id":"t0:decode:M1:W4A8","numerical_identity":"synthetic-integer-v1",
                                "kernel_plan_identity":"synthetic-kernel-v1","prefill_identity":"synthetic-floating-prefill",
                                "state_history_identity":"synthetic-canonical-history-contract","input_storage":"fp32"}
        selected = next(x for x in self.plan["candidates"] if x["id"] == self.plan["runtime"]["candidate_id"])
        self.plan["runtime"]["numerical_identity"] = c.integer_profile(selected,"fp32")
        self.observations = {"schema":"strix-numerical-observations-v1","plan_sha256":"", "cases":{}}
        for case in self.plan["cases"]:
            reference = self.record(case, False)
            result = self.record(case, True)
            self.plan["references"][case["id"]] = {"status":"recorded","artifact":reference}
            self.observations["cases"][case["id"]] = {"result":result, "replay":self.record(case, True, restored=True) if case["kind"] in {"state","context"} else None}
        self.rebind()

    def raw(self, value, name=None):
        self.serial += 1; path = self.root / (name or f"artifact-{self.serial}")
        path.write_bytes(value)
        return c.pin(path, c.MAX_LOGITS_FILE)

    def json(self, value, name=None):
        return self.raw(c.canonical(value), name)

    def logits(self, rows=None):
        rows = rows or [[0.0,2.0,1.0],[3.0,0.0,1.0]]
        return self.raw(b"".join(c.canonical({"position":i,"target_token":token,"logits":row})+b"\n"
                                 for i,(token,row) in enumerate(zip([1,0], rows))))

    def record(self, case, candidate, restored=False):
        data = {"schema":"strix-numerical-record-v1","model_sha256":self.model["sha256"],"prompt_sha256":c.identity(case),
                "executable":self.exe,"source_manifest":self.sources,"numerical_identity":self.plan["runtime"]["numerical_identity"] if candidate else "synthetic-independent-oracle",
                "kernel_plan_identity":"synthetic-kernel-v1","prefill_identity":"synthetic-floating-prefill",
                "state_history_identity":"synthetic-canonical-history-contract","history_sha256":"a"*64,"input_tokens":[0,1],
                "logits":self.logits(),"teacher_forced_target_tokens":[1,0],"generation_logits":self.logits(),"output_tokens":[1,0],
                "output_text":json.dumps(case["expected"]) if case["kind"] in {"json","tool","context"} else "synthetic fixture text",
                "state":{name:self.raw(b"fixture-state-"+name.encode()) for name in c.STATE_CLASSES},
                "capture_contract":{"origin":"fresh","history_contract":c.FRESH_RAW_HISTORY,"restoration_record":None},
                "timings_ms":{name:1.0 for name in ("input_widen","activation_prepare","weight_repack","projection","scale_correction",
                                                    "output_conversion","status_sync","logits_commit","complete_wall")}}
        if restored:
            history = "synthetic-explicit-restored-prefix-history"
            evidence = {"schema":"strix-declared-state-restoration-v1","origin":"restored","model_sha256":data["model_sha256"],
                        "executable_sha256":data["executable"]["sha256"],"source_manifest_sha256":data["source_manifest"]["sha256"],
                        "history_sha256":data["history_sha256"],"state_history_identity":data["state_history_identity"],
                        **{name:data[name] for name in ("numerical_identity","kernel_plan_identity","prefill_identity")},
                        "history_contract":history,"restore_log":self.raw(b"synthetic restore event, never native proof")}
            data["capture_contract"] = {"origin":"restored","history_contract":history,"restoration_record":self.json(evidence)}
        return self.json(data)

    def change(self, case_id, mutate, which="result"):
        value = c.load(self.observations["cases"][case_id][which]["path"])
        mutate(value)
        self.observations["cases"][case_id][which] = self.json(value)

    def rebind(self):
        self.observations["plan_sha256"] = c.identity(self.plan)


class Tests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(prefix="strix-numerical-test-")
        self.addCleanup(self.temp.cleanup)
        self.f = Fixture(self.temp.name)

    def test_valid_independent_metrics_but_never_qualification(self):
        result = c.evaluate(self.f.plan, self.f.observations)
        self.assertEqual(result["status"], "pass")
        self.assertFalse(result["qualification"]); self.assertFalse(result["native_execution_claim"])
        metrics = result["cases"]["teacher-forced"]["metrics"]
        self.assertEqual(metrics["max_mean_kl"], 0.0)
        self.assertEqual(metrics["teacher_forced_positions"], 2)

    def test_unknown_reference_and_runtime_never_pass(self):
        self.f.plan["references"]["greedy-ties"] = {"status":"unknown","reason":"baseline not measured"}; self.f.rebind()
        self.assertEqual(c.evaluate(self.f.plan,self.f.observations)["status"],"unknown")

    def test_fresh_duplicates_do_not_establish_restored_state_and_missing_stays_unknown(self):
        self.f.observations["cases"]["recurrent-resume"]["replay"] = self.f.observations["cases"]["recurrent-resume"]["result"]
        with self.assertRaises(c.Invalid): c.evaluate(self.f.plan,self.f.observations)
        self.f.observations["cases"]["recurrent-resume"]["replay"] = None
        self.assertEqual(c.evaluate(self.f.plan,self.f.observations)["status"],"unknown")

    def test_restoration_record_profile_and_nested_log_are_identity_bound(self):
        replay = c.load(self.f.observations["cases"]["recurrent-resume"]["replay"]["path"])
        evidence = c.load(replay["capture_contract"]["restoration_record"]["path"])
        evidence["numerical_identity"] = "foreign profile"
        replay["capture_contract"]["restoration_record"] = self.f.json(evidence)
        self.f.observations["cases"]["recurrent-resume"]["replay"] = self.f.json(replay)
        with self.assertRaises(c.Invalid): c.evaluate(self.f.plan,self.f.observations)
        evidence["numerical_identity"] = self.f.plan["runtime"]["numerical_identity"]
        replay["capture_contract"]["restoration_record"] = self.f.json(evidence)
        self.f.observations["cases"]["recurrent-resume"]["replay"] = self.f.json(replay)
        Path(evidence["restore_log"]["path"]).write_bytes(b"changed restoration log")
        with self.assertRaises(c.Invalid): c.evaluate(self.f.plan,self.f.observations)
        self.f.plan["runtime"] = {"status":"unknown"}; self.f.rebind()
        self.assertEqual(c.evaluate(self.f.plan,self.f.observations)["status"],"unknown")

    def test_empty_subset_extra_case_rejected(self):
        for value in ({}, {"teacher-forced":self.f.observations["cases"]["teacher-forced"]},
                      {**self.f.observations["cases"],"extra":{}}):
            obs = copy.deepcopy(self.f.observations); obs["cases"] = value
            with self.assertRaises(c.Invalid): c.evaluate(self.f.plan,obs)

    def test_wrong_model_runtime_source_numerics_history(self):
        for field, value in (("model_sha256","b"*64),("numerical_identity","wrong"),
                             ("prefill_identity","wrong"),("state_history_identity","wrong"),("kernel_plan_identity","wrong")):
            original = copy.deepcopy(self.f.observations)
            self.f.change("teacher-forced", lambda data: data.update({field:value}))
            with self.assertRaises(c.Invalid): c.evaluate(self.f.plan,self.f.observations)
            self.f.observations = original
        self.f.change("teacher-forced", lambda data: data.update(executable=self.f.raw(b"other binary")))
        with self.assertRaises(c.Invalid): c.evaluate(self.f.plan,self.f.observations)

    def test_recorded_json_syntax_is_not_semantic_success(self):
        self.f.change("json-object", lambda data: data.update(output_text='{"answer":41,"valid":true}'))
        self.assertEqual(c.evaluate(self.f.plan,self.f.observations)["status"],"fail")
        self.assertFalse(c.semantic_equal({"valid":True},{"valid":1}))
        self.assertTrue(c.semantic_equal({"n":2},{"n":2.0}))

    def test_tool_argument_tamper_fails(self):
        self.f.change("tool-arguments", lambda data: data.update(output_text='{"name":"add","arguments":{"a":3,"b":2}}'))
        self.assertEqual(c.evaluate(self.f.plan,self.f.observations)["status"],"fail")

    def test_replay_state_and_history_mismatch(self):
        self.f.change("recurrent-resume", lambda data: data["state"].update(recurrent=self.f.raw(b"different-state")), "replay")
        self.assertEqual(c.evaluate(self.f.plan,self.f.observations)["status"],"fail")
        self.f.change("long-context", lambda data: data.update(history_sha256="b"*64),"replay")
        with self.assertRaises(c.Invalid): c.evaluate(self.f.plan,self.f.observations)

    def test_exact_replay_rejects_signed_zero_logit_change(self):
        self.f.change("recurrent-resume", lambda data:data.update(logits=self.f.logits([[-0.0,2,1],[3,0,1]])), "replay")
        result = c.evaluate(self.f.plan,self.f.observations)
        self.assertEqual(result["status"],"fail")
        self.assertFalse(result["cases"]["recurrent-resume"]["within_plan_state_match"])

    def test_exact_replay_checks_actual_generation_logits_not_only_teacherforced_rows(self):
        self.f.change("recurrent-resume", lambda data:data.update(generation_logits=self.f.logits([[0.25,2,1],[3,0,1]])), "replay")
        result = c.evaluate(self.f.plan,self.f.observations)
        self.assertEqual(result["status"],"fail")
        self.assertFalse(result["cases"]["recurrent-resume"]["within_plan_state_match"])

    def test_unmeasured_diagnostic_costs_remain_unknown(self):
        self.f.change("teacher-forced", lambda data:data.update(timings_ms={name:None for name in data["timings_ms"]}))
        result = c.evaluate(self.f.plan,self.f.observations)
        self.assertEqual(result["status"],"pass")
        self.assertIsNone(result["cases"]["teacher-forced"]["timings_ms"]["activation_prepare"])
        self.assertFalse(result["performance_evidence"])

    def test_context_min_and_actual_vocab_bound(self):
        for value in ([0], [0,3]):
            original = copy.deepcopy(self.f.observations)
            self.f.change("long-context", lambda data: data.update(input_tokens=value))
            with self.assertRaises(c.Invalid): c.evaluate(self.f.plan,self.f.observations)
            self.f.observations = original

    def test_empty_truncated_nonfinite_logit_trace_rejected(self):
        traces = [b"\n", c.canonical({"position":0,"target_token":1,"logits":[0,2]})+b"\n",
                  b'{"position":0,"target_token":1,"logits":[0,NaN,1]}\n',
                  c.canonical({"position":0,"target_token":1,"logits":[0,2,1]})+b"\n"]
        for raw in traces:
            original = copy.deepcopy(self.f.observations)
            self.f.change("teacher-forced",lambda data:data.update(logits=self.f.raw(raw)))
            with self.assertRaises(c.Invalid): c.evaluate(self.f.plan,self.f.observations)
            self.f.observations = original

    def test_generated_token_must_match_generation_logits(self):
        self.f.change("greedy-ties",lambda data:data.update(output_tokens=[2,0]))
        with self.assertRaises(c.Invalid): c.evaluate(self.f.plan,self.f.observations)

    def test_teacher_forced_and_generated_are_distinct(self):
        self.assertEqual(c.token_edit_rate([1,2,3],[1,3]),1/3)
        a = self.f.logits([[0,2,1],[3,0,1]])
        b = self.f.logits([[0,2.02,1],[3,0,1]])
        metrics = c.compare_logits(a,b,3)
        import math
        p = [math.exp(x)/sum(math.exp(y) for y in [0,2,1]) for x in [0,2,1]]
        q = [math.exp(x)/sum(math.exp(y) for y in [0,2.02,1]) for x in [0,2.02,1]]
        independent = sum(x*math.log(x/y) for x,y in zip(p,q))/2
        self.assertAlmostEqual(metrics["max_mean_kl"],independent,places=14)
        self.assertAlmostEqual(metrics["max_abs_logit_error"],0.02)

    def test_near_tie_greedy_and_large_error_fail(self):
        self.f.change("teacher-forced",lambda data:data.update(logits=self.f.logits([[0,1,2],[3,0,1]])))
        result = c.evaluate(self.f.plan,self.f.observations)
        self.assertEqual(result["status"],"fail")
        self.assertEqual(result["cases"]["teacher-forced"]["metrics"]["max_teacher_forced_argmax_mismatch_rate"],0.5)

    def test_identity_threshold_and_phase_graph_substitution(self):
        self.f.plan["thresholds"]["max_abs_logit_error"] = 9.0
        with self.assertRaises(c.Invalid): c.evaluate(self.f.plan,self.f.observations)
        for candidate_id, mode in (("t0:prefill:M16:W4A8","eager_integer_lm_head"),
                                   ("t0:verify:M2:W4A8","eager_integer_lm_head"),
                                   ("t0:decode:M1:W4A4","graph_float")):
            plan = copy.deepcopy(self.f.plan); plan["runtime"].update(candidate_id=candidate_id,mode=mode)
            with self.assertRaises(c.Invalid): c.validate_plan(plan)

    def test_audit_missing_unknown_identity_and_model_drift(self):
        audit = c.load(self.f.receipt["tensor_audit"]["path"]); audit["missing"] = 1
        receipt = copy.deepcopy(self.f.receipt); receipt["tensor_audit"] = self.f.json(audit)
        with self.assertRaises(c.Invalid): c.verify_audit(receipt)
        manifest = c.load(self.f.receipt["manifest"]["path"]); manifest["unknown_tensor_count"] = 1
        receipt = copy.deepcopy(self.f.receipt); receipt["manifest"] = self.f.json(manifest)
        with self.assertRaises(c.Invalid): c.verify_audit(receipt)
        Path(self.f.model["path"]).write_bytes(b"changed model")
        with self.assertRaises(c.Invalid): c.validate_plan(self.f.plan)

    def test_actual_shapes_storage_groups_padding_and_costs(self):
        manifest = {"tensors":[{"role":"output_head","name":"real-shape","dimensions":[2560,153600],"ggml_type":23,"byte_size":208896000},
                                {"role":"moe_expert","name":"K640","dimensions":[640,2560,4],"ggml_type":23,"byte_size":1}]}
        candidates = c.candidate_matrix(manifest)
        candidate = next(x for x in candidates if x["id"]=="t0:decode:M1:W4A4")
        self.assertEqual(candidate["operand"],"iu8")
        self.assertEqual(candidate["costs"]["column_tiles"],150)
        self.assertEqual(candidate["costs"]["preparation_elements_per_complete_projection"],384000)
        self.assertEqual(candidate["costs"]["internal_allocation_count"],12)
        self.assertTrue(candidate["costs"]["within_256MiB_requested_peak"])
        self.assertEqual(candidate["costs"]["storage_block_elements"],256)
        self.assertEqual(candidate["costs"]["scale_group_elements"],32)
        self.assertFalse(next(x for x in candidates if x["id"]=="t1:decode:M1:W4A8")["mathematically_representable"])
        manifest["tensors"][1]["ggml_type"] = 2
        manifest["tensors"][1]["byte_size"] = 4 * 2560 * 20 * 18
        candidate = next(x for x in c.candidate_matrix(manifest) if x["id"]=="t1:decode:M1:W4A4")
        self.assertTrue(candidate["mathematically_representable"]); self.assertEqual(candidate["operand"],"iu4")
        self.assertEqual(candidate["costs"]["expert_rows"],"unknown_requires_observed_histogram")
        self.assertIsNone(next(x for x in candidates if x["activation_class"]=="W4A16")["costs"]["wmma_tile"])

    def test_source_nested_result_logit_drift_rejected_after_evaluation(self):
        original = c.compare_logits; count = 0
        def drift(*args):
            nonlocal count
            value = original(*args); count += 1
            if count == 1:
                record = c.load(self.f.observations["cases"]["teacher-forced"]["result"]["path"])
                Path(record["logits"]["path"]).write_bytes(b"changed nested result")
            return value
        with mock.patch.object(c,"compare_logits",side_effect=drift):
            with self.assertRaises(c.Invalid): c.evaluate(self.f.plan,self.f.observations)

    def test_private_git_symlink_and_fresh_output(self):
        git = self.f.root / "checkout"; git.mkdir(); (git / ".git").mkdir()
        with self.assertRaises(c.Invalid): c.write_new({},git / "out.json")
        outside = self.f.root / "fresh.json"; c.write_new({},outside)
        with self.assertRaises(FileExistsError): c.write_new({},outside)
        try:
            link = self.f.root / "link"; link.symlink_to(git,target_is_directory=True)
        except OSError:
            return
        with self.assertRaises(c.Invalid): c.write_new({},link / "out.json")

    def test_json_duplicate_nonfinite_and_size_bounds(self):
        for raw in (b'{"a":1,"a":2}', b'{"a":Infinity}'):
            with self.assertRaises(c.Invalid): c.parse_json(raw)
        with self.assertRaises(c.Invalid): c.load(self.f.prompts["path"],4)

    def test_cli_unknown_fail_exit_codes_and_new_reports(self):
        plan = self.f.json(self.f.plan); observations = self.f.json(self.f.observations)
        output = self.f.root / "report.json"
        self.assertEqual(c.main(["evaluate","--plan",plan["path"],"--observations",observations["path"],"--out",str(output)]),0)
        self.f.change("json-object",lambda data:data.update(output_text='{}'))
        observations = self.f.json(self.f.observations)
        self.assertEqual(c.main(["evaluate","--plan",plan["path"],"--observations",observations["path"],"--out",str(self.f.root/"fail.json")]),4)
        self.f.plan["runtime"] = {"status":"unknown"}; self.f.rebind()
        plan = self.f.json(self.f.plan); observations = self.f.json(self.f.observations)
        self.assertEqual(c.main(["evaluate","--plan",plan["path"],"--observations",observations["path"],"--out",str(self.f.root/"unknown.json")]),3)

    def test_host_audit_exact_tool_and_model_prepost_binding(self):
        manifest = self.f.raw(b"synthetic-native-host-tool", "strix-model-manifest")
        audit = self.f.raw(b"synthetic-native-host-tool-two", "strix-audit-execution-tensors")
        def fake(tool, model, output, expected):
            self.assertEqual(c.pin(tool,c.MAX_LOGITS_FILE),expected)
            source = self.f.receipt["manifest"] if Path(tool).name == "strix-model-manifest" else self.f.receipt["tensor_audit"]
            Path(output).write_bytes(Path(source["path"]).read_bytes())
            return expected
        with mock.patch.object(c,"run_host_tool",side_effect=fake):
            result = c.audit_model(self.f.model["path"],self.f.model["sha256"],manifest["path"],manifest["sha256"],
                                   audit["path"],audit["sha256"],self.f.root/"audited")
            self.assertFalse(result["native_execution"])
            with self.assertRaises(c.Invalid):
                c.audit_model(self.f.model["path"],self.f.model["sha256"],manifest["path"],manifest["sha256"],
                              audit["path"],audit["sha256"],self.f.root/"audited")
        with self.assertRaises(c.Invalid):
            c.audit_model(self.f.model["path"],"b"*64,manifest["path"],manifest["sha256"],audit["path"],audit["sha256"],self.f.root/"wrong-model")
        with self.assertRaises(c.Invalid):
            c.audit_model(self.f.model["path"],self.f.model["sha256"],manifest["path"],"b"*64,audit["path"],audit["sha256"],self.f.root/"wrong-tool")

    def test_float32_control_is_distinct_from_requested_W4A16(self):
        plan = copy.deepcopy(self.f.plan)
        manifest = c.load(self.f.receipt["manifest"]["path"])
        manifest["tensors"][0]["ggml_type"] = 20
        receipt = copy.deepcopy(self.f.receipt); receipt["manifest"] = self.f.json(manifest)
        plan["audit_receipt"] = self.f.json(receipt); plan["candidates"] = c.candidate_matrix(manifest)
        plan["runtime"].update(mode="graph_float",candidate_id="t0:decode:M1:FP32_activation_control",numerical_identity=c.floating_profile("fp32"))
        c.validate_plan(plan)
        plan["runtime"].update(input_storage="bf16",candidate_id="t0:decode:M1:BF16_activation_control",numerical_identity=c.floating_profile("bf16"))
        c.validate_plan(plan)
        plan["runtime"]["input_storage"] = "fp32"
        with self.assertRaises(c.Invalid): c.validate_plan(plan)
        plan["runtime"]["candidate_id"] = "t0:decode:M1:W4A16"
        with self.assertRaises(c.Invalid): c.validate_plan(plan)

    def test_actual_floating_launcher_formats_and_integer_formats_are_separate(self):
        for kind, block, size in ((0,1,4),(1,1,2),(8,32,34),(20,32,18),(23,256,136),(30,1,2),(2,32,18),(99,1,1)):
            tensor = {"role":"output_head","name":"head","dimensions":[2560,153600],"ggml_type":kind,
                      "byte_size":153600*(2560//block)*size}
            candidates = c.candidate_matrix({"tensors":[tensor]})
            control = next(x for x in candidates if x["id"] == "t0:decode:M1:FP32_activation_control")
            integer = next(x for x in candidates if x["id"] == "t0:decode:M1:W4A8")
            self.assertEqual(control["mathematically_representable"], kind in {0,1,8,20,23,30})
            self.assertEqual(integer["mathematically_representable"], kind in {2,23})
            self.assertEqual(control["ggml_type"],kind)
            bf16 = next(x for x in candidates if x["id"] == "t0:decode:M1:BF16_activation_control")
            self.assertEqual(bf16["mathematically_representable"],control["mathematically_representable"])
            self.assertIsNone(control["costs"]["scale_group_elements"])
            self.assertIsNone(control["costs"]["staged_weight_bytes_per_largest_tile"])
            if control["mathematically_representable"]:
                self.assertEqual(control["costs"]["checkpoint_bytes_per_expert_projection"],tensor["byte_size"])
        tensor["ggml_type"] = 8; tensor["dimensions"][0] = 640
        tensor["byte_size"] = 153600*20*34
        self.assertTrue(next(x for x in c.candidate_matrix({"tensors":[tensor]}) if x["activation_class"]=="FP32_activation_control")["mathematically_representable"])
        tensor["byte_size"] -= 1
        self.assertFalse(next(x for x in c.candidate_matrix({"tensors":[tensor]}) if x["activation_class"]=="FP32_activation_control")["mathematically_representable"])
        tensor.update(dimensions=[32,3,1],ggml_type=2,byte_size=54)
        integer = next(x for x in c.candidate_matrix({"tensors":[tensor]}) if x["id"]=="t0:decode:M1:W4A8")
        self.assertEqual(integer["integration"],"projection_diagnostic_only")
        alternate = next(x for x in c.candidate_matrix({"tensors":[tensor]}) if x["id"]=="t0:decode:M1:W4A4:iu8")
        default = next(x for x in c.candidate_matrix({"tensors":[tensor]}) if x["id"]=="t0:decode:M1:W4A4")
        self.assertEqual(alternate["operand"],"iu8"); self.assertEqual(default["operand"],"iu4")
        self.assertEqual(alternate["costs"],default["costs"])
        self.assertIn(":symmetric_a4:iu8:",c.integer_profile(alternate,"bf16"))

    def test_floating_profile_exact_storage_and_scoped_arithmetic_descriptor(self):
        expected = ("strix-experimental-float-lm-head-v1:decode:C1:M1:bf16-source-widen-f32:"
                    "original-ggml-weight-decode-f32:256-thread-feature-stride:fp32-sum-binary-tree:"
                    "fp32-logits:compiler-bound-arithmetic:no-whole-model-precision-claim")
        self.assertEqual(c.floating_profile("bf16"),expected)
        self.assertNotEqual(c.floating_profile("fp32"),expected)
        with self.assertRaises(c.Invalid): c.floating_profile("float16")

    def test_unknown_plan_byte_admission_is_checked_before_creating_output(self):
        plan = c.prepare_plan(self.f.receipt_pin["path"],self.f.prompts["path"])
        self.assertEqual(plan["runtime"],{"status":"unknown"})
        self.assertTrue(all(x["status"]=="unknown" for x in plan["references"].values()))
        encoded_size = len(c.canonical(plan))+1
        rejected = self.f.root/"never-created"/"plan.json"
        with self.assertRaises(c.Invalid): c.write_new(plan,rejected,maximum=encoded_size-1)
        self.assertFalse(rejected.parent.exists())
        accepted = self.f.root/"admitted-plan.json"
        c.write_new(plan,accepted,maximum=encoded_size)
        self.assertEqual(accepted.stat().st_size,encoded_size)
        self.assertEqual(c.load_plan(accepted),plan)
        c.validate_plan(c.load_plan(accepted))
        with self.assertRaises(c.Invalid): c.load(accepted,maximum=encoded_size-1)
        self.assertEqual(c.MAX_PLAN_JSON,256*1024*1024)
        self.assertEqual(c.MAX_JSON,16*1024*1024)

    def test_long_context_text_has_explicit_utf8_byte_bound_without_token_inference(self):
        prompts = c.load(self.f.prompts["path"])
        case = next(x for x in prompts["cases"] if x["kind"]=="context")
        case["text"] = "Neutral text. "*6000 + "界"*20000
        case["context_min_tokens"] = 16384
        self.assertGreater(len(case["text"].encode("utf-8")),65536)
        self.assertEqual(c.validate_prompts(prompts)["cases"][-1]["context_min_tokens"],16384)
        case["text"] = "界"*((1024*1024)//3+1)
        with self.assertRaises(c.Invalid): c.validate_prompts(prompts)

    def test_exact_current_cpp_profile_identity_and_unknown_code_rejection(self):
        candidate = next(x for x in self.f.plan["candidates"] if x["id"] == "t0:decode:M1:W4A8")
        self.assertIn(":11490292264919929327:",c.integer_profile(candidate,"fp32"))
        candidate = next(x for x in self.f.plan["candidates"] if x["id"] == "t0:decode:M1:W4A4")
        self.assertIn(":13887973429326095187:",c.integer_profile(candidate,"bf16"))
        for encoding in ("iq4_xs","unknown"):
            bad = copy.deepcopy(candidate); bad["encoding"] = encoding
            with self.assertRaises(c.Invalid): c.integer_profile(bad,"fp32")
        plan = copy.deepcopy(self.f.plan)
        plan["runtime"]["candidate_id"] = "t0:decode:M1:W4A4"
        with self.assertRaises(c.Invalid): c.validate_plan(plan)

    def test_nested_source_and_reference_state_drift(self):
        source = c.load(self.f.sources["path"])["files"][0]
        Path(source["path"]).write_bytes(b"source altered")
        with self.assertRaises(c.Invalid): c.evaluate(self.f.plan,self.f.observations)


if __name__ == "__main__":
    unittest.main()
