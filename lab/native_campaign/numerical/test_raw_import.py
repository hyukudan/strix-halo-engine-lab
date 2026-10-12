from __future__ import annotations

import copy
import hashlib
from pathlib import Path
import struct
import sys
import tempfile
import types
import unittest
from unittest import mock

sys.dont_write_bytecode = True
sys.path.insert(0, str(Path(__file__).resolve().parent))
import campaign as c
import raw_import as r
from test_campaign import Fixture


def direct_fnv(words):
    result = 14695981039346656037
    for word in words:
        for byte in word.to_bytes(8,"little"):
            result ^= byte
            result = (result * 1099511628211) % (2**64)
    return result


class RawFixture:
    def __init__(self, root, runtime_mode="eager_float", activation_class="W4A8", restored=False, each=False):
        self.root = Path(root); self.base = Fixture(root)
        plan = self.plan = copy.deepcopy(self.base.plan)
        manifest = c.load(self.base.receipt["manifest"]["path"])
        manifest["tensors"][0]["ggml_type"] = 2 if runtime_mode == "eager_integer_lm_head" else 20
        audit = copy.deepcopy(self.base.receipt); audit["manifest"] = self.base.json(manifest)
        plan["audit_receipt"] = self.base.json(audit); plan["candidates"] = c.candidate_matrix(manifest)
        plan["runtime"].update(mode=runtime_mode,candidate_id="t0:decode:M1:BF16_activation_control",input_storage="bf16",
                               numerical_identity=c.floating_profile("bf16"),kernel_plan_identity="71")
        if runtime_mode == "eager_integer_lm_head":
            plan["runtime"]["candidate_id"] = "t0:decode:M1:"+activation_class
            selected = next(x for x in plan["candidates"] if x["id"]==plan["runtime"]["candidate_id"])
            plan["runtime"]["numerical_identity"] = c.integer_profile(selected,"bf16")
        self.case = next(x for x in plan["cases"] if x["id"] == "greedy-ties")
        self.tf = self.raw("teacher",False); self.gr = self.raw("greedy",True,each=each,restored=restored)
        self.text = self.base.raw(b"external decoded fixture text", "decoded.txt")
        tokenizer = {"schema":"strix-external-tokenization-v1","model_sha256":plan["model_sha256"],
                     "prompt_sha256":c.identity(self.case),"input_tokens":[0,1],"output_tokens":[1,0],
                     "output_text_sha256":self.text["sha256"],"tokenizer_identity":"synthetic-external-tokenizer",
                     "asserted_by":"external_operator"}
        raw = c.load(self.gr/"receipt.json")
        self.binding = {"schema":"strix-raw-import-binding-v1","plan_sha256":c.identity(plan),"case_id":self.case["id"],
                        "compile_receipt":self.base.json({"synthetic_fixture":True}),
                        "compile_binding":{"executable_sha256":plan["runtime"]["executable"]["sha256"],
                                           "source_manifest_sha256":plan["runtime"]["source_manifest"]["sha256"]},
                        "tokenization":self.base.json(tokenizer),"output_text":self.text,
                        "observed":{k:raw[k] for k in r.OBSERVED},"teacher_forced_targets":[1,0]}

    def relative(self, path, data):
        path.write_bytes(data); p = c.pin(path,c.MAX_LOGITS_FILE)
        return {**p,"path":path.name}

    def raw(self, name, greedy, each=False, restored=False):
        root = self.root/name; root.mkdir()
        history = [0,1,1] if greedy else [0,1,1,0]
        steps = []
        for pos, token in enumerate(history):
            values = [0.,0.,2.] if pos==0 else [0.,2.,1.] if pos==1 else [3.,0.,1.] if pos==2 else [0.,0.,2.]
            steps.append({"position":pos,"input_token":token,"role":"prefix" if greedy and pos<2 else "generation_fed" if greedy else "teacher_forced",
                          "history_length":pos+1,"next_position":pos+1,"argmax_token":max(range(3),key=values.__getitem__),
                          "teacher_forced_target":None if greedy or pos+1==len(history) else history[pos+1],
                          "logits_count":3,"logits_dtype":"fp32-le",
                          "logits":self.relative(root/f"step-{pos}.f32le",struct.pack("<3f",*values))})
        states = []
        count = len(history)
        schedule = [("initial",0)]
        if each:
            for x in range(count):
                schedule.append((f"post-{x}",x+1))
                if restored and x == 1:
                    schedule += [("checkpoint",2),("restored",2)]
        else:
            if restored:
                schedule += [("checkpoint",2),("restored",2)]
            schedule.append(("final",count))
        for kind, position in schedule:
            allocations = []
            for i, (component,state_class,flag,storage) in enumerate(r.COMPONENTS):
                allocations.append({"component":component,"state_class":state_class,"state_flag":flag,"storage":storage,
                                    "layout_offset":i*32,"provisional":False,
                                    "artifact":self.relative(root/f"{kind}-{component}.raw",bytes([position,i,3,4]))})
            states.append({"kind":kind,"next_position":position,"committed_tokens":position,"state_valid":63,
                           "state_generations":[position]*6,"allocations":allocations})
        receipt = {"schema":"strix-raw-eager-baseline-v1","complete":True,"qualified":False,"performance_evidence":False,
                   "source_binding":"external_compile_receipt_required","tokenizer_binding":"external_raw_tokens_only",
                   "evidence_kind":"injected-fixture","mode":"raw_greedy" if greedy else "raw_teacher_forced",
                   "state_policy":"each" if each else "final","model_sha256":self.plan["model_sha256"],"model_content_identity":"b"*64,
                   "executable_sha256":self.plan["runtime"]["executable"]["sha256"],"gpu_identity":"fixture","gpu_architecture":"fixture",
                   "hip_runtime_version":0,"hip_driver_version":0,"execution_identity":61,"kernel_plan_identity":71,
                   "runtime_compatibility_identity":81,"layout_identity":91,"context_tokens":16,"vocabulary_size":3,
                   "activation_storage":"bf16",
                   "execution_mode":self.plan["runtime"]["mode"],"numerical_identity":self.plan["runtime"]["numerical_identity"],
                   "capture_origin":"fresh_initialized","restore":None,
                   "supplied_tokens":[0,1] if greedy else history,"generated_output_tokens":[1,0] if greedy else [],
                   "maximum_generated_tokens":self.case["max_new_tokens"],"eos_token":0,"stop_reason":"eos" if greedy else "raw_input_exhausted",
                   "steps_executed":count,"final_position":count,"argmax_rule":"first_index_on_exact_tie",
                   "history_contract":r.FRESH_HISTORY[self.plan["runtime"]["mode"]],"state_encoding":"opaque_native_bytes","native_byte_order":"little",
                   "steps":self.relative(root/"steps.jsonl",b"".join(c.canonical(x)+b"\n" for x in steps)),
                   "states":self.relative(root/"states.jsonl",b"".join(c.canonical(x)+b"\n" for x in states)),
                   "history":self.relative(root/"history.u32le",struct.pack(f"<{count}I",*history))}
        if restored:
            interval = {"begin":0,"end":2,"producer":"serial_row","numerical_plan_identity":61,
                        "reduction_plan_identity":71,"state_update_identity":61}
            history_id = direct_fnv([0x53545249584c494e,1,0,2,0,61,71,61])
            lineage = {"history_identity":history_id,"tainted":False,"intervals":[interval]}
            event = {"schema":"strix-raw-prefix-restore-v1","prefix_identity":direct_fnv([61,81,91,52,2,2,history_id,2,0,1]),
                     "snapshot_plan_identity":52,"prefix_token_count":2,"prefix_tokens_sha256":hashlib.sha256(struct.pack("<2I",0,1)).hexdigest(),
                     "mutation_position":3,"restored_position":2,"execution_identity":61,"kernel_plan_identity":71,
                     "runtime_compatibility_identity":81,"layout_identity":91,"execution_mode":receipt["execution_mode"],
                     "numerical_identity":receipt["numerical_identity"],"source_states_kind":"checkpoint","restored_states_kind":"restored",
                     "complete_component_restore":True,"source_state_generations":[2]*6,"restored_state_generations":[2]*6,
                     "source_lineage":lineage,"restored_lineage":copy.deepcopy(lineage)}
            receipt.update(capture_origin="fresh_then_prefix_restored",history_contract=r.RESTORED_HISTORIES[receipt["execution_mode"]],
                           restore=self.relative(root/"restore.json",c.canonical(event)))
        (root/"receipt.json").write_bytes(c.canonical(receipt)); return root

    def mutate_restore(self, change):
        receipt = c.load(self.gr/"receipt.json"); path = self.gr/receipt["restore"]["path"]
        event = c.load(path); change(event)
        receipt["restore"] = self.relative(path,c.canonical(event))
        (self.gr/"receipt.json").write_bytes(c.canonical(receipt))

    def mutate_receipt(self, root, change):
        path = root/"receipt.json"; data = c.load(path); change(data); path.write_bytes(c.canonical(data))

    def mutate_rows(self, root, key, change):
        receipt = c.load(root/"receipt.json"); path = root/receipt[key]["path"]
        rows = [c.parse_json(x) for x in path.read_bytes().splitlines()]; change(rows)
        receipt[key] = self.relative(path,b"".join(c.canonical(x)+b"\n" for x in rows))
        (root/"receipt.json").write_bytes(c.canonical(receipt))

    def run(self, out="result"):
        plan_path = self.base.json(self.plan)["path"]; binding_path = self.base.json(self.binding)["path"]
        return r.import_record(plan_path,self.case["id"],self.tf,self.gr,binding_path,self.root/out)


class Tests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(prefix="strix-raw-import-"); self.addCleanup(self.temp.cleanup)
        self.f = RawFixture(self.temp.name)

    def test_true_teacherforced_and_greedy_rows_final_state_and_unknown_cost(self):
        receipt = self.f.run(); data = c.load(receipt["record"]["path"])
        self.assertFalse(receipt["qualification"]); self.assertFalse(receipt["native_execution_claim"])
        self.assertEqual(receipt["evidence_kind"],"injected-fixture")
        self.assertEqual(data["input_tokens"],[0,1]); self.assertEqual(data["output_tokens"],[1,0])
        self.assertEqual(data["capture_contract"],{"origin":"fresh","history_contract":c.FRESH_RAW_HISTORY,"restoration_record":None})
        self.assertEqual(data["teacher_forced_target_tokens"],[1,0])
        self.assertEqual([x["logits"] for x in c.logits_rows(data["logits"],3)],[[0.,2.,1.],[3.,0.,1.]])
        raw = c.load(self.f.gr/"receipt.json"); self.assertEqual(data["history_sha256"],raw["history"]["sha256"])
        self.assertEqual(raw["final_position"],3)  # EOS emitted but never fed; teacher-forced final position is4.
        blob = Path(data["state"]["convolution"]["path"]).read_bytes()
        self.assertIn(b"gdn_convolution",blob); self.assertIn(b"ple_convolution",blob)
        self.assertIn(bytes([3,2,3,4]),blob); self.assertNotIn(bytes([4,2,3,4]),blob)
        self.assertTrue(all(x is None for x in data["timings_ms"].values()))
        self.assertFalse((self.f.root/"result"/"INCOMPLETE").exists())

    def test_marker_and_escape_and_truncation_reject(self):
        (self.f.gr/"INCOMPLETE").write_text("failed")
        with self.assertRaises(c.Invalid): self.f.run()
        (self.f.gr/"INCOMPLETE").unlink()
        self.f.mutate_receipt(self.f.gr,lambda x:x["steps"].update(path="../steps.jsonl"))
        with self.assertRaises(c.Invalid): self.f.run()

    def test_pair_observed_model_source_phase_and_integer_rejection(self):
        for field in ("model_sha256","executable_sha256","gpu_identity","layout_identity","kernel_plan_identity"):
            original = (self.f.gr/"receipt.json").read_bytes()
            self.f.mutate_receipt(self.f.gr,lambda x:x.update({field:"c"*64 if isinstance(x[field],str) else x[field]+1}))
            with self.assertRaises(c.Invalid): self.f.run()
            (self.f.gr/"receipt.json").write_bytes(original)
        self.f.plan["runtime"]["mode"] = "graph_float"
        self.f.binding["plan_sha256"] = c.identity(self.f.plan)
        with self.assertRaises(c.Invalid): self.f.run()

    def test_external_decoded_text_and_prompt_binding_cannot_substitute(self):
        tokenizer = c.load(self.f.binding["tokenization"]["path"]); tokenizer["output_tokens"] = [0,1]
        self.f.binding["tokenization"] = self.f.base.json(tokenizer)
        with self.assertRaises(c.Invalid): self.f.run()
        self.f.binding["compile_binding"]["source_manifest_sha256"] = "c"*64
        with self.assertRaises(c.Invalid): self.f.run()

    def test_targets_not_argmax_and_no_feed_of_final_emitted_token(self):
        self.f.binding["teacher_forced_targets"] = [2,0]
        with self.assertRaises(c.Invalid): self.f.run()
        self.f.binding["teacher_forced_targets"] = [1,0]
        self.f.mutate_receipt(self.f.gr,lambda x:x.update(generated_output_tokens=[1,1,0]))
        with self.assertRaises(c.Invalid): self.f.run()

    def test_nonfinite_false_argmax_partial_vocab_and_nested_hash_reject(self):
        original = (self.f.gr/"step-1.f32le").read_bytes()
        for payload in (struct.pack("<3f",0,float("nan"),1),struct.pack("<3f",3,0,1),struct.pack("<2f",0,2)):
            (self.f.gr/"step-1.f32le").write_bytes(payload)
            with self.assertRaises(c.Invalid): self.f.run()
        (self.f.gr/"step-1.f32le").write_bytes(original)
        self.f.mutate_rows(self.f.gr,"steps",lambda rows:rows[1].update(argmax_token=0))
        with self.assertRaises(c.Invalid): self.f.run()

    def test_repinned_nonfinite_and_wrong_binary_extent_do_not_become_valid(self):
        original = (self.f.gr/"receipt.json").read_bytes(); steps = (self.f.gr/"steps.jsonl").read_bytes()
        for payload in (struct.pack("<3f",0,float("inf"),1),struct.pack("<2f",0,2)):
            p = self.f.relative(self.f.gr/"step-1.f32le",payload)
            self.f.mutate_rows(self.f.gr,"steps",lambda rows:rows[1].update(logits=p))
            with self.assertRaises(c.Invalid): self.f.run()
            (self.f.gr/"receipt.json").write_bytes(original); (self.f.gr/"steps.jsonl").write_bytes(steps)

    def test_actual_bf16_control_and_architecture_feature_tags_keep_exact_binding(self):
        self.f.plan["runtime"].update(input_storage="bf16",candidate_id="t0:decode:M1:BF16_activation_control")
        self.f.binding["plan_sha256"] = c.identity(self.f.plan)
        self.assertEqual(self.f.run()["evidence_kind"],"injected-fixture")
        for root in (self.f.tf,self.f.gr):
            self.f.mutate_receipt(root,lambda x:x.update(evidence_kind="native-eager-raw",gpu_identity="1"*32,
                                                       gpu_architecture="gfx1151:sramecc-:xnack-",hip_runtime_version=70200,hip_driver_version=70200))
        raw = c.load(self.f.gr/"receipt.json"); self.f.binding["observed"] = {k:raw[k] for k in r.OBSERVED}
        result = self.f.run("declared-native")
        self.assertFalse(result["native_execution_claim"]); self.assertFalse(result["qualification"])
        self.f.mutate_receipt(self.f.gr,lambda x:x.update(gpu_architecture="gfx1151:sramecc+:xnack-"))
        with self.assertRaises(c.Invalid): self.f.run("drift")

    def test_bf16_capture_cannot_bind_fp32_or_unknown_storage(self):
        self.f.plan["runtime"].update(input_storage="fp32",candidate_id="t0:decode:M1:FP32_activation_control",numerical_identity=c.floating_profile("fp32"))
        self.f.binding["plan_sha256"] = c.identity(self.f.plan)
        with self.assertRaises(c.Invalid): self.f.run()

    def test_actual_explicit_graph_and_integer_head_modes_import_without_qualification(self):
        for mode, activation in (("graph_float","W4A8"),("eager_integer_lm_head","W4A8"),("eager_integer_lm_head","W4A4"),
                                 ("eager_integer_lm_head","W4A4:iu8")):
            root = self.f.root/(mode+activation.replace(":","_")); root.mkdir()
            fixture = RawFixture(root,runtime_mode=mode,activation_class=activation)
            result = fixture.run()
            self.assertEqual(result["execution_mode"],mode)
            self.assertEqual(result["numerical_identity"],fixture.plan["runtime"]["numerical_identity"])
            self.assertFalse(result["qualification"]); self.assertFalse(result["native_execution_claim"])
            self.assertEqual(result["evidence_kind"],"injected-fixture")
            self.assertEqual(c.load(result["record"]["path"])["capture_contract"]["origin"],"fresh")
            for capture in (fixture.tf,fixture.gr):
                fixture.mutate_receipt(capture,lambda x:x.update(evidence_kind=r.NATIVE_MODES[mode],gpu_identity="1"*32,
                                                                 gpu_architecture="gfx1151",hip_runtime_version=70200,hip_driver_version=70200))
            raw = c.load(fixture.gr/"receipt.json"); fixture.binding["observed"] = {k:raw[k] for k in r.OBSERVED}
            declared = fixture.run("native-shaped-declaration")
            self.assertEqual(declared["evidence_kind"],r.NATIVE_MODES[mode])
            self.assertFalse(declared["native_execution_claim"]); self.assertFalse(declared["qualification"])

    def test_mode_profile_native_kind_and_missing_observation_cannot_relabel_path(self):
        original = (self.f.gr/"receipt.json").read_bytes()
        for change in (lambda x:x.update(execution_mode="graph_float"),
                       lambda x:x.update(numerical_identity="foreign arithmetic"),
                       lambda x:x.update(evidence_kind="native-integer-eager-raw"),
                       lambda x:x.pop("execution_mode"),
                       lambda x:x.update(execution_mode="graph_integer_lm_head")):
            self.f.mutate_receipt(self.f.gr,change)
            with self.assertRaises(c.Invalid): self.f.run()
            (self.f.gr/"receipt.json").write_bytes(original)
        self.f.plan["runtime"].update(input_storage="bf16",candidate_id="t0:decode:M1:BF16_activation_control",numerical_identity=c.floating_profile("bf16"))
        self.f.binding["plan_sha256"] = c.identity(self.f.plan)
        self.f.mutate_receipt(self.f.gr,lambda x:x.update(activation_storage="fp32"))
        with self.assertRaises(c.Invalid): self.f.run()
        self.f.mutate_receipt(self.f.gr,lambda x:x.update(activation_storage="float16"))
        with self.assertRaises(c.Invalid): self.f.run()

    def test_missing_convolution_overlap_and_generation_regression(self):
        original = (self.f.gr/"receipt.json").read_bytes(); rows = (self.f.gr/"states.jsonl").read_bytes()
        for change in (lambda xs:xs[-1]["allocations"].pop(3),
                       lambda xs:xs[-1]["allocations"][1].update(layout_offset=0),
                       lambda xs:xs[0].update(state_generations=[4]*6)):
            self.f.mutate_rows(self.f.gr,"states",change)
            with self.assertRaises(c.Invalid): self.f.run()
            (self.f.gr/"receipt.json").write_bytes(original); (self.f.gr/"states.jsonl").write_bytes(rows)

    def test_each_state_policy_and_initial_pair_identity(self):
        self.f.gr = self.f.raw("greedy-each",True,each=True)
        self.assertEqual(self.f.run()["evidence_kind"],"injected-fixture")
        path = self.f.gr/"initial-attention_kv.raw"; path.write_bytes(b"diff")
        self.f.mutate_rows(self.f.gr,"states",lambda rows:rows[0]["allocations"][0].update(artifact=self.f.relative(path,b"diff")))
        with self.assertRaises(c.Invalid): self.f.run("other")

    def test_native_fixture_claim_requires_actual_identity_but_never_qualifies(self):
        self.f.mutate_receipt(self.f.gr,lambda x:x.update(evidence_kind="native-eager-raw"))
        with self.assertRaises(c.Invalid): self.f.run()

    def test_drift_after_conversion_leaves_incomplete_private_output(self):
        original = r.Raw.recheck; calls = 0
        def drift(raw):
            nonlocal calls
            calls += 1
            if calls == 3:
                (self.f.gr/"final-ple_convolution.raw").write_bytes(b"drft")
            original(raw)
        with mock.patch.object(r.Raw,"recheck",side_effect=drift,autospec=True):
            with self.assertRaises(c.Invalid): self.f.run()
        self.assertTrue((self.f.root/"result"/"INCOMPLETE").exists())

    def test_existing_output_and_invalid_closed_binding_reject(self):
        (self.f.root/"result").mkdir()
        with self.assertRaises(c.Invalid): self.f.run()
        self.f.binding["extra"] = True
        with self.assertRaises(c.Invalid): self.f.run("other")

    def test_actual_restore_event_imports_eager_float_and_integer_head_final_and_each(self):
        for mode, each in (("eager_float",False),("eager_float",True),("eager_integer_lm_head",False),("graph_float",False),("graph_float",True)):
            root = self.f.root/(mode+str(each)); root.mkdir()
            fixture = RawFixture(root,runtime_mode=mode,restored=True,each=each)
            receipt = fixture.run(); data = c.load(receipt["record"]["path"])
            self.assertEqual(data["capture_contract"]["origin"],"restored")
            self.assertEqual(receipt["capture_origin"],"fresh_then_prefix_restored")
            self.assertEqual(receipt["restore"]["sha256"],c.pin(fixture.gr/"restore.json")["sha256"])
            declaration = c.load(data["capture_contract"]["restoration_record"]["path"])
            self.assertEqual(declaration["restore_log"],receipt["restore"])
            self.assertFalse(receipt["qualification"]); self.assertFalse(receipt["native_execution_claim"])

    def test_restore_event_prefix_runtime_identity_and_full_lineage_tampering_reject(self):
        root = self.f.root/"restore-tamper"; root.mkdir(); fixture = RawFixture(root,restored=True)
        original = (fixture.gr/"receipt.json").read_bytes(); event = (fixture.gr/"restore.json").read_bytes()
        def fully_rehashed_foreign_reduction(value):
            identity = direct_fnv([0x53545249584c494e,1,0,2,0,61,72,61])
            for name in ("source_lineage","restored_lineage"):
                value[name]["intervals"][0]["reduction_plan_identity"] = 72
                value[name]["history_identity"] = identity
            value["prefix_identity"] = direct_fnv([61,81,91,52,2,2,identity,2,0,1])
        for change in (lambda x:x.update(prefix_tokens_sha256="c"*64),lambda x:x.update(prefix_identity=x["prefix_identity"]+1),
                       lambda x:x.update(snapshot_plan_identity=53),lambda x:x.update(restored_position=1),
                       lambda x:x.update(mutation_position=17),lambda x:x.update(mutation_position=2),
                       lambda x:x.update(numerical_identity="foreign profile"),
                       lambda x:x.update(complete_component_restore=False),lambda x:x["source_lineage"].update(tainted=True),
                       lambda x:x["source_lineage"]["intervals"][0].update(end=1),
                       lambda x:x["source_lineage"]["intervals"][0].update(producer="prefill_chunk"),
                       lambda x:x["restored_lineage"]["intervals"][0].update(reduction_plan_identity=72),
                       fully_rehashed_foreign_reduction):
            fixture.mutate_restore(change)
            with self.assertRaises(c.Invalid): fixture.run()
            (fixture.gr/"receipt.json").write_bytes(original); (fixture.gr/"restore.json").write_bytes(event)

    def test_all_component_restore_bytes_and_generations_required(self):
        root = self.f.root/"restore-components"; root.mkdir(); fixture = RawFixture(root,restored=True)
        rows = (fixture.gr/"states.jsonl").read_bytes(); receipt = (fixture.gr/"receipt.json").read_bytes()
        changed = fixture.relative(fixture.gr/"restored-ple_convolution.raw",b"diff")
        fixture.mutate_rows(fixture.gr,"states",lambda xs:xs[2]["allocations"][3].update(artifact=changed))
        with self.assertRaises(c.Invalid): fixture.run()
        (fixture.gr/"states.jsonl").write_bytes(rows); (fixture.gr/"receipt.json").write_bytes(receipt)
        (fixture.gr/"restored-ple_convolution.raw").write_bytes(bytes([2,3,3,4]))
        fixture.mutate_restore(lambda x:x.update(restored_state_generations=[3]*6))
        with self.assertRaises(c.Invalid): fixture.run()

    def test_cross_graph_eager_restore_and_fresh_relabel_without_native_event_rejected(self):
        root = self.f.root/"graph-restore"; root.mkdir(); fixture = RawFixture(root,runtime_mode="graph_float",restored=True)
        fixture.plan["runtime"]["mode"] = "eager_float"; fixture.binding["plan_sha256"] = c.identity(fixture.plan)
        with self.assertRaises(c.Invalid): fixture.run()
        self.f.mutate_receipt(self.f.gr,lambda x:x.update(capture_origin="fresh_then_prefix_restored",history_contract=r.RESTORED_HISTORY))
        with self.assertRaises(c.Invalid): self.f.run()

    def test_restore_event_artifact_drift_after_conversion_keeps_import_incomplete(self):
        root = self.f.root/"restore-drift"; root.mkdir(); fixture = RawFixture(root,restored=True)
        original = r.Raw.recheck; calls = 0
        def drift(raw):
            nonlocal calls
            calls += 1
            if calls == 3:
                (fixture.gr/"restore.json").write_bytes(b"changed event")
            original(raw)
        with mock.patch.object(r.Raw,"recheck",side_effect=drift,autospec=True):
            with self.assertRaises(c.Invalid): fixture.run()
        self.assertTrue((root/"result"/"INCOMPLETE").exists())

    def test_mocked_optional_token_adapter_contract_binds_actual_prefix_and_rendered_bytes(self):
        expected = {"model":self.f.plan["audit_receipt"],"prompt_contract_sha256":c.identity(self.f.case),
                    "prompt":{"sha256":hashlib.sha256(self.f.case["text"].encode()).hexdigest()},"controls":{"vocabulary_size":3}}
        expected["model"] = self.f.base.model
        module = types.ModuleType("tensormeld.llamacpp_tokenization")
        calls = []
        def validate(value,*,require_observed):
            calls.append(require_observed)
            if value["execution_kind"] != "trusted-local-upstream-tool": raise ValueError("fixture cannot bind observed tokens")
            return types.SimpleNamespace(config=expected)
        module.validate_tokenization_receipt = validate
        package = types.ModuleType("tensormeld"); package.__path__=[]
        proof = {"tokens":[0,1],"result_sha256":"a"*64,"execution_kind":"trusted-local-upstream-tool"}
        self.f.binding["input_token_evidence"] = self.f.base.json(proof)
        # This mocks the validator contract only; no upstream/tool execution.
        with mock.patch.dict(sys.modules,{"tensormeld":package,"tensormeld.llamacpp_tokenization":module}):
            receipt = self.f.run()
            self.assertEqual(receipt["input_token_binding"],"observed-upstream-tool")
            self.assertFalse(receipt["native_execution_claim"]); self.assertFalse(receipt["qualification"])
            self.assertTrue(calls and all(calls))
            proof["tokens"]=[1,0]; self.f.binding["input_token_evidence"] = self.f.base.json(proof)
            with self.assertRaises(c.Invalid): self.f.run("wrong-prefix")
            proof["tokens"]=[0,1]; proof["execution_kind"]="injected-fixture"
            self.f.binding["input_token_evidence"] = self.f.base.json(proof)
            with self.assertRaises(c.Invalid): self.f.run("injected-proof")


if __name__ == "__main__":
    unittest.main()
