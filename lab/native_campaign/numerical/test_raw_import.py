from __future__ import annotations

import copy
from pathlib import Path
import struct
import sys
import tempfile
import unittest
from unittest import mock

sys.dont_write_bytecode = True
sys.path.insert(0, str(Path(__file__).resolve().parent))
import campaign as c
import raw_import as r
from test_campaign import Fixture


class RawFixture:
    def __init__(self, root):
        self.root = Path(root); self.base = Fixture(root)
        plan = self.plan = copy.deepcopy(self.base.plan)
        manifest = c.load(self.base.receipt["manifest"]["path"])
        manifest["tensors"][0]["ggml_type"] = 20
        audit = copy.deepcopy(self.base.receipt); audit["manifest"] = self.base.json(manifest)
        plan["audit_receipt"] = self.base.json(audit); plan["candidates"] = c.candidate_matrix(manifest)
        plan["runtime"].update(mode="eager_float",candidate_id="t0:decode:M1:BF16_activation_control",input_storage="bf16",
                               numerical_identity="synthetic-explicit-floating-profile",kernel_plan_identity="71")
        self.case = next(x for x in plan["cases"] if x["id"] == "greedy-ties")
        self.tf = self.raw("teacher",False); self.gr = self.raw("greedy",True)
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

    def raw(self, name, greedy, each=False):
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
        for kind, position in [("initial",0)] + ([(f"post-{x}",x+1) for x in range(count)] if each else [("final",count)]):
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
                   "supplied_tokens":[0,1] if greedy else history,"generated_output_tokens":[1,0] if greedy else [],
                   "maximum_generated_tokens":self.case["max_new_tokens"],"eos_token":0,"stop_reason":"eos" if greedy else "raw_input_exhausted",
                   "steps_executed":count,"final_position":count,"argmax_rule":"first_index_on_exact_tie",
                   "history_contract":"single_row_eager_from_fresh_initialized_state","state_encoding":"opaque_native_bytes","native_byte_order":"little",
                   "steps":self.relative(root/"steps.jsonl",b"".join(c.canonical(x)+b"\n" for x in steps)),
                   "states":self.relative(root/"states.jsonl",b"".join(c.canonical(x)+b"\n" for x in states)),
                   "history":self.relative(root/"history.u32le",struct.pack(f"<{count}I",*history))}
        (root/"receipt.json").write_bytes(c.canonical(receipt)); return root

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
        self.f.plan["runtime"].update(input_storage="fp32",candidate_id="t0:decode:M1:FP32_activation_control")
        self.f.binding["plan_sha256"] = c.identity(self.f.plan)
        with self.assertRaises(c.Invalid): self.f.run()
        self.f.plan["runtime"].update(input_storage="bf16",candidate_id="t0:decode:M1:BF16_activation_control")
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


if __name__ == "__main__":
    unittest.main()
