"""CPU-only source-generation tests. Synthetic vectors are never GPU evidence."""
import copy
import importlib.util
import pathlib
import tempfile
import unittest

ROOT=pathlib.Path(__file__).resolve().parents[1]
SPEC=importlib.util.spec_from_file_location('capture_prepare',ROOT/'tools/prepare_private_hc_capture.py')
P=importlib.util.module_from_spec(SPEC);SPEC.loader.exec_module(P)

class PrepareTests(unittest.TestCase):
    def setUp(self):
        self.source=(ROOT/'apps/hip_decode_token_main.cpp').read_text(encoding='utf8')
        self.reference={'prompt':{'token_ids':list(range(43))}}
        self.golden={'stage':'finished','all_generated_tokens_equal':True,'samples':[{'generated_tokens':list(range(512))}]}
    def render(self,source=None,reference=None,golden=None,output=None):
        return P.render(source or self.source,output or ROOT/'not-created-test-output/capture',reference or self.reference,golden or self.golden,'a'*40,'b'*64,'c'*64)
    def test_actual_app_seams_and_no_input_mutation(self):
        before=self.source;text=self.render();self.assertEqual(self.source,before)
        self.assertEqual(text.count('private_hc_capture::Dispatcher dispatcher('),1)
        self.assertEqual(text.count('dispatcher.finish(result.generated_tokens,result.steps_executed,result.final_position);'),1)
        self.assertIn('execute_native_token',text)
        self.assertNotIn('class Dispatcher final',text) # header is a separate input
    def test_changed_or_duplicate_seams_fail_closed(self):
        needle='    strix::DirectLaunchDispatcher dispatcher(launch_plan);'
        for source in [self.source.replace(needle,''),self.source+needle,self.source.replace('        double elapsed_total','        double different_total')]:
            with self.assertRaises(ValueError):self.render(source=source)
    def test_boolean_token_and_length_mutants(self):
        for mutate in [lambda r:r['prompt']['token_ids'].__setitem__(0,True),lambda r:r['prompt']['token_ids'].pop()]:
            value=copy.deepcopy(self.reference);mutate(value)
            with self.assertRaises(ValueError):self.render(reference=value)
        for mutate in [lambda g:g.update(stage='failed'),lambda g:g['samples'][0]['generated_tokens'].pop(),lambda g:g['samples'].append({'generated_tokens':[0]*512})]:
            value=copy.deepcopy(self.golden);mutate(value)
            with self.assertRaises(ValueError):self.render(golden=value)
    def test_absolute_path_and_digest_gate(self):
        with self.assertRaises(ValueError):self.render(output=pathlib.Path('relative'))
        with self.assertRaises(ValueError):P.render(self.source,ROOT/'out',self.reference,self.golden,'wrong','b'*64,'c'*64)
    def test_full_qualified_recipe_guard(self):
        text=self.render()
        for flag in ['parallel_moe_router','virtual_q8_lm_head','virtual_q8_gdn_input','prefetch_q8_gdn_input','virtual_q8_gdn_output','q4k_a8g32_gate_up','grouped_exact_q51_down','wave_moe_router_topk','lane_chain_hyper_output','prefetch_q8_qsa_input','prefetch_q8_shared_expert','deferred_moe_status']:
            self.assertIn('!options.'+flag,text)
        self.assertIn('DecodeIntegerHead::none',text)
        self.assertIn('options.tokens!=std::vector<std::uint32_t>',text)
        guard=text.split('private HC capture requires qualified all5 eager registered43/20')[0]
        for flag in ['q51_slot_parallel_down','prefetch_hc_projection_next_tile']:
            self.assertIn('options.'+flag,guard)
            self.assertNotIn('!options.'+flag,guard)
    def test_duplicate_json_key_rejected(self):
        with tempfile.TemporaryDirectory()as temp:
            path=pathlib.Path(temp)/'duplicate.json';path.write_text('{"a":1,"a":2}')
            with self.assertRaises(ValueError):P.load(path)

if __name__=='__main__':unittest.main()
