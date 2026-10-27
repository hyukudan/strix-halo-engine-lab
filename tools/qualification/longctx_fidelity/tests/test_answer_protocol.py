"""Answer protocols, family budgets and paired short controls (CPU only)."""
import json
from pathlib import Path
import unittest

from tools.qualification.longctx_fidelity import tasks
from tools.qualification.longctx_fidelity.build import build_case
from tools.qualification.longctx_fidelity.common import NATIVE_CAP, ids_bytes, sha
from tools.qualification.longctx_fidelity.plan import matrix
from tools.qualification.longctx_fidelity.score import exact, extract_answer, ordered_pairs, paired_controls, score_case
from tools.qualification.longctx_fidelity.tok import FakeTokenizer, PREFIX, SUFFIX

FIXTURES = Path(__file__).parent / "fixtures"


def measured(case, output):
    receipt = {"status": "COMPLETE", "case_id": case["case_id"],
               "input_ids_sha256": case["ids_sha256"],
               "ground_truth_sha256": case["ground_truth_sha256"], "generated_tokens": [1]}
    return score_case(case, receipt, output)


class ProtocolTests(unittest.TestCase):
    def test_last_answer_line_and_decoys(self):
        self.assertEqual(extract_answer('Answer: decoy\nReasoning mentions Answer: wrong\nAnswer: `42`.\n'), ('42', True))
        for text in ('Reasoning says Answer: 42', '> Answer: 42', '42', 'answer: 42'):
            self.assertEqual(extract_answer(text), (None, False))
        for text in ('Answer: "42."', "Answer: '42'.", 'Answer: ```42```.', 'Answer:  42.  '):
            self.assertEqual(extract_answer(text), ('42', True))
        self.assertEqual(extract_answer('Answer: 42\nAnswer: wrong'), ('wrong', True))
        self.assertEqual(extract_answer('Answer: 42..'), ('42.', True))

    def test_missing_line_is_measured_wrong_and_records_extraction(self):
        c = build_case('aggregate', 8192, .5, 104729, FakeTokenizer())[2]
        answer = c['ground_truth']
        for output in (answer, 'Reasoning mentions Answer: ' + answer):
            row = measured(c, output)
            self.assertEqual(row['status'], 'COMPLETE')
            self.assertFalse(row['exact'])
            self.assertFalse(row['answer_line_found'])
            self.assertIsNone(row['extracted_answer'])
        row = measured(c, 'Answer: decoy\nAnswer: "' + answer + '".')
        self.assertTrue(row['exact'])
        self.assertTrue(row['answer_line_found'])
        self.assertEqual(row['extracted_answer'], answer)
        self.assertFalse(measured(c, 'Answer: '+answer+'\nAnswer: wrong')['exact'])
        self.assertFalse(measured(c, 'Answer: '+answer+'..')['exact'])
        self.assertEqual(score_case(c)['status'], 'PLANNED')

    def test_multikey_shapes_fence_order_and_truncation(self):
        answer = [['a','one'], ['b','two']]
        objects = [{'key': k, 'value': v} for k, v in answer]
        for shape in (answer, objects):
            text = json.dumps(shape)
            for output in (text, 'Here is the result: '+text+' Done.', '[Result] '+text+' Done.', 'Result:\n```json\n'+text+'\n```\nDone.'):
                self.assertTrue(exact(answer, output)['exact'])
            reordered = exact(answer, json.dumps(shape[::-1]))
            self.assertFalse(reordered['exact'])
            self.assertEqual(reordered['per_value'], [True, True])
            for truncated in (text[:-1], '```json\n'+text[:-1], '```json\n'+text):
                self.assertFalse(exact(answer, truncated)['exact'])
        for wrong in ('{"answer": [["a","one"],["b","two"]]}', '[["a","one"],["a","one"]]', '[{"key":"a","value":"one","extra":0}]',
                      '[{"key":"a","key":"b","value":"two"}]',
                      '[["a","one"],{"key":"b","value":"two"}]', '[{"key":"a","value":1}]'):
            self.assertFalse(exact(answer, wrong)['exact'])

    def test_archived_answer_excerpts(self):
        fixture = json.loads((FIXTURES/'truncated_objects.json').read_text())
        result = exact(fixture['answer'], fixture['text'])
        self.assertFalse(result['exact'])
        self.assertEqual(result['answer_error'], 'invalid_json_or_shape')
        # For diagnosis only: closing the complete prefix proves its values/order.
        prefix = fixture['text'].split('```json\n', 1)[1].rsplit(',\n  {', 1)[0] + '\n]'
        present = ordered_pairs(prefix)
        self.assertEqual(present, fixture['answer'][:fixture['complete_objects']])
        self.assertTrue(exact(present, prefix)['exact'])
        self.assertTrue(exact(fixture['answer'], json.dumps([
            {'key': k, 'value': v} for k, v in fixture['answer']]))['exact'])
        c = build_case('vartrack', 8192, .25, 130363, FakeTokenizer())[2]
        row = measured(c, (FIXTURES/'reasoning_excerpt.txt').read_text())
        self.assertEqual(row['status'], 'COMPLETE')
        self.assertFalse(row['answer_line_found'])
        self.assertFalse(row['exact'])

    def test_family_defaults_and_budget(self):
        expected = {'vartrack':4096, 'vartrack256':8192, 'multihop':1024, 'multihop2':1024,
                    'multihop8':1024, 'aggregate':1024, 'aggregate128':1024,
                    'multikey':2048, 'multikey32':8192, 'niah_single':32}
        for family, g in expected.items():
            self.assertEqual(tasks.default_generate(family), g)
            self.assertEqual(matrix(['native'], [32768], [family], [1], [.5])[0]['generate'], g)
            c = build_case(family, 32768, .5, 104729, FakeTokenizer())[2]
            self.assertEqual(c['generate'], g)
            with self.assertRaisesRegex(ValueError, 'budget'):
                build_case(family, NATIVE_CAP-g+1, .5, 104729, FakeTokenizer())
            with self.assertRaisesRegex(ValueError, 'budget'):
                matrix(['native'], [NATIVE_CAP-g+1], [family], [1], [.5])
        self.assertEqual(build_case('aggregate',8192,.5,104729,FakeTokenizer(),17)[2]['generate'],17)

    def test_protocol_questions_and_unchanged_symbol_edges(self):
        for family in (*tasks.FAMILIES, 'vartrack256','multihop2','multihop8','aggregate128','multikey32'):
            task = tasks.generate(family,32768,.5,104729)
            symbol = tasks.generate(family,32768,.5,104729,'symbol')
            self.assertNotIn('Answer:', symbol['question'])
            if tasks.reasoning_protocol(family):
                self.assertTrue(task['question'].endswith('End your reply with one final line: Answer: <value>'))
            if family.startswith('multikey'):
                self.assertIn('[[key,value],...]', task['question'])
            edge = build_case(family,262143,.95,104729,FakeTokenizer(),1,variant='symbol')[2]
            self.assertEqual(edge['generate'],1)
            self.assertTrue(measured(edge,edge['ground_truth'])['exact'])

    def test_additional_changed_family_goldens(self):
        for family, depth, expected in (
            ('vartrack',8192,'331c7c46cfecce8d34c079e79634b949c593d7c39248e7a70fe0d36a2e96be26'),
            ('multikey',32768,'2f22bb2587247731b5070116d90423e9c9219b9ab2ea3d140c091fcfd5899bff')):
            self.assertEqual(sha(ids_bytes(build_case(family,depth,.5,104729,FakeTokenizer())[1])),expected)

    def test_short_construction_same_immutable_task(self):
        for family in (*tasks.FAMILIES, 'vartrack256','multihop2','multihop8','aggregate128','multikey32'):
            for variant in ('normal','symbol'):
                tok = FakeTokenizer()
                text, ids, c = build_case(family,32768,.75,104729,tok,variant=variant,short=True)
                task = tasks.generate(family,32768,.75,104729,variant)
                ordered = sorted(zip(task['positions'],task['facts'],range(len(task['facts']))))
                expected = PREFIX+task['rule']+'\nEvidence and unrelated observations follow.\n'
                expected += ''.join('\n'+fact+'\n' for _,fact,_ in ordered)
                expected += '\nQuestion: '+task['question']+SUFFIX
                self.assertEqual(text,expected)
                self.assertEqual(c['depth'],tok.count(expected))
                self.assertEqual(c['requested_depth'],32768)
                self.assertEqual(c['ground_truth'],task['answer'])
                self.assertTrue(c['case_id'].endswith('.short'))
                self.assertEqual(c['filler']['generated_chars'],0)
                self.assertEqual(c['filler']['padding_chars'],0)
                self.assertEqual(len(ids),c['depth'])
                for span in c['spans']:
                    self.assertIn(task['facts'][span['fact_index']], tok.decode(ids[span['start']:span['end']]))
                other = build_case(family,65536,.75,104729,tok,variant=variant,short=True)
                self.assertEqual(other[0],text)
                self.assertEqual(other[2]['case_id'],c['case_id'])

    def test_pairing_complete_same_family_seed_layout_only(self):
        def row(depth=8192, short=False, seed=104729, position=.5, correct=True):
            c = build_case('aggregate',depth,position,seed,FakeTokenizer(),short=short)[2]
            return measured(c, 'Answer: '+(c['ground_truth'] if correct else 'wrong'))
        short, long = row(short=True), row(correct=False)
        pairs = paired_controls([short,long])
        self.assertEqual(len(pairs),1)
        self.assertEqual(pairs[0]['long_minus_short_exact'],-1)
        self.assertEqual(pairs[0]['long_minus_short_per_value'],[-1])
        self.assertEqual(pairs[0]['short_case_id'],short['case_id'])
        self.assertEqual(len(paired_controls([short,long,row(32768)])),2)
        self.assertEqual(paired_controls([short,row(seed=1),row(position=.75)]),[])
        missing = dict(short,status='INCOMPLETE')
        self.assertEqual(paired_controls([missing,long]),[])
        self.assertEqual(paired_controls([short,dict(short),long]),[])
        other = dict(long,layout=dict(long['layout'],variant='symbol'))
        self.assertEqual(paired_controls([short,other]),[])


class HardwareShapeTests(unittest.TestCase):
    def test_inline_code_before_final_fence(self):
        # Shape observed on hardware: reasoning with inline `code`, then one fenced answer.
        from tools.qualification.longctx_fidelity.score import exact
        truth = [["k-00", "v0"], ["k-01", "v1"]]
        text = ("Check `k-00`: found `version=9`, `value=v0`.\n"
                "```json\n[\n  [\"k-00\", \"v0\"],\n  [\"k-01\", \"v1\"]\n]\n```")
        self.assertTrue(exact(truth, text)["exact"])
        self.assertFalse(exact(truth, text[:-3])["exact"])  # unterminated fence
        earlier = "```json\n[[\"k-00\", \"vX\"]]\n```\nCorrected:\n" + text
        self.assertTrue(exact(truth, earlier)["exact"])  # last complete fence wins
