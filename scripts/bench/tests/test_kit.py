#!/usr/bin/env python3
"""CPU receipt/identity tests; synthetic timings are confined to tests."""
import copy
import importlib.util
import json
import os
from pathlib import Path
import re
import shlex
import shutil
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

sys.dont_write_bytecode = True

HERE = Path(__file__).resolve().parent
SSE = HERE / 'fixtures/sse'
spec = importlib.util.spec_from_file_location('kit', HERE.parent / 'lib/kit.py')
k = importlib.util.module_from_spec(spec)
spec.loader.exec_module(k)


class KitTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.out = Path(self.tmp.name)

    def test_effective_formats_and_identity(self):
        glm=k.parse_effective('effective baseline=0\neffective identity=r1b (default)\neffective env HSA_DISABLE_FRAGMENT_ALLOCATOR=1 (default, item allocator)\n')
        self.assertEqual(glm['environment'],{'HSA_DISABLE_FRAGMENT_ALLOCATOR':'1'})
        self.assertEqual(glm['settings']['identity'],'r1b')
        mini=k.parse_effective('minimax defaults: STRIX_MM_BASELINE=1, the 22 optimization items are off (an explicit item value wins)\napp=rank\nSTRIX_MM_INDEX_CONC=0\n')
        self.assertEqual(mini['environment']['STRIX_MM_INDEX_CONC'],'0')
        rtx=k.parse_effective('{"load_serial":false,"draft_vocab":"auto:vocab","draft_keep":64}')
        self.assertEqual(rtx['environment'],{'DG_LOAD_SERIAL':'0'})
        with self.assertRaisesRegex(ValueError,'missing effective'):
            k.parse_effective('a GPU timing receipt is not a config')
        with self.assertRaisesRegex(ValueError,'draft vocab'):
            k.parse_effective('{"load_serial":false,"draft_vocab":"full"}')
        # The recorded environment, including app-specific differences, enters both keys.
        inv=dict(build_options={},binary_sha256s=[],model_files=[],repo_commit='0'*40,
                 tracked_dirty=False, hardware=dict(rocm_version=None,hip_version=None,kernel='test'))
        k.dump(self.out/'local-inventory.json',inv)
        env={'decode':{'STRIX_TEST':'1'},'prefill':{'STRIX_TEST':'0'}}
        k.dump(self.out/'effective-environment.json',env)
        with patch.object(k,'ENGINE','glm-pair'), patch.dict(k.os.environ,{'BENCH_PROFILE':'default','BENCH_SUITE':'quick','BENCH_PROMPTS':'kit'}):
            k.initialize(self.out)
        identity=k.read(self.out/'identity.json')
        self.assertEqual(identity['identity']['environment'],env)
        self.assertEqual(identity['reference_key']['environment'],env)

    def test_qwen_notices_do_not_hide_effective_json(self):
        # Engine stderr notice lines precede the stdout JSON in the captured text.
        native_status = dict(rope_mode='native', factor=0, original_max=262144, effective_max=262144)
        env = {'STRIX_QPF2K_SCHED': {'value': '12', 'source': 'automatic'},
               'STRIX_QWEN_ROPE_YARN': {'value': 'off', 'source': 'baseline'}}
        baseline = ('Qwen baseline: suppressed YaRN request (STRIX_QWEN_ROPE_YARN=2); rotary=native\n' +
                    json.dumps(dict(rotary=native_status, suppressed_yarn_request='STRIX_QWEN_ROPE_YARN=2',
                                    admitted=True, environment=env, argv=['--context', '8192', '--baseline'])))
        parsed = k.parse_effective(baseline)
        self.assertEqual(parsed['environment']['STRIX_QPF2K_SCHED'], '12')
        self.assertEqual(parsed['environment']['STRIX_QWEN_ROPE_YARN'], 'off')
        self.assertEqual(k.qwen_rotary_reference({'decode': parsed}), {})
        payload = dict(rope_mode='yarn', factor=2, original_max=262144, effective_max=524288,
                       rotary_dim=64, index_rotary_dim=64, theta=10000000, mrope_sections=[11,11,10],
                       mrope_interleaved=True, beta_fast=32, beta_slow=1, truncate=True,
                       attention_factor=1.0693147180559945, math_version='qwen-yarn-hf64-v1',
                       generator='fp64-libm-to-f32-rne', table_sha256='1'*64)
        active = ('Qwen experimental YaRN: factor=2 context=300000 T=8192; qualification=experimental; '
                  'verified PLE credit=28322006848 bytes\n' +
                  json.dumps(dict(rotary=payload, admission={'host_reserve_bytes': 8589934592}, admitted=True,
                                  environment={'STRIX_QPF2K_SCHED': {'value': '12', 'source': 'automatic'}},
                                  argv=['--context', '300000', '--rope-yarn', '2'])))
        parsed = k.parse_effective(active)
        self.assertEqual(parsed['arguments'][-2:], ['--rope-yarn', '2'])
        self.assertNotEqual(k.qwen_rotary_reference({'decode': parsed}), {})

    def test_rotary_reference_identity(self):
        native = {'decode': {'settings': {}}}
        self.assertEqual(k.qwen_rotary_reference(native), {})
        payload = dict(rope_mode='yarn', factor=2, original_max=262144, effective_max=524288,
                       rotary_dim=64, index_rotary_dim=64, theta=10000000, mrope_sections=[11,11,10],
                       mrope_interleaved=True, beta_fast=32, beta_slow=1, truncate=True,
                       attention_factor=1.0693147180559945, math_version='qwen-yarn-hf64-v1',
                       generator='fp64-libm-to-f32-rne', table_sha256='1'*64)
        config = {'decode': {'settings': {'rotary': payload}}}
        key = k.qwen_rotary_reference(config)
        signals = [dict(arguments=['--rope-yarn','2']), dict(arguments=['--rope-yarn=2']),
                   dict(requested_args=['--rope-yarn','2']), dict(requested_args={'--rope-yarn':'2'}), dict(requested_arguments='--rope-yarn 2'),
                   dict(environment={'STRIX_QWEN_ROPE_YARN':'2'}),
                   dict(settings=dict(reason='experimental-yarn')),
                   dict(settings=dict(environment={'STRIX_QWEN_ROPE_YARN':dict(value='2',source='explicit')})),
                   dict(requested=dict(argv=['--rope-yarn','2'])),
                   dict(settings=dict(resolver_reason='experimental-yarn'))]
        for signal in signals:
            for native_payload in (None, dict(rope_mode='native')):
                changed = copy.deepcopy(signal)
                changed.setdefault('settings', {})
                if native_payload is not None: changed['settings']['rotary'] = native_payload
                with self.subTest(signal=signal, native=native_payload), self.assertRaises(ValueError):
                    k.qwen_rotary_reference({'decode':changed})
            changed = copy.deepcopy(signal)
            changed.setdefault('settings', {})['rotary'] = payload
            self.assertEqual(key, k.qwen_rotary_reference({'decode':changed}))
        for signal in (dict(arguments=['--rope-yarn','4']), dict(arguments=['--rope-yarn','off']),
                       dict(environment={'STRIX_QWEN_ROPE_YARN':'4'}),
                       dict(requested_args=['--rope-yarn=4'])):
            changed = dict(signal, settings=dict(rotary=payload))
            with self.assertRaises(ValueError): k.qwen_rotary_reference({'decode':changed})
        removed = copy.deepcopy(config)
        removed['decode']['settings']['reason'] = 'experimental-yarn'
        removed['decode']['arguments'] = ['--context','300000','--rope-yarn','2']
        del removed['decode']['settings']['rotary']
        with self.assertRaises(ValueError): k.qwen_rotary_reference(removed)

        for field in payload:
            changed = copy.deepcopy(config)
            value = changed['decode']['settings']['rotary'][field]
            if field == 'table_sha256': value = '2'*64
            elif field == 'rope_mode': continue
            elif isinstance(value, bool): value = not value
            elif isinstance(value, (int, float)): value += 1
            elif isinstance(value, list): value[0] += 1
            else: value += '-changed'
            changed['decode']['settings']['rotary'][field] = value
            self.assertNotEqual(key, k.qwen_rotary_reference(changed), field)
        for missing in payload:
            changed = copy.deepcopy(config)
            del changed['decode']['settings']['rotary'][missing]
            with self.assertRaises(ValueError): k.qwen_rotary_reference(changed)
        changed = copy.deepcopy(config)
        changed['decode']['settings']['rotary']['table_sha256'] = None
        with self.assertRaises(ValueError): k.qwen_rotary_reference(changed)
        changed = copy.deepcopy(config)
        changed['decode']['settings']['rotary']['factor'] = 4
        changed['decode']['settings']['rotary']['effective_max'] = 1048576
        self.assertNotEqual(key, k.qwen_rotary_reference(changed))

    def test_rotary_refusal_does_not_publish(self):
        inv=dict(build_options={},binary_sha256s=[],model_files=[],repo_commit='0'*40,
                 tracked_dirty=False,hardware=dict(rocm_version=None,hip_version=None,kernel='test'))
        k.dump(self.out/'local-inventory.json',inv)
        cases = [dict(rotary={}), dict(rotary=dict(factor=2)),
                 dict(rope_mode='yarn',rotary=dict(rope_mode='native')),
                 dict(rope_mode='native',rotary=dict(rope_mode='yarn')),
                 dict(rotary=dict(rope_mode='native',factor=2)),
                 dict(rotary=dict(rope_mode='native',effective_max=524288)),
                 dict(rotary=dict(rope_mode='native',table_sha256='1'*64)),
                 dict(rotary=dict(rope_mode='native',descriptor_sha256='1'*64)),
                 dict(rotary=dict(rope_mode='native'),rotary_identity=dict(rope_mode='yarn'))]
        for settings in cases:
            k.dump(self.out/'effective-decode.json',dict(settings=settings))
            with patch.object(k,'ENGINE','qwen'), patch.dict(k.os.environ,{'BENCH_PROFILE':'default','BENCH_SUITE':'quick','BENCH_PROMPTS':'kit'}):
                with self.assertRaises(ValueError): k.initialize(self.out)
            self.assertFalse((self.out/'identity.json').exists(), settings)
        native=dict(rope_mode='native',factor=0,original_max=262144,effective_max=262144,
                    rotary_dim=64,index_rotary_dim=64,theta=10000000,mrope_sections=[11,11,10],
                    mrope_interleaved=True,beta_fast=32,beta_slow=1,truncate=True,attention_factor=1,
                    math_version='qwen-yarn-hf64-v1',generator='fp64-libm-to-f32-rne',table_sha256=None,table_bytes=0)
        self.assertEqual(k.qwen_rotary_reference({'decode':dict(settings=dict(rotary=native))}), {})
        native['descriptor_sha256']=k.hashlib.sha256(b'').hexdigest()
        with patch.object(k,'ENGINE','qwen'), patch.dict(k.os.environ,{'BENCH_PROFILE':'default','BENCH_SUITE':'quick','BENCH_PROMPTS':'kit'}):
            k.dump(self.out/'effective-decode.json',dict(settings={}))
            k.initialize(self.out)
            historical=k.read(self.out/'identity.json')['reference_sha256']
            k.dump(self.out/'effective-decode.json',dict(settings=dict(rotary=native)))
            k.initialize(self.out)
            self.assertEqual(k.read(self.out/'identity.json')['reference_sha256'],historical)


    def test_qwen_auto_config_and_admission(self):
        config={'admitted':True,'environment':{'STRIX_PREFILL_PIPE':{'value':'255','source':'automatic'}},
                'argv':['--context','8192'],'T':1024,'context':8192,
                'decode_profile':'dense-a8','prefill_profile':'fast-prefill-f16wmma-v1'}
        text='Qwen auto-prefill: automatic T1024 context=8192\n'+json.dumps(config)
        parsed=k.parse_effective(text)
        self.assertEqual(parsed['environment'],{'STRIX_PREFILL_PIPE':'255'})
        self.assertEqual(parsed['settings']['environment']['STRIX_PREFILL_PIPE']['source'],'automatic')
        log=self.out/'prefill.log'
        log.write_text(text+'\n'+json.dumps({'generated_tokens':[1],'prompt_ms':100}),encoding='utf-8')
        with patch.object(k,'ENGINE','qwen'), patch.dict(k.os.environ,{'BENCH_SELECTED_PROFILE':'default'}):
            k.ingest(self.out,'prefill-1024','ttft',1,log)
        sample=json.loads((self.out/'samples.jsonl').read_text())
        self.assertEqual(sample['auto_prefill']['T'],1024)
        self.assertEqual(sample['auto_prefill']['context'],8192)
        self.assertIsNone(sample['engine_digest'])
        for text in ('Qwen auto-prefill declined: test-capacity; using serial prefill', 'no notice'):
            with self.assertRaises(ValueError): k.qwen_admission(text)
        inv=dict(build_options={},binary_sha256s=[],model_files=[],repo_commit='0'*40,
                 tracked_dirty=False,hardware=dict(rocm_version=None,hip_version=None,kernel='test'))
        k.dump(self.out/'local-inventory.json',inv)
        k.dump(self.out/'effective-prefill.json',parsed)
        env={'prefill':parsed['environment']}
        k.dump(self.out/'effective-environment.json',env)
        with patch.object(k,'ENGINE','qwen'), patch.dict(k.os.environ,{'BENCH_PROFILE':'default','BENCH_SUITE':'quick','BENCH_PROMPTS':'kit'}):
            k.initialize(self.out)
        identity=k.read(self.out/'identity.json')
        self.assertEqual(identity['identity']['environment'],env)
        self.assertEqual(identity['reference_key']['environment'],env)
        self.assertEqual(identity['reference_key']['profile_pairs']['prefill']['prefill_profile'],'fast-prefill-f16wmma-v1')
        self.assertEqual(identity['reference_sha256'], '294a5ed80404d42c03741e36165dc54202753397446161f7ba16a30a1956a40e')
        with patch.dict(k.os.environ,{'BENCH_SELECTED_PROFILE':'baseline'}):
            self.assertEqual(k.parse_effective('{"load_serial":true,"draft_vocab":"full"}')['environment'],{'DG_LOAD_SERIAL':'1'})

    def test_model_hash_cache(self):
        asset = self.out / 'model.bin'
        asset.write_bytes(b'model')
        calls = []
        def counted(path):
            calls.append(path)
            return k.sha(path)
        cache = k.ModelHashes(self.out, 'cached', counted)
        first = cache.digest(asset)
        cache.save()
        self.assertTrue((self.out / '.bench-sha256-cache.json').is_file())
        cache = k.ModelHashes(self.out, 'cached', counted)
        self.assertEqual(cache.digest(asset), first)
        self.assertEqual(len(calls), 1)
        st = asset.stat()
        k.os.utime(asset, ns=(st.st_atime_ns, st.st_mtime_ns + 1000000000))
        self.assertEqual(cache.digest(asset), first)
        self.assertEqual(len(calls), 2)
        self.assertIsNone(k.ModelHashes(self.out, 'skip', counted).digest(asset))
        self.assertEqual(len(calls), 2)
        k.ModelHashes(self.out, 'full', counted).digest(asset)
        self.assertEqual(len(calls), 3)

    def test_resolved_parameters_and_validation(self):
        import contextlib, io
        out = io.StringIO()
        args = ['glm-pair','full','2k,16k','','1,4,8','2','0','','kit','prefill','0']
        with contextlib.redirect_stdout(out): k.interface.resolve(args,k.KIT)
        self.assertIn('context_values=(2048 16384)',out.getvalue())
        self.assertIn('profile=r1b',out.getvalue())
        self.assertIn('only=prefill',out.getvalue())
        args[7] = 'invalid'
        with self.assertRaises(ValueError): k.interface.resolve(args,k.KIT)

    def test_batched_glm_and_minimax_receipts(self):
        log = self.out / 'batch.log'
        rows = [dict(prompt_id='chat',mode='greedy',generation=2,warmup=True),
                dict(prompt_id='chat',mode='greedy',generation=2,warmup=False)]
        block = ('rank 0 pf dev free delta +1 MiB\n'
                 'rank 0 prefill gemv: 2 tokens 1.0 s, logits fnv abc, state fnv def\n'
                 'rank 0 decode: 1 steps (20.0 tok/s), tokens fnv abc\nrank 0 tokens: 1 2\n')
        log.write_text(block*2)
        with patch.object(k,'ENGINE','glm-pair'): k.ingest_batch(self.out,rows,log)
        self.assertEqual(len((self.out/'samples.jsonl').read_text().splitlines()),1)
        row = dict(prompt_id='chat',mode='greedy',generation=2,warmup=False,name='d0')
        log.write_text('rank 0 d0: decode 1 steps at ctx 2: median(steps 1-1) 50.0 ms/token = 20.0 tok/s; mean 50 ms\nrank 0 d0: tokens fnv abc ids: 1 2\nrank 0 d0: state fnv abc, final rows fnv def\n')
        with patch.object(k,'ENGINE','minimax-pair'): k.ingest_batch(self.out,[row],log)
        self.assertEqual(len((self.out/'samples.jsonl').read_text().splitlines()),2)
        with patch.object(k,'ENGINE','minimax-pair'), self.assertRaises(ValueError):
            k.ingest_batch(self.out,[dict(row,name='missing')],log)

    def test_glm_spv1_current_receipts(self):
        log=self.out/'spv1.log'
        text=('SPV1_BEGIN rank=0 N=2\n'
              'SPV1_PLAIN rank=0 prompt=code pass=A0 gen=2 ms_token=50 hash=0000000000000abc\n'
              'SPV1_RESULT rank=0 prompt=code K=7 gen=2 ms_token=25 equal=1 hash=0000000000000abc\n'
              'SPV1_PLAIN rank=0 prompt=code pass=A1 gen=2 ms_token=51 hash=0000000000000abc\n'
              'SPV1_DONE rank=0 status=PASS prompts=1 arms=1\n')
        log.write_text(text,encoding='ascii')
        rows=[dict(prompt_id='code',mode=m,generation=2,warmup=False) for m in ('greedy','spec')]
        with patch.object(k,'ENGINE','glm-pair'): k.ingest_batch(self.out,rows,log)
        samples=[json.loads(x) for x in (self.out/'samples.jsonl').read_text().splitlines()]
        self.assertEqual([x['values'] for x in samples],[[20],[40]])
        self.assertTrue(all(x['token_ids_sha256'] is None for x in samples))
        fixture=k.read(HERE/'fixtures/results.json')
        fixture['engine']='glm-pair'
        k.dump(self.out/'identity.json',fixture)
        k.dump(self.out/'prompts.json',fixture['prompts'])
        k.finalize(self.out)
        result=k.read(self.out/'results.json')
        self.assertEqual(result['exactness'],[])
        self.assertIn('kit exactness unqualified',result['skipped'][0])
        with self.assertRaisesRegex(ValueError,'without full token IDs'): k.finalize(self.out,True)
        log.write_text(text.replace('equal=1','equal=0'),encoding='ascii')
        with patch.object(k,'ENGINE','glm-pair'),self.assertRaisesRegex(ValueError,'equality mismatch'):
            k.ingest_batch(self.out,rows,log)

    def test_glm_mtp_batch(self):
        log = self.out/'mtp.log'
        log.write_text('rank 0 spec-run: K 7 | plain 50.0 ms/token, spec 25.0 ms/token (2x) | tokens EQUAL to plain (fnv abc)\nrank 0 spec-run plain tokens: 1 2\nrank 0 spec-run tokens: 1 2\n')
        rows = [dict(prompt_id='chat',mode=mode,generation=2,warmup=False) for mode in ('greedy','spec')]
        with patch.object(k,'ENGINE','glm-pair'): k.ingest_batch(self.out,rows,log)
        data=[json.loads(line) for line in (self.out/'samples.jsonl').read_text().splitlines()]
        self.assertEqual([r['values'] for r in data],[[20],[40]])

    def test_hash_is_canonical(self):
        self.assertEqual(k.id_hash([1, 2, 3]), 'a615eeaee21de5179de080de8c3052c8da901138406ba71c38c032845f7d54f4')
        with self.assertRaises(AssertionError):
            k.id_hash([True])

    def test_qwen_receipt(self):
        log = self.out / 'receipt.log'
        log.write_text('diagnostic\n' + json.dumps({'generated_tokens': [1, 2], 'generation_step_elapsed_ms_total': 50}), encoding='utf-8')
        with patch.object(k, 'ENGINE', 'qwen'):
            k.ingest(self.out, 'code', 'plain', 2, log)
        row = json.loads((self.out / 'samples.jsonl').read_text())
        self.assertEqual(row['values'], [40])
        self.assertEqual(row['actual_tokens'], 2)

    def test_missing_timing_fails_closed(self):
        log = self.out / 'receipt.log'
        log.write_text('{"generated_tokens":[1]}', encoding='utf-8')
        with patch.object(k, 'ENGINE', 'qwen'), self.assertRaises(ValueError):
            k.ingest(self.out, 'code', 'spec', 1, log)

    def test_deepseek_native_plan_and_report(self):
        import contextlib, io
        names=[x['id'] for x in k.interface.prompt_set('kit',k.KIT)]
        params=dict(profile='default',contexts=[3,5],gen_tokens=2,warmup=1,repeat=2,only=['decode','prefill'])
        for n in params['contexts']:
            for name in names: (self.out/f'{name}-ctx{n}.ids').write_text(','.join(['1']*n))
        output=io.StringIO()
        with patch.object(k,'ENGINE','deepseek-pair'), patch.dict(k.os.environ,{'BENCH_PARAMS':json.dumps(params)}), contextlib.redirect_stdout(output):
            k.interface.plan(self.out,False,k)
        plan=output.getvalue()
        self.assertEqual(plan.count('deepseek_native_decode '),2)
        self.assertNotIn('strix-deepseek41-serve',plan)
        self.assertIn('--transport usb4auto',plan)
        for mode in ('greedy','spec'):
            rows=k.read(self.out/f'decode-{mode}.manifest.json')
            lines=(self.out/f'decode-{mode}.ids').read_text().splitlines()
            self.assertEqual(len(rows),2*len(names)*3)
            self.assertEqual(sum(x['warmup'] for x in rows),2*len(names))
            self.assertEqual([len(line.split(',')) for line in lines],[int(x['prompt_id'].rsplit('ctx',1)[1]) for x in rows])
        fixture=k.read(HERE/'fixtures/results.json')
        fixture['engine']='deepseek-pair'
        row=fixture['measurements'][0]
        fixture['measurements']=[dict(row,prompt_id='code-ctx3',mode=mode,timing_scope='native') for mode in ('greedy','spec')]
        fixture['measurements'].append(dict(row,prompt_id='http-c1',mode='aggregate-wall',timing_scope='http-end-to-end'))
        # The rewritten cells are a greedy/spec pair; the fixture's lone plain exactness cell would be a partial pair.
        fixture['exactness']=[]
        k.dump(self.out/'results.json',fixture)
        k.report(self.out/'results.json')
        text=(self.out/'results.md').read_text()
        for value in ('34.5 tok/s','70.6 tok/s','workload differs','HTTP end-to-end'): self.assertIn(value,text)

    def test_pair_receipt_and_prefill(self):
        log = self.out / 'pair.log'
        log.write_text('tokens (3, fnv 0000000000000001): 1 2 3\nprompt 16 tokens in 1.00 s; generation 2 tokens in 0.050 s = 40.00 tok/s\n', encoding='utf-8')
        with patch.object(k, 'ENGINE', 'deepseek-pair'):
            k.ingest(self.out, 'code', 'greedy', 2, log)
            log.write_text('ttft warm-up run: 1024 rows, wall 999.000 ms\nttft run 0: 1024 rows, wall 2000.000 ms (run 1999.0 ms), next token 4, logits_fnv 0000000000000001\nttft run 1: 1024 rows, wall 2200.000 ms (run 2199.0 ms), next token 4, logits_fnv 0000000000000001\n', encoding='utf-8')
            k.ingest(self.out, 'prefill-1024', 'ttft', 1, log)
        rows = [json.loads(x) for x in (self.out / 'samples.jsonl').read_text().splitlines()]
        self.assertEqual([x['values'] for x in rows], [[40], [2], [2.2]])

    def test_rtx_receipt(self):
        log = self.out / 'rtx.log'
        events = [{'event': 'chat_end', 'name': 'chats0#1-0-x', 'toks': [1, 2], 'digest': 'abc'},
                  {'event': 'chat_end', 'name': 'schats0#1-0-x', 'toks': [1, 2], 'digest': 'abc'},
                  {'event': 'summary', 'phase': 'chats0#1-0-x', 'tok_s': 10},
                  {'event': 'spec_summary', 'phase': 'schats0#1-0-x', 'tok_s': 20}]
        log.write_text('\n'.join(json.dumps(x) for x in events), encoding='utf-8')
        with patch.object(k, 'ENGINE', 'deepseek-rtx'):
            k.ingest(self.out, 'code', 'rtx', 1, log)
            log.write_text('{"event":"run","seconds":2.5,"next_token":8,"logits_fnv":"abc"}\n', encoding='utf-8')
            k.ingest(self.out, 'prefill-1024', 'ttft', 1, log)
        self.assertEqual(len((self.out / 'samples.jsonl').read_text().splitlines()), 3)

    def test_schema_rejects_wrong_stats(self):
        d = k.read(HERE / 'fixtures/results.json')
        d['measurements'][0]['median'] = 0
        with self.assertRaises(ValueError):
            k.validate(d)

    def test_profile_pins_and_schema(self):
        schema = k.read(HERE.parent / 'results.schema.json')
        fixture = k.read(HERE / 'fixtures/results.json')
        self.assertTrue(set(schema['required']).issubset(fixture))
        for engine, (_, names) in k.PROFILES.items():
            profile = (HERE.parent / 'engines' / (engine + '.sh')).read_text()
            for name in names:
                if name in k.PINS:
                    size, digest = k.PINS[name]
                    self.assertIn(str(size) if size is not None else chr(110)+chr(117)+chr(108)+chr(108), profile)
                    self.assertIn(digest, profile)

    def test_reference_verdicts(self):
        fixture = k.read(HERE / 'fixtures/results.json')
        identity = copy.deepcopy(fixture)
        for key in ('measurements', 'exactness', 'prompts'):
            identity.pop(key)
        k.dump(self.out / 'identity.json', identity)
        k.dump(self.out / 'prompts.json', fixture['prompts'])
        sample = {'prompt_id': 'code', 'mode': 'plain', 'generate_length': 2, 'actual_tokens': 2,
                  'token_ids_sha256': k.id_hash([1, 2]), 'engine_digest': None,
                  'values': [40], 'unit': 'tok/s', 'log': 'fixture.log'}
        (self.out / 'samples.jsonl').write_text(json.dumps(sample) + '\n', encoding='utf-8')
        k.finalize(self.out, True)
        d = k.read(self.out / 'results.json')
        self.assertEqual(d['exactness'][0]['verdict'], 'NO-REFERENCE')
        refs = k.read(self.out / 'candidate-refs.json')
        actual_read = k.read
        def read_refs(path):
            return refs if Path(path).parent.name == 'refs' else actual_read(path)
        with patch.object(k, 'read', read_refs):
            k.finalize(self.out)
            self.assertEqual(actual_read(self.out / 'results.json')['exactness'][0]['verdict'], 'EXACT')
            # A self-built binary changes the full identity but not the reference key.
            identity['identity_sha256'] = '2' * 64
            k.dump(self.out / 'identity.json', identity)
            k.finalize(self.out)
            self.assertEqual(actual_read(self.out / 'results.json')['exactness'][0]['verdict'], 'EXACT')
            sample['token_ids_sha256'] = k.id_hash([2, 3])
            (self.out / 'samples.jsonl').write_text(json.dumps(sample) + '\n', encoding='utf-8')
            k.finalize(self.out)
            self.assertEqual(actual_read(self.out / 'results.json')['exactness'][0]['verdict'], 'MISMATCH')

    def test_redaction(self):
        address = '.'.join(['1'] * 4)
        mac = ':'.join(['ab'] * 6)
        text = f'model /path/to/models/model.gguf peer user@node-b address {address} MAC {mac} speed 40 tok/s'
        clean = k.sanitize(text)
        for private in ('/path/to', 'user@node-b', address, mac):
            self.assertNotIn(private, clean)
        self.assertIn('40 tok/s', clean)

    POLICY = {'requested': True, 'session_requested': False, 'effective': True, 'runner': 'http-mtp-fixed'}
    STARTUP = 'Qwen serve-prefill: explicit T8192; arithmetic preserved\n'

    def serve(self, fixture, unique=True, label='serve', claim='', startup=STARTUP, policy=POLICY, inspected=None, **env):
        """Run the HTTP client against a recorded server stream; each request gets its own completion id."""
        (self.out / ('server' + label[len('serve'):] + '.log')).write_text(startup + '{"event":"READY"}\n', encoding='ascii')
        (self.out / 'http-measurements.json').unlink(missing_ok=True)
        (self.out / ('effective-' + label + '.json')).unlink(missing_ok=True)
        if inspected is not None:
            k.dump(self.out / ('effective-' + label + '.json'), {'settings': inspected})
        for prompt in ('chat', 'code', 'prose'):
            (self.out / (prompt + '.ids')).write_text('1,2,3\n', encoding='ascii')
        text = (SSE / fixture).read_bytes()
        calls = iter(range(1000))
        class Response:
            status = 200
            def __init__(self, body):
                self.body = body
            def __enter__(self):
                return self
            def __exit__(self, *args):
                pass
            def __iter__(self):
                yield from self.body.splitlines(keepends=True)
            def read(self):
                return self.body
        def urlopen(request, *args, **kwargs):
            if isinstance(request, str) and request.endswith('/v1/capabilities'):
                if policy is None:
                    raise k.urllib.error.HTTPError(request, 404, 'not found', {}, None)
                return Response(json.dumps({'object': 'strix.capabilities', 'cache_policy': policy}).encode())
            return Response(text.replace(b'cmpl-ID', b'cmpl-%d' % next(calls)) if unique else text)
        with patch.object(k.urllib.request, 'urlopen', urlopen), patch.dict(k.os.environ, {'BENCH_REPEATS': '1', **env}):
            k.http(self.out, 18880, label, claim)
        return k.read(self.out / 'http-measurements.json')

    def test_http_route_read_and_claims(self):
        rows = self.serve('qwen-mtp.sse', claim='8192', inspected={'T': 8192, 'cache_policy': dict(self.POLICY, effective=None)})
        route = rows[0]['route']
        self.assertEqual((route['label'], route['status'], route['T'], route['T_source'], route['cache_policy_source']),
                         ('default', 'read', 8192, 'startup-log', 'capabilities'))
        self.assertEqual(route['claimed'], {'pinned_T': 8192, 'inspected_T': 8192,
                                            'cache_requested': {'requested': True, 'session_requested': False}})
        self.assertEqual({r['prompt_id'] for r in rows}, {'http-c1', 'http-c4', 'http-c8'})
        # Servers without the startup line or capabilities are labelled unread; nothing is claimed for them.
        rows = self.serve('minimax.sse', startup='', policy=None)
        self.assertEqual((rows[0]['route']['status'], rows[0]['route']['T'], rows[0]['route']['cache_policy']), ('unread', None, None))
        for kwargs, error in ((dict(claim='8192', inspected={'T': 2048}), 'inspected_T 2048 differs'),
                              (dict(claim='8192', inspected={'T': 8192}, startup=''), 'pinned_T 8192 differs'),
                              (dict(claim='8192', inspected={'cache_policy': dict(self.POLICY, requested=False)}), 'cache policy'),
                              (dict(claim='8192', inspected={'cache_policy': self.POLICY}, policy=None), 'cache policy'),
                              (dict(claim='8192', startup='Qwen serve-prefill: automatic T2048\n'), 'pinned_T 8192 differs'),
                              (dict(BENCH_ENGINE='qwen'), 'no pinned T')):
            with self.subTest(error=error), self.assertRaisesRegex(ValueError, error):
                self.serve('qwen-mtp.sse', **kwargs)

    def test_http_serve_cell_pins_t(self):
        rows = self.serve('qwen-mtp.sse', claim='8192', BENCH_ENGINE='qwen')
        self.assertEqual({r['prompt_id'] for r in rows}, {'http-c1', 'http-c4', 'http-c8'})
        self.assertEqual({(r['route']['label'], r['route']['T']) for r in rows}, {('default', 8192)})
        # Every Qwen serve cell pins T, keeps the http-cN ids and claims what the server resolved.
        good = k.read(HERE / 'fixtures/results-witnessed.json')
        for change, error in ((lambda d: d['measurements'][2]['route'].update(label='tiled'), 'invalid serve route'),
                              (lambda d: d['measurements'][2]['route']['claimed'].pop('pinned_T'), 'no pinned T'),
                              (lambda d: d['measurements'][2]['route']['claimed'].update(pinned_T=2048), 'pinned_T 2048 differs'),
                              (lambda d: d['measurements'][2].update(prompt_id='http-tiled-c1'), 'cell id'),
                              (lambda d: d['measurements'][2]['route'].update(T=None), 'status'),
                              (lambda d: d['measurements'][2]['route']['claimed'].update(inspected_T=4096), 'inspected_T 4096 differs'),
                              (lambda d: d['measurements'][2]['route']['cache_policy'].update(requested=False), 'cache policy')):
            d = copy.deepcopy(good)
            change(d)
            with self.subTest(error=error), self.assertRaisesRegex(ValueError, error):
                k.validate(d)

    def test_http_sse_usage_and_concurrency(self):
        rows = self.serve('qwen-mtp.sse', BENCH_DRAFT_ROLE='positive-control')
        # aggregate-wall, ttft and per-token tpot for each of C1/C4/C8
        self.assertEqual(len(rows), 9)
        for c in (1, 4, 8):
            requests = k.read(self.out / f'http-c{c}.json')['requests'][0]
            self.assertEqual(len(requests), 8)
            self.assertTrue(all(x['completion_tokens'] == 4 and x['ttft_s'] > 0 for x in requests))
            self.assertTrue(all(len(x['emission_times_s']) == 3 and 'completion_id' not in x for x in requests))
        self.assertEqual({row['timing_scope'] for row in rows},{'http-end-to-end'})
        # Eight requests of 6 proposed, 2 verify cycles and 3 accepted drafts each.
        self.assertEqual({json.dumps(row['draft'], sort_keys=True) for row in rows},
                         {json.dumps(dict(role='positive-control', status='drafted', proposed=48, verified=16, accepted=24), sort_keys=True)})
        self.assertEqual([row['tpot_scope'] for row in rows if row['mode'] == 'tpot'], ['per-token-request-mean'] * 3)

    def test_http_positive_control_and_conflated_requests(self):
        for fixture, status in (('qwen-no-draft.sse', 'no-draft'), ('qwen-declined.sse', 'declined'), ('minimax.sse', 'unwitnessed')):
            with self.subTest(fixture=fixture):
                with self.assertRaisesRegex(ValueError, 'positive-control.*' + status):
                    self.serve(fixture, BENCH_DRAFT_ROLE='positive-control')
                # Observed cells keep the legitimate label instead of passing as measured MTP.
                rows = self.serve(fixture)
                self.assertEqual({row['draft']['status'] for row in rows}, {status})
                self.assertEqual({row['draft']['role'] for row in rows}, {'observed'})
        with self.assertRaisesRegex(ValueError, 'conflated'):
            self.serve('qwen-mtp.sse', unique=False)

    def test_sse_fixtures_fail_closed(self):
        def parse(name, clock=None, prompt=(1, 2, 3)):
            ticks = iter(x / 10 for x in range(1, 1000))
            return k.sse_receipt((SSE / name).read_bytes().splitlines(keepends=True), list(prompt), 0,
                                 clock or (lambda: next(ticks)))
        r = parse('qwen-mtp.sse')
        self.assertEqual(r['draft'], dict(status='drafted', proposed=6, verified=2, accepted=3))
        self.assertEqual(r['completion_tokens'], 4)
        self.assertEqual(len(r['emission_times_s']), 3)
        # TPOT is per token: three chunks carried four tokens, so the span divides by three tokens, not two gaps.
        self.assertAlmostEqual(r['tpot_s'], (r['emission_times_s'][-1] - r['emission_times_s'][0]) / 3)
        self.assertEqual(r['engine_digest'], {'final_state_digest': 'ab' * 16})
        self.assertEqual(parse('minimax.sse')['draft']['status'], 'unwitnessed')
        self.assertEqual(parse('qwen-no-draft.sse')['draft']['status'], 'no-draft')
        self.assertEqual(parse('qwen-declined.sse')['draft']['status'], 'declined')
        for name, error in (('truncated-done.sse', 'truncated'), ('truncated-event.sse', 'truncated'),
                            ('truncated-ids.sse', 'token IDs'), ('conflated.sse', 'conflated'),
                            ('partial-witness.sse', 'partial')):
            with self.subTest(name=name), self.assertRaisesRegex(ValueError, error):
                parse(name)
        with self.assertRaisesRegex(ValueError, 'prompt token count'):
            parse('qwen-mtp.sse', prompt=(1, 2))
        backwards = iter([0.5, 0.4, 0.3, 0.2, 0.1, 0.0] + [0.0] * 20)
        with self.assertRaisesRegex(ValueError, 'non-monotonic'):
            parse('qwen-mtp.sse', clock=lambda: next(backwards))
        with patch.dict(k.os.environ, {'BENCH_EVIDENCE': 'ids,state'}):
            self.assertEqual(parse('qwen-mtp.sse')['evidence_scope'], ['ids', 'state'])
            with self.assertRaisesRegex(ValueError, 'state'):
                parse('minimax.sse')
        with patch.dict(k.os.environ, {'BENCH_EVIDENCE': 'logits'}), self.assertRaisesRegex(ValueError, 'logits'):
            parse('qwen-mtp.sse')

    def test_draft_witness_labels(self):
        self.assertEqual(k.draft_witness({'mtp_cycles': 3, 'mtp_proposed_candidates': 9, 'mtp_accepted_candidates': 4})['status'], 'drafted')
        self.assertEqual(k.draft_witness({})['status'], 'unwitnessed')
        for bad in ({'mtp_cycles': 1}, {'mtp_cycles': 1, 'mtp_proposed': 2, 'mtp_accepted': 3},
                    {'mtp_cycles': 0, 'mtp_proposed': 2, 'mtp_accepted': 1}, {'mtp_cycles': 1, 'mtp_proposed': -1, 'mtp_accepted': 0}):
            with self.subTest(bad=bad), self.assertRaises(ValueError):
                k.draft_witness(bad)
        drafted = k.draft_witness({'mtp_cycles': 1, 'mtp_proposed': 2, 'mtp_accepted': 1})
        declined = k.draft_witness({'mtp_cycles': 1, 'mtp_proposed': 2, 'mtp_accepted': 0})
        # A positive control is judged on the whole cell; a declined request beside a drafted one still drafts.
        self.assertEqual(k.draft_total([drafted, declined], 'positive-control')['accepted'], 1)
        self.assertEqual(k.draft_total([declined], 'observed')['status'], 'declined')
        for cell in ([declined], [drafted, k.draft_witness({})], []):
            with self.subTest(cell=cell), self.assertRaisesRegex(ValueError, 'positive-control'):
                k.draft_total(cell, 'positive-control')

    def test_qwen_spec_receipt_draft_witness(self):
        log = self.out / 'spec.log'
        def spec(**counters):
            log.write_text(json.dumps({'generated_tokens': [1, 2], 'mtp_generation_cycle_elapsed_ms': 25, **counters}), encoding='utf-8')
            k.ingest(self.out, 'code', 'spec', 2, log)
            return json.loads((self.out / 'samples.jsonl').read_text().splitlines()[-1])
        with patch.object(k, 'ENGINE', 'qwen'):
            row = spec(mtp_cycles=1, mtp_proposed_candidates=7, mtp_accepted_candidates=1)
            self.assertEqual(row['draft'], dict(status='drafted', proposed=7, verified=1, accepted=1, role='observed'))
            self.assertEqual(spec()['draft']['status'], 'unwitnessed')
            with self.assertRaisesRegex(ValueError, 'partial'):
                spec(mtp_proposed_candidates=7)
            log.write_text(json.dumps({'generated_tokens': [1, 2], 'generation_step_elapsed_ms_total': 50}), encoding='utf-8')
            k.ingest(self.out, 'code', 'plain', 2, log)
            self.assertNotIn('draft', json.loads((self.out / 'samples.jsonl').read_text().splitlines()[-1]))

    def test_ingest_evidence_scope_fails_closed(self):
        log = self.out / 'receipt.log'
        log.write_text(json.dumps({'generated_tokens': [1, 2], 'generation_step_elapsed_ms_total': 50,
                                   'final_logits_digest': 'cd' * 16}), encoding='utf-8')
        with patch.object(k, 'ENGINE', 'qwen'):
            with patch.dict(k.os.environ, {'BENCH_EVIDENCE': 'ids,logits'}):
                k.ingest(self.out, 'code', 'plain', 2, log)
            with patch.dict(k.os.environ, {'BENCH_EVIDENCE': 'ids,logits,state'}), self.assertRaisesRegex(ValueError, 'state'):
                k.ingest(self.out, 'code', 'plain', 2, log)
            with patch.dict(k.os.environ, {'BENCH_EVIDENCE': 'ids,weights'}), self.assertRaisesRegex(ValueError, 'BENCH_EVIDENCE'):
                k.ingest(self.out, 'code', 'plain', 2, log)
        self.assertEqual(json.loads((self.out / 'samples.jsonl').read_text())['evidence_scope'], ['ids', 'logits'])
        # GLM MTP receipts carry hashes, not token IDs: a run claiming IDs cannot record them.
        log.write_text('rank 0 spec-run: K 7 | plain 50.0 ms/token, spec 25.0 ms/token (2x) | tokens EQUAL to plain (fnv abc)\n'
                       'rank 0 spec-run plain tokens: 1 2\nrank 0 spec-run tokens: 1 2\n')
        rows = [dict(prompt_id='chat', mode=mode, generation=2, warmup=False) for mode in ('greedy', 'spec')]
        spv1 = self.out / 'spv1.log'
        spv1.write_text('SPV1_BEGIN\nSPV1_PLAIN rank=0 prompt=chat gen=2 pass=A0 hash=1 ms_token=50\n'
                        'SPV1_RESULT rank=0 prompt=chat gen=2 hash=1 equal=1 ms_token=25\n'
                        'SPV1_PLAIN rank=0 prompt=chat gen=2 pass=A1 hash=1 ms_token=50\nSPV1_DONE rank=0 status=PASS\n')
        with patch.object(k, 'ENGINE', 'glm-pair'), patch.dict(k.os.environ, {'BENCH_EVIDENCE': 'ids'}):
            k.ingest_batch(self.out, rows, log)
            with self.assertRaisesRegex(ValueError, 'token IDs'):
                k.ingest_batch(self.out, rows, spv1)

    def finalize_samples(self, samples):
        fixture = k.read(HERE / 'fixtures/results.json')
        identity = copy.deepcopy(fixture)
        for key in ('measurements', 'exactness', 'prompts'):
            identity.pop(key)
        k.dump(self.out / 'identity.json', identity)
        k.dump(self.out / 'prompts.json', fixture['prompts'])
        (self.out / 'samples.jsonl').write_text(''.join(json.dumps(s) + '\n' for s in samples), encoding='utf-8')
        k.finalize(self.out)
        return k.read(self.out / 'results.json')

    def test_finalize_cells_fail_closed(self):
        base = {'prompt_id': 'code', 'mode': 'plain', 'generate_length': 2, 'actual_tokens': 2,
                'token_ids_sha256': k.id_hash([1, 2]), 'engine_digest': None,
                'values': [40], 'unit': 'tok/s', 'log': 'fixture.log'}
        drafted = dict(status='drafted', proposed=4, verified=2, accepted=1, role='positive-control')
        spec = dict(base, mode='spec', draft=drafted)
        d = self.finalize_samples([base, spec, dict(spec, values=[41])])
        self.assertEqual([e['verdict'] for e in d['exactness']], ['NO-REFERENCE'] * 2)
        self.assertEqual([e['reference_found'] for e in d['exactness']], [False] * 2)
        self.assertEqual(d['measurements'][1]['draft'], dict(drafted, proposed=8, verified=4, accepted=2))
        self.assertTrue(d['exactness'][1]['plain_tokens_equal'])
        for samples, error in (([base, dict(base, evidence_scope=['ids'])], 'conflated'),
                               ([base, spec, dict(spec, draft=dict(drafted, role='observed'))], 'conflated'),
                               ([base, spec, dict(base, mode='spec')], 'conflated'),
                               ([base, dict(spec, draft=dict(drafted, status='no-draft', proposed=0, accepted=0))], 'positive-control'),
                               ([base, dict(spec, prompt_id='code', generate_length=3)], 'partial pair')):
            with self.subTest(error=error), self.assertRaisesRegex(ValueError, error):
                self.finalize_samples(samples)

    def test_witnessed_results_fail_closed(self):
        good = k.read(HERE / 'fixtures/results-witnessed.json')
        k.validate(good)
        def broken(change):
            d = copy.deepcopy(good)
            change(d)
            return d
        cases = {
            'positive-control': lambda d: d['measurements'][1]['draft'].update(status='no-draft', proposed=0, accepted=0),
            'draft label': lambda d: d['exactness'][1]['draft'].update(accepted=0),
            'unwitnessed': lambda d: d['measurements'][2]['draft'].update(status='unwitnessed'),
            'invalid draft': lambda d: d['measurements'][3]['draft'].pop('role'),
            'logits': lambda d: d['exactness'][0]['engine_digest'].pop('final_logits_digest'),
            'state': lambda d: d['exactness'][1].update(engine_digest={'final_logits_digest': 'cd' * 16}),
            'token IDs': lambda d: d['exactness'][0].update(actual_tokens=0),
            'invalid exactness digest': lambda d: d['exactness'][0].update(token_ids_sha256=None),
            'conflated identities: duplicate exactness': lambda d: d['exactness'].append(copy.deepcopy(d['exactness'][0])),
            'conflated identities: duplicate measurement': lambda d: d['measurements'].append(copy.deepcopy(d['measurements'][2])),
            'conflated identities: exactness prompt': lambda d: d['exactness'][1].update(prompt_sha256='0' * 64),
            'conflated identities: reference_sha256': lambda d: d['reference_key'].update(profile='other'),
            'conflated identities: duplicate prompt': lambda d: d['prompts'].append(dict(d['prompts'][0])),
            "partial pair outcome: \\('exactness'": lambda d: d['exactness'].pop(),
            "partial pair outcome: \\('measurement'": lambda d: d['measurements'].pop(0),
            'plain comparison': lambda d: d['exactness'][1].pop('plain_tokens_equal'),
            'EXACT requires': lambda d: d['exactness'][0].update(verdict='EXACT'),
            'NO-REFERENCE contradicts': lambda d: d['exactness'][0].update(reference_found=True),
            'NO-REFERENCE contradicts a found reference or an observed': lambda d: d['exactness'][0].update(repeat_stable=False),
            'complete status': lambda d: (d['exactness'][0].update(verdict='MISMATCH')),
            'bad engine': lambda d: d.update(engine='Qwen Pair'),
        }
        for error, change in cases.items():
            with self.subTest(error=error), self.assertRaisesRegex(ValueError, error):
                k.validate(broken(change))
        # A found reference may pass; an absent reference stays NO-REFERENCE and never becomes EXACT.
        exact = broken(lambda d: [e.update(verdict='EXACT', reference_found=True) for e in d['exactness']])
        k.validate(exact)
        # A failed run keeps its successful cells even when a pair is incomplete.
        k.validate(broken(lambda d: (d['exactness'].pop(), d.update(status='failed'))))
        # Engines are an open set: a new adapter name needs no schema or validator change.
        k.validate(broken(lambda d: d.update(engine='future-adapter')))

    @unittest.skipUnless(importlib.util.find_spec('jsonschema'), 'jsonschema not installed')
    def test_json_schema_mirrors_witness_rules(self):
        import jsonschema
        schema = k.read(HERE.parent / 'results.schema.json')
        good = k.read(HERE / 'fixtures/results-witnessed.json')
        for name in ('results.json', 'results-witnessed.json'):
            jsonschema.validate(k.read(HERE / 'fixtures' / name), schema)
        jsonschema.validate(dict(good, engine='future-adapter'), schema)
        for change in (lambda d: d['measurements'][1]['draft'].update(status='no-draft', proposed=0, accepted=0),
                       lambda d: d['measurements'][1]['draft'].update(accepted=0),
                       lambda d: d['exactness'][0].update(verdict='EXACT'),
                       lambda d: d['exactness'][0].update(reference_found=True),
                       lambda d: d['exactness'][0].update(engine_digest=None),
                       lambda d: d['measurements'][2]['draft'].update(status='unwitnessed'),
                       lambda d: d['measurements'][2]['route'].update(T=None),
                       lambda d: d['measurements'][2]['route'].update(label='tiled'),
                       lambda d: d['measurements'][2]['route']['claimed'].pop('pinned_T')):
            d = copy.deepcopy(good)
            change(d)
            with self.assertRaises(jsonschema.ValidationError):
                jsonschema.validate(d, schema)

    def test_run_identity_checked_up_front(self):
        self.assertRegex(k.run_identity(), '^[0-9a-f]{40}$')
        # A tree without .git has no commit: the run refuses before any launch.
        with patch.object(k, 'ROOT', self.out), self.assertRaisesRegex(ValueError, 'run identity: no repository commit'):
            k.main(['run-identity'])

    def test_pair_identity_mismatch(self):
        data = {'repo_commit': '0' * 40, 'binary_sha256s': [], 'model_files': [],
                'build_options': {}, 'hardware': {'rocm_version': 'test'}}
        k.dump(self.out / 'a.json', data)
        data['repo_commit'] = '1' * 40
        k.dump(self.out / 'b.json', data)
        with self.assertRaisesRegex(ValueError, 'repo_commit'):
            k.main(['compare', str(self.out / 'a.json'), str(self.out / 'b.json')])


# Controllers that apps/hip_decode_options.hpp refuses under a pinned --numerical-profile
# (validate_numerical_profile_cli): any presence, or a value other than the ordinary one.
REFUSED = ('--gpu-greedy-token', '--integer-lm-head', '--prompt-lookup', '--serve-prefix-u32le',
           '--serve-tensor-cohort-rows', '--serve-prefill-rows', '--prefix-cache-roundtrip',
           '--moe-token-cost-policy', '--qsa-hierarchical-selection', '--prefetch-hc-projection-next-tile',
           '--ranked-moe-router-topk', '--offline-tuning-cache', '--offline-compile-provenance')
ORDINARY = {'--execution-mode': 'eager', '--serve-concurrency': '1', '--gdn-recurrent-layout': 'baseline',
            '--gdn-recurrent-storage': 'key-value', '--qsa-gqa-heads-per-block': '1'}


V5_ORGANIZATION = ('--parallel-hyper-mix', '--staged-hyper-projections', '--fused-staged-hyper-mix',
                   '--typed-fused-hyper-projections', '--prefetch-typed-hyper-rows', '--prefetch-q8-hyper-output',
                   '--lane-chain-hyper-output', '--prefetch-q8-qsa-input', '--parallel-moe-router',
                   '--wave-moe-router-topk', '--virtual-q8-gdn-input', '--prefetch-q8-gdn-input',
                   '--virtual-q8-gdn-output', '--deferred-moe-status', '--prefetch-q8-shared-expert',
                   '--q4k-a8g32-gate-up', '--grouped-exact-q51-down', '--virtual-q8-lm-head',
                   '--q51-slot-parallel-down', '--batched-moe-activation', '--wave8-q8-gdn-input',
                   '--wave8-q8-gdn-output', '--prefetch-q8-lm-head', '--wave8-q8-lm-head', '--wave8-q8-qsa-output')


def refused(argv):
    """Combinations the binary's option checks refuse for a built Qwen command."""
    found = []
    if '--numerical-profile' in argv:
        found += [x for x in REFUSED if x in argv]
        found += [f'{x} {argv[argv.index(x) + 1]}' for x, value in ORDINARY.items()
                  if x in argv and argv[argv.index(x) + 1] != value]
        if '--mtp' in argv and '--mtp-window-prefix' not in argv:
            found.append('--mtp without --mtp-window-prefix')
        if any(x in argv for x in ('--known-prefix-window', '--known-prefix-rows', '--known-prefix-expert-union',
                                   '--known-prefix-prefill-rows')) and '--known-prefix-full-rows' not in argv:
            found.append('known-prefix organization without --known-prefix-full-rows')
        if '--known-prefix-window' in argv:
            found += [f'known-prefix window without {x}' for x in V5_ORGANIZATION if x not in argv]
        if '--mtp-calibration-rows' in argv and '--known-prefix-window' not in argv:
            found.append('MTP calibration rows without the tiled prompt')
        if '--http-serve' in argv and '--known-prefix-window' not in argv:
            found.append('--http-serve without the tiled MTP route')
    # The known-prefix window and MTP checks refuse GPU greedy selection with or without a profile.
    if '--gpu-greedy-token' in argv and ('--known-prefix-window' in argv or '--mtp' in argv):
        found.append('--gpu-greedy-token with a known-prefix window or MTP')
    return found


@unittest.skipUnless(shutil.which('bash'), 'bash required for dry-run plans')
class QwenCommandTests(unittest.TestCase):
    """Every Qwen command the kit builds, taken from offline dry-run previews."""
    @classmethod
    def setUpClass(cls):
        root = HERE.parents[2]
        runs = [([p], {}) for p in ('default', 'plain', 'baseline', 'gsq', 'mtp-k1', 'mtp-k4')]
        runs += [(['default', 'serve'], {})]
        cls.commands = []
        for args, env in runs:
            only = ['--only', 'serve'] if 'serve' in args else ['--only', 'decode,prefill,exactness', '--serve']
            result = subprocess.run(['bash', 'scripts/bench/bench.sh', 'run', '--engine', 'qwen', '--suite', 'quick',
                                     '--profile', args[0], *only, '--dry-run',
                                     '--model-dir', os.environ.get('BENCH_QWEN_MODEL_DIR', 'models')],
                                    capture_output=True, text=True, cwd=root, timeout=600,
                                    env={'BENCH_PYTHON': Path(sys.executable).as_posix(), **os.environ, **env})
            assert result.returncode == 0, result.stderr
            for line in result.stdout.splitlines():
                words = shlex.split(line) if 'strix-hip-decode-token' in line else []
                at = next((i for i, w in enumerate(words) if w.endswith(('strix-hip-decode-token', 'strix-hip-decode-token.exe'))), None)
                if at is not None:
                    cls.commands.append((args[0], [w for w in words[:at] if '=' in w], words[at:]))

    def test_no_refused_controller_combinations(self):
        self.assertGreater(len(self.commands), 10)
        self.assertTrue(any('--numerical-profile' in argv and '--known-prefix-window' in argv and '--mtp' not in argv
                            for _, _, argv in self.commands), 'plain tiled decode command missing')
        for profile, _, argv in self.commands:
            with self.subTest(profile=profile, argv=' '.join(argv[3:8])):
                self.assertEqual(refused(argv), [])
        # The rule itself catches the combination the binary refused.
        self.assertIn('--gpu-greedy-token', refused(['bin', '--numerical-profile', 'x', '--gpu-greedy-token']))
        self.assertIn('known-prefix window without --wave8-q8-qsa-output',
                      refused(['bin', '--numerical-profile', 'x', '--known-prefix-window', '--known-prefix-full-rows']))
        self.assertIn('MTP calibration rows without the tiled prompt',
                      refused(['bin', '--numerical-profile', 'x', '--mtp', '--mtp-window-prefix', '--mtp-calibration-rows', '16']))
        self.assertIn('--http-serve without the tiled MTP route', refused(['bin', '--numerical-profile', 'x', '--http-serve']))

    @unittest.skipUnless(os.environ.get('BENCH_QWEN_BIN'), 'BENCH_QWEN_BIN not set')
    def test_commands_pass_binary_print_effective_config(self):
        """Host-only inspection of every built command. Run from a directory holding the 'models' folder:
        the commands use relative models/ paths. Run-time placeholders get stand-in values: 64-hex for the
        content identity and hashes (all ones: the binary rejects an all-zero hash), a temporary file for the
        golden and other output paths."""
        def ids(name):
            n = re.search(r'(?:ctx|prefill-)([0-9]+)$', name)
            return ['1'] * (int(n[1]) if n else 32)
        self.scratch = tempfile.mkdtemp()
        self.addCleanup(shutil.rmtree, self.scratch, True)
        for profile, assignments, argv in self.commands:
            words = [os.environ['BENCH_QWEN_BIN']]
            for w in argv[1:]:
                if w.startswith('TOKEN_IDS_'):
                    words += ids(w[len('TOKEN_IDS_'):])
                elif w.startswith('MEASURED_'):
                    words.append('1' * 64)
                elif w.startswith('DETERMINISTIC_PROMPTS'):
                    path = os.path.join(self.scratch, w.replace('/', '_'))
                    # Non-empty, so a served golden reaches the digest check instead of the empty-file refusal.
                    Path(path).write_bytes(b'placeholder')
                    words.append(path)
                else:
                    words.append(w)
            env = {**os.environ, **dict(x.split('=', 1) for x in assignments)}
            with self.subTest(profile=profile, argv=' '.join(argv[3:8])):
                result = subprocess.run([*words, '--print-effective-config'], capture_output=True, text=True, env=env, timeout=600)
                # A server command passes option validation, then checks the stand-in golden against its digest.
                if '--http-serve' in argv and result.returncode != 0:
                    self.assertIn('asset size or digest differs', result.stderr, result.stderr[-2000:])
                else:
                    self.assertEqual(result.returncode, 0, result.stderr[-2000:])


if __name__ == '__main__':
    unittest.main(verbosity=2)
