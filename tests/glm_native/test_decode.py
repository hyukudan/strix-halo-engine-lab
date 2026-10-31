"""CPU-only decode tests: cache head contract, decode window logic, receiver snapshot mode.

No device, CUDA import, socket or model data. Temporary files live under --build.
"""
import argparse
import contextlib
import hashlib
import importlib.util
import io
import json
import os
from pathlib import Path
import struct
import subprocess
import tempfile
import time
import types
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[2]
RECEIPT = None


def load(name, path):
    spec = importlib.util.spec_from_file_location(name, ROOT/path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


cache = load('cache', 'src/glm/rtxpf/producer/cache.py')
dw = load('decode_window', 'src/glm/rtxpf/producer/decode_qualification_window.py')
CHAT_ALL = len(dw.CHAT_KINDS)


class CacheHeadTests(unittest.TestCase):
    def test_root_names_k6_contract(self):
        plain, head = cache.required_schema(), cache.required_schema(head=True)
        self.assertEqual(set(head) - set(plain), {'lm_head.mul1', 'lm_head.trellis', 'lm_head.suh', 'lm_head.svh'})
        self.assertFalse(any(n.startswith('model.language_model.lm_head') for n in head))
        self.assertEqual(head['lm_head.trellis'], ('I16', [256, 9680, 96]))
        self.assertEqual(head['lm_head.mul1'], ('I32', []))
        sizes = cache.SIZES
        for name, (dtype, shape) in head.items():
            if name in cache.HEAD_BYTES:
                n = sizes[dtype]
                for d in shape:
                    n *= d
                self.assertEqual(n, cache.HEAD_BYTES[name], name)
        self.assertEqual(sum(cache.HEAD_BYTES.values()), 476109316)
        self.assertEqual(cache.blob_of('lm_head.trellis'), 'head.bin')
        self.assertEqual(cache.blob_of('model.language_model.norm.weight'), 'base.bin')
        self.assertEqual(cache.blob_of('model.language_model.layers.10.mlp.experts.5.up_proj.suh'), 'experts-10.bin')

    def test_cache_key_separation(self):
        plain = hashlib.sha256(cache.canonical(cache.key_contract('a'*64))).hexdigest()
        legacy = hashlib.sha256(cache.canonical(dict(cache.VERSION, model_sha256='a'*64))).hexdigest()
        head = hashlib.sha256(cache.canonical(cache.key_contract('a'*64, True))).hexdigest()
        self.assertEqual(plain, legacy, 'prefill cache key unchanged')
        self.assertNotEqual(plain, head)

    def manifest(self, td, head):
        sources = dict(config={}, files=[])
        model = hashlib.sha256(cache.canonical(sources)).hexdigest()
        m = dict(cache.key_contract(model, head), sources=sources,
                 cache_key=hashlib.sha256(cache.canonical(cache.key_contract(model, head))).hexdigest(), blobs={}, tensors={})
        (td/'manifest.json').write_bytes(cache.canonical(m))
        return m

    def test_missing_head_rejected_and_inventory_checked(self):
        with tempfile.TemporaryDirectory(dir=BUILD) as td:
            td = Path(td)
            self.manifest(td, False)
            with self.assertRaisesRegex(ValueError, 'head-capable'):
                cache.verify(td, require_head=True)
            with self.assertRaisesRegex(ValueError, 'inventory'):
                cache.verify(td)  # empty tensor inventory still rejected
            m = self.manifest(td, True)
            m['head'] = 'other'
            (td/'manifest.json').write_bytes(cache.canonical(m))
            with self.assertRaisesRegex(ValueError, 'head contract'):
                cache.verify(td)


class WindowTests(unittest.TestCase):
    def args(self, prompt_dir=Path('PROMPTS')):
        return types.SimpleNamespace(build=Path('.'), model=Path('MODEL'), cache=None, cache_manifest_sha256=None,
                                     prompt_dir=prompt_dir, steps=4, bench_steps=4, expert_tier='reference',
                                     timeout=1800, allow_gpu=False, check_inputs=False, unit_file=None,
                                     tier_vram_gib=None, tier_ram_gib=None, unit_manifest_sha256=None)

    def record(self, steps=4, n=1024, file_bytes=4096):
        e = lambda c, t: dict(t=t, frontier=n-1+t, rank0_sha256=c*64, rank1_sha256=c*64, rank0_bytes=2097152, rank1_bytes=2097152,
                              replay_sha256='f'*64, components=[[0, 0, 2097152]])
        gen = list(range(11, 12+steps))
        inc = [e('a', t) for t in range(steps+1)]
        stepped = [dict(token=y, position=n-1+i, logits_sha256=('%x' % (y % 16))*64, experts=dict(file_bytes=file_bytes))
                   for i, y in enumerate(gen)]
        fresh = []
        for t in range(1, steps+1):
            f = e('a', t)
            f['next'] = dict(token=gen[t], logits_sha256=stepped[t]['logits_sha256'], position=n-1+t)
            fresh.append(f)
        return dict(prompt_tokens=n, legacy_prefill=dict(rank0_sha256='a'*64, rank1_sha256='a'*64),
                    incremental_exports=inc, incremental_steps=stepped, generated=gen, fresh=fresh)

    def bench(self, order=('cold', 'warm'), steps=4, file_bytes=4096, n=1024):
        arm = lambda name: dict(arm=name, tokens=[5]*steps, logits_sha256=['c'*64]*steps, generation_loop_s=1.0,
                                tokens_per_s=steps/1.0,
                                steps=[dict(token=5, logits_sha256='c'*64, position=n-1+j, experts=dict(file_bytes=file_bytes))
                                       for j in range(steps)],
                                final_state=dict(rank0_sha256='d'*64, rank1_sha256='d'*64, replay_sha256='e'*64,
                                                 rank0_bytes=1, rank1_bytes=1, frontier=n + steps - 1, components=[[0, 0, 1]]))
        return dict(prompt_tokens=n, arms=[arm(x) for x in order])

    @staticmethod
    def prefill_state(n):
        """A complete prefill state signature at frontier n - 1 (session_digest fields)."""
        return dict(rank0_sha256='a'*64, rank1_sha256='b'*64, rank0_bytes=10, rank1_bytes=10, replay_sha256='d'*64,
                    frontier=n - 1, components=[[0, 0, 10]])

    def test_plan_starts_nothing(self):
        a = self.args()
        a.expert_tier, a.unit_file, a.unit_manifest_sha256, a.tier_ram_gib = 'tierstream', Path('U'), 'f'*64, 32.0
        with patch.object(dw.subprocess, 'Popen') as popen, contextlib.redirect_stdout(io.StringIO()) as out:
            self.assertEqual(dw.run(a), 0)
            popen.assert_not_called()
        plan = json.loads(out.getvalue())
        self.assertIn('--decode-commands', plan['command'])
        self.assertIn('--require-file-tier', plan['command'])
        self.assertEqual([i['op'] for i in plan['items']][:3], ['qualify', 'qualify', 'compare'])
        self.assertEqual(len(plan['items']), 2 + 1 + 2 + 2, 'gate, compare, TTFT 1K/8K, one chat kind at 1K/8K (cached path only)')
        self.assertLessEqual(plan['expected_wall_seconds'], dw.MAX_SECONDS)

    def test_uuid_deadline_and_source_checked_before_process(self):
        a = self.args()
        a.allow_gpu = True
        with patch.dict(dw.os.environ, {'CUDA_VISIBLE_DEVICES': '0'}), patch.object(dw, 'start') as start:
            with self.assertRaises(ValueError):
                dw.run(a)
            start.assert_not_called()
        for t in (0, 1801):
            a.timeout = t
            with self.assertRaises(ValueError):
                dw.run(a)
        a.timeout, a.cache = 100, Path('CACHE')
        with self.assertRaises(ValueError):
            dw.run(a)
        with self.assertRaises(SystemExit), contextlib.redirect_stderr(io.StringIO()):
            dw.arguments(['--build', '.', '--model', 'M', '--prompt-dir', 'P', '--steps', '2'])

    def test_judge_exact_mismatch_missing(self):
        self.assertEqual(dw.judge_qualify(self.record(), 4, 1024), ('EXACT', []))
        for mutate, why in ((lambda r: r['fresh'][0].update(rank1_sha256='b'*64), 'state SHA'),
                            (lambda r: r['fresh'][1]['next'].update(logits_sha256='0'*64), 'logits'),
                            (lambda r: r['fresh'][0]['next'].update(token=99), 'token'),
                            (lambda r: r['fresh'][1].update(replay_sha256='0'*64), 'replay'),
                            (lambda r: r['fresh'][0].update(components=[[1, 0, 2097152]]), 'component'),
                            (lambda r: r['legacy_prefill'].update(rank0_sha256='c'*64), 'prefill producer')):
            r = self.record()
            mutate(r)
            v, problems = dw.judge_qualify(r, 4, 1024)
            self.assertEqual(v, 'MISMATCH')
            self.assertTrue(any(why in p for p in problems), (why, problems))
        r = self.record()
        del r['fresh'][1]
        self.assertEqual(dw.judge_qualify(r, 4, 1024)[0], 'NO-REFERENCE')
        r = self.record()
        r['incremental_steps'][0]['logits_sha256'] = 'short'
        self.assertEqual(dw.judge_qualify(r, 4, 1024)[0], 'NO-REFERENCE')

    def test_coverage_required(self):
        self.assertEqual(dw.qualify_coverage(self.record(4, 1024), 1024, 4), [])
        self.assertEqual(dw.qualify_coverage(self.record(4, 8192), 8192, 4), [])
        self.assertTrue(any('B32' in m for m in dw.qualify_coverage(self.record(4, 1000), 1000, 4)))
        self.assertTrue(any('frontier' in m for m in dw.qualify_coverage(self.record(4, 16), 1024, 4)))
        self.assertTrue(any('sparse' in m for m in dw.qualify_coverage(self.record(4, 1024), 8192, 4)))
        self.assertTrue(any('file-tier' in m for m in dw.qualify_coverage(self.record(4, 1024, 0), 1024, 4)))
        self.assertTrue(dw.qualify_coverage(dict(prompt_tokens=1024), 1024, 4))

    def test_bench_requires_digests_labels_counts(self):
        order = ['cold', 'warm']
        self.assertEqual(dw.judge_bench(self.bench(), order, 4, 1024), ('EXACT', [], []))
        self.assertEqual(dw.judge_bench(self.bench(order=('warm', 'cold')), order, 4, 1024)[0], 'NO-REFERENCE')
        self.assertEqual(dw.judge_bench(self.bench(steps=3), order, 4, 1024)[0], 'NO-REFERENCE')
        for mutate, why in ((lambda b: self.rebind(b['arms'][1], logits=['c'*64, 'c'*64, '0'*64, 'c'*64]), 'logits'),
                            (lambda b: b['arms'][1]['final_state'].update(rank1_sha256='0'*64), 'final state'),
                            (lambda b: b['arms'][0]['final_state'].update(replay_sha256='0'*64), 'replay'),
                            (lambda b: self.rebind(b['arms'][1], tokens=[5, 5, 5, 6]), 'tokens')):
            b = self.bench()
            b['arms'] = [dict(x, logits_sha256=list(x['logits_sha256']), final_state=dict(x['final_state'])) for x in b['arms']]
            mutate(b)
            v, problems, _ = dw.judge_bench(b, order, 4, 1024)
            self.assertEqual(v, 'MISMATCH')
            self.assertTrue(any(why in p for p in problems), (why, problems))
        b = self.bench()
        del b['arms'][0]['final_state']
        self.assertEqual(dw.judge_bench(b, order, 4, 1024)[0], 'NO-REFERENCE')
        # every arm's complete final state signature, frontier from the REQUEST
        for field in ('frontier', 'rank0_bytes', 'rank1_bytes', 'replay_sha256', 'components'):
            for arm in (0, 1):
                b = self.bench()
                b['arms'] = [dict(x, final_state=dict(x['final_state'])) for x in b['arms']]
                del b['arms'][arm]['final_state'][field]
                self.assertEqual(dw.judge_bench(b, order, 4, 1024)[0], 'NO-REFERENCE', (field, arm))
        wrong = self.bench()
        wrong['arms'] = [dict(x, final_state=dict(x['final_state'], frontier=1026)) for x in wrong['arms']]  # 1024 + 4 - 1 = 1027
        self.assertEqual(dw.judge_bench(wrong, order, 4, 1024)[0], 'NO-REFERENCE')
        self.assertEqual(dw.judge_bench(dict(self.bench(), prompt_tokens=None), order, 4, 1024)[0], 'NO-REFERENCE')
        # an 8192 case must not accept a 1024-token response (frontier from the plan)
        self.assertEqual(dw.judge_bench(self.bench(n=1024), order, 4, 8192)[0], 'NO-REFERENCE')
        self.assertEqual(dw.judge_bench(self.bench(n=8192), order, 4, 8192)[0], 'EXACT')
        self.assertEqual(dw.judge_bench(self.bench(n=8192), order, 4, 1024)[0], 'NO-REFERENCE')
        bytes_differ = self.bench()
        bytes_differ['arms'] = [dict(x, final_state=dict(x['final_state'], rank1_bytes=1 + i)) for i, x in enumerate(bytes_differ['arms'])]
        self.assertEqual(dw.judge_bench(bytes_differ, order, 4, 1024)[0], 'MISMATCH')
        # comparators never raise: an arm without a validated signature is NO-REFERENCE with a reason
        bad = [dict(op='bench', tokens=1024, sweep_tile=t, verdict='EXACT', arms=[dict(arm='cold', tokens_per_s=1.0)]) for t in (2048, 8192)]
        out = dw.sweep_compare(bad)
        self.assertEqual(out[0]['verdict'], 'NO-REFERENCE')
        self.assertIn('reason', out[0])
        ttft_bad = [dict(op='ttft', tokens=8192, sweep_tile=t, verdict='EXACT', token=5, prefill_state=None, row_tile=t) for t in (2048, 8192)]
        self.assertEqual(dw.sweep_compare(ttft_bad)[0]['verdict'], 'NO-REFERENCE')
        self.assertEqual(dw.tile_compare([{k: v for k, v in r.items() if k != 'sweep_tile'} for r in ttft_bad])[0]['verdict'],
                         'NO-REFERENCE')
        self.assertEqual(dw.tile_compare(ttft_bad), [])  # no state at all: not a tile arm
        self.assertEqual(dw.judge_bench(self.bench(file_bytes=0), order, 4, 1024)[2], ['cold: no file-tier expert reads', 'warm: no file-tier expert reads'])

    def prompts(self, td, chat=CHAT_ALL):
        d = Path(td)/'prompts'
        d.mkdir()
        for n in dw.QUALIFY:
            (d/f'q{n}.u32').write_bytes(struct.pack(f'<{n}I', *([7]*n)))
            for k in dw.CHAT_KINDS[:chat]:
                (d/f'chat-{k}-{n}.u32').write_bytes(struct.pack(f'<{n}I', *([9]*n)))
        return d

    def test_inputs_exact_lengths_and_interleaved_order(self):
        with tempfile.TemporaryDirectory(dir=BUILD) as td:
            a = self.cached(self.args(self.prompts(td, chat=2)), td)
            pins = dw.check_inputs(a)
            self.assertEqual(sum(1 for p in pins.values() if not p.get('missing')), 5)  # q1024, q8192, 2 chat prompts + unit file
            self.assertEqual(sum(1 for p in pins.values() if p.get('missing')), 0)
            orders = [i[3] for i in dw.prompt_plan(a) if i[0] == 'bench']
            self.assertEqual(orders, [['cold', 'warm'], ['cold', 'warm']])
            for bad in (struct.pack('<2I', 1, 2), struct.pack('<1023I', *([1]*1023)), struct.pack('<1025I', *([1]*1025)),
                        struct.pack('<1024I', *([154880]*1024))):
                (a.prompt_dir/'q1024.u32').write_bytes(bad)
                with self.assertRaises(ValueError):
                    dw.check_inputs(a)
            (a.prompt_dir/'chat-explanation-8192.u32').write_bytes(struct.pack('<16I', *range(16)))
            (a.prompt_dir/'q1024.u32').write_bytes(struct.pack('<1024I', *([1]*1024)))
            with self.assertRaises(ValueError):
                dw.check_inputs(a)  # undersized chat prompt is rejected, not silently accepted
            (a.prompt_dir/'q1024.u32').unlink()
            with self.assertRaises(ValueError):
                dw.check_inputs(a)

    def cached(self, a, td):
        """The cached tierstream product path (the only backend timed arms run on)."""
        a.expert_tier, a.tier_ram_gib, a.unit_manifest_sha256 = 'tierstream', 32.0, 'f'*64
        a.unit_file = Path(td)/'units.bin'
        a.unit_file.write_bytes(b'u')
        Path(str(a.unit_file)+'.units.json').write_text('{}')
        return a

    def run_execute(self, td, replies, left, chat=CHAT_ALL):
        a = self.cached(self.args(self.prompts(td, chat)), td)
        out = Path(td)/'out'
        out.mkdir()
        ready = dict(ready=True, producer_executable_sha256='e'*64,
                     health=dict(decode_identity=dw.IDENTITY, row_tile=2048, residency=dict(decode=dict(expert_tier='tierstream', tier_plan=dict(
                         three_tier=True, reference_compare=True, unit_manifest_sha256='f'*64)))))
        sent = []

        class FakeService:
            def __init__(self, *_):
                self.proc = types.SimpleNamespace(poll=lambda: 0)
                self.replies = [ready] + list(replies)

            def read(self, deadline):
                return self.replies.pop(0)

            def send(self, command):
                sent.append(command)

        deadline = types.SimpleNamespace(left=lambda: left)
        with patch.object(dw, 'Service', FakeService), patch.object(dw, 'stop'), patch.object(dw, 'snapshot_check'), \
             contextlib.redirect_stdout(io.StringIO()) as printed:
            rc = dw.execute(a, out, deadline, dict(service_sha256='e'*64), dw.check_inputs(a))
        return rc, json.loads(printed.getvalue()), sent

    def full_replies(self, chat=CHAT_ALL):
        replies = [self.record(4, 1024), self.record(4, 8192), self.compare_record()]
        replies += [dict(backend='tierstream', prompt_tokens=n, ttft_s=1.5, prefill_s=1.4, token=5, row_tile=2048, allocated_row_tile=2048,
                         prefill_state=self.prefill_state(n))
                    for n in dw.QUALIFY]
        if chat:
            replies += [self.bench(order=('cold', 'warm'), n=n) for n in dw.QUALIFY]
        return replies

    def test_complete_workload_is_exact(self):
        with tempfile.TemporaryDirectory(dir=BUILD) as td:
            rc, summary, sent = self.run_execute(td, self.full_replies(), left=100000)
            self.assertEqual((rc, summary['verdict'], summary['completion']), (0, 'EXACT', 'COMPLETE'))
            self.assertEqual(len(sent), 2 + 1 + 2 + 2 + 1)
            compare = [r for r in summary['results'] if r['op'] == 'compare']
            self.assertEqual([(r['prompt'], r.get('timed')) for r in compare], [('q1024.u32', False)])
            self.assertTrue(all('wall_s' not in r for r in compare))

    def test_missing_chat_or_coverage_is_incomplete(self):
        with tempfile.TemporaryDirectory(dir=BUILD) as td:
            rc, summary, sent = self.run_execute(td, self.full_replies(chat=0), left=100000, chat=0)
            self.assertEqual((rc, summary['verdict'], summary['exactness_of_completed']), (4, 'INCOMPLETE', 'EXACT'))
            self.assertEqual(sum('planned prompt missing' in g for g in summary['gaps']), 2)
        replies = self.full_replies()
        replies[1] = self.record(4, 8192, file_bytes=0)
        with tempfile.TemporaryDirectory(dir=BUILD) as td:
            rc, summary, _ = self.run_execute(td, replies, left=100000)
            self.assertEqual((rc, summary['verdict'], summary['exactness_of_completed']), (4, 'INCOMPLETE', 'EXACT'))
            self.assertTrue(any('file-tier' in g for g in summary['gaps']))

    def test_tierstream_command_health_and_cache_summary(self):
        a = self.args()
        a.expert_tier, a.unit_file, a.tier_ram_gib, a.unit_manifest_sha256 = 'tierstream', Path('OUT_DIR/glm-units.bin'), 32.0, 'f'*64
        cmd = dw.service_command(a, Path('svc.exe'))
        self.assertEqual(cmd[cmd.index('--expert-tier')+1:cmd.index('--expert-tier')+7],
                         ['tierstream', '--unit-file', str(a.unit_file), '--unit-manifest-sha256', 'f'*64, '--reference-compare'])
        self.assertEqual(cmd[cmd.index('--tier-ram-gib')+1], '32.0')
        # Tierstream mode: prefill keeps no resident/pinned expert copies (all layers file tier).
        self.assertEqual((cmd[cmd.index('--max-device-layers')+1], cmd[cmd.index('--max-pinned-layers')+1]), ('0', '0'))
        self.assertIn('--require-file-tier', cmd)
        ref = self.args()
        rcmd = dw.service_command(ref, Path('svc.exe'))
        self.assertEqual(rcmd[rcmd.index('--max-device-layers')+1], '19')
        # Dry plan: same options, --plan-budget instead of --allow-gpu, device figures passed through.
        a.tier_vram_gib, a.device_total_gib, a.device_free_gib = 24.0, 95.5, 95.4
        seen = {}
        def fake_run(cmd, **kw):
            seen['cmd'] = cmd
            return types.SimpleNamespace(stdout='{"verdict":"ADMITTED"}\n', stderr='', returncode=0)
        real = dw.subprocess.run
        dw.subprocess.run = fake_run
        try:
            self.assertEqual(dw.plan_budget(a), 0)
        finally:
            dw.subprocess.run = real
        self.assertEqual(seen['cmd'][-5:], ['--plan-budget', '--device-total-gib', '95.5', '--device-free-gib', '95.4'])
        self.assertNotIn('--allow-gpu', seen['cmd'])
        self.assertEqual(seen['cmd'][seen['cmd'].index('--tier-vram-gib')+1], '24.0')
        with self.assertRaises(ValueError):
            dw.plan_budget(ref)  # the reference backend has no tierstream budget
        granted = dict(residency=dict(device_layers=0, pinned_layers=0, decode=dict(expert_tier='tierstream', tier_plan=dict(
            three_tier=True, reference_compare=True, unit_manifest_sha256='f'*64, device_budget_bytes=24 << 30, host_budget_bytes=32 << 30))))
        self.assertEqual(dw.health_coverage(granted, a), [])
        granted['residency']['device_layers'] = 19
        granted['residency']['decode']['tier_plan']['device_budget_bytes'] = 4 << 30
        gaps = dw.health_coverage(granted, a)
        self.assertTrue(any('resident/pinned expert copies' in g for g in gaps) and any('device_budget_bytes' in g for g in gaps))
        a.tier_vram_gib = None
        # Whole-device caps: passed through; the arena-only RAM default is not applied; receipts gated.
        whole = dw.arguments(['--build', '.', '--model', 'M', '--prompt-dir', 'P', '--vram-total-gib', '24', '--ram-total-gib', '32'])
        self.assertEqual((whole.vram_total_gib, whole.ram_total_gib, whole.tier_ram_gib), (24.0, 32.0, None))
        self.assertEqual(whole.capacity, dw.WHOLE_CARD_CAPACITY)  # 24 GiB card: plan at the 8K capacity it needs
        whole.prompt_dir = Path('P')
        self.assertEqual([i[0] for i in dw.prompt_plan(whole)], ['qualify', 'qualify', 'compare', 'ttft', 'ttft', 'bench', 'bench'])
        self.assertLessEqual(dw.expected_seconds(whole), dw.MAX_SECONDS)  # startup prior included
        for bad in (['--capacity', '8000'], ['--capacity', '8321']):
            with self.assertRaises(SystemExit):
                dw.arguments(['--build', '.', '--model', 'M', '--prompt-dir', 'P'] + bad)
        whole.unit_file, whole.unit_manifest_sha256 = Path('OUT_DIR/glm-units.bin'), 'f'*64
        wcmd = dw.service_command(whole, Path('svc.exe'))
        self.assertEqual((wcmd[wcmd.index('--vram-total-gib')+1], wcmd[wcmd.index('--ram-total-gib')+1]), ('24.0', '32.0'))
        self.assertEqual(wcmd[wcmd.index('--capacity')+1], str(dw.WHOLE_CARD_CAPACITY))
        self.assertNotIn('--tier-ram-gib', wcmd)
        with self.assertRaises(SystemExit):
            dw.arguments(['--build', '.', '--model', 'M', '--prompt-dir', 'P', '--expert-tier', 'reference', '--vram-total-gib', '24'])
        peaks = {}
        ok = dict(memory=dict(measured_device_peak_bytes=23 << 30, host_peak_commit_bytes=30 << 30, within_caps=True))
        self.assertEqual(dw.memory_gate(ok, whole, peaks), [])
        over = dict(memory=dict(measured_device_peak_bytes=(24 << 30) + 1, host_peak_commit_bytes=33 << 30, within_caps=False))
        gaps = dw.memory_gate(over, whole, peaks)
        self.assertTrue(any('--vram-total-gib' in g for g in gaps) and any('--ram-total-gib' in g for g in gaps))
        self.assertEqual(peaks['measured_device_peak_bytes'], (24 << 30) + 1)
        self.assertEqual(dw.memory_gate({}, whole, {}), ['receipt has no memory report'])
        self.assertEqual(dw.memory_gate({}, a, {}), [])  # no caps: no memory gate
        # In whole-device mode the arena is what the caps leave: no exact-arena gap.
        whole.tier_vram_gib = 4.0
        wh = dict(residency=dict(device_layers=0, pinned_layers=0, decode=dict(expert_tier='tierstream', tier_plan=dict(
            three_tier=True, reference_compare=True, unit_manifest_sha256='f'*64, device_budget_bytes=3 << 30, host_budget_bytes=20 << 30))))
        self.assertEqual(dw.health_coverage(wh, whole), [])
        health = lambda three, tier='tierstream', compare=True, pin='f'*64: dict(residency=dict(decode=dict(
            expert_tier=tier, tier_plan=dict(three_tier=three, reference_compare=compare, unit_manifest_sha256=pin))))
        self.assertEqual(dw.health_coverage(health(True), a), [])
        self.assertTrue(dw.health_coverage(health(False), a))
        self.assertTrue(any('reference backend' in g for g in dw.health_coverage(health(True, compare=False), a)))
        for bad in (health(True, 'reference'), health(True, pin='0'*64)):
            with self.assertRaises(ValueError):
                dw.health_coverage(bad, a)
        arm = dict(expert_totals=dict(lookups=1376, misses=120, device_hits=1256, resurrected=8, covered=12, ram_copies=60,
                                      file_reads=40, file_bytes=40*4096, h2d_bytes=100*4096))
        c = dw.cache_summary(arm, 4)
        self.assertEqual((c['misses_per_token'], c['device_hits'], c['resurrected'], c['covered'], c['ram_copies'], c['file_reads'],
                          c['file_bytes_per_token']), (30.0, 1256, 8, 12, 60, 40, 10*4096))
        self.assertNotIn('pinned_hits', c)
        plan = [i[:3] for i in dw.prompt_plan(a)]
        self.assertEqual([i[0] for i in plan], ['qualify']*2 + ['compare', 'ttft', 'ttft', 'bench', 'bench'])
        self.assertEqual({(i[1], i[2].name) for i in plan if i[0] == 'compare'}, {(1024, 'q1024.u32')})
        self.assertIn('compare q1024.u32', dw.plan(a)['run_order'])
        self.assertTrue(any('2 planned chat case(s)' in c for c in dw.plan(a)['coverage']))  # coverage text from the actual schedule
        parsed = dw.arguments(['--build', '.', '--model', 'M', '--prompt-dir', 'P'])
        self.assertEqual((parsed.expert_tier, parsed.tier_ram_gib), ('tierstream', 32.0))
        with tempfile.TemporaryDirectory(dir=BUILD) as td:
            a.prompt_dir = self.prompts(td, chat=0)
            with self.assertRaisesRegex(ValueError, 'unit-file'):
                dw.check_inputs(a)  # unit file absent
            a.unit_file = Path(td)/'u.bin'
            a.unit_file.write_bytes(b'x')
            with self.assertRaisesRegex(ValueError, 'unit-file'):
                dw.check_inputs(a)  # manifest absent
            Path(str(a.unit_file)+'.units.json').write_text('{}')
            a.unit_manifest_sha256 = None
            with self.assertRaises(ValueError):
                dw.check_inputs(a)  # unpinned manifest
            a.unit_manifest_sha256 = 'f'*64
            self.assertEqual(dw.check_inputs(a)['unit_file']['bytes'], 1)

    def compare_record(self, steps=4, n=1024):
        arm = lambda name: dict(arm=name, backend=name, tokens=[3]*steps, logits_sha256=['a'*64]*steps,
                                steps=[dict(token=3, logits_sha256='a'*64, position=n-1+j) for j in range(steps)],
                                final_state=dict(rank0_sha256='b'*64, rank1_sha256='b'*64, rank0_bytes=1, rank1_bytes=1,
                                                 replay_sha256='c'*64, frontier=n + steps - 1, components=[[0, 0, 1]]))
        return dict(prompt_tokens=n, arms=[arm('reference'), arm('tierstream')])

    @staticmethod
    def rebind(arm, tokens=None, logits=None):
        """Changes an arm's tokens/logits together with its step records (they are bound)."""
        if tokens is not None:
            arm['tokens'] = tokens
        if logits is not None:
            arm['logits_sha256'] = logits
        arm['steps'] = [dict(s, token=arm['tokens'][i], logits_sha256=arm['logits_sha256'][i]) for i, s in enumerate(arm['steps'])]

    def test_cache_summary_keeps_every_counter(self):
        totals = dict(lookups=1376, misses=120, device_hits=1256, resurrected=8, covered=12, fetched=70, prefetched=9,
                      bank_serves=3, ram_copies=39, file_reads=40, file_bytes=40*4096, file_read_ns=123456789,
                      h2d_bytes=79*4096, evicted=33, disk_drops=5, writebacks=21, stalled=2, big=2**63+1)
        self.assertEqual(len(set(totals.values())), len(totals), 'distinct nonzero inputs')
        c = dw.cache_summary(dict(expert_totals=totals), 4)
        for key, value in totals.items():
            self.assertEqual(c[key], value, key)
            self.assertEqual(c['per_token'][key], value / 4, key)
        self.assertEqual((c['misses_per_token'], c['file_bytes_per_token'], c['h2d_bytes_per_token']), (30.0, 40960.0, 79*1024.0))
        ref = dw.cache_summary(dict(expert_totals=dict(device_selections=10, pinned_selections=6, file_selections=2)), 2)
        self.assertEqual((ref['device_hits'], ref['ram_copies'], ref['file_reads'], ref['device_selections']), (10, 6, 2, 10))

    def test_service_receipt_through_cache_summary(self):
        if RECEIPT is None:
            self.skipTest('no --receipt (CTest passes the fixture artifact of producer_decode_host)')
        self.assertTrue(RECEIPT.is_file(), 'fixture setup did not publish the receipt artifact')
        arm = json.loads(RECEIPT.read_text())
        self.assertEqual(arm['published_by'], 'glm_native_decode_host')
        self.assertLess(abs(time.time() - arm['written_unix_s']), 3600, 'stale receipt artifact')
        self.assertTrue(arm['run_nonce'])
        c = dw.cache_summary(arm, len(arm['steps']))
        self.assertEqual(set(arm['expert_totals']) - set(c), set(), 'every aggregated counter survives')
        self.assertEqual((c['lookups'], c['misses'], c['misses_per_token'], c['device_hits'], c['resurrected'], c['covered'],
                          c['ram_copies'], c['file_reads'], c['fetched'], c['prefetched'], c['bank_serves']),
                         (688, 80, 40.0, 608, 7, 6, 50, 15, 55, 10, 2))
        self.assertEqual((c['file_bytes_per_token'], c['h2d_bytes_per_token']), (15*12648448/2, 65*12648448/2))

    def test_reference_comparison_judge(self):
        self.assertEqual(dw.judge_compare(self.compare_record(), 4, 1024), ('EXACT', []))
        for mutate, why in ((lambda r: self.rebind(r['arms'][1], tokens=[3, 3, 3, 4]), 'tokens'),
                            (lambda r: self.rebind(r['arms'][1], logits=['a'*64]*3+['0'*64]), 'logits'),
                            (lambda r: r['arms'][1]['final_state'].update(rank1_sha256='0'*64), 'final state'),
                            (lambda r: r['arms'][1]['final_state'].update(replay_sha256='0'*64), 'replay')):
            r = self.compare_record()
            mutate(r)
            v, problems = dw.judge_compare(r, 4, 1024)
            self.assertEqual(v, 'MISMATCH')
            self.assertTrue(any(why in q for q in problems), (why, problems))
        r = self.compare_record()
        r['arms'][1]['backend'] = 'reference'
        self.assertEqual(dw.judge_compare(r, 4, 1024)[0], 'NO-REFERENCE')
        self.assertEqual(dw.judge_compare(self.compare_record(3), 4, 1024)[0], 'NO-REFERENCE')
        self.assertEqual(dw.judge_compare(dict(arms=[]), 4, 1024)[0], 'NO-REFERENCE')

    def test_no_timed_reference_arm(self):
        # The reference backend runs only inside the exactness compare
        # (untimed, q1024 only); every timed arm is the cached tierstream path.
        a = self.args()
        with self.assertRaises(ValueError):
            dw.prompt_plan(a)  # no standalone reference-backend run at all
        a.expert_tier = 'tierstream'
        items = dw.prompt_plan(a)
        self.assertTrue(any(i[0] == 'bench' for i in items))
        self.assertEqual([(i[1], i[2].name) for i in items if i[0] == 'compare'], [(1024, 'q1024.u32')])
        self.assertTrue(all(i[0] in ('qualify', 'compare', 'ttft', 'bench') for i in items))
        self.assertEqual(dw.COMPARE, (('q', 1024),))
        with self.assertRaises(SystemExit):
            dw.arguments(['--build', '.', '--model', 'M', '--prompt-dir', 'P', '--expert-tier', 'reference'])

    def test_full_card_mode(self):
        with patch.object(dw, 'host_total_gib', return_value=191.6):
            a = dw.arguments(['--build', '.', '--model', 'M', '--prompt-dir', 'P', '--full-card'])
        self.assertEqual((a.vram_total_gib, a.ram_total_gib, a.tier_ram_gib, a.full_card, a.capacity), (93.0, 175.0, None, True, dw.CAPACITY))
        with self.assertRaises(SystemExit), patch.object(dw, 'host_total_gib', return_value=191.6):
            dw.arguments(['--build', '.', '--model', 'M', '--prompt-dir', 'P', '--full-card', '--capacity', '8320'])
        with patch.object(dw, 'host_total_gib', return_value=191.6):
            given = dw.arguments(['--build', '.', '--model', 'M', '--prompt-dir', 'P', '--full-card', '--ram-total-gib', '120'])
        self.assertEqual(given.ram_total_gib, 120.0)
        with self.assertRaises(SystemExit):
            dw.arguments(['--build', '.', '--model', 'M', '--prompt-dir', 'P', '--full-card', '--expert-tier', 'reference'])
        items = dw.prompt_plan(a)
        self.assertEqual([i[0] for i in items], ['qualify', 'qualify', 'compare'] + ['ttft'] * 5 + ['bench'] * 2)
        self.assertLessEqual(dw.expected_seconds(a), dw.MAX_SECONDS)  # the full-card schedule fits the deadline (startup excluded)
        self.assertLessEqual(dw.FULL_CARD_VRAM_GIB * (1 << 30), 99.86e9)  # within the runtime's usable device
        # TTFT tile arms: 8K at 8192 vs 2048, 32K at the allocated (largest admitted) tile vs 2048.
        self.assertEqual([(i[1], i[2].name, i[3]) for i in items if i[0] == 'ttft'],
                         [(1024, 'q1024.u32', None), (8192, 'q8192.u32', 8192), (8192, 'q8192.u32', 2048),
                          (32000, 'ttft-32000.u32', None), (32000, 'ttft-32000.u32', 2048)])
        self.assertEqual(dw.expected_seconds(a), 1780)
        self.assertEqual([(i[1], i[2].name, i[3]) for i in items if i[0] == 'bench'],
                         [(1024, 'chat-explanation-1024.u32', ['cold', 'warm']), (8192, 'chat-explanation-8192.u32', ['cold', 'warm'])])
        a.unit_file, a.unit_manifest_sha256 = Path('U'), 'f'*64
        cmd = dw.service_command(a, Path('svc.exe'))
        self.assertEqual((cmd[cmd.index('--vram-total-gib')+1], cmd[cmd.index('--ram-total-gib')+1]), ('93.0', '175.0'))
        ok = dict(backend='tierstream', prompt_tokens=32000, ttft_s=12.5, prefill_s=12.0, token=7, row_tile=4096, allocated_row_tile=4096,
                  prefill_state=self.prefill_state(32000))
        self.assertEqual(dw.judge_ttft(ok, 32000, 4096, 4096), ('EXACT', []))
        for bad in (dict(ok, backend='reference'), dict(ok, prompt_tokens=1024), dict(ok, ttft_s=0), dict(ok, token=None)):
            self.assertEqual(dw.judge_ttft(bad, 32000, 4096, 4096)[0], 'NO-REFERENCE')
        # The worker re-parses its arguments: --full-card must be forwarded.
        a.timeout, a.allow_gpu, a.check_inputs, a.build, a.prompt_dir, a.cache = 1800, False, True, Path('.'), Path('P'), None
        seen = {}

        class Job:
            def __call__(self):
                return self

            def __enter__(self):
                return self

            def __exit__(self, *_):
                return False

            def start(self, argv):
                seen['argv'] = argv
                return None, None

            def wait(self, *_):
                return 0
        with patch.object(dw.window, 'windows_job', return_value=Job()), patch.object(Path, 'mkdir'), patch.object(dw.os, 'name', 'nt'):
            dw.run(a)
        self.assertIn('--full-card', seen['argv'])

    def test_schedule_over_timeout_refused(self):
        a = self.args()
        a.expert_tier, a.unit_file, a.unit_manifest_sha256, a.tier_ram_gib = 'tierstream', Path('U'), 'f'*64, 32.0
        a.check_inputs, a.timeout = True, 600
        with self.assertRaisesRegex(ValueError, 'timeout'):
            dw.run(a)  # refused before any process or output directory

    def test_deadline_estimate_marks_incomplete(self):
        with tempfile.TemporaryDirectory(dir=BUILD) as td:
            rc, summary, sent = self.run_execute(td, [], left=200)
            self.assertEqual(rc, 4)
            self.assertEqual(summary['verdict'], 'INCOMPLETE')
            self.assertEqual(sent, [dict(op='quit')])

    def test_compare_evidence_and_signatures(self):
        # 1. compare: planned n, reported length, complete final signatures at n + steps - 1
        c = self.compare_record()
        self.assertEqual(dw.judge_compare(c, 4, 1024), ('EXACT', []))
        for mutate in (lambda r: r.pop('prompt_tokens'), lambda r: r.update(prompt_tokens=16),
                       lambda r: r['arms'][0]['final_state'].pop('frontier'), lambda r: r['arms'][0]['final_state'].update(frontier=1026),
                       lambda r: r['arms'][1]['final_state'].pop('rank0_bytes'), lambda r: r['arms'][1]['final_state'].update(components=[])):
            r = self.compare_record()
            mutate(r)
            self.assertEqual(dw.judge_compare(r, 4, 1024)[0], 'NO-REFERENCE')
        r = self.compare_record()
        r['arms'][1]['final_state']['rank1_bytes'] = 2
        self.assertEqual(dw.judge_compare(r, 4, 1024)[0], 'MISMATCH')
        r = self.compare_record()
        r['arms'][1]['final_state']['rank0_bytes'] = 2  # != the sum of its component extents
        self.assertEqual(dw.judge_compare(r, 4, 1024)[0], 'NO-REFERENCE')
        # 2. qualify: prefill producer digests and the initial / every state signature validated
        for mutate in (lambda r: [r['legacy_prefill'].pop(k) for k in ('rank0_sha256', 'rank1_sha256')],
                       lambda r: [e.pop('rank0_sha256') for e in r['incremental_exports'][:1]],
                       lambda r: r['incremental_exports'][0].pop('rank0_bytes'),
                       lambda r: r['fresh'][1].pop('rank1_bytes'),
                       lambda r: [e.update(components=[]) for e in r['incremental_exports'] + r['fresh']],
                       lambda r: r['incremental_exports'][0].update(frontier=1024)):
            r = self.record()
            mutate(r)
            self.assertEqual(dw.judge_qualify(r, 4, 1024)[0], 'NO-REFERENCE')
        # 3. token IDs: ints in [0, vocab), each generated token bound to its step incl. index 0
        r = self.record()
        r['generated'] = [None] * 5
        r['incremental_steps'] = [dict(s, token=None) for s in r['incremental_steps']]
        self.assertEqual(dw.judge_qualify(r, 4, 1024)[0], 'NO-REFERENCE')
        r = self.record()
        r['incremental_steps'][0]['token'] = r['generated'][0] + 1
        self.assertEqual(dw.judge_qualify(r, 4, 1024)[0], 'NO-REFERENCE')
        for kind, rec, judge in (('bench', self.bench(), lambda x: dw.judge_bench(x, ['cold', 'warm'], 4, 1024)[0]),
                                 ('compare', self.compare_record(), lambda x: dw.judge_compare(x, 4, 1024)[0])):
            for bad in ([None] * 4, [154880] * 4, [-1] * 4, ['5'] * 4):
                x = json.loads(json.dumps(rec))
                for arm in x['arms']:
                    self.rebind(arm, tokens=list(bad))
                self.assertEqual(judge(x), 'NO-REFERENCE', (kind, bad))
        # 4. a materialization/execution exception still prints the summary (non-zero exit)
        with tempfile.TemporaryDirectory(dir=BUILD) as td:
            bench = self.bench()
            del bench['arms'][0]['tokens_per_s']
            replies = self.full_replies()[:5] + [bench] + self.full_replies()[6:]
            rc, summary, _ = self.run_execute(td, replies, left=100000)
            self.assertNotEqual(rc, 0)
            self.assertTrue(any(r['op'] == 'bench' and r['verdict'] == 'NO-REFERENCE' for r in summary['results']))
        with tempfile.TemporaryDirectory(dir=BUILD) as td:
            rc, summary, _ = self.run_execute(td, self.full_replies()[:3], left=100000)  # the service dies after the gate
            self.assertEqual(rc, 4)
            self.assertEqual(summary['verdict'], 'INCOMPLETE')
            self.assertIn('failure', summary)
            self.assertTrue(any(r['verdict'] == 'INCOMPLETE' and r.get('reason', '').startswith('not run') for r in summary['results']))
        # 5. tile: see TileCompareTests (reported tile == effective requested tile, allocation > 0)

    def test_uncompared_items_and_summary_failures(self):
        """Comparison items that could not be compared and a failing summarize() are structured
        run failures with a non-zero exit (finalize, called directly)."""
        def run_finalize(td, results, sweep=False, **patches):
            a = self.cached(self.args(self.prompts(td, CHAT_ALL)), td)
            if sweep:
                a.full_card, a.capacity, a.tile_sweep, a.sweep_note = True, 32768, [2048, 8192], ''
            out = Path(td)/'out'
            out.mkdir(exist_ok=True)
            with contextlib.redirect_stdout(io.StringIO()) as printed:
                rc = dw.finalize(a, out, results, [], {}, 1.0, [], [], set(), None, sweep)
            return rc, json.loads(printed.getvalue())
        # 1. EXACT TTFT arms without their stored evidence: tile_compare marks the item
        #    internal_error -> run failure, non-zero exit
        broken = [dict(op='ttft', tokens=8192, prompt='q8192.u32', verdict='EXACT', evidence=None) for _ in range(2)]
        with tempfile.TemporaryDirectory(dir=BUILD) as td:
            rc, summary = run_finalize(td, broken)
        self.assertNotEqual(rc, 0)
        self.assertTrue(summary['tile_compare'][0]['internal_error'])
        self.assertTrue(any(e.startswith('tile_compare') for e in summary['failure']['summary_builders']))
        # the same for the sweep comparison
        sweep_broken = [dict(op='bench', tokens=1024, prompt='chat-explanation-1024.u32', verdict='EXACT', sweep_tile=t, evidence=None)
                        for t in (2048, 8192)]
        with tempfile.TemporaryDirectory(dir=BUILD) as td:
            rc, summary = run_finalize(td, sweep_broken, sweep=True)
        self.assertNotEqual(rc, 0)
        self.assertTrue(any(e.startswith('tile_sweep') for e in summary['failure']['summary_builders']))
        # 2. summarize() raising: the same structured failure, verdict NO-REFERENCE, exit 3
        with tempfile.TemporaryDirectory(dir=BUILD) as td, patch.object(dw, 'summarize', side_effect=RuntimeError('summary broke')):
            rc, summary = run_finalize(td, [])
        self.assertEqual((rc, summary['verdict']), (3, 'NO-REFERENCE'))
        self.assertTrue(any('summary broke' in e for e in summary['failure']['summary_builders']))
        # a clean finalize with nothing to compare stays failure-free
        with tempfile.TemporaryDirectory(dir=BUILD) as td:
            rc, summary = run_finalize(td, [])
        self.assertNotIn('failure', summary)

    def test_positions_and_final_state_evidence(self):
        # 1. positions are evidence: every arm step from index 0, every fresh next token
        for kind, rec, judge in (('bench', self.bench(), lambda x: dw.judge_bench(x, ['cold', 'warm'], 4, 1024)[0]),
                                 ('compare', self.compare_record(), lambda x: dw.judge_compare(x, 4, 1024)[0])):
            for j in (0, 3):
                for bad in (None, 1023 + j + 1, str(1023 + j)):
                    x = json.loads(json.dumps(rec))
                    x['arms'][1]['steps'][j]['position'] = bad
                    self.assertEqual(judge(x), 'NO-REFERENCE', (kind, j, bad))
            self.assertEqual(dw.evidence_arms(rec, 1024, 4, [a['arm'] for a in rec['arms']], kind)['arms'][0]['positions'],
                             [1023, 1024, 1025, 1026])
        for k in (0, 3):
            r = self.record()
            r['fresh'][k]['next']['position'] += 1
            self.assertEqual(dw.judge_qualify(r, 4, 1024)[0], 'NO-REFERENCE')
            r = self.record()
            del r['fresh'][k]['next']['position']
            self.assertEqual(dw.judge_qualify(r, 4, 1024)[0], 'NO-REFERENCE')
        self.assertEqual([f['next_position'] for f in dw.evidence_qualify(self.record(), 1024, 4)['fresh']], [1024, 1025, 1026, 1027])
        # 2. provenance / plan construction failures are inside the outer try: summary printed, non-zero
        with tempfile.TemporaryDirectory(dir=BUILD) as td, patch.object(dw, 'plan', side_effect=RuntimeError('plan broke')):
            rc, summary, sent = self.run_execute(td, self.full_replies(), left=100000)
        self.assertNotEqual(rc, 0)
        self.assertEqual(summary['failure']['error'], 'plan broke')
        self.assertEqual(sent, [])
        # 3. a failing summary builder is a run failure (never EXACT, never exit 0)
        with tempfile.TemporaryDirectory(dir=BUILD) as td, patch.object(dw, 'tile_compare', side_effect=RuntimeError('builder broke')):
            rc, summary, _ = self.run_execute(td, self.full_replies(), left=100000)
        self.assertEqual((rc, summary['verdict']), (4, 'INCOMPLETE'))
        self.assertTrue(any('builder broke' in e for e in summary['failure']['summary_builders']))
        with tempfile.TemporaryDirectory(dir=BUILD) as td, patch.object(dw, 'expected_seconds', side_effect=RuntimeError('meta broke')):
            rc, summary, _ = self.run_execute(td, self.full_replies(), left=100000)
        self.assertEqual(rc, 4)
        # 4. the service is stopped even when the results write fails
        real_write = Path.write_text

        def failing_write(path, *args, **kw):
            if path.name == 'results.json':
                raise OSError('disk full')
            return real_write(path, *args, **kw)
        with tempfile.TemporaryDirectory(dir=BUILD) as td, patch.object(Path, 'write_text', failing_write), \
                patch.object(dw, 'stop') as stopped:
            a = self.cached(self.args(self.prompts(td, CHAT_ALL)), td)
            out = Path(td)/'out'
            out.mkdir()
            replies = self.full_replies()
            ready = dict(ready=True, producer_executable_sha256='e'*64, health=dict(decode_identity=dw.IDENTITY, row_tile=2048,
                         residency=dict(decode=dict(expert_tier='tierstream', tier_plan=dict(three_tier=True, reference_compare=True,
                                                                                             unit_manifest_sha256='f'*64)))))

            class FakeService:
                def __init__(self, *_):
                    self.proc = types.SimpleNamespace(poll=lambda: 0)
                    self.replies = [ready] + list(replies)

                def read(self, deadline):
                    return self.replies.pop(0)

                def send(self, command):
                    pass
            with patch.object(dw, 'Service', FakeService), patch.object(dw, 'snapshot_check'), \
                    contextlib.redirect_stdout(io.StringIO()) as printed:
                rc = dw.execute(a, out, types.SimpleNamespace(left=lambda: 100000), dict(service_sha256='e'*64), dw.check_inputs(a))
            self.assertEqual(stopped.call_count, 1)
            self.assertNotEqual(rc, 0)
            self.assertIn('failure', json.loads(printed.getvalue()))

    def test_every_field_mutation_is_rejected(self):
        """Table-driven: every field of every record kind, deleted / null / wrong type /
        off by one. No mutation may leave an EXACT verdict (or, for TTFT, BITWISE-EQUAL
        against the unmutated record), except value changes of timings and cache
        counters, which are not exactness evidence."""
        timing = {'generation_loop_s', 'tokens_per_s', 'ttft_s', 'prefill_s'}

        def paths(x, path=()):
            yield path
            if isinstance(x, dict):
                for k, v in x.items():
                    yield from paths(v, path + (k,))
            elif isinstance(x, list):
                for i, v in enumerate(x):
                    yield from paths(v, path + (i,))

        def mutations(v):
            out = [('null', None)]
            if isinstance(v, bool):
                out += [('type', 'x')]
            elif isinstance(v, int):
                out += [('type', str(v)), ('off-by-one', v + 1), ('off-by-one', v - 1)]
            elif isinstance(v, float):
                out += [('type', str(v)), ('off-by-one', v + 1)]
            elif isinstance(v, str):
                out += [('type', 7), ('off-by-one', ('b' if v[:1] != 'b' else 'c') + v[1:])]
            elif isinstance(v, list):
                out += [('type', {}), ('off-by-one', v[:-1])]
            elif isinstance(v, dict):
                out += [('type', [])]
            return out

        def mutated(rec, path, how, value):
            x = json.loads(json.dumps(rec))
            parent = x
            for k in path[:-1]:
                parent = parent[k]
            if how == 'delete':
                del parent[path[-1]]
            else:
                parent[path[-1]] = value
            return x

        tile = TileCompareTests()
        kinds = [
            ('qualify', self.record(), lambda x: dw.judge_qualify(x, 4, 1024)[0], None),
            ('compare', self.compare_record(), lambda x: dw.judge_compare(x, 4, 1024)[0], None),
            ('bench', self.bench(), lambda x: dw.judge_bench(x, ['cold', 'warm'], 4, 1024)[0], None),
            ('ttft', tile.rec(tile=4096), lambda x: dw.judge_ttft(x, 8192, 4096, 8192)[0],
             lambda x: dw.tile_compare([tile.result(tile=2048), dict(op='ttft', tokens=8192, verdict='EXACT',
                                                                      evidence=dw.evidence_ttft(x, 8192, 4096, 8192))])[0]['verdict']),
        ]
        checked = 0
        for kind, rec, judge, against in kinds:
            self.assertEqual(judge(rec), 'EXACT', kind)
            for path in paths(rec):
                if not path:
                    continue
                parent = rec
                for k in path[:-1]:
                    parent = parent[k]
                value = parent[path[-1]]
                tolerated = 'experts' in path or (path[-1] in timing)
                # The qualify initial state (t=0) has no second witness for rank 1's bytes, the
                # replay checkpoint or the component layer/index numbers (the prefill producer
                # reports rank hashes only): well-formedness is all that can be checked there.
                unreferenced = kind == 'qualify' and path[:2] == ('incremental_exports', 0) and (
                    path[2:] in (('rank1_bytes',), ('replay_sha256',)) or (len(path) == 5 and path[2] == 'components' and path[4] in (0, 1))
                ) or kind == 'qualify' and path == ('incremental_steps', 0, 'logits_sha256')  # step 0's logits: no fresh witness (fresh starts at t=1)
                for how, new in [('delete', None)] + mutations(value):
                    x = mutated(rec, path, how, new)
                    verdict = judge(x)
                    checked += 1
                    if verdict != 'EXACT':
                        continue
                    if against is not None and against(x) == 'DIFFERENT':
                        continue  # a valid record of a different state: detected by the comparison
                    if 'experts' in path or (tolerated and how == 'off-by-one'):
                        continue  # cache counters / a changed timing value: not exactness evidence
                    if unreferenced and how == 'off-by-one':
                        continue  # a different but well-formed value of a field nothing else witnesses
                    self.fail(f'{kind}: {how} at {path!r} still EXACT')
        self.assertGreater(checked, 1000)

    def test_summary_always_prints(self):
        """Mutated replies of every record kind through execute: a JSON summary is printed
        every time and a damaged run never exits 0."""
        base = self.full_replies()
        targets = [(0, 'qualify'), (2, 'compare'), (3, 'ttft'), (5, 'bench')]
        for index, kind in targets:
            for key in list(base[index]):
                for value in ('delete', None, [], 'x'):
                    replies = json.loads(json.dumps(base))
                    if value == 'delete':
                        del replies[index][key]
                    else:
                        replies[index][key] = value
                    with tempfile.TemporaryDirectory(dir=BUILD) as td:
                        rc, summary, _ = self.run_execute(td, replies, left=100000)
                    self.assertIn('verdict', summary, (kind, key, value))
                    if key not in ('ttft_s', 'prefill_s') or value != 'delete':
                        self.assertNotEqual((rc, summary['verdict']), (0, 'EXACT'), (kind, key, value))

    def test_tile_sweep_one_start_per_tile(self):
        """--tile-sweep: one service start per tile (--row-tile T), the gate on the first
        tile only, TTFT + bench on every tile, cross-tile digests in the summary."""
        with tempfile.TemporaryDirectory(dir=BUILD) as td:
            a = self.cached(self.args(self.prompts(td, CHAT_ALL)), td)
            a.full_card, a.capacity, a.tile_sweep, a.sweep_note = True, 32768, [2048, 8192], ''
            out = Path(td)/'out'
            out.mkdir()
            starts, sent = [], []
            gate = [self.record(4, 1024), self.record(4, 8192), self.compare_record()]
            test = self

            class FakeService:
                def __init__(self, command, _log):
                    tile = int(command[command.index('--row-tile') + 1])
                    starts.append(tile)
                    self.proc = types.SimpleNamespace(poll=lambda: 0)
                    ready = dict(ready=True, producer_executable_sha256='e'*64, health=dict(
                        decode_identity=dw.IDENTITY, row_tile=tile, startup_ms=50000 if len(starts) > 1 else 180000,
                        residency=dict(decode=dict(expert_tier='tierstream', tier_plan=dict(
                            three_tier=True, reference_compare=True, unit_manifest_sha256='f'*64)))))
                    state = dict(rank0_sha256='a'*64, rank1_sha256=('b' if tile == 2048 else 'c')*64, rank0_bytes=10, rank1_bytes=10,
                                 replay_sha256='d'*64, frontier=8191, components=[[0, 0, 10]])
                    per = [dict(backend='tierstream', prompt_tokens=8192, ttft_s=2.0, prefill_s=1.9, token=5, row_tile=tile, allocated_row_tile=tile,
                                prefill_state=state)]  # ttft-32000.u32 is not among the test prompts
                    per += [test.bench(order=('cold', 'warm'), n=n) for n in dw.QUALIFY]
                    self.replies = [ready] + (list(gate) if len(starts) == 1 else []) + per

                def read(self, deadline):
                    return self.replies.pop(0)

                def send(self, command):
                    sent.append((starts[-1], command['op']))

            deadline = types.SimpleNamespace(left=lambda: 100000)
            with patch.object(dw, 'Service', FakeService), patch.object(dw, 'stop'), patch.object(dw, 'snapshot_check'), \
                 contextlib.redirect_stdout(io.StringIO()) as printed:
                dw.execute(a, out, deadline, dict(service_sha256='e'*64), dw.check_inputs(a))
            summary = json.loads(printed.getvalue())
            self.assertEqual(starts, [2048, 8192])
            self.assertEqual(sent, [(2048, 'qualify'), (2048, 'qualify'), (2048, 'compare'), (2048, 'ttft'), (2048, 'bench'),
                                    (2048, 'bench'), (2048, 'quit'), (8192, 'ttft'), (8192, 'bench'), (8192, 'bench'), (8192, 'quit')])
            sweep = summary['tile_sweep']
            self.assertEqual([(s['tile'], s['allocated_row_tile'], s['startup_s']) for s in sweep['starts']], [(2048, 2048, 180.0), (8192, 8192, 50.0)])
            verdicts = {(c['op'], c['tokens']): c['verdict'] for c in sweep['comparison']}
            self.assertEqual(verdicts[('ttft', 8192)], 'DIFFERENT')       # rank1 state differs across tiles
            self.assertEqual(verdicts[('bench', 1024)], 'BITWISE-EQUAL')  # same tokens, logits, final state
            self.assertTrue((out/'tile-2048'/'health.json').is_file() and (out/'tile-8192'/'ttft-q8192.json').is_file())
            self.assertTrue(any(r['prompt'] == 'ttft-32000.u32' and r['verdict'] == 'INCOMPLETE' for r in summary['results']))

    def test_worker_receives_tile_and_sweep_options(self):
        """The parent forwards the startup tile, the fitted sweep, its note and the
        placement options to the worker (dry --check-inputs launch, argv captured)."""
        with tempfile.TemporaryDirectory(dir=BUILD) as td:
            build, prompts = Path(td)/'b', Path(td)/'p'
            build.mkdir()
            prompts.mkdir()
            with patch.object(dw, 'host_total_gib', return_value=191.6):
                a = dw.arguments(['--build', str(build), '--model', 'M', '--prompt-dir', str(prompts), '--full-card',
                                  '--tile-sweep', '2048,4096,8192', '--timeout', '1500', '--tier-policy', 'lru',
                                  '--routes-out', 'r.txt', '--check-inputs'])
            captured = {}

            class Job:
                def __enter__(self):
                    return self

                def __exit__(self, *exc):
                    return False

                def start(self, cmd):
                    captured['cmd'] = cmd
                    return object(), None

                def wait(self, process, left):
                    return 0
            with patch.object(dw.window, 'windows_job', return_value=Job):
                self.assertEqual(dw.run(a), 0)
            worker = captured['cmd'][7:]
            with patch.object(dw, 'host_total_gib', return_value=191.6):
                b = dw.arguments(worker)
            self.assertEqual((b.tile_sweep, b.sweep_note, b.tier_policy, b.routes_out, b.full_card),
                             ([2048, 8192], a.sweep_note, 'lru', 'r.txt', True))
            self.assertIn('4096 arm dropped', b.sweep_note)
            with patch.object(dw, 'host_total_gib', return_value=191.6):
                c = dw.arguments(['--build', str(build), '--model', 'M', '--prompt-dir', str(prompts), '--full-card',
                                  '--row-tile', '8192', '--check-inputs'])
            self.assertEqual(dw.worker_options(c)[dw.worker_options(c).index('--row-tile') + 1], '8192')

    def test_profile_switch_through_harness(self):
        """--profile / --allow-pending reach the service and the worker; --print-effective-config runs the service's
        print mode (no model read, no CUDA call) and relays its JSON; the summary records the ready event's profile."""
        with patch.object(dw, 'host_total_gib', return_value=191.6):
            a = dw.arguments(['--build', '.', '--model', 'M', '--prompt-dir', 'P', '--full-card', '--profile', 'fast', '--allow-pending'])
        a.unit_file, a.unit_manifest_sha256 = Path('U'), 'f'*64
        cmd = dw.service_command(a, Path('svc.exe'))
        self.assertEqual(cmd[cmd.index('--profile') + 1], 'fast')
        self.assertIn('--allow-pending', cmd)
        opts = dw.worker_options(a)
        self.assertEqual(opts[opts.index('--profile') + 1], 'fast')
        self.assertIn('--allow-pending', opts)
        with patch.object(dw, 'host_total_gib', return_value=191.6), self.assertRaises(SystemExit), contextlib.redirect_stderr(io.StringIO()):
            dw.arguments(['--build', '.', '--model', 'M', '--prompt-dir', 'P', '--profile', 'turbo'])
        plain = dw.service_command(dw.arguments(['--build', '.', '--model', 'M', '--prompt-dir', 'P']), Path('svc.exe'))
        self.assertNotIn('--profile', plain)  # no --profile: the service's default (= today)
        seen = {}

        def fake_run(cmd, **kw):
            seen['cmd'] = cmd
            return types.SimpleNamespace(stdout='{"profile":"fast","levers":{}}\n', stderr='', returncode=0)
        with patch.object(dw.subprocess, 'run', fake_run), contextlib.redirect_stdout(io.StringIO()) as printed:
            a.print_effective_config, a.allow_gpu = True, False
            self.assertEqual(dw.run(a), 0)
        self.assertEqual(seen['cmd'][-1], '--print-effective-config')
        self.assertNotIn('--allow-gpu', seen['cmd'])
        self.assertEqual(json.loads(printed.getvalue())['profile'], 'fast')
        # the summary records the first start's profile resolution (from the ready event)
        with tempfile.TemporaryDirectory(dir=BUILD) as td:
            rc, summary, _ = self.run_execute(td, self.full_replies(), left=100000)
        self.assertIn('profile', summary)

    def test_tile_sweep_arguments_and_estimate(self):
        with patch.object(dw, 'host_total_gib', return_value=191.6):
            a = dw.arguments(['--build', '.', '--model', 'M', '--prompt-dir', 'P', '--full-card', '--tile-sweep', '2048,4096,8192'])
        self.assertEqual((a.tile_sweep, a.sweep_note), ([2048, 4096, 8192], ''))
        self.assertLessEqual(dw.expected_seconds(a), dw.MAX_SECONDS)
        self.assertEqual([(t, [i[0] for i in items]) for t, items in dw.sweep_segments(a)],
                         [(2048, ['qualify', 'qualify', 'compare', 'ttft', 'ttft', 'bench', 'bench']),
                          (4096, ['ttft', 'ttft', 'bench', 'bench']), (8192, ['ttft', 'ttft', 'bench', 'bench'])])
        with patch.object(dw, 'host_total_gib', return_value=191.6):  # a shorter deadline drops the 4096 arm first
            short = dw.arguments(['--build', '.', '--model', 'M', '--prompt-dir', 'P', '--full-card', '--tile-sweep', '2048,4096,8192',
                                  '--timeout', '1500'])
        self.assertEqual(short.tile_sweep, [2048, 8192])
        self.assertIn('4096 arm dropped', short.sweep_note)
        for bad in (['--tile-sweep', '2048,8192', '--timeout', '900'], ['--tile-sweep', '2048,2048'], ['--tile-sweep', '2048,8200'],
                    ['--tile-sweep', '2048,8192', '--row-tile', '4096']):
            with self.assertRaises(SystemExit), patch.object(dw, 'host_total_gib', return_value=191.6), contextlib.redirect_stderr(io.StringIO()):
                dw.arguments(['--build', '.', '--model', 'M', '--prompt-dir', 'P', '--full-card'] + bad)
        with self.assertRaises(SystemExit), contextlib.redirect_stderr(io.StringIO()):
            dw.arguments(['--build', '.', '--model', 'M', '--prompt-dir', 'P', '--tile-sweep', '2048,8192'])  # full card only

    def test_mismatch_stops_qualification(self):
        bad = self.record(4)
        bad['fresh'][0]['rank0_sha256'] = 'b'*64
        with tempfile.TemporaryDirectory(dir=BUILD) as td:
            rc, summary, sent = self.run_execute(td, [bad], left=100000)
            self.assertEqual((rc, summary['verdict']), (3, 'MISMATCH'))
            self.assertEqual([c['op'] for c in sent], ['qualify', 'quit'])
            self.assertIn('export_dir', sent[0])


census = load('glm_route_census', 'src/glm/rtxpf/producer/glm_route_census.py')


class TileCompareTests(unittest.TestCase):
    STATE = dict(rank0_sha256='a' * 64, rank1_sha256='b' * 64, rank0_bytes=10, rank1_bytes=10, replay_sha256='d' * 64,
                 frontier=8191, components=[[0, 0, 10]])

    def rec(self, n=8192, tile=2048, state=None, token=7):
        return dict(backend='tierstream', prompt_tokens=n, ttft_s=2.0, prefill_s=1.9, token=token, row_tile=tile,
                    allocated_row_tile=8192, prefill_state=dict(state or self.STATE, frontier=n - 1))

    def result(self, n=8192, tile=2048, **kw):
        """A judged TTFT result as execute materializes it (canonical evidence stored)."""
        r = self.rec(n, tile, **kw)
        v, problems = dw.judge_ttft(r, n, tile, 8192)
        return dict(op='ttft', tokens=n, verdict=v, problems=problems,
                    evidence=dw.evidence_ttft(r, n, tile, 8192) if v == 'EXACT' else None)

    def test_tile_compare(self):
        same = dw.tile_compare([self.result(tile=8192), self.result(tile=2048)])
        self.assertEqual([(c['tokens'], c['row_tiles'], c['verdict']) for c in same], [(8192, [8192, 2048], 'BITWISE-EQUAL')])
        for kw in (dict(state=dict(self.STATE, rank1_sha256='c' * 64)), dict(token=8), dict(state=dict(self.STATE, replay_sha256='e' * 64)),
                   dict(state=dict(self.STATE, components=[[0, 0, 11]], rank0_bytes=11)), dict(state=dict(self.STATE, rank1_bytes=11)),
                   dict(state=dict(self.STATE, components=[[1, 0, 10]]))):
            diff = dw.tile_compare([self.result(tile=8192), self.result(tile=2048, **kw)])
            self.assertEqual(diff[0]['verdict'], 'DIFFERENT', kw)
            self.assertIn('FAST-profile', diff[0]['meaning'])
        # one tile only, or an arm that is not EXACT: nothing to compare
        self.assertEqual(dw.tile_compare([self.result(tile=2048), self.result(tile=2048)]), [])
        self.assertEqual(dw.tile_compare([self.result(tile=8192), dict(self.result(tile=2048), verdict='INCOMPLETE')]), [])

    def test_ttft_requires_complete_state_evidence(self):
        good = self.rec()
        self.assertEqual(dw.judge_ttft(good, 8192, 2048, 8192), ('EXACT', []))
        for bad in ({}, dict(replay_sha256=None), dict(rank1_sha256='x'), dict(frontier=8192), dict(components=[]),
                    dict(rank0_bytes=0)):
            rec = dict(good, prefill_state=dict(good['prefill_state'], **bad)) if bad else dict(good, prefill_state=None)
            self.assertEqual(dw.judge_ttft(rec, 8192, 2048, 8192)[0], 'NO-REFERENCE', bad)
        # the reported tile is an int and EQUALS the effective requested tile; the allocation is a
        # positive multiple of 32 (0 / None / misaligned refused)
        for tile in ([2048], '2048', 2050, 0, None, 4096):
            self.assertEqual(dw.judge_ttft(dict(good, row_tile=tile), 8192, 2048, 8192)[0], 'NO-REFERENCE', tile)
        for alloc in (0, None, 8200, '8192', 1024):
            self.assertEqual(dw.judge_ttft(good, 8192, 2048, alloc)[0], 'NO-REFERENCE', alloc)
        self.assertEqual(dw.judge_ttft(dict(good, allocated_row_tile=4096), 8192, 2048, 8192)[0], 'NO-REFERENCE')
        # planned tiles 2048 / 8192 both REPORTING 2048 never compare as equal
        both_2048 = [self.result(tile=2048), dict(op='ttft', tokens=8192, **dict(zip(('verdict', 'problems'), dw.judge_ttft(
            self.rec(tile=2048), 8192, 8192, 8192))))]
        self.assertEqual(both_2048[1]['verdict'], 'NO-REFERENCE')
        self.assertEqual(dw.tile_compare(both_2048), [])
        # malformed results never kill the comparison
        weird = [dict(self.result(tile=8192), evidence=dict(kind='ttft')), self.result(tile=2048)]
        self.assertEqual(dw.tile_compare(weird)[0]['verdict'], 'NO-REFERENCE')
        self.assertEqual(dw.tile_compare([None, dict(op='ttft')])[0]['verdict'], 'NO-REFERENCE')
        self.assertEqual(dw.sweep_compare([None]), [])
        self.assertEqual(dw.sweep_compare([None, dict(op='bench', sweep_tile=2048)])[0]['verdict'], 'NO-REFERENCE')
        self.assertEqual(dw.judge_ttft(dict(good, prefill_state=None), 8192, 2048, 8192)[0], 'NO-REFERENCE')

    def test_ttft_tile_capped_by_allocation(self):
        self.assertEqual(dw.ttft_tile(8192, 4096)[0], 4096)
        self.assertIn('exceeds the startup allocation 4096', dw.ttft_tile(8192, 4096)[1])
        self.assertEqual(dw.ttft_tile(2048, 4096), (2048, 'requested 2048'))
        self.assertEqual(dw.ttft_tile(None, 4096)[0], None)
        self.assertIn('allocated tile 4096', dw.ttft_tile(None, 4096)[1])


class PlacementInputTests(unittest.TestCase):
    def test_route_census_from_receipts(self):
        line = ' '.join(f'{layer}:' + ','.join(str((layer * 7 + k) % 288) for k in range(8)) for layer in range(3, 46))
        rec = {'prompt_tokens': 8192, 'arms': [{'steps': [{'routes': line}, {'routes': line}, {'experts': {}}]}],
               'other': {'steps': [{'token': 1}]}}
        with tempfile.TemporaryDirectory() as td:
            src, out = Path(td)/'bench.json', Path(td)/'routes.txt'
            src.write_text(json.dumps(rec), encoding='utf-8')
            self.assertEqual(census.main([str(src), '-o', str(out)]), 0)
            text = out.read_text(encoding='utf-8').splitlines()
            self.assertEqual(text, ['# seq bench.json:/arms[0] rows 0', line, line])
            src.write_text(json.dumps({'steps': [{'routes': '3:1,2 4:x'}]}), encoding='utf-8')
            with self.assertRaises(ValueError):
                census.census([src])
            src.write_text(json.dumps({'steps': [{'token': 1}]}), encoding='utf-8')
            self.assertEqual(census.main([str(src), '-o', str(out)]), 2)

    def test_window_passes_placement_flags(self):
        a = argparse.Namespace(model=Path('m'), cache=None, expert_tier='tierstream', unit_file=Path('u.bin'),
                               unit_manifest_sha256='a' * 64, vram_total_gib=93.0, ram_total_gib=None,
                               tier_vram_gib=None, tier_ram_gib=None, tier_spare_percent=5.0, tier_policy='auto',
                               tier_predict='1', expert_residency=None, expert_routes='r.txt', routes_out='o.txt',
                               capacity=None, full_card=False)
        cmd = dw.service_command(a, Path('svc.exe'))
        for flag, value in (('--tier-spare-percent', '5.0'), ('--tier-policy', 'auto'), ('--tier-predict', '1'),
                            ('--expert-routes', 'r.txt'), ('--routes-out', 'o.txt')):
            self.assertEqual(cmd[cmd.index(flag) + 1], value)
        self.assertNotIn('--expert-residency', cmd)
        self.assertNotIn('--row-tile', cmd)  # default: the service's 5% rule
        a.row_tile = 8192
        cmd = dw.service_command(a, Path('svc.exe'))
        self.assertEqual(cmd[cmd.index('--row-tile') + 1], '8192')
        with patch.object(dw, 'host_total_gib', return_value=191.6):
            parsed = dw.arguments(['--build', '.', '--model', 'M', '--prompt-dir', 'P', '--full-card', '--row-tile', '8192'])
        self.assertEqual(parsed.row_tile, 8192)
        for bad in ('8200', '0', '65536'):
            with self.assertRaises(SystemExit), patch.object(dw, 'host_total_gib', return_value=191.6), contextlib.redirect_stderr(io.StringIO()):
                dw.arguments(['--build', '.', '--model', 'M', '--prompt-dir', 'P', '--full-card', '--row-tile', bad])
        # with 8192 allocated the full-card arms run 8K at 8192 vs 2048 and 32K at 8192 vs 2048
        self.assertEqual([dw.ttft_tile(t, 8192)[0] for t in (8192, 2048, None, 2048)], [8192, 2048, None, 2048])

    def test_memory_gate_reports_raw_and_backing(self):
        peaks = {}
        a = argparse.Namespace(vram_total_gib=93.0, ram_total_gib=111.0)
        rec = {'memory': {'measured_device_peak_bytes': 98_701_794_188, 'host_peak_commit_bytes': 79_552_000_000,
                          'process_peak_commit_bytes_raw': 175_587_450_880, 'wddm_device_backing_bytes': 96_037_952_396,
                          'within_caps': True}}
        self.assertEqual(dw.memory_gate(rec, a, peaks), [])
        self.assertEqual(peaks['process_peak_commit_bytes_raw'], 175_587_450_880)
        self.assertEqual(peaks['wddm_device_backing_bytes'], 96_037_952_396)


class SnapshotTests(unittest.TestCase):
    def test_receiver_snapshot_layout_and_hash(self):
        receiver = BUILD/'glm_native_receiver.exe'
        P = 5
        kda = 34*(32*65536 + 32*4608)
        dsa = 12*(P*1024 + 2*P*256 + (P//4)*512)
        with tempfile.TemporaryDirectory(dir=BUILD) as td:
            path = Path(td)/'state-r0.bin'
            h = hashlib.sha256()
            with path.open('wb') as f:
                f.write(struct.pack('<QQ', P, P+1))
                block = bytes(range(256))*4096
                left = kda + dsa
                while left:
                    b = block[:min(left, len(block))]
                    f.write(b)
                    h.update(b)
                    left -= len(b)
            out = subprocess.run([str(receiver), '--snapshot', str(path), '0', str(P), str(P+1)], capture_output=True, text=True)
            self.assertEqual(out.returncode, 0, out.stderr)
            self.assertEqual(out.stdout.split(), ['SNAPSHOT', h.hexdigest(), str(kda+dsa)])
            bad = subprocess.run([str(receiver), '--snapshot', str(path), '0', str(P+1), str(P+2)], capture_output=True, text=True)
            self.assertNotEqual(bad.returncode, 0)
            with path.open('r+b') as f:
                f.truncate(16 + kda + dsa - 1)
            short = subprocess.run([str(receiver), '--snapshot', str(path), '0', str(P), str(P+1)], capture_output=True, text=True)
            self.assertNotEqual(short.returncode, 0)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--build', type=Path, required=True)
    parser.add_argument('--receipt', type=Path, help='fixture artifact published by glm_native_decode_host')
    parsed = parser.parse_args()
    BUILD, RECEIPT = parsed.build.resolve(), parsed.receipt
    result = unittest.TextTestRunner(verbosity=2).run(unittest.defaultTestLoader.loadTestsFromModule(__import__('__main__')))
    raise SystemExit(0 if result.wasSuccessful() else 1)
