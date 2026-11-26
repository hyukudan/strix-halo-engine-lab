import contextlib
import csv
import io
import json
import os
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'lib'))
import campaign
import interface
import kit

MODEL = campaign.MODELS[0]
HW = campaign.HARDWARE[0]
CELL = campaign.CELLS[0]
DIGEST = 'a' * 64


class DefaultCampaignTest(unittest.TestCase):
    def ledger_row(self, tier='exact', value='9', status='record', digest=DIGEST):
        return (MODEL, HW, tier, CELL, status, value, 'tok/s', digest, 'fixture.md:4', 'test fixture')

    def ledger(self, *rows):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'private.tsv'
            with path.open('w', encoding='utf-8-sig', newline='') as stream:
                writer = csv.writer(stream, delimiter='\t')
                writer.writerow(campaign.LEDGER_FIELDS)
                writer.writerows(rows)
            return campaign.read_ledger(path)

    def receipt(self, tier, value, digest=DIGEST, valid=True):
        return dict(model=MODEL, hardware=HW, tier=tier, cell=CELL, value=value,
                    workload_sha256=digest, unit='tok/s', valid=valid)

    def verdict(self, ledger, rows):
        stdout = io.StringIO()
        with patch.object(campaign, 'CELLS', (CELL,)), contextlib.redirect_stdout(stdout):
            rc = campaign.compare(ledger, rows, {(MODEL, HW)})
        return rc, list(csv.DictReader(io.StringIO(stdout.getvalue()), delimiter='\t'))

    def test_exact_match_and_nondecreasing(self):
        ledger = self.ledger(*(self.ledger_row(tier) for tier in campaign.TIERS))
        rc, rows = self.verdict(ledger, [self.receipt(tier, n) for tier, n in zip(campaign.TIERS, (9, 10, 10))])
        self.assertEqual(rc, 0)
        self.assertEqual([r['reached'] for r in rows], ['yes'] * 3)

    def test_under_record_and_tier_regression_are_independent(self):
        ledger = self.ledger(*(self.ledger_row(tier, value) for tier, value in zip(campaign.TIERS, ('10', '12', '9'))))
        rc, rows = self.verdict(ledger, [self.receipt(tier, n) for tier, n in zip(campaign.TIERS, (9, 12, 11))])
        self.assertEqual(rc, 1)
        self.assertEqual([r['status'] for r in rows], ['under-record', 'ok', 'slower-than-lower-tier'])
        self.assertEqual([r['reached'] for r in rows], ['no', 'yes', 'yes'])

    def test_historical_and_different_hash_never_compare(self):
        ledger = self.ledger(self.ledger_row(status='historical', digest=''),
                             self.ledger_row(tier='fast', digest='b' * 64))
        rc, rows = self.verdict(ledger, [self.receipt('exact', 12), self.receipt('fast', 12)])
        self.assertEqual(rc, 1)
        self.assertEqual([r['reached'] for r in rows], ['no'] * 3)
        self.assertEqual(rows[0]['status'], 'record-workload-unverified')
        self.assertEqual(rows[1]['status'], 'record-workload-unverified')
        self.assertEqual(rows[2]['status'], 'default-unavailable,no-record')

    def test_rejects_bad_ledger_and_duplicate_defaults(self):
        for args in ({'value': 'nan'}, {'value': '-1'}, {'digest': 'a'},
                     {'status': 'missing', 'digest': ''}, {'status': 'historical'}):
            with self.subTest(args=args), self.assertRaisesRegex(ValueError, 'ledger line 2'):
                self.ledger(self.ledger_row(**args))
        row = self.ledger_row()
        with self.assertRaisesRegex(ValueError, 'duplicate comparable'):
            self.ledger(row, row)
        receipt = self.receipt('exact', 12)
        with self.assertRaisesRegex(ValueError, 'duplicate default'):
            self.verdict(self.ledger(row), [receipt, receipt])

    def test_pair_ratio_requires_matching_valid_workload(self):
        out = io.StringIO()
        with patch.object(campaign, 'CELLS', (CELL,)), contextlib.redirect_stdout(out):
            paired = dict(self.receipt('exact', 18), hardware='strix-two')
            campaign.compare({}, [self.receipt('exact', 12), paired], {(MODEL, HW), (MODEL, 'strix-two')})
        rows = list(csv.DictReader(io.StringIO(out.getvalue()), delimiter='\t'))
        self.assertEqual(next(r for r in rows if r['hardware'] == 'strix-two')['pair_over_one'], '1.5')

    def test_response_requires_full_ids_and_native_logits(self):
        body = {'choices': [{'finish_reason': 'length'}], 'usage': {'prompt_tokens': 42, 'completion_tokens': 2},
                'strix': {'token_ids': [12, 13]}, 'timings': {'decode_ms': 5, 'logits_fnv': 'a' * 16}}
        reply = campaign.generation_response(body, 2, kit.id_hash)
        self.assertEqual(reply['decode_ms'], 5)
        self.assertEqual(reply['ids_sha256'], kit.id_hash([12, 13]))
        self.assertEqual(reply['logits'], {'logits_fnv': 'a' * 16})
        body['strix']['token_ids'] = [12]
        self.assertIsNone(campaign.generation_response(body, 2, kit.id_hash)['ids_sha256'])
        with self.assertRaisesRegex(ValueError, 'token count'):
            campaign.generation_response(body, 1, kit.id_hash)
        body['choices'][0]['finish_reason'] = 'stop'
        body['usage']['completion_tokens'] = 1
        body['strix']['token_ids'] = [12, 13]
        self.assertEqual(campaign.generation_response(body, 5, kit.id_hash)['ids_sha256'], kit.id_hash([12, 13]))

    def test_wall_uses_all_request_tokens_and_evictions_fail(self):
        payload = {'choices': [{'finish_reason': 'stop'}], 'usage': {'prompt_tokens': 42, 'completion_tokens': 2},
                   'strix': {'token_ids': [12, 13], 'logits_fnv': '1' * 16}}
        prompts = dict.fromkeys(campaign.PROMPTS, 'prompt')
        with patch.object(campaign, 'http', return_value=payload):
            row = {}
            changes = iter([[{'a': 0}], [{'a': 1}]])
            campaign.measure_cell(row, 'c4', prompts, [], 8080, campaign.time.monotonic() + 5,
                                  kit.id_hash, lambda: next(changes))
        self.assertAlmostEqual(row['value'], 8 / row['wall_s'])
        self.assertEqual(len(row['requests']), 4)
        self.assertFalse(row['valid'])
        self.assertIn('eviction counter changed', row['errors'])
        self.assertFalse(campaign.eviction_clean([], []))
        self.assertFalse(campaign.eviction_clean([{'a': 0}], [{'a': 0}, {'b': 0}]))

    def test_fresh_exact_prefill_rows(self):
        seed = [4, 5, 6, 4, 7, 8]
        route, rows = campaign.cell_payloads('prefill-32768', {}, seed)
        _, shorter = campaign.cell_payloads('prefill-8192', {}, seed)
        self.assertEqual(route, '/v1/completions')
        self.assertTrue(all(len(r['prompt']) == 32768 and r['max_tokens'] == 1 for r in rows))
        self.assertEqual([r['prompt'][0] for r in shorter + rows], [5, 6, 7, 8])

    def test_dry_run_covers_every_arm_without_looking_at_device_or_ledger(self):
        out = io.StringIO()
        with patch.object(campaign.subprocess, 'Popen', side_effect=AssertionError('process launched')), \
                contextlib.redirect_stdout(out):
            self.assertEqual(campaign.main(['default', '--dry-run', '--ledger', 'missing.tsv'], kit.KIT,
                                           kit.id_hash, kit.sha), 0)
        lines = out.getvalue().splitlines()
        commands = [line.split('] ', 1)[1] for line in lines if line.startswith('[')]
        self.assertEqual(len(commands), 72)
        self.assertTrue(all(cmd.startswith('strix serve ') and len(cmd.split()) == 5 for cmd in commands))
        self.assertEqual(sum(line.startswith('  c16:') for line in lines), 72)
        self.assertEqual(sum(line.startswith('  rank 1 product:') for line in lines), 24)


class PublicCampaignTest(unittest.TestCase):
    def test_fixed_plan_rotates_prompt_order_and_arm_order(self):
        params = dict(profile='turbo', native_tier=interface.deepseek_campaign_tier('turbo', kit.KIT), only=['decode', 'prefill', 'exactness'], contexts=[1024, 8192, 32768],
                      gen_tokens=256, repeat=3, warmup=1)
        texts = interface.prompt_set('kit', kit.KIT)
        with tempfile.TemporaryDirectory() as name, patch.object(kit, 'ENGINE', 'deepseek-pair'), \
                patch.dict(os.environ, {'BENCH_PARAMS': json.dumps(params), 'BENCH_PROMPTS': 'kit'}), \
                contextlib.redirect_stdout(io.StringIO()) as plan:
            out = Path(name)
            for i, row in enumerate(texts, 1):
                (out / (row['id'] + '.ids')).write_text(str(i), encoding='ascii')
            interface.plan(out, False, kit)
            first = kit.read(out / 'decode-greedy-r1.manifest.json')
            second = kit.read(out / 'decode-spec-r2.manifest.json')
            self.assertEqual([r['prompt_id'] for r in first[:8]], [p['id'] for p in texts])
            self.assertEqual([r['prompt_id'] for r in second[:8]], [p['id'] for p in texts[1:] + texts[:1]])
            self.assertTrue(all(r['warmup'] for r in second[:8]))
            self.assertTrue(all(not r['warmup'] and r['round'] == 2 for r in second[8:]))
        self.assertEqual([line.split()[1] for line in plan.getvalue().splitlines() if line.startswith('deepseek_native_decode')],
                         ['decode-greedy-r1', 'decode-spec-r1', 'decode-spec-r2', 'decode-greedy-r2',
                          'decode-greedy-r3', 'decode-spec-r3'])
        # The TURBO preset's native tier is turbo (models/deepseek-v4.1-flash.toml [presets.turbo]).
        self.assertEqual(params['native_tier'], 'turbo')
        self.assertEqual(plan.getvalue().count('--profile turbo'), 3)
        self.assertEqual(set(sum(kit.DEEPSEEK_GROUPS.values(), ())), {x['id'] for x in texts})

    def test_schema_accepts_every_campaign_native_tier(self):
        schema = kit.read(Path(kit.__file__).resolve().parents[1] / 'results.schema.json')

        def find(node):
            if isinstance(node, dict):
                if 'native_tier' in node and isinstance(node['native_tier'], dict) and 'enum' in node['native_tier']:
                    return node['native_tier']['enum']
                return next((r for r in map(find, node.values()) if r is not None), None)
            if isinstance(node, list):
                return next((r for r in map(find, node) if r is not None), None)
            return None

        allowed = find(schema)
        self.assertIsNotNone(allowed)
        self.assertEqual(set(allowed), set(interface.DEEPSEEK_CAMPAIGN_TIERS))
        self.assertIn(interface.deepseek_campaign_tier('turbo', kit.KIT), allowed)

    def test_public_group_aggregation_requires_all_measured_jobs(self):
        measurements = [
            dict(prompt_id=name, mode=mode, **kit.stats([1, 2, 3]))
            for members in kit.DEEPSEEK_GROUPS.values() for name in members for mode in ('greedy', 'spec')
        ]
        measurements.extend(dict(prompt_id=f'prefill-{n}', mode='ttft', **kit.stats([4, 5, 6]))
                            for n in (1024, 8192, 32768))
        with tempfile.TemporaryDirectory() as name:
            out = Path(name)
            kit.dump(out / 'commands.json', [{'rank': rank} for _ in range(9) for rank in (0, 1)])
            data = dict(params=dict(profile='fast', native_tier='fast'), status='complete', measurements=measurements)
            kit.attach_deepseek_campaign(out, data)
            groups = data['public_campaign']['group_measurements']
            self.assertEqual(len(groups), 6)
            self.assertEqual([r['repeat_count'] for r in groups], [12, 12, 6, 6, 6, 6])
            self.assertTrue(all(r['median'] == 2 for r in groups))
            data['measurements'].pop()
            with self.assertRaisesRegex(ValueError, 'incomplete issue 116'):
                kit.attach_deepseek_campaign(out, data)

    def test_public_commands_normalize_roots_and_keep_arguments(self):
        with tempfile.TemporaryDirectory() as name:
            out = Path(name)
            kit.campaign_command(out, 'decode-spec-r1', 1, ['/tree/build', '/weights', '/results'],
                                 ['env', 'STRIX_DS41_PACKS=/weights', '/tree/build/engine', '--mode', 'spec',
                                  '--prompt-file', '/results/input.ids', '--vocab', 'auto:/weights/vocab'])
            command = kit.read(out / 'commands.json')[0]
        self.assertEqual(command['argv'],
                         ['env', 'STRIX_DS41_PACKS=<MODEL_DIR>', '<BIN_DIR>/engine', '--mode', 'spec',
                          '--prompt-file', '<RUN_DIR>/input.ids', '--vocab', 'auto:<MODEL_DIR>/vocab'])

    def test_public_profile_rejects_noncanonical_selection(self):
        with self.assertRaisesRegex(ValueError, 'fixed protocol'):
            interface.resolve(['deepseek-pair', 'quick', '', '', '1', '', '', 'exact', 'kit', 'decode', '0'], kit.KIT)


if __name__ == '__main__':
    unittest.main()
