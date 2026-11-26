import contextlib
import io
import json
from pathlib import Path
import sys
import tempfile
import unittest

KIT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(KIT / 'lib'))
import band_edges as band
import campaign
from kit import id_hash


class BandEdgeTests(unittest.TestCase):
    def setUp(self):
        self.inventory = band.load_inventory(KIT)

    def test_plan_for_every_row_includes_exact_points_and_abba(self):
        output = io.StringIO()
        with contextlib.redirect_stdout(output):
            band.plan(self.inventory, None, 18080, Path('edge-results'))
        plan = output.getvalue()
        self.assertEqual(plan.count(' band-edges sample '),
                         4 * sum(len(band.points(row)) for row in self.inventory.values()))
        self.assertEqual(plan.count('# Start on rank 1:'), 4 * sum(
            len(band.points(row)) for row in self.inventory.values() if band.is_pair(row)))
        self.assertIn('STRIX_DS41_DECODER_REPLAY=256', plan)
        self.assertIn('STRIX_GLM_PF_WMMA=1', plan)
        self.assertIn('STRIX_QPF2K_MOE_ROUTE_PRUNE=cap9', plan)
        self.assertIn('STRIX_INK_MOE_W4A4=iu4', plan)
        self.assertIn('--wide-exact 64', plan)
        self.assertIn('historical 24K EXACT control exceeds this limit', plan)

    def test_verdict_uses_winning_suffix(self):
        points = [{'tokens': 10, 'wins': True}, {'tokens': 20, 'wins': False}, {'tokens': 30, 'wins': True}]
        self.assertEqual(band.verdict(points, 10, 30), 'RAISE-LOWER-EDGE-TO 30')
        points[1]['wins'] = True
        self.assertEqual(band.verdict(points, 10, 30), 'KEEP')
        points[2]['wins'] = False
        self.assertEqual(band.verdict(points, 10, 30), 'RAISE-LOWER-EDGE-TO 31')
        self.assertEqual(band.verdict([{'tokens': 10, 'error': 'timeout'}], 10, 30), 'INCOMPLETE')

    def test_native_and_http_parser(self):
        self.assertEqual(band.native_ms('prompt 32 tokens: 22.6 ms\n', 32), 22.6)
        with self.assertRaises(ValueError):
            band.native_ms('prompt 31 tokens: 22.6 ms\n', 32)
        response = dict(usage=dict(prompt_tokens=32, completion_tokens=1, prompt_tokens_details=dict(cached_tokens=0)),
                        choices=[dict(finish_reason='length')], timings=dict(prefill_ms=21.0))
        self.assertEqual(band.server_ms(response, 32, 1, campaign, id_hash), 21.0)
        response['usage']['prompt_tokens_details']['cached_tokens'] = 31
        with self.assertRaises(ValueError):
            band.server_ms(response, 32, 1, campaign, id_hash)

    def test_receipts_grade_raw_samples_and_reject_tampering(self):
        key = 'qwen35-hip-decode/wide_rows/1'
        row = self.inventory[key]
        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp)
            for n in band.points(row):
                point = out / key / str(n)
                point.mkdir(parents=True)
                for step, (side, ms) in enumerate(zip(band.ABBA, [105.0, 92.0, 90.0, 100.0]), 1):
                    raw = point / f'{step}.log'
                    raw.write_text(f'prompt {n} tokens: {ms} ms\n', encoding='utf-8')
                    band.write(point / f'{step}.json',
                               dict(row=key, step=step, side=side, length=n, device='gfx1151|fixture', capacity=2304,
                                    prompt_sha256=id_hash([1] * n), power_state='fixture', weights='fixtures',
                                    build='fixture', commit='fixture', generation=1, scope='native-prefill',
                                    wall_seconds=10, prefill_ms=ms, raw=raw.name, raw_sha256=band.digest(raw)))
            output = io.StringIO()
            with contextlib.redirect_stdout(output):
                self.assertEqual(band.report(self.inventory, [key], out, campaign, id_hash), 0)
            receipt = json.loads(output.getvalue())['receipts'][0]
            self.assertEqual(receipt['verdict'], 'KEEP')
            self.assertEqual(receipt['timings'][0]['prefill_ms']['A'], [105.0, 100.0])
            point = out / key / '1'
            (point / '2.log').write_text('prompt 1 tokens: 10.0 ms\n', encoding='utf-8')
            with self.assertRaisesRegex(ValueError, 'hash mismatch'):
                band.check_point(point, key, row, 1, campaign, id_hash)
            output = io.StringIO()
            with contextlib.redirect_stdout(output):
                self.assertEqual(band.report(self.inventory, [key], out, campaign, id_hash), 1)
            self.assertEqual(json.loads(output.getvalue())['receipts'][0]['verdict'], 'INCOMPLETE')


if __name__ == '__main__':
    unittest.main()
