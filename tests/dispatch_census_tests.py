#!/usr/bin/env python3
"""CPU tests for tools/dispatch_census.py on a small synthetic rocprofv3 trace (tests/data/dispatch_census)."""
import importlib.util
import io
import json
from pathlib import Path
import sys
import tempfile
import unittest
from contextlib import redirect_stdout, redirect_stderr

sys.dont_write_bytecode = True
ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location('dispatch_census', ROOT / 'tools/dispatch_census.py')
C = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(C)
DATA = ROOT / 'tests/data/dispatch_census'

# Fixture layout (ns): init 0-1000 | marker 10000-10100 | window 1: kA 10200-12200, kB 12300-13300,
# kA 13000-14000 (second queue, overlaps kB) | marker 20000-20100 | window 2: kA 20500-22500, kC 23000-23500.


class DispatchCensusTests(unittest.TestCase):
    def test_csv_and_json_parse_to_the_same_records(self):
        fmt_csv, csv_records = C.load_trace(DATA / 'kernel_trace.csv')
        fmt_json, json_records = C.load_trace(DATA / 'results.json')
        self.assertEqual((fmt_csv, fmt_json), ('rocprofv3-csv', 'rocprofv3-json'))
        self.assertEqual(csv_records, json_records)
        self.assertEqual(len(csv_records), 8)
        self.assertEqual([r[0] for r in csv_records], sorted(r[0] for r in csv_records))

    def test_whole_trace_busy_is_an_interval_union(self):
        out = C.run(DATA / 'kernel_trace.csv')
        self.assertEqual(out['dispatches'], 8)
        self.assertAlmostEqual(out['trace']['span_ms'], 0.0235)
        self.assertAlmostEqual(out['trace']['gpu_busy_ms'], 0.0074)
        self.assertAlmostEqual(out['trace']['host_gap_share'], (23500 - 7400) / 23500)
        self.assertAlmostEqual(out['trace']['max_gap_us'], 9.0)
        self.assertNotIn('windows', out)
        self.assertEqual(out['source']['trace'], 'kernel_trace.csv')
        self.assertRegex(out['source']['trace_sha256'], '^[0-9a-f]{64}$')

    def test_boundary_kernel_windows(self):
        out = C.run(DATA / 'kernel_trace.csv', window_kernel='^marker_kernel$', tokens=5)
        w = out['windows']
        self.assertEqual((w['count'], w['prologue_launches'], w['cut_by']), (2, 1, 'boundary-kernel'))
        rows = w['per_window']
        self.assertEqual([r['launches'] for r in rows], [3, 2])
        self.assertAlmostEqual(rows[0]['span_ms'], 0.010)
        self.assertAlmostEqual(rows[0]['gpu_busy_ms'], 0.0037)
        self.assertAlmostEqual(rows[0]['host_gap_ms'], 0.0063)
        self.assertAlmostEqual(rows[0]['max_gap_us'], 6.0)  # trailing idle 14000 -> next boundary 20000
        self.assertAlmostEqual(rows[1]['span_ms'], 0.0035)
        self.assertAlmostEqual(rows[1]['gpu_busy_ms'], 0.0025)
        self.assertAlmostEqual(w['host_gap_share'], 7300 / 13500)
        self.assertEqual(w['launches']['median'], 2.5)
        self.assertEqual(out['per_token']['launches'], 1.0)
        self.assertAlmostEqual(out['per_token']['gpu_busy_ms'], 0.0062 / 5)
        self.assertEqual(out['top_by_count'][0], dict(out['top_by_count'][0], name='void kA<2>', count=3))
        self.assertAlmostEqual(out['top_by_time'][0]['total_us'], 5.0)
        self.assertEqual([x['name'] for x in out['top_by_time']], ['void kA<2>', 'kB', 'kC'])
        self.assertEqual(out['distinct_kernels'], 3)

    def test_boundary_counted_and_drop_last(self):
        out = C.run(DATA / 'results.json', window_kernel='marker', include_boundary=True, drop_last=True)
        self.assertEqual(out['windows']['count'], 1)
        self.assertEqual(out['windows']['per_window'][0]['launches'], 4)
        self.assertAlmostEqual(out['windows']['per_window'][0]['gpu_busy_ms'], 0.0038)

    def test_marker_windows(self):
        out = C.run(DATA / 'kernel_trace.csv', marker_file=DATA / 'marker_api_trace.csv', marker_regex='^window$')
        w = out['windows']
        self.assertEqual((w['cut_by'], w['count'], w['prologue_launches']), ('markers', 2, 1))
        self.assertEqual([r['launches'] for r in w['per_window']], [4, 3])
        self.assertEqual(out['source']['markers'], 'marker_api_trace.csv')

    def test_full_names_and_record_list_json(self):
        with tempfile.TemporaryDirectory() as tmp:
            p = Path(tmp) / 'list.json'
            p.write_text(json.dumps([{'name': 'k(int)', 'start': 0, 'end': 10}, {'name': 'k(float)', 'start': 20, 'end': 30}]))
            full = C.run(p, names='full')
            base = C.run(p)
        self.assertEqual(full['distinct_kernels'], 2)
        self.assertEqual(base['distinct_kernels'], 1)
        self.assertEqual(full['source']['format'], 'records-json')

    def test_base_name(self):
        self.assertEqual(C.base_name('void f<(anonymous namespace)::X, 2>(float*, int)'), 'void f<(anonymous namespace)::X, 2>')
        self.assertEqual(C.base_name('plain'), 'plain')
        self.assertEqual(C.base_name('(lambda)'), '(lambda)')

    def test_errors(self):
        with self.assertRaisesRegex(ValueError, 'window kernel'):
            C.run(DATA / 'kernel_trace.csv', window_kernel='absent')
        with self.assertRaisesRegex(ValueError, 'positive'):
            C.run(DATA / 'kernel_trace.csv', tokens=0)
        with tempfile.TemporaryDirectory() as tmp:
            p = Path(tmp) / 'bad.csv'
            p.write_text('"Kernel_Name","Start_Timestamp"\n"k",1\n')
            with self.assertRaisesRegex(ValueError, 'end timestamp'):
                C.load_trace(p)

    def test_cli_writes_json(self):
        with tempfile.TemporaryDirectory() as tmp:
            o = Path(tmp) / 'census.json'
            self.assertEqual(C.main([str(DATA / 'kernel_trace.csv'), '--window-kernel', 'marker', '--tokens', '2', '-o', str(o)]), 0)
            data = json.loads(o.read_text())
        self.assertEqual(data['schema'], 'dispatch-census-v1')
        self.assertEqual(data['per_token']['launches'], 2.5)
        err = io.StringIO()
        with redirect_stderr(err), redirect_stdout(io.StringIO()):
            self.assertEqual(C.main([str(DATA / 'kernel_trace.csv'), '--window-kernel', 'nope']), 1)
        self.assertIn('no dispatch matches', err.getvalue())

    def write_csv(self, tmp, rows, name='trace.csv'):
        p = Path(tmp) / name
        p.write_text('"Kernel_Name","Start_Timestamp","End_Timestamp"\n' +
                     ''.join(f'"{n}",{a},{b}\n' for n, a, b in rows), encoding='utf-8')
        return p

    def test_carry_in_work_is_clipped_to_windows(self):
        # Sol finding 1: a kernel launched in window 1 runs across the boundary into window 2.
        rows = [('mark', 0, 1), ('long', 10, 1000), ('mark', 100, 101), ('short', 1100, 1200)]
        with tempfile.TemporaryDirectory() as tmp:
            out = C.run(self.write_csv(tmp, rows), window_kernel='^mark$')
        w = out['windows']
        self.assertEqual([r['launches'] for r in w['per_window']], [1, 1])
        self.assertAlmostEqual(w['per_window'][0]['gpu_busy_ms'], 90 / 1e6)
        self.assertAlmostEqual(w['per_window'][1]['gpu_busy_ms'], (1000 - 100 + 100) / 1e6)
        for r in w['per_window'] + [w]:
            self.assertGreaterEqual(r['host_gap_share'], 0)
            self.assertLessEqual(r['host_gap_share'], 1)

    def test_window_gaps_include_bounds_and_empty_windows(self):
        # Sol finding 5: leading/trailing idle and empty windows count toward max_gap_us.
        rows = [('mark', 0, 1), ('k', 500, 600), ('mark', 1000, 1001), ('mark', 4000, 4001), ('k', 4100, 4200)]
        with tempfile.TemporaryDirectory() as tmp:
            out = C.run(self.write_csv(tmp, rows), window_kernel='^mark$')
        gaps = [r['max_gap_us'] for r in out['windows']['per_window']]
        self.assertEqual(gaps, [0.5, 3.0, 0.1])
        self.assertEqual(out['windows']['per_window'][1]['launches'], 0)
        self.assertEqual(out['windows']['per_window'][1]['host_gap_share'], 1.0)

    def test_final_window_ends_at_last_dispatch_including_boundaries(self):
        # Sol r2 finding 3: the trace end comes from every dispatch; boundary exclusion is for measured work only.
        rows = [('mark', 0, 1), ('k', 10, 20), ('mark', 100, 200)]
        with tempfile.TemporaryDirectory() as tmp:
            out = C.run(self.write_csv(tmp, rows), window_kernel='^mark$')
        last = out['windows']['per_window'][1]
        self.assertAlmostEqual(last['span_ms'], 100 / 1e6)
        self.assertEqual((last['launches'], last['gpu_busy_ms'], last['host_gap_share']), (0, 0, 1.0))
        self.assertAlmostEqual(last['max_gap_us'], 0.1)

    def test_zero_duration_dispatches_are_ignored_in_gaps(self):
        # Sol r2 finding 4: [0,0],[4,4] carry no work; the idle gap before [10,11] is 10 ns.
        with tempfile.TemporaryDirectory() as tmp:
            out = C.run(self.write_csv(tmp, [('z', 0, 0), ('z', 4, 4), ('k', 10, 11)]))
        self.assertAlmostEqual(out['trace']['max_gap_us'], 0.010)
        self.assertAlmostEqual(out['trace']['gpu_busy_ms'], 1 / 1e6)
        self.assertEqual(out['dispatches'], 3)

    def test_nanosecond_timestamps_are_exact_integers(self):
        # Sol finding 6: timestamps above 2**53 ns must not round through float, markers included.
        n = 10 ** 16
        with tempfile.TemporaryDirectory() as tmp:
            trace = self.write_csv(tmp, [('k', n, n + 1), ('k', n + 1, n + 3)])
            fmt, records = C.load_trace(trace)
            self.assertEqual(records, [(n, n + 1, 'k'), (n + 1, n + 3, 'k')])
            markers = Path(tmp) / 'm.csv'
            markers.write_text('"Function","Start_Timestamp"\n"w",%d\n"w",%d\n' % (n, n + 1), encoding='utf-8')
            self.assertEqual(C.load_markers(markers, 'w'), [n, n + 1])
            out = C.run(trace, marker_file=markers, marker_regex='w')
        self.assertEqual([r['launches'] for r in out['windows']['per_window']], [1, 1])
        self.assertEqual(C.parse_ns('1.6E+4'), 16000)
        with self.assertRaisesRegex(ValueError, 'non-integer'):
            C.parse_ns('1.5')


if __name__ == '__main__':
    unittest.main(verbosity=2)
