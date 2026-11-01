#!/usr/bin/env python3
"""Dispatch census: rocprofv3 kernel trace -> JSON launch/busy/gap ledger (stdlib only, CPU only).

Reads the kernel trace that `rocprofv3 --kernel-trace` writes:
  * CSV  (`--output-format csv`, file `*_kernel_trace.csv`): columns Kernel_Name, Start_Timestamp,
    End_Timestamp (ns); older rocprof `results.csv` columns KernelName, BeginNs, EndNs are also accepted.
  * JSON (`--output-format json`, file `*_results.json`): `rocprofiler-sdk-tool[].buffer_records.kernel_dispatch`
    with names from `kernel_symbols`; a plain list of {"name","start","end"} records (ns) is also accepted.

Windows (one decode token, one MTP verify window, one benchmark call) are cut either
  * by a boundary kernel (`--window-kernel REGEX`): every matching dispatch starts a new window, or
  * by markers (`--markers marker_api_trace.csv --marker-regex REGEX`): every matching roctx marker/range start
    (rocprofv3 `--marker-trace`) starts a new window.
Dispatches before the first boundary are the prologue and are reported apart from the windows.

Per window: launches (by start time), span (window start to next window start, or to the trace end for the final window),
GPU busy (union of kernel intervals clipped to the window: carry-in work counts where it runs, concurrent streams are
not double counted), host gap = span - busy, largest idle gap including both window edges.
`--tokens N` (the engine-reported generated token count) adds launches and busy/gap time per token.
The output names only the trace basename and its SHA-256; no paths.
"""
import argparse
import bisect
import csv
from decimal import Decimal, InvalidOperation
import hashlib
import json
from pathlib import Path
import re
import statistics
import sys

SCHEMA = 'dispatch-census-v1'

NAME_KEYS = ('Kernel_Name', 'KernelName', 'kernel_name', 'Name', 'name')
START_KEYS = ('Start_Timestamp', 'BeginNs', 'start_timestamp', 'Start', 'start')
END_KEYS = ('End_Timestamp', 'EndNs', 'end_timestamp', 'End', 'end')
MARKER_NAME_KEYS = ('Function', 'Message', 'Marker_Name', 'Name', 'name', 'message')


def parse_ns(value):
    """Exact integer nanoseconds: no float round trip (timestamps exceed 2**53 ns after ~104 days of uptime)."""
    if isinstance(value, bool):
        raise ValueError('bad timestamp: ' + repr(value))
    if isinstance(value, int):
        return value
    text = str(value).strip()
    if re.fullmatch(r'-?[0-9]+', text):
        return int(text)
    try:
        exact = Decimal(text)
    except InvalidOperation:
        raise ValueError('bad timestamp: ' + repr(value)) from None
    if not exact.is_finite() or exact != exact.to_integral_value():
        raise ValueError('non-integer nanosecond timestamp: ' + repr(value))
    return int(exact)


def _pick(row, keys, what):
    for key in keys:
        if key in row and row[key] not in (None, ''):
            return row[key]
    raise ValueError('trace row lacks ' + what + ' (expected one of ' + ', '.join(keys) + ')')


def base_name(name):
    """Drop a trailing C++ parameter list `(...)`, keep template arguments."""
    name = name.strip()
    if name.endswith(')'):
        depth = 0
        for i in range(len(name) - 1, -1, -1):
            if name[i] == ')':
                depth += 1
            elif name[i] == '(':
                depth -= 1
                if depth == 0:
                    if i > 0:
                        name = name[:i].rstrip()
                    break
    return name


def _records_from_rows(rows):
    out = []
    for row in rows:
        start = parse_ns(_pick(row, START_KEYS, 'a start timestamp'))
        end = parse_ns(_pick(row, END_KEYS, 'an end timestamp'))
        if end < start:
            raise ValueError('kernel record ends before it starts')
        out.append((start, end, str(_pick(row, NAME_KEYS, 'a kernel name'))))
    return out


def _records_from_sdk_json(obj):
    tools = obj.get('rocprofiler-sdk-tool')
    if isinstance(tools, dict):
        tools = [tools]
    out = []
    for tool in tools:
        symbols = tool.get('kernel_symbols', [])
        if isinstance(symbols, dict):
            symbols = list(symbols.values())
        names = {}
        for s in symbols:
            kid = s.get('kernel_id')
            names[kid] = s.get('formatted_kernel_name') or s.get('kernel_name') or s.get('truncated_kernel_name') or str(kid)
        for rec in tool.get('buffer_records', {}).get('kernel_dispatch', []):
            info = rec.get('dispatch_info', {})
            kid = info.get('kernel_id', rec.get('kernel_id'))
            start, end = parse_ns(rec['start_timestamp']), parse_ns(rec['end_timestamp'])
            if end < start:
                raise ValueError('kernel record ends before it starts')
            out.append((start, end, names.get(kid, 'kernel_id:' + str(kid))))
    return out


def load_trace(path):
    """Return (format, [(start_ns, end_ns, name)]) sorted by start then end."""
    text = Path(path).read_text(encoding='utf-8', errors='replace')
    stripped = text.lstrip()
    if stripped.startswith(('{', '[')):
        obj = json.loads(text)
        if isinstance(obj, dict) and 'rocprofiler-sdk-tool' in obj:
            fmt, records = 'rocprofv3-json', _records_from_sdk_json(obj)
        elif isinstance(obj, list):
            fmt, records = 'records-json', _records_from_rows(obj)
        else:
            raise ValueError('unrecognised JSON trace (expected rocprofiler-sdk-tool or a record list)')
    else:
        fmt, records = 'rocprofv3-csv', _records_from_rows(csv.DictReader(text.splitlines()))
    if not records:
        raise ValueError('trace contains no kernel dispatches')
    records.sort(key=lambda r: (r[0], r[1]))
    return fmt, records


def load_markers(path, pattern):
    rx = re.compile(pattern)
    starts = []
    for row in csv.DictReader(Path(path).read_text(encoding='utf-8', errors='replace').splitlines()):
        name = str(_pick(row, MARKER_NAME_KEYS, 'a marker name'))
        if rx.search(name):
            starts.append(parse_ns(_pick(row, START_KEYS, 'a marker timestamp')))
    if not starts:
        raise ValueError('no marker matches ' + pattern)
    return sorted(starts)


def _bounded(intervals, lo, hi):
    """(busy_ns, max_idle_ns) of intervals already clipped to [lo, hi], counting leading/trailing idle;
    an empty window is one idle gap of its whole span. Zero-duration intervals carry no work and are ignored."""
    busy, worst, reach = 0, 0, lo
    for s, e in sorted(intervals):
        if e <= s:
            continue
        if s > reach:
            worst = max(worst, s - reach)
        if e > reach:
            busy += e - max(s, reach)
            reach = e
    worst = max(worst, hi - reach)
    return busy, worst


def _summary(values):
    if not values:
        return None
    return {'median': statistics.median(values), 'mean': statistics.fmean(values), 'min': min(values), 'max': max(values)}


def _segment(records, starts, include_boundary, boundary_rx):
    """Split records at the given window start timestamps -> (prologue, [(start_ns, next_start_ns|None, records)])."""
    prologue, windows = [], [[] for _ in starts]
    idx = -1
    for r in records:
        while idx + 1 < len(starts) and r[0] >= starts[idx + 1]:
            idx += 1
        if boundary_rx is not None and not include_boundary and boundary_rx.search(r[2]):
            continue
        (prologue if idx < 0 else windows[idx]).append(r)
    bounds = [(starts[i], starts[i + 1] if i + 1 < len(starts) else None, windows[i]) for i in range(len(starts))]
    return prologue, bounds


def top(records, limit):
    by = {}
    for s, e, name in records:
        c = by.setdefault(name, [0, 0])
        c[0] += 1
        c[1] += e - s
    total = sum(v[1] for v in by.values()) or 1
    rows = [{'name': n, 'count': v[0], 'total_us': v[1] / 1000, 'mean_us': v[1] / 1000 / v[0], 'time_share': v[1] / total}
            for n, v in by.items()]
    return (sorted(rows, key=lambda x: (-x['count'], x['name']))[:limit],
            sorted(rows, key=lambda x: (-x['total_us'], x['name']))[:limit])


def census(records, window_kernel=None, markers=None, tokens=None, include_boundary=False,
           names='base', top_n=15, drop_last=False):
    if names == 'base':
        records = [(s, e, base_name(n)) for s, e, n in records]
    boundary_rx = re.compile(window_kernel) if window_kernel else None
    result = {'schema': SCHEMA, 'dispatches': len(records)}
    trace_start, trace_end = records[0][0], max(e for _, e, _ in records)
    span = trace_end - trace_start
    busy, trace_gap = _bounded([(s, e) for s, e, _ in records], trace_start, trace_end)
    result['trace'] = {'span_ms': span / 1e6, 'gpu_busy_ms': busy / 1e6, 'host_gap_ms': (span - busy) / 1e6,
                       'gpu_busy_share': busy / span if span else None,
                       'host_gap_share': (span - busy) / span if span else None, 'max_gap_us': trace_gap / 1e3}
    measured = records
    if boundary_rx is not None or markers is not None:
        if markers is not None:
            starts, how = list(markers), 'markers'
        else:
            starts, how = [r[0] for r in records if boundary_rx.search(r[2])], 'boundary-kernel'
            if not starts:
                raise ValueError('no dispatch matches window kernel ' + window_kernel)
        prologue, bounds = _segment(records, starts, include_boundary, boundary_rx)
        # Busy time is the union of dispatch intervals intersected with each window, so work carried in from the
        # previous window (or the prologue) counts where it runs; launches stay attributed by start time.
        eligible = [r for r in records if include_boundary or boundary_rx is None or not boundary_rx.search(r[2])]
        # Window bounds come from every dispatch (boundary kernels included); the exclusion is for measured work only.
        edges = [(start, nxt if nxt is not None else max(trace_end, start)) for start, nxt, _ in bounds]
        lows = [lo for lo, _ in edges]
        clipped = [[] for _ in edges]
        for s_, e_, _ in eligible:
            i = max(bisect.bisect_right(lows, s_) - 1, 0)
            while i < len(edges) and edges[i][0] < e_:
                lo, hi = edges[i]
                if s_ < hi and e_ > lo and e_ > s_:
                    clipped[i].append((max(s_, lo), min(e_, hi)))
                i += 1
        if drop_last and bounds:
            bounds, edges, clipped = bounds[:-1], edges[:-1], clipped[:-1]
        rows, spans, busys = [], 0, 0
        for (lo, hi), (_, _, recs), parts in zip(edges, bounds, clipped):
            wspan = hi - lo
            wbusy, wgap = _bounded(parts, lo, hi)
            spans += wspan
            busys += wbusy
            rows.append({'launches': len(recs), 'span_ms': wspan / 1e6, 'gpu_busy_ms': wbusy / 1e6,
                         'host_gap_ms': (wspan - wbusy) / 1e6,
                         'host_gap_share': ((wspan - wbusy) / wspan) if wspan else None,
                         'max_gap_us': wgap / 1e3})
        measured = [r for _, _, recs in bounds for r in recs]
        spans, busys = spans / 1e6, busys / 1e6
        result['windows'] = {
            'cut_by': how, 'boundary': window_kernel if how == 'boundary-kernel' else None,
            'boundary_counted': bool(include_boundary) if how == 'boundary-kernel' else None,
            'count': len(rows), 'prologue_launches': len(prologue), 'last_dropped': bool(drop_last),
            'launches': _summary([r['launches'] for r in rows]),
            'span_ms': _summary([r['span_ms'] for r in rows]),
            'gpu_busy_ms': _summary([r['gpu_busy_ms'] for r in rows]),
            'host_gap_ms': _summary([r['host_gap_ms'] for r in rows]),
            'total_span_ms': spans, 'total_gpu_busy_ms': busys,
            'host_gap_share': (spans - busys) / spans if spans else None,
            'per_window': rows}
    if tokens is not None:
        if tokens <= 0:
            raise ValueError('--tokens must be positive')
        scope = result.get('windows')
        wall = scope['total_span_ms'] if scope else span / 1e6
        gpu = scope['total_gpu_busy_ms'] if scope else busy / 1e6
        result['per_token'] = {'tokens': tokens, 'source': 'caller (engine-reported count)',
                               'launches': len(measured) / tokens, 'span_ms': wall / tokens,
                               'gpu_busy_ms': gpu / tokens, 'host_gap_ms': (wall - gpu) / tokens}
    by_count, by_time = top(measured, top_n)
    result['top_by_count'] = by_count
    result['top_by_time'] = by_time
    result['distinct_kernels'] = len({n for _, _, n in measured})
    return result


def run(trace, **options):
    marker_file = options.pop('marker_file', None)
    marker_regex = options.pop('marker_regex', None)
    fmt, records = load_trace(trace)
    markers = load_markers(marker_file, marker_regex or '.') if marker_file else None
    out = census(records, markers=markers, **options)
    digest = hashlib.sha256(Path(trace).read_bytes()).hexdigest()
    out['source'] = {'format': fmt, 'trace': Path(trace).name, 'trace_sha256': digest}
    if marker_file:
        out['source']['markers'] = Path(marker_file).name
        out['source']['marker_regex'] = marker_regex
    return out


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    ap.add_argument('trace', help='rocprofv3 kernel trace (CSV or JSON)')
    ap.add_argument('--window-kernel', help='regex: each matching dispatch starts a window')
    ap.add_argument('--include-boundary', action='store_true', help='count the boundary kernel inside its window')
    ap.add_argument('--markers', help='rocprofv3 marker_api_trace.csv; matching markers start windows')
    ap.add_argument('--marker-regex', default='.', help='regex on the marker name (default: every marker)')
    ap.add_argument('--drop-last', action='store_true', help='drop the final (possibly partial) window')
    ap.add_argument('--tokens', type=int, help='engine-reported token count covered by the measured windows/trace')
    ap.add_argument('--names', choices=('base', 'full'), default='base', help='group by name without (args) or full name')
    ap.add_argument('--top', type=int, default=15)
    ap.add_argument('-o', '--output', help='write JSON here (default stdout)')
    a = ap.parse_args(argv)
    if a.window_kernel and a.markers:
        ap.error('use --window-kernel or --markers, not both')
    try:
        result = run(a.trace, window_kernel=a.window_kernel, marker_file=a.markers, marker_regex=a.marker_regex,
                     tokens=a.tokens, include_boundary=a.include_boundary, names=a.names, top_n=a.top,
                     drop_last=a.drop_last)
    except (ValueError, KeyError, OSError, re.error) as exc:
        print('dispatch_census: ' + str(exc), file=sys.stderr)
        return 1
    text = json.dumps(result, indent=2, ensure_ascii=True) + '\n'
    if a.output:
        Path(a.output).write_text(text, encoding='utf-8', newline='\n')
    else:
        sys.stdout.write(text)
    return 0


if __name__ == '__main__':
    sys.exit(main())
