"""Resident product-default campaign and its private receipt/record comparison."""
import argparse
from concurrent.futures import ThreadPoolExecutor
from contextlib import ExitStack, contextmanager
import csv
import hashlib
import json
import math
import os
from pathlib import Path
import re
import selectors
import shlex
import signal
import shutil
import socket
from statistics import median
import subprocess
import sys
import threading
import time
from urllib.request import ProxyHandler, Request, build_opener

from interface import prompt_set

MODELS = (
    'inkling2-mini', 'qwen3.8-flash-next', 'qwen3.8-27b', 'deepseek-v4.1-flash',
    'glm-5.3-flash', 'minimax-m3.1-flash', 'hy4-flash', 'nemotron-3.5-bolt-e4b',
)
HARDWARE = ('strix-one', 'strix-two', 'rtx')
TIERS = ('exact', 'fast', 'turbo')
PROMPTS = ('chat', 'code', 'prose')
CELLS = ('prefill-8192', 'prefill-32768', *('decode-' + p for p in PROMPTS), 'c2', 'c4', 'c8', 'c16')
MEASUREMENT_ORDER = (*CELLS[2:], *CELLS[:2])
LEDGER_FIELDS = ('model', 'hardware', 'tier', 'cell', 'status', 'value', 'unit',
                 'workload_sha256', 'source', 'workload')
GENERATION = 64
SHA256 = re.compile('[0-9a-f]{64}')
LOGITS = re.compile('(?:[0-9a-f]{16}|[0-9a-f]{64})')
EVIC_DIRECTORY = Path('/sys/class/kfd/kfd/proc')


def workload_hash(value):
    data = json.dumps(value, sort_keys=True, separators=(',', ':'), ensure_ascii=False)
    return hashlib.sha256(data.encode('utf-8')).hexdigest()


def positive(value):
    if type(value) not in (int, float) or not math.isfinite(value) or value <= 0:
        raise ValueError('expected a finite positive measurement')
    return value


def cell_key(row, historical=False):
    key = tuple(row[k] for k in LEDGER_FIELDS[:4])
    model, hardware, tier, cell = key
    if model not in MODELS or hardware not in HARDWARE or tier not in TIERS:
        raise ValueError(f'unknown campaign arm: {key}')
    if cell not in CELLS and not (historical and cell == 'decode-other'):
        raise ValueError(f'unknown campaign cell: {cell}')
    return key


def read_ledger(path):
    records = {}
    references = set()
    with path.open(encoding='utf-8-sig', newline='') as stream:
        reader = csv.DictReader(stream, delimiter='\t')
        if tuple(reader.fieldnames or ()) != LEDGER_FIELDS:
            raise ValueError('invalid ledger header')
        for row in reader:
            try:
                if None in row or any(value is None for value in row.values()):
                    raise ValueError('wrong number of TSV columns')
                kind = row['status']
                key = cell_key(row, historical=kind == 'historical')
                if row['unit'] != 'tok/s':
                    raise ValueError('expected tok/s')
                digest = row['workload_sha256']
                if kind == 'missing':
                    if row['value'] or digest:
                        raise ValueError('missing rows cannot have a number or workload hash')
                elif kind in ('historical', 'record'):
                    row['value'] = positive(float(row['value']))
                    if not row['source'].strip() or not row['workload'].strip():
                        raise ValueError('record needs a source line and workload description')
                    if kind == 'record':
                        if not SHA256.fullmatch(digest) or (*key, digest) in references:
                            raise ValueError('missing or duplicate comparable workload hash')
                        references.add((*key, digest))
                    elif digest:
                        raise ValueError('historical rows cannot assert a workload hash')
                else:
                    raise ValueError(f'unknown ledger status: {kind}')
                records.setdefault(key, []).append(row)
            except ValueError as exc:
                raise ValueError(f'ledger line {reader.line_num}: {exc}') from exc
    return records


def compare(ledger, receipts, arms):
    rows = {}
    for row in receipts:
        key = cell_key(row)
        if key[:2] not in arms or key in rows:
            raise ValueError(f'unexpected or duplicate default receipt: {key}')
        if row.get('value') is not None:
            positive(row['value'])
            if row.get('unit') != 'tok/s' or not SHA256.fullmatch(row.get('workload_sha256', '')):
                raise ValueError(f'invalid default workload identity: {key}')
        elif row.get('valid'):
            raise ValueError(f'valid receipt has no measurement: {key}')
        rows[key] = row

    print('model\thardware\tcell\ttier\tdefault_tok_s\trecord_tok_s\treached\tstatus\tsource\tpair_over_one')
    failed = False
    for model, hardware in sorted(arms):
        for cell in CELLS:
            previous = None
            for tier in TIERS:
                key = (model, hardware, tier, cell)
                row = rows.get(key, {})
                value = row.get('value')
                digest = row.get('workload_sha256')
                candidates = ledger.get(key, [])
                match = next((r for r in candidates if r['status'] == 'record' and r['workload_sha256'] == digest), None)
                context = next((r for r in candidates if r['status'] != 'missing'), None)
                record = match or context
                reached = bool(row.get('valid') and match and value >= match['value'])
                issues = []
                if not row.get('valid'):
                    issues.append('default-unavailable' if value is None else 'default-unverified')
                if not match:
                    issues.append('record-workload-unverified' if context else 'no-record')
                elif row.get('valid') and not reached:
                    issues.append('under-record')
                if row.get('valid') and previous:
                    previous_value, previous_digest = previous
                    if digest != previous_digest:
                        issues.append('tier-workload-mismatch')
                    elif value < previous_value:
                        issues.append('slower-than-lower-tier')
                previous = (value, digest) if row.get('valid') else None
                ratio = ''
                if hardware == 'strix-two' and row.get('valid'):
                    one = rows.get((model, 'strix-one', tier, cell), {})
                    if one.get('valid') and one['workload_sha256'] == digest:
                        ratio = f"{value / one['value']:.6g}"
                failed |= bool(issues)
                print(model, hardware, cell, tier, '' if value is None else f'{value:.9g}',
                      '' if record is None else f"{record['value']:.9g}", 'yes' if reached else 'no',
                      ','.join(issues) or 'ok', record['source'] if record else '', ratio, sep='\t')
    return int(failed)


def eviction(pid, counter):
    paths = [counter] if counter else sorted((EVIC_DIRECTORY / str(pid)).glob('stats_*/evicted_ms'))
    if not paths:
        raise ValueError('eviction counter unavailable')
    values = {p.name if counter else p.parent.name: int(p.read_text(encoding='ascii').strip()) for p in paths}
    if any(value < 0 for value in values.values()):
        raise ValueError('invalid negative eviction counter')
    return values


def eviction_clean(before, after):
    return bool(before and all(before) and before == after)


def default_command(binary, model, tier):
    return [binary, 'serve', model, '--profile', tier]


@contextmanager
def process(argv, log, pipe=False):
    with log.open('xb') as output:
        proc = subprocess.Popen(argv, stdin=subprocess.PIPE if pipe else subprocess.DEVNULL,
                                stdout=subprocess.PIPE if pipe else output, stderr=output,
                                start_new_session=True)
        try:
            yield proc
        finally:
            if pipe:
                proc.stdin.close()
            try:
                proc.wait(timeout=3 if pipe else 0)
            except subprocess.TimeoutExpired:
                proc.terminate() if pipe else os.killpg(proc.pid, signal.SIGTERM)
                try:
                    proc.wait(timeout=3)
                except subprocess.TimeoutExpired:
                    proc.kill() if pipe else os.killpg(proc.pid, signal.SIGKILL)
                    proc.wait()
            if pipe:
                proc.stdout.close()


def require_default_environment():
    overrides = sorted(key for key in os.environ if key.startswith(('STRIX_', 'ROCR_', 'HIP_', 'HSA_', 'CUDA_'))
                       or key == 'GPU_DEVICE_ORDINAL')
    if overrides:
        raise ValueError('start with a clean engine environment; found ' + ', '.join(overrides))


def worker(args):
    require_default_environment()
    # Keep the rank tied to the control channel. The engine writes a separate log on the peer.
    with process(default_command(args.strix, args.model, args.tier), args.log) as proc:
        for line in sys.stdin:
            if line.strip() != 'eviction':
                raise ValueError('unknown worker message')
            try:
                if proc.poll() is not None:
                    raise ValueError('peer product command exited')
                print(json.dumps({'counters': eviction(proc.pid, None)}), flush=True)
            except (ValueError, OSError) as exc:
                print(json.dumps({'error': str(exc)}), flush=True)


def peer_command(args, kit, model, tier):
    repo = args.peer_repo
    helper = (kit.relative_to(kit.parents[1]) / 'lib' / 'kit.py').as_posix()
    argv = ['python3', helper, 'campaign', 'worker', '--strix', args.peer_strix, '--model', model,
            '--tier', tier, '--log', f'{args.peer_logs}/{model}-{tier}.log']
    shell = 'cd ' + shlex.quote(repo) + ' && exec ' + shlex.join(argv)
    return ['ssh', '-o', 'BatchMode=yes', '-o', 'ConnectTimeout=10', '--', args.peer,
            'bash -c ' + shlex.quote(shell)]


def snapshot(local, peer, counter, deadline):
    if local.poll() is not None:
        raise ValueError('product command exited')
    result = [eviction(local.pid, counter)]
    if peer:
        peer.stdin.write(b'eviction\n')
        peer.stdin.flush()
        with selectors.DefaultSelector() as selector:
            selector.register(peer.stdout, selectors.EVENT_READ)
            if not selector.select(max(0, min(5, deadline - time.monotonic()))):
                raise ValueError('peer eviction counter timed out')
        response = json.loads(peer.stdout.readline())
        if 'error' in response:
            raise ValueError(response['error'])
        result.append(response['counters'])
    return result


def http(port, route, deadline, payload=None):
    remaining = deadline - time.monotonic()
    if remaining <= 0:
        raise TimeoutError('measurement window exceeded')
    data = json.dumps(payload).encode('utf-8') if payload is not None else None
    req = Request(f'http://127.0.0.1:{port}{route}', data, {'Content-Type': 'application/json'})
    with build_opener(ProxyHandler({})).open(req, timeout=remaining) as response:
        return json.load(response)


def token_vector(ids):
    if not isinstance(ids, list) or not ids or any(type(x) is not int or not 0 <= x < 2**32 for x in ids):
        raise ValueError('complete token vector unavailable')
    return ids


def generation_response(response, limit, id_hash):
    usage = response['usage']
    count = usage['completion_tokens']
    if type(count) is not int or not 0 <= count <= limit:
        raise ValueError('invalid completion token count')
    outcome = response['choices'][0]['finish_reason']
    if outcome not in ('stop', 'length') or (outcome == 'length' and count != limit):
        raise ValueError('incomplete generation')
    engine = response.get('strix', {})
    timings = response.get('timings', {})
    details = usage.get('prompt_tokens_details', {})
    if type(usage['prompt_tokens']) is not int or usage['prompt_tokens'] <= 0:
        raise ValueError('invalid prompt token count')
    ids = engine.get('token_ids', engine.get('generated_ids'))
    ids_sha = None
    if ids and len(token_vector(ids)) in ({count, count + 1} if outcome == 'stop' else {count}):
        ids_sha = id_hash(ids)
    data = {**timings, **engine}
    digests = {key: data[key] for key in ('logits_sha256', 'logits_digest', 'logits_fnv')
               if isinstance(data.get(key), str) and LOGITS.fullmatch(data[key])}
    return dict(completion_tokens=count, prompt_tokens=usage['prompt_tokens'],
                cached_tokens=details.get('cached_tokens'), imported_tokens=details.get('imported_tokens', 0),
                decode_ms=engine.get('decode_ms', timings.get('decode_ms')),
                prefill_ms=engine.get('prefill_ms', timings.get('prefill_ms')),
                ids_sha256=ids_sha, logits=digests, finish_reason=outcome)


def prefill_offsets(seed):
    offsets = {}
    for i, token in enumerate(seed):
        offsets.setdefault(token, i)
    if len(offsets) < 5:
        raise ValueError('prefill seed needs four distinct starts beyond the short chat prompt')
    return list(offsets.values())[1:5]


def cell_payloads(cell, prompts, seed=()):
    if cell.startswith('prefill-'):
        length = int(cell.split('-')[1])
        vectors = []
        offsets = prefill_offsets(seed)
        for offset in offsets[:2] if length == 8192 else offsets[2:]:
            rotated = seed[offset:] + seed[:offset]
            vectors.append((rotated * ((length + len(rotated) - 1) // len(rotated)))[:length])
        return '/v1/completions', [dict(prompt=ids, max_tokens=1) for ids in vectors]
    names = [cell.removeprefix('decode-')] if cell.startswith('decode-') else [
        PROMPTS[i % len(PROMPTS)] for i in range(int(cell[1:]))]
    return '/v1/chat/completions', [dict(messages=[dict(role='user', content=prompts[name])],
                                        max_tokens=GENERATION) for name in names]


def measure_cell(result, cell, prompts, seed, port, deadline, id_hash, counters):
    route, payloads = cell_payloads(cell, prompts, seed)
    for payload in payloads:
        payload.update(temperature=0, stream=False, return_token_ids=True)
    scope = ('engine-prefill' if cell.startswith('prefill-') else
             'engine-decode' if cell.startswith('decode-') else 'wall-complete-cohort')
    identity = workload_hash(dict(route=route, requests=payloads, scope=scope))
    result.update(workload_sha256=identity, unit='tok/s', scope=scope, requests=[], eviction=[])
    def request(payload):
        response = http(port, route, deadline, payload)
        parsed = generation_response(response, payload['max_tokens'], id_hash)
        parsed['raw'] = response
        return parsed
    def timed(requests, submit):
        before = counters()
        start = time.monotonic()
        replies = submit(requests)
        wall = time.monotonic() - start
        after = counters()
        result['eviction'].append(dict(before=before, after=after))
        result['requests'].extend(replies)
        return replies, wall
    if scope == 'wall-complete-cohort':
        with ThreadPoolExecutor(max_workers=len(payloads)) as pool:
            ready = threading.Barrier(len(payloads) + 1, timeout=max(0.1, deadline - time.monotonic()))
            def launch(payload):
                ready.wait()
                return request(payload)
            futures = [pool.submit(launch, payload) for payload in payloads]
            def submit(_):
                ready.wait()
                return [future.result() for future in futures]
            replies, wall = timed(payloads, submit)
        result['wall_s'] = wall
        result['value'] = positive(sum(r['completion_tokens'] for r in replies) / wall)
    elif scope == 'engine-prefill':
        rates = []
        for payload in payloads:
            replies, _ = timed([payload], lambda requests: list(map(request, requests)))
            reply = replies[0]
            if (reply['prompt_tokens'], reply['cached_tokens'], reply['imported_tokens']) != (len(payload['prompt']), 0, 0):
                raise ValueError('exact prefill row count or fresh-prefill evidence differs')
            rates.append(reply['prompt_tokens'] * 1000 / positive(reply['prefill_ms']))
        result.update(value=median(rates), repeat_values=rates)
    else:
        replies, _ = timed(payloads, lambda requests: list(map(request, requests)))
        reply = replies[0]
        result['value'] = positive((reply['completion_tokens'] - 1) * 1000 / positive(reply['decode_ms']))
    result['valid'] = True
    result['errors'] = []
    if not all(eviction_clean(row['before'], row['after']) for row in result['eviction']):
        result['errors'].append('eviction counter changed')
    if any(not row['ids_sha256'] or not row['logits'] for row in result['requests']):
        result['errors'].append('missing token IDs or native logits digest')
    if time.monotonic() > deadline:
        result['errors'].append('measurement window exceeded')
    result['valid'] = not result['errors']


def append_receipt(out, row):
    with (out / 'cells.jsonl').open('a', encoding='utf-8') as stream:
        stream.write(json.dumps(row, allow_nan=False) + '\n')


def run_tier(args, model, tier, prompts, out, local, peer, id_hash):
    start = time.monotonic() + args.startup_timeout
    while local.poll() is None:
        try:
            health = http(args.port, '/health', min(start, time.monotonic() + 1))
            if health.get('status') in ('ok', 'ready') or health.get('ok') is True:
                break
        except (OSError, ValueError):
            pass
        if time.monotonic() >= start:
            raise TimeoutError('resident server readiness timed out')
        time.sleep(0.25)
    else:
        raise ValueError('resident product route did not start; see server log')
    deadline = time.monotonic() + args.seconds
    seed = []
    seed_error = ''
    try:
        seed = token_vector(http(args.port, '/tokenize', deadline,
                                 {'messages': [dict(role='user', content=prompts['chat'])]})['tokens'])
        prefill_offsets(seed)
    except (ValueError, OSError, KeyError) as exc:
        seed_error = str(exc)
    def counters():
        try:
            return snapshot(local, peer, args.eviction_counter, deadline)
        except (ValueError, OSError) as exc:
            print(f'{model}/{tier}: {exc}', file=sys.stderr)
            return None
    rows = []
    # Short decode and cohorts get first use of the bounded window. Long prefill can consume the balance.
    for cell in MEASUREMENT_ORDER:
        row = dict(model=model, hardware=args.hardware, tier=tier, cell=cell, valid=False)
        try:
            if cell.startswith('prefill-') and seed_error:
                raise ValueError(seed_error)
            measure_cell(row, cell, prompts, seed, args.port, deadline, id_hash, counters)
        except (ValueError, KeyError, TypeError, IndexError, OSError) as exc:
            row['errors'] = [str(exc)]
        rows.append(row)
        append_receipt(out, row)
    return rows


def run(args, kit, id_hash, sha):
    models = args.model or MODELS
    hardware = (args.hardware,) if args.hardware else HARDWARE
    if args.dry_run:
        for hw in hardware:
            for model in models:
                for tier in TIERS:
                    print(f'[{hw} {tier}] {shlex.join(default_command(args.strix, model, tier))}')
                    if hw == 'strix-two':
                        print('  rank 1:', shlex.join(peer_command(args, kit, model, tier)))
                        print('  rank 1 product:', shlex.join(default_command(args.peer_strix, model, tier)))
                    print(f'  GET http://127.0.0.1:{args.port}/health; POST /tokenize for prefill seed')
                    for cell in MEASUREMENT_ORDER:
                        route, payloads = cell_payloads(cell, dict.fromkeys(PROMPTS), [1, 2, 3, 4, 5])
                        print(f'  {cell}: POST {route}, {len(payloads)} request(s), max_tokens={payloads[0]["max_tokens"]}')
        return 0
    if not args.hardware or not args.out or not args.ledger or not args.power_state:
        raise ValueError('a live campaign requires hardware, power state, private ledger and output paths')
    if not sys.platform.startswith('linux'):
        raise ValueError('resident capture and process cleanup require Linux')
    if args.hardware == 'strix-two' and args.peer == 'PEER':
        raise ValueError('two-Strix campaign requires an explicit peer')
    if args.strix.startswith('-') or args.peer.startswith('-') or args.peer_strix.startswith('-'):
        raise ValueError('invalid executable or peer')
    require_default_environment()
    if args.eviction_counter and args.hardware != 'rtx':
        raise ValueError('Strix eviction checks use the engine process KFD counter')
    root = kit.parents[1]
    for path in (args.out, args.ledger):
        if path.resolve().is_relative_to(root):
            raise ValueError('put private ledger and receipts outside the engine worktree')
    ledger = read_ledger(args.ledger)
    prompts = {row['id']: row['text'] for row in prompt_set('kit', kit) if row['id'] in PROMPTS}
    binary = shutil.which(args.strix)
    if not binary:
        raise ValueError('product launcher is unavailable')
    source = subprocess.check_output(['git', 'rev-parse', 'HEAD'], text=True).strip()
    args.out.mkdir(parents=True)
    arms = {(model, args.hardware) for model in models}
    launch_files = {model: Path('launch') / (model + '.toml') for model in models}
    (args.out / 'campaign.json').write_text(json.dumps(dict(
        source=source, binary_sha256=sha(Path(binary)), hardware=args.hardware,
        power_state=args.power_state, models=list(models), seconds=args.seconds, kernel=os.uname().release,
        launch_sha256s={m: sha(p) for m, p in launch_files.items() if p.is_file()},
        commands=[default_command(args.strix, model, tier) for model in models for tier in TIERS],
    ), indent=2) + '\n', encoding='utf-8')
    receipts = []
    for model in models:
        for tier in TIERS:
            try:
                # A pre-existing listener could serve a different model or profile.
                with socket.socket() as probe:
                    probe.bind(('127.0.0.1', args.port))
                with ExitStack() as stack:
                    prefix = args.out / f'{model}-{tier}'
                    peer = stack.enter_context(process(peer_command(args, kit, model, tier),
                                                       Path(str(prefix) + '.peer.log'), pipe=True)) if args.hardware == 'strix-two' else None
                    local = stack.enter_context(process(default_command(args.strix, model, tier),
                                                        Path(str(prefix) + '.server.log')))
                    receipts.extend(run_tier(args, model, tier, prompts, args.out, local, peer, id_hash))
            except (ValueError, OSError) as exc:
                print(f'{model}/{tier}: {exc}', file=sys.stderr)
                append_receipt(args.out, dict(model=model, hardware=args.hardware, tier=tier, launch_error=str(exc)))
    return compare(ledger, receipts, arms)


def stop_on_signal(number, _frame):
    # Unwind both process contexts so the rank and HTTP server do not outlive the job.
    raise SystemExit(128 + number)


def main(argv, kit, id_hash, sha):
    if argv and argv[0] == 'band-edges':
        import band_edges
        return band_edges.main(argv[1:], kit, id_hash, sys.modules[__name__])
    parser = argparse.ArgumentParser(prog='bench.sh campaign')
    parser.epilog = 'Use band-edges plan, sample or report for ladder timing.'
    actions = parser.add_subparsers(dest='action', required=True)
    default = actions.add_parser('default')
    default.add_argument('--dry-run', action='store_true')
    default.add_argument('--model', choices=MODELS, action='append')
    default.add_argument('--hardware', choices=HARDWARE)
    default.add_argument('--out', type=Path)
    default.add_argument('--ledger', type=Path)
    default.add_argument('--power-state')
    default.add_argument('--strix', default='strix')
    default.add_argument('--port', type=int, default=8080)
    default.add_argument('--seconds', type=int, default=120)
    default.add_argument('--startup-timeout', type=int, default=120)
    default.add_argument('--eviction-counter', type=Path)
    default.add_argument('--peer', default='PEER')
    default.add_argument('--peer-repo', default='REPO')
    default.add_argument('--peer-strix', default='strix')
    default.add_argument('--peer-logs', default='.')
    comparison = actions.add_parser('compare')
    comparison.add_argument('--ledger', type=Path, required=True)
    comparison.add_argument('receipts', type=Path, nargs='+', help='campaign output directories')
    rank = actions.add_parser('worker', help=argparse.SUPPRESS)
    rank.add_argument('--model', choices=MODELS, required=True)
    rank.add_argument('--tier', choices=TIERS, required=True)
    rank.add_argument('--strix', required=True)
    rank.add_argument('--log', type=Path, required=True)
    args = parser.parse_args(argv)
    if args.action == 'default':
        if not (0 < args.port <= 65535 and args.seconds > 0 and args.startup_timeout > 0):
            parser.error('port must be 1..65535 and timeouts must be positive')
        if args.model and len(set(args.model)) != len(args.model):
            parser.error('model must be selected at most once')
        if not args.dry_run:
            signal.signal(signal.SIGTERM, stop_on_signal)
        return run(args, kit, id_hash, sha)
    if args.action == 'worker':
        signal.signal(signal.SIGTERM, stop_on_signal)
        worker(args)
        return 0
    rows, arms = [], set()
    for directory in args.receipts:
        manifest = json.loads((directory / 'campaign.json').read_text(encoding='utf-8'))
        arms.update((model, manifest['hardware']) for model in manifest['models'])
        path = directory / 'cells.jsonl'
        if path.is_file():
            rows.extend(row for row in map(json.loads, path.read_text(encoding='utf-8').splitlines()) if 'cell' in row)
    return compare(read_ledger(args.ledger), rows, arms)
