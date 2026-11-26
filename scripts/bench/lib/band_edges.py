"""Plan, capture and compare bounded ABBA prefill samples from the ladder inventory."""

import argparse
from contextlib import contextmanager
import hashlib
import json
import os
from pathlib import Path
import re
import shlex
import signal
import statistics
import subprocess
import time

ABBA = 'ABBA'
LIMIT = 115  # Five more seconds for termination, within the two-minute turn.
D1 = (
    'STRIX_DS41_FAST_ATTN=2', 'STRIX_DS41_FAST_DENSE=1', 'STRIX_DS41_FAST_MOE=1',
    'STRIX_DS41_FAST_MOE_F16=1', 'STRIX_DS41_FAST_ENGRAM=1', 'STRIX_DS41_FAST_SHARED=1',
    'STRIX_DS41_FAST_I4=0',
)


def load_inventory(kit):
    data = json.loads((kit.parents[1] / 'docs/band-edges.json').read_text(encoding='utf-8'))
    if data['schema'] != 'strix-band-edges/v1':
        raise ValueError('unknown band inventory schema')
    rows = {f"{r['engine']}/{r['lever']}/{r['band'][0]}": r for r in data['rows'] if r['band_bound']}
    if len(rows) != sum(r['band_bound'] for r in data['rows']):
        raise ValueError('duplicate band selector')
    return rows


def points(row):
    lo, hi = row['band']
    return tuple(dict.fromkeys((lo, (lo + hi) // 2, hi)))


def recipe(row, side, rank, port, capacity):
    """The candidate executes the measured increment over its reference composition."""
    engine = row['engine']
    upper = side == 'B'
    if engine == 'deepseek41-pair':
        replay = row['lever'] == 'decoder_replay'
        d1 = replay or upper
        overrides = D1 if d1 else ()
        overrides += (f'STRIX_DS41_DECODER_REPLAY={256 if replay and upper else 0}',)
        return ['env', *overrides, 'strix-deepseek41-serve', '$BAND_MODEL', '--rank', str(rank),
                '--transport', 'usb4auto', '--profile', 'fast' if d1 else 'exact',
                '--slots', '1', '--context', str(capacity), '--cache-profile', 'off',
                *(['--http-bind', '127.0.0.1', '--http-port', str(port)] if rank == 0 else [])]
    if engine == 'glm-pair':
        tier = row['tier'] if upper else row['reference_tier']
        overrides = ['env', f'STRIX_GLM_PF_WMMA={int(upper)}']
        return [*overrides, 'strix-glm5-serve', '$BAND_MODEL', '--world', '2', '--rank', str(rank),
                '--transport', 'usb4-auto', '--profile', tier, '--mtp-spec', '1',
                '--slots', '1',
                '--capacity', str(capacity), '--max-rows', '8192', '--prefix-cache-gib', '0',
                *(['--host', '127.0.0.1', '--port', str(port)] if rank == 0 else [])]
    if engine == 'inkling2-mini-hip':
        return ['env', f'STRIX_INK_MOE_W4A4={"iu4" if upper else "0"}', 'strix-serve',
                '--model', row['model'], '--profile', 'fast', '--bind', '127.0.0.1',
                '--port', str(port), '--context', str(capacity), '--slots', '1', '--',
                '--checkpoint', '$BAND_MODEL', '--tokenizer', '$BAND_TOKENIZER', '--backend', 'hip']
    if engine == 'qwen-hip':
        # Keep the FAST chain fixed. The TURBO row prices its cap9 increment.
        return ['env', 'STRIX_MTP=0', f'STRIX_QPF2K_MOE_ROUTE_PRUNE={"cap9" if upper else "off"}',
                'strix-hip-decode-token', '$BAND_MODEL', '--profile', 'fast', '--http-serve',
                '--http-bind', '127.0.0.1', '--http-port', str(port), '--http-batch', '0',
                '--context', str(capacity), '--prefix-cache', 'off', '--session-cache', 'off']
    if engine == 'qwen35-hip-decode':
        return ['strix-hip-qwen35-decode', '--model', '$BAND_MODEL', '--profile', row['tier'] if upper else 'exact',
                '--prompt-ids', '@$PROMPT_FILE', '--no-mtp', '--gen', '1', '--max-pos', str(capacity),
                '--wide' if upper else '--wide-exact', '512' if upper else '64']
    raise ValueError(f'no fixture for {engine}')


def is_pair(row):
    return row['engine'] in ('deepseek41-pair', 'glm-pair')


def is_cli(row):
    return row['engine'] == 'qwen35-hip-decode'


def capacity(row):
    return row['band'][1] + 256


def command(words):
    # Only the recipe's fixed shell variables expand. Quote file names and selectors literally.
    return ' '.join(f'"{w}"' if w in ('$BAND_MODEL', '$BAND_TOKENIZER', '$PROMPT_FILE', '@$PROMPT_FILE',
                                     '$BAND_DEVICE_PRODUCT', '$BAND_BUILD_ID', '$BAND_WEIGHTS_ID',
                                     '$BAND_POWER_STATE', '$SERVER_PID') else shlex.quote(w) for w in words)


def plan(inventory, selected, port, out):
    prefix = ['bash', 'scripts/bench/bench.sh', 'campaign', 'band-edges']
    if not selected:
        selected = inventory
    for key in selected:
        row = inventory[key]
        reference = 'D1 without replay' if row['lever'] == 'decoder_replay' else row['reference_tier'].upper()
        print(f"\n# {key}: {row['tier'].upper()} increment over {reference}")
        print(f'# Resident requests or native CLI turns: {LIMIT}s plus termination per sample.')
        if is_cli(row) and row['band'][1] == 24576:
            print('# The historical 24K EXACT control exceeds this limit. A timeout supplies no timing evidence.')
        for n in points(row):
            labels = [label for label, point in zip(('lower', 'middle', 'upper'),
                      (row['band'][0], sum(row['band']) // 2, row['band'][1])) if point == n]
            print(f"# {', '.join(labels)}: {n} prompt tokens")
            for step, side in enumerate(ABBA, 1):
                print(f'# {step}: {side}')
                if is_cli(row):
                    native = recipe(row, side, 0, port, capacity(row))
                    native = [f'@prompts/{n}.ids' if word == '@$PROMPT_FILE' else word for word in native]
                    print('# Native command under the sample deadline: ' + command(native))
                else:
                    if is_pair(row):
                        print('# Start on rank 1:')
                        print(command(recipe(row, side, 1, port, capacity(row))))
                    print('# Start on rank 0 (or the sole node), then wait for readiness:')
                    print(command(recipe(row, side, 0, port, capacity(row))) + ' & SERVER_PID=$!')
                    print(f'# Wait for curl -fsS http://127.0.0.1:{port}/health to report ready.')
                print(command([*prefix, 'sample', '--row', key, '--length', str(n), '--step', str(step),
                               '--port', str(port), '--out', str(out), '--prompt', f'prompts/{n}.ids',
                               '--device-product', '$BAND_DEVICE_PRODUCT', '--build-id', '$BAND_BUILD_ID',
                               '--weights-id', '$BAND_WEIGHTS_ID', '--power-state', '$BAND_POWER_STATE',
                               '--model', '$BAND_MODEL',
                               *(['--tokenizer', '$BAND_TOKENIZER'] if row['engine'] == 'inkling2-mini-hip' else []),
                               *([] if is_cli(row) else ['--server-pid', '$SERVER_PID'])]))
                if not is_cli(row):
                    print('kill -TERM "$SERVER_PID" 2>/dev/null || true; wait "$SERVER_PID" || true')
                    if is_pair(row):
                        print('# Stop rank 1 before starting the next sample.')
    print(command([*prefix, 'report', '--out', str(out),
                   *[word for key in selected for word in ('--row', key)]]))


def read_ids(path, count):
    parts = re.split(r'[,\s]+', path.read_text(encoding='utf-8').strip())
    ids = [int(part) for part in parts]
    if len(ids) != count or any(not 0 <= token < 2**32 for token in ids):
        raise ValueError(f'{path}: expected exactly {count} unsigned token IDs')
    return ids


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write(path, value):
    with path.open('x', encoding='utf-8') as stream:
        json.dump(value, stream, indent=2, allow_nan=False)
        stream.write('\n')


@contextmanager
def timer(seconds):
    def expired(_signum, _frame):
        raise TimeoutError('band sample deadline exceeded')

    previous = signal.signal(signal.SIGALRM, expired)
    signal.alarm(seconds)
    try:
        yield
    finally:
        signal.alarm(0)
        signal.signal(signal.SIGALRM, previous)


def verify_resident(pid, argv):
    # Verify the lead process actually uses the planned arm. Avoid recording ambient environment contents.
    proc = Path('/proc') / str(pid)
    actual = os.fsdecode((proc / 'cmdline').read_bytes()).rstrip('\0').split('\0')
    overrides = {}
    if argv[0] == 'env':
        argv = argv[1:]
        while '=' in argv[0]:
            key, value = argv[0].split('=', 1)
            overrides[key] = value
            argv = argv[1:]
    environment = dict(entry.split('=', 1) for entry in
                       os.fsdecode((proc / 'environ').read_bytes()).split('\0') if '=' in entry)
    if Path(actual[0]).name != argv[0] or actual[1:] != argv[1:] or any(
            environment.get(key) != value for key, value in overrides.items()):
        raise ValueError('resident process does not match the planned command and overrides')
    ambient = sorted((key, value) for key, value in environment.items()
                     if key.startswith(('STRIX_', 'HIP_', 'HSA_', 'ROCR_')) and key not in overrides)
    return hashlib.sha256(json.dumps(ambient).encode('utf-8')).hexdigest()


def native_ms(raw, count):
    timing = re.findall(r'^prompt (\d+) tokens: (\d+(?:\.\d+)?) ms$', raw, re.MULTILINE)
    if len(timing) != 1 or int(timing[0][0]) != count:
        raise ValueError('missing or mismatched native prefill timing')
    return float(timing[0][1])


def server_ms(response, count, gen, campaign, id_hash):
    reply = campaign.generation_response(response, gen, id_hash)
    if (reply['prompt_tokens'], reply['cached_tokens'], reply['imported_tokens']) != (count, 0, 0):
        raise ValueError('resident sample must account for a fresh exact-length prompt')
    return campaign.positive(reply['prefill_ms'])


def sample(args, row, campaign, id_hash):
    if args.length not in points(row):
        raise ValueError('expected a nominal lower, middle or upper prompt length')
    if not args.device_product.startswith(row['device']) or not all(
            value.strip() for value in (args.power_state, args.build_id, args.weights_id)):
        raise ValueError('record the device product, power state, build and weights identity')
    ids = read_ids(args.prompt, args.length)
    directory = args.out / args.row / str(args.length)
    directory.mkdir(parents=True, exist_ok=True)
    out = directory / f'{args.step}.json'
    if out.exists() or any(not (directory / f'{step}.json').is_file() for step in range(1, args.step)):
        raise ValueError('capture samples once and in ABBA order')
    side = ABBA[args.step - 1]
    argv = recipe(row, side, 0, args.port, capacity(row))
    if '$BAND_TOKENIZER' in argv and args.tokenizer is None:
        raise ValueError('--tokenizer is required for Inkling')
    substitutions = {'$BAND_MODEL': str(args.model), '$BAND_TOKENIZER': str(args.tokenizer)}
    argv = [substitutions.get(word, word) for word in argv]
    record = dict(row=args.row, step=args.step, side=side, length=args.length, prompt_sha256=id_hash(ids),
                  build=args.build_id, weights=args.weights_id, device=args.device_product, power_state=args.power_state,
                  commit=subprocess.check_output(['git', 'rev-parse', 'HEAD'], text=True).strip(), argv=argv, capacity=capacity(row), generation=1,
                  scope='native-prefill' if is_cli(row) else 'resident-prefill')
    start = time.monotonic()
    try:
        if is_cli(row):
            fixture = out.with_suffix('.ids')
            fixture.write_text(','.join(map(str, ids)) + '\n', encoding='ascii')
            actual = ['@' + str(fixture) if word == '@$PROMPT_FILE' else word for word in argv]
            raw = directory / f'{args.step}.log'
            with raw.open('xb') as log:
                subprocess.run(actual, stdout=log, stderr=subprocess.STDOUT, timeout=LIMIT, check=True)
            elapsed = native_ms(raw.read_text(encoding='utf-8', errors='replace'), args.length)
        else:
            if args.server_pid is None or args.server_pid <= 1:
                raise ValueError('--server-pid must name the local resident server process')
            record['resident_environment'] = verify_resident(args.server_pid, argv)
            raw = directory / f'{args.step}.http.json'
            with timer(LIMIT):
                response = campaign.http(args.port, '/v1/completions', start + LIMIT,
                                         dict(prompt=ids, max_tokens=1, temperature=0, stream=False,
                                              return_token_ids=True))
            write(raw, response)
            elapsed = server_ms(response, args.length, 1, campaign, id_hash)
            record['server_pid'] = args.server_pid
        record.update(prefill_ms=campaign.positive(elapsed), raw=raw.name, raw_sha256=digest(raw))
    except (TimeoutError, subprocess.TimeoutExpired, OSError, ValueError, KeyError, subprocess.CalledProcessError) as exc:
        if isinstance(exc, TimeoutError) and not is_cli(row) and args.server_pid and args.server_pid > 1:
            # A disconnected client may leave device work running. Abort the resident process on expiry.
            try:
                os.kill(args.server_pid, signal.SIGKILL)
            except ProcessLookupError:
                pass
        record['error'] = str(exc)
    record['wall_seconds'] = time.monotonic() - start
    write(out, record)
    return int('error' in record)


def check_point(path, key, row, n, campaign, id_hash):
    rows = [json.loads((path / f'{i}.json').read_text(encoding='utf-8')) for i in range(1, 5)]
    context = [(r['prompt_sha256'], r['device'], r['power_state'], r['build'], r['weights'], r['commit'],
                r['capacity'], r['generation'], r['scope'], r.get('resident_environment')) for r in rows]
    if len(set(context)) != 1:
        raise ValueError('mixed prompt, hardware, power, build or workload in ABBA block')
    values = []
    for step, (side, r) in enumerate(zip(ABBA, rows), 1):
        if (r['row'], r['length'], r['step'], r['side']) != (key, n, step, side) or 'error' in r:
            raise ValueError(f'missing valid {side} sample {step}: {r.get("error", "wrong metadata")}')
        if not 0 < r['wall_seconds'] <= 120:
            raise ValueError('sample exceeded the two-minute limit')
        raw = path / (f'{step}.log' if is_cli(row) else f'{step}.http.json')
        if raw.name != r['raw'] or digest(raw) != r['raw_sha256']:
            raise ValueError('raw sample hash mismatch')
        text = raw.read_text(encoding='utf-8', errors='replace')
        observed = native_ms(text, n) if is_cli(row) else server_ms(json.loads(text), n, 1, campaign, id_hash)
        if observed != campaign.positive(r['prefill_ms']):
            raise ValueError('raw timing differs from sample receipt')
        values.append(observed)
    a0, b0, b1, a1 = values
    ratios = [a0 / b0, a1 / b1]
    return dict(tokens=n, samples=rows, prefill_ms=dict(A=[a0, a1], B=[b0, b1]),
                paired_speedup=ratios, speedup=statistics.median([a0, a1]) / statistics.median([b0, b1]),
                wins=min(ratios) > 1 + row['noise'])


def verdict(measurements, lo, hi):
    if any('error' in point for point in measurements):
        return 'INCOMPLETE'
    floor = hi + 1
    for point in reversed(measurements):
        if not point['wins']:
            break
        floor = point['tokens']
    return 'KEEP' if floor == lo else f'RAISE-LOWER-EDGE-TO {floor}'


def report(inventory, selected, out, campaign, id_hash):
    result = []
    for key in selected or inventory:
        row = inventory[key]
        measurements = []
        for n in points(row):
            try:
                measurements.append(check_point(out / key / str(n), key, row, n, campaign, id_hash))
            except (OSError, ValueError, KeyError, TypeError) as exc:
                measurements.append(dict(tokens=n, error=str(exc)))
        configs = {(r['build'], r['weights'], r['commit'], r['device'], r['power_state'])
                   for p in measurements for r in p.get('samples', [])}
        lo, hi = row['band']
        decision = 'INCOMPLETE' if len(configs) != 1 else verdict(measurements, lo, hi)
        result.append(dict(row=key, ladder=row, order=ABBA, timings=measurements, verdict=decision))
    print(json.dumps(dict(schema='strix-band-timing/v1', receipts=result), indent=2, allow_nan=False))
    return int(any(r['verdict'] == 'INCOMPLETE' for r in result))


def main(argv, kit, id_hash, campaign):
    inventory = load_inventory(kit)
    parser = argparse.ArgumentParser(prog='bench.sh campaign band-edges')
    actions = parser.add_subparsers(dest='action', required=True)
    for name in ('plan', 'report'):
        action = actions.add_parser(name)
        action.add_argument('--row', choices=inventory, action='append')
        action.add_argument('--out', type=Path, default=Path('band-edge-receipts'))
        if name == 'plan':
            action.add_argument('--port', type=int, default=18080)
    capture = actions.add_parser('sample')
    capture.add_argument('--row', choices=inventory, required=True)
    capture.add_argument('--step', type=int, choices=range(1, 5), required=True)
    capture.add_argument('--length', type=int, required=True)
    capture.add_argument('--out', type=Path, required=True)
    capture.add_argument('--prompt', type=Path, required=True)
    capture.add_argument('--port', type=int, default=18080)
    capture.add_argument('--server-pid', type=int)
    capture.add_argument('--model', required=True)
    capture.add_argument('--tokenizer')
    for arg in ('device-product', 'power-state', 'build-id', 'weights-id'):
        capture.add_argument('--' + arg, required=True)
    args = parser.parse_args(argv)
    if args.action != 'report' and not 0 < args.port <= 65535:
        parser.error('port must be 1..65535')
    if args.action == 'plan':
        plan(inventory, args.row, args.port, args.out)
        return 0
    if args.action == 'report':
        return report(inventory, args.row, args.out, campaign, id_hash)
    try:
        return sample(args, inventory[args.row], campaign, id_hash)
    except (OSError, ValueError) as exc:
        parser.error(str(exc))
