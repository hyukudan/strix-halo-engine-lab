#!/usr/bin/env python3
"""Prepare/run the native visibility profile and compare two complete receipts.

No SSH setup, driver configuration, reset, installation or qualification writes.
The operator supplies exact observed identities and establishes the SSH tunnel.
"""
import argparse
import json
from pathlib import Path
import shlex
import subprocess
import sys

PROFILE = 'gpu-normal-loads-v1'
MASK = 0xa6d2b8439f5071ce
MAX_JSON = 16 * 1024 * 1024
VALUE_KEYS = {
    'campaign-id', 'run-id', 'role', 'dma-device', 'local-endpoint', 'peer-endpoint',
    'physical-link', 'kernel-release', 'driver-build-id', 'gpu-uuid',
    'gpu-runtime-version', 'gpu-driver-version', 'epoch', 'request-id',
    'payload-bytes', 'control-mode', 'control-port', 'gpu-device', 'ring-frames',
    'seed', 'timeout-ms', 'maximum-polls', 'gpu-timeout-ms', 'gpu-maximum-polls',
    'gpu-pause-ms', 'model-plan-id', 'generation', 'boundary-id', 'dtype-id',
    'sequence', 'iterations', 'payload-sizes', 'producer-delay-ms', 'consumer-delay-ms',
}
FLAGS = {'alternate-directions', 'duplex-pairs'}


def require(condition, message):
    if not condition:
        raise ValueError(message)


def unique_object(pairs):
    result = {}
    for key, value in pairs:
        require(key not in result, f'duplicate JSON key: {key}')
        result[key] = value
    return result


def load(path):
    with Path(path).open('rb') as stream:
        data = stream.read(MAX_JSON + 1)
    require(len(data) <= MAX_JSON, 'JSON exceeds 16 MiB')
    result = json.loads(data, object_pairs_hook=unique_object)
    require(isinstance(result, dict), 'JSON must be an object')
    return result


def arguments(config):
    require(config.get('gpu-visibility') is True, 'gpu-visibility must be true')
    require(not set(config) - VALUE_KEYS - FLAGS - {'gpu-visibility'}, 'unknown configuration key')
    args = ['--experimental-commissioning', '--gpu-visibility']
    for key, value in config.items():
        if key == 'gpu-visibility':
            continue
        if key in FLAGS:
            require(type(value) is bool, f'{key} must be boolean')
            if value:
                args.append('--' + key)
        else:
            require(type(value) in (str, int) and str(value) != '', f'invalid {key}')
            args.extend(['--' + key, str(value)])
    return args


def checked_receipt(record):
    require(record.get('production_qualified') is False, 'unexpected qualification declaration')
    require(record.get('cleanup') == 'released' and record.get('error') == '', 'failed or retained run')
    if record.get('schema') == 'strix-direct-gpu-commissioning-session-v2':
        rows = record.get('transfers')
        require(isinstance(rows, list) and 2 <= len(rows) <= 128, 'missing/bad session transfers')
        require(record.get('requested_transfers') == len(rows) == record.get('completed_count'), 'partial session')
        require(record.get('schedule') in ('sequential', 'alternating', 'duplex_pairs'), 'unknown schedule')
        require(record.get('concurrency') == (2 if record['schedule'] == 'duplex_pairs' else 1), 'unexpected concurrency')
    else:
        require(record.get('schema') == 'strix-direct-gpu-commissioning-v1', 'not an executed native receipt')
        rows = [record]
    nonces = set()
    local_identity = ('device_identity', 'device_instance_generation', 'gpu_uuid',
                      'kernel_identity', 'driver_build_id', 'connection_epoch',
                      'pool_binding_identity', 'import_generation',
                      'local_endpoint_identity', 'peer_endpoint_identity')
    for index, row in enumerate(rows):
        require(isinstance(row, dict), 'invalid transfer')
        require(row.get('schema') == ('strix-direct-gpu-commissioning-transfer-v2' if len(rows) > 1 else 'strix-direct-gpu-commissioning-v1'), 'wrong transfer schema')
        require(row.get('sequence') == index + 1 and row.get('completed_count') == 1, 'missing/unordered transfer')
        require(row.get('production_qualified') is False and row.get('executed') is True, 'transfer did not execute')
        require(row.get('cleanup') == ('import_held' if len(rows) > 1 else 'released') and row.get('error') == '', 'transfer failed/retained')
        require(row.get('gpu_visibility_profile') == PROFILE and row.get('gpu_verifier_self_test') is True, 'GPU profile/self-test absent')
        for name in ('controller_frames_complete', 'consumer_credit'):
            require(row.get(name) is True, f'incomplete {name}')
        for name in ('connection_epoch', 'pool_binding_identity', 'import_generation', 'device_instance_generation', 'payload_bytes'):
            require(type(row.get(name)) is int and row[name] > 0, f'missing {name}')
        for name in ('campaign_id', 'run_id', 'device_identity', 'gpu_uuid', 'kernel_identity', 'driver_identity', 'driver_build_id', 'abi_identity', 'local_endpoint_identity', 'peer_endpoint_identity', 'physical_link_id'):
            require(isinstance(row.get(name), str) and row[name], f'missing {name}')
        require('observed_payload_sha256' in row and row['observed_payload_sha256'] is None, 'GPU-only profile must not manufacture a payload readback hash')
        if index:
            require(all(row[key] == rows[0][key] for key in local_identity), 'session identity changed')
        if len(rows) > 1:
            require(row['pool_binding_identity'] == record.get('pool_binding_identity') and
                    row['import_generation'] == record.get('import_generation'), 'session import mismatch')
        require(row.get('role') in ('sender', 'receiver'), 'unknown role')
        if row['role'] == 'receiver':
            for name in ('gpu_acquire_complete', 'consumer_complete', 'receive_reposted', 'full_payload_verified'):
                require(row.get(name) is True, f'incomplete {name}')
            for name, stage in (('raw_rx_gpu', 3), ('consumer_gpu', 4)):
                observation = row.get(name)
                require(isinstance(observation, dict), f'missing {name}')
                for key in ('nonce', 'started', 'finished', 'epoch', 'sequence', 'stage', 'checked_bytes', 'mismatches'):
                    require(type(observation.get(key)) is int, f'invalid GPU counter: {key}')
                nonce = observation.get('nonce')
                require(type(nonce) is int and 3 < nonce < 2**64 and nonce not in nonces, 'stale/duplicate GPU result')
                nonces.add(nonce)
                require(observation.get('executed') is True and observation.get('passed') is True, f'failed {name}')
                require(observation.get('started') == nonce and observation.get('finished') == nonce ^ MASK, 'unfinished GPU result')
                require(observation.get('epoch') == row['connection_epoch'] and observation.get('sequence') == row['sequence'] and observation.get('stage') == stage, 'foreign GPU result')
                require(observation.get('checked_bytes') == row['payload_bytes'] and observation.get('mismatches') == 0 and observation.get('first_bad') is None, 'partial or mismatched GPU payload')
        else:
            require(row.get('local_tx_complete') is True and row.get('transport_ack') is True, 'sender lacks local/remote completion')
            for name in ('raw_rx_gpu', 'consumer_gpu'):
                require(isinstance(row.get(name), dict) and row[name].get('executed') is False and row[name].get('passed') is False, 'sender claims unexecuted receiver work')
    return rows


def compare(a, b):
    left, right = checked_receipt(a), checked_receipt(b)
    require(len(left) == len(right), 'different transfer counts')
    shared = ('campaign_id', 'run_id', 'sequence', 'connection_epoch', 'physical_link_id',
              'payload_bytes', 'pattern_seed', 'pattern_version', 'model_plan_id',
              'request_id', 'generation', 'boundary_id', 'shape_fingerprint', 'dtype_id',
              'expected_payload_sha256', 'gpu_visibility_profile', 'producer_delay_ms', 'consumer_delay_ms')
    if len(left) > 1:
        require(a['schedule'] == b['schedule'], 'different schedules')
    total = 0
    for x, y in zip(left, right):
        require(x['role'] != y['role'], 'roles are not complementary')
        for key in shared:
            require(key in x and key in y and x[key] == y[key], f'peer disagreement: {key}')
        require(x['local_endpoint_identity'] == y['peer_endpoint_identity'] and x['peer_endpoint_identity'] == y['local_endpoint_identity'], 'wrong endpoint pair')
        require(x['local_endpoint_identity'] != x['peer_endpoint_identity'] and x['gpu_uuid'] != y['gpu_uuid'], 'not distinct GPU endpoints')
        total += x['payload_bytes']
    return {'schema': 'strix-gpu-visibility-comparison-v1', 'record_comparison_passed': True,
            'transfers': len(left), 'receiver_verified_bytes': total, 'production_qualified': False,
            'scope': 'Recorded sequential/paired-duplex GPU visibility only; no model or eight-slot qualification.'}


def prepare_config(gpu, driver, args):
    require(gpu.get('schema') == 'strix-direct-gpu-device-observation-v1' and
            gpu.get('gpu_architecture') == 'gfx1151', 'incompatible GPU observation')
    for field in ('qualified', 'production_qualified', 'native_ready', 'executed_payload', 'allocation_performed', 'export_performed'):
        require(gpu.get(field) is False, f'not a metadata-only GPU observation: {field}')
    require(driver.get('schema') == 'strix-direct-gpu-driver-observation-v1' and
            driver.get('query_observed') is True and driver.get('status') == 'observed' and
            driver.get('fd_retained') is False and driver.get('fd_close_requested') is True and
            driver.get('error') == '', 'driver observation failed or retained its FD')
    for field in ('qualified', 'production_qualified', 'native_ready', 'executed_payload'):
        require(driver.get(field) is False, f'not a metadata-only driver observation: {field}')
    require(type(gpu.get('gpu_index')) is int and 0 <= gpu['gpu_index'] <= 15, 'invalid GPU index')
    require(1 <= args.control_port <= 65535 and 0 <= args.seed < 2**64 and
            0 <= args.producer_delay_ms <= 5000 and 0 <= args.consumer_delay_ms <= 5000 and
            args.producer_delay_ms + args.consumer_delay_ms < 10000, 'invalid control/stress policy')
    require(isinstance(driver.get('canonical_path'), str) and driver['canonical_path'].startswith('/'), 'missing absolute device path')
    query = driver.get('query')
    require(isinstance(query, dict) and query.get('abi_major') == 1 and query.get('abi_minor') == 0 and
            query.get('abi_cookie') == 0x5354475055303031 and
            query.get('frame_bytes') == 4096 and query.get('raw_state') == 0,
            'incompatible or active driver query')
    require(query.get('import_generation') == 0 and query.get('pool_binding_identity') == 0 and
            query.get('connection_epoch') == 0, 'observation describes an imported session')
    require(type(query.get('ring_frames')) is int and 32 <= query['ring_frames'] <= 4096, 'invalid observed ring size')
    require(1 <= args.epoch < 2**64 and 1 <= args.request_id < 2**64, 'invalid shared epoch/request ID')
    require(1 <= args.iterations <= 128 and 1 <= args.payload_bytes <= (query['ring_frames'] - 1) * 4096 and
            args.iterations * args.payload_bytes <= 128 * 1024 * 1024, 'test exceeds ring/session bounds')
    require(not args.duplex_pairs or (args.iterations >= 2 and args.iterations % 2 == 0), 'duplex requires even iterations')
    require(not args.alternate_directions or args.iterations >= 2, 'alternation requires at least two iterations')
    config = {'gpu-visibility': True, 'campaign-id': args.campaign_id, 'run-id': args.run_id,
              'role': args.role, 'dma-device': driver['canonical_path'],
              'local-endpoint': query['local_endpoint_identity'], 'peer-endpoint': query['peer_endpoint_identity'],
              'physical-link': args.physical_link, 'kernel-release': driver['kernel_identity'],
              'driver-build-id': driver['driver_build_id'], 'gpu-uuid': gpu['gpu_uuid'],
              'gpu-device': gpu['gpu_index'], 'gpu-runtime-version': gpu['gpu_runtime_version'],
              'gpu-driver-version': gpu['gpu_driver_version'], 'ring-frames': query['ring_frames'],
              'epoch': args.epoch, 'request-id': args.request_id, 'payload-bytes': args.payload_bytes,
              'iterations': args.iterations, 'seed': args.seed,
              'control-mode': 'listen' if args.role == 'sender' else 'connect', 'control-port': args.control_port,
              'producer-delay-ms': args.producer_delay_ms, 'consumer-delay-ms': args.consumer_delay_ms}
    if args.duplex_pairs:
        config['duplex-pairs'] = True
    if args.alternate_directions:
        config['alternate-directions'] = True
    # Do not copy the observer's per-open instance UUID/import state: the
    # commissioning executable observes and binds its own newly opened FD.
    arguments(config)
    return config


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest='command', required=True)
    for command in ('plan', 'run'):
        p = sub.add_parser(command)
        p.add_argument('--binary', type=Path, required=True)
        p.add_argument('--config', type=Path, required=True)
        if command == 'run':
            p.add_argument('--output', type=Path, required=True)
            p.add_argument('--configured-gpu-direct', action='store_true', required=True,
                           help='Acknowledge exclusive operator-controlled ConfigFS gpu_direct setup')
    p = sub.add_parser('prepare')
    p.add_argument('--gpu-observation', type=Path, required=True)
    p.add_argument('--driver-observation', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--campaign-id', required=True)
    p.add_argument('--run-id', required=True)
    p.add_argument('--role', choices=('sender', 'receiver'), required=True)
    p.add_argument('--physical-link', required=True)
    p.add_argument('--epoch', type=int, required=True)
    p.add_argument('--request-id', type=int, required=True)
    p.add_argument('--control-port', type=int, required=True)
    p.add_argument('--payload-bytes', type=int, default=65537)
    p.add_argument('--iterations', type=int, default=32)
    p.add_argument('--seed', type=int, default=0)
    p.add_argument('--producer-delay-ms', type=int, default=0)
    p.add_argument('--consumer-delay-ms', type=int, default=0)
    mode=p.add_mutually_exclusive_group()
    mode.add_argument('--duplex-pairs', action='store_true')
    mode.add_argument('--alternate-directions', action='store_true')
    p = sub.add_parser('compare')
    p.add_argument('host_a', type=Path)
    p.add_argument('host_b', type=Path)
    args = parser.parse_args()
    if args.command == 'prepare':
        config=prepare_config(load(args.gpu_observation), load(args.driver_observation), args)
        args.output.parent.mkdir(parents=True, exist_ok=True)
        with args.output.open('x', encoding='utf-8') as out:
            json.dump(config, out, indent=2)
            out.write('\n')
        print(f'Expected configuration: {args.output}. Run plan to validate it with the native executable.')
        return 0
    if args.command == 'compare':
        print(json.dumps(compare(load(args.host_a), load(args.host_b)), indent=2))
        return 0
    command = [str(args.binary.resolve()), *arguments(load(args.config))]
    # The real executable validates the complete CLI and emits expectations.
    # --check-config never opens GPU/DMA/socket resources in that executable.
    check = subprocess.run([*command, '--check-config'], capture_output=True, text=True, check=False)
    require(check.returncode == 0, f'native configuration rejected: {check.stdout} {check.stderr}')
    parsed = json.loads(check.stdout, object_pairs_hook=unique_object)
    require(parsed.get('executed') is False, 'preflight was not configuration-only')
    if args.command == 'plan':
        print(check.stdout, end='')
        print(shlex.join(command), file=sys.stderr)
        return 0
    args.output.parent.mkdir(parents=True, exist_ok=True)
    # Exclusive creation: never overwrite another run's evidence. Stderr stays
    # on the terminal; stdout (including a native failure receipt) is preserved.
    with args.output.open('x', encoding='utf-8') as out:
        result = subprocess.run(command, stdout=out, check=False)
    if result.returncode:
        print('Native run failed. Receipt preserved; process exit does not prove DMA/GPU cleanup.', file=sys.stderr)
        return result.returncode if result.returncode > 0 else 1
    checked_receipt(load(args.output))
    print(f'Local receipt complete: {args.output}. Compare it with the other host; no qualification was enabled.')
    return 0


if __name__ == '__main__':
    try:
        raise SystemExit(main())
    except (OSError, ValueError, KeyError, TypeError, subprocess.SubprocessError) as exc:
        print(f'visibility: {exc}', file=sys.stderr)
        raise SystemExit(1)
