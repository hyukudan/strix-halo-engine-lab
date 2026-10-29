#!/usr/bin/env python3
"""Small synthetic GGUF checks for placement and the CPU read benchmark; no model assets required."""
import argparse
import json
import pathlib
import struct
import subprocess
import sys
import tempfile


def string(value):
    raw = value.encode()
    return struct.pack('<Q', len(raw)) + raw


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--bench', required=True)
    ap.add_argument('--helper', default=str(pathlib.Path(__file__).resolve().parents[1] / 'tools/ds41_stripe_ranges.py'))
    args = ap.parse_args()
    # Keep fixtures/builds inside the worktree, including on Windows with a system TEMP elsewhere.
    with tempfile.TemporaryDirectory(prefix='read-tools-', dir=pathlib.Path.cwd()) as temp:
        root = pathlib.Path(temp)
        model = root / 'model.gguf'
        entries = []
        off = 0
        for layer in range(40):
            for part in ('gate', 'up', 'down'):
                entries.append((f'blk.{layer}.ffn_{part}_exps.weight', [256, 384], off))
                off += 256 * 384 * 4
        for layer in (7, 29):
            entries.append((f'blk.{layer}.engram_embd.weight', [66, 128], off))
            off += 264 * 128
        header = b'GGUF' + struct.pack('<IQQ', 3, len(entries), 0)
        for name, dims, offset in entries:
            header += string(name) + struct.pack('<I', len(dims))
            header += struct.pack('<' + 'Q' * len(dims), *dims) + struct.pack('<IQ', 0, offset)
        data = (len(header) + 31) // 32 * 32
        with model.open('wb') as out:
            out.write(header)
            out.truncate(data + off)
        placement = json.loads(subprocess.check_output([sys.executable, '-I', args.helper, str(model)]))
        assert len(placement['ranges']) == 122
        assert all(r['policy'] == 'interleave' for r in placement['ranges'])
        assert {r['name'] for r in placement['ranges'] if '.engram_' in r['name']} == {
            'blk.7.engram_embd.weight', 'blk.29.engram_embd.weight'}
        for policy in ('interleave', 'interleave-even', 'all'):
            row = json.loads(subprocess.check_output([sys.executable, '-I', args.helper, str(model), '--policy', policy]))
            assert len(row['ranges']) == 122 and all(r['policy'] == policy for r in row['ranges'])
        for flag, count in (('--no-engram', 120), ('--no-experts', 2)):
            row = json.loads(subprocess.check_output([sys.executable, '-I', args.helper, str(model), flag]))
            assert len(row['ranges']) == count
        for mode in ('prefill', 'units', 'engram'):
            receipt = root / f'{mode}.json'
            command = [args.bench, str(model), '--mode', mode, '--tokens', '1', '--jobs', '3',
                       '--weight-dirs', '', '--weight-verify', 'always', '--json', str(receipt)]
            row = json.loads(subprocess.check_output(command))
            assert row == json.loads(receipt.read_text())
            assert row['seconds'] > 0 and row['gb_s'] > 0 and row['location_bytes'][0] > 0
            assert row['host_gib'] == 0 and row['page_hits'] == 0 and row['verify_mode'] == 'always'
            assert row['layer']['count'] == (40 if mode != 'engram' else 0)
            assert row['read']['count'] == {'prefill': 120, 'units': 720, 'engram': 0}[mode]
            assert row['job']['count'] == (6 if mode == 'engram' else 0)
        for value in ('0', '-1', '33', '100000', '99999999999999999999999', '1x'):
            bad = subprocess.run([args.bench, str(model), '--drive-readers', value], capture_output=True)
            assert bad.returncode == 2 and b'1..32' in bad.stderr
        bad = subprocess.run([sys.executable, '-I', args.helper, str(model), '--policy', 'invalid'], capture_output=True)
        assert bad.returncode != 0
        model.write_bytes(b'GGUF' + struct.pack('<IQQ', 99, 0, 0))
        bad = subprocess.run([sys.executable, '-I', args.helper, str(model)], capture_output=True)
        assert bad.returncode != 0
    print('Placement and read-benchmark CPU checks passed')


if __name__ == '__main__':
    main()
