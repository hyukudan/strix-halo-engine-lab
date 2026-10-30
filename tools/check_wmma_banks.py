#!/usr/bin/env python3
"""Checks the pinned-register WMMA kernels (STRIX_QPF2K_WMMA_PIN) of a built binary or code object.

Usage: check_wmma_banks.py BINARY [--llvm DIR] [--max-vgpr N]

Extracts the gfx1151 code objects (llvm-objdump --offloading, in a temporary directory), finds the pinned kernels (every
WMMA reads A from v[183:190] and B from v[166:173] or v[158:165]; an unpinned kernel may use one of these ranges by
chance) and requires for each:
  - every WMMA: D == C (tied accumulator), the D range disjoint from the A and B ranges (RDNA3.5: an earlier D
    overlapping a later A/B needs a separating instruction), and A, B and C starting in three different VGPR banks
    (start % 4);
  - zero private segment (scratch) and at most --max-vgpr VGPRs (default 192: two form-4 blocks per WGP).
Exit status 0 = all pinned kernels pass (prints one line per kernel), 1 = a violation, 2 = no pinned kernel found.
"""
import argparse
import collections
import os
import re
import shutil
import subprocess
import sys
import tempfile

PIN_A = (183, 190)
PIN_B = {(166, 173), (158, 165)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('binary')
    ap.add_argument('--llvm', default='/opt/rocm/core-10.1/lib/llvm/bin')
    ap.add_argument('--max-vgpr', type=int, default=192)
    args = ap.parse_args()
    tool = lambda name: os.path.join(args.llvm, name)
    work = tempfile.mkdtemp()
    try:
        copy = os.path.join(work, 'b')
        shutil.copy(args.binary, copy)
        subprocess.run([tool('llvm-objdump'), '--offloading', copy], cwd=work, capture_output=True, check=False)
        objects = [os.path.join(work, f) for f in os.listdir(work) if 'gfx1151' in f]
        if not objects:  # a bare code object
            objects = [copy]
        dis, notes = [], []
        for obj in objects:
            dis.append(subprocess.run([tool('llvm-objdump'), '-d', '--no-show-raw-insn', obj], capture_output=True, text=True).stdout)
            notes.append(subprocess.run([tool('llvm-readelf'), '--notes', obj], capture_output=True, text=True).stdout)
    finally:
        shutil.rmtree(work)

    meta = collections.defaultdict(dict)
    name = None
    for line in '\n'.join(notes).splitlines():
        m = re.match(r'\s*\.name:\s+(\S+)', line)
        if m:
            name = m.group(1)
            continue
        m = re.match(r'\s*\.(vgpr_count|private_segment_fixed_size):\s+(\d+)', line)
        if m and name:
            meta[name][m.group(1)] = int(m.group(2))

    wmma = collections.defaultdict(list)
    kernel = None
    for line in '\n'.join(dis).splitlines():
        m = re.match(r'^[0-9a-f]+ <(.+)>:', line)
        if m:
            kernel = m.group(1)
            continue
        if kernel and 'v_wmma_f32_16x16x16_f16' in line:
            ranges = [(int(a), int(b)) for a, b in re.findall(r'v\[(\d+):(\d+)\]', line)]
            if len(ranges) == 4:
                wmma[kernel].append(ranges)

    pinned = {k: v for k, v in wmma.items() if all(r[1] == PIN_A and r[2] in PIN_B for r in v)}
    if not pinned:
        print('no pinned WMMA kernel found')
        return 2
    failed = 0
    for k, insts in sorted(pinned.items()):
        problems = []
        for d, a, b, c in insts:
            if d != c:
                problems.append(f'D {d} != C {c}')
            if not (d[1] < a[0] or a[1] < d[0]) or not (d[1] < b[0] or b[1] < d[0]):
                problems.append(f'D {d} overlaps A {a} / B {b}')
            if len({a[0] % 4, b[0] % 4, c[0] % 4}) != 3:
                problems.append(f'banks A {a[0] % 4} B {b[0] % 4} C {c[0] % 4}')
        md = meta.get(k, {})
        vgpr, scratch = md.get('vgpr_count'), md.get('private_segment_fixed_size')
        if vgpr is None or vgpr > args.max_vgpr:
            problems.append(f'vgpr {vgpr} > {args.max_vgpr}')
        if scratch != 0:
            problems.append(f'scratch {scratch}')
        status = 'PASS' if not problems else 'FAIL ' + '; '.join(sorted(set(problems))[:4])
        failed += bool(problems)
        print(f'{status}: {len(insts)} WMMA, vgpr {vgpr}, scratch {scratch}: {k}')
    print(f'{len(pinned)} pinned kernels, {failed} failed')
    return 1 if failed else 0


if __name__ == '__main__':
    sys.exit(main())
