#!/usr/bin/env python3
"""Extract rotary code objects, per-symbol .text hashes and compiler resources.

No device is opened. Keep output outside Git. Run with the same ROCm toolchain
and CMake options for each revision; compare the resulting manifests by the
demangled kernel name. Raw bytes, disassembly and complete metadata are retained.
"""
import argparse
import hashlib
import json
from pathlib import Path
import re
import shutil
import struct
import subprocess


def elf_sections(data):
    if data[:6] != b'\x7fELF\x02\x01':
        raise ValueError('expected little-endian ELF64')
    offset = struct.unpack_from('<Q', data, 40)[0]
    size, count, strings = struct.unpack_from('<HHH', data, 58)
    sections = [struct.unpack_from('<IIQQQQIIQQ', data, offset + i * size) for i in range(count)]
    names = sections[strings]
    names = data[names[4]:names[4] + names[5]]
    return sections, [names[s[0]:].split(b'\0', 1)[0].decode() for s in sections]


def extract(args):
    out = args.out.resolve()
    out.mkdir(parents=True, exist_ok=True)
    obj = out / 'input.o'
    shutil.copyfile(args.object, obj)
    llvm = args.llvm
    subprocess.run([str(llvm / 'llvm-objdump'), '--offloading', str(obj)], check=True, capture_output=True)
    images = list(out.glob('input.o.*hip*gfx1151*'))
    if len(images) != 1:
        raise ValueError('expected exactly one gfx1151 device image')
    device = images[0]
    data = device.read_bytes()
    sections, names = elf_sections(data)
    text = sections[names.index('.text')]
    (out / 'text.bin').write_bytes(data[text[4]:text[4] + text[5]])
    notes = subprocess.check_output([str(llvm / 'llvm-readelf'), '--notes', str(device)], text=True)
    (out / 'metadata.txt').write_text(notes)
    metadata = {}
    for entry in notes.split('  - .args:')[1:]:
        name = re.search(r'^    .name:\s+(\S+)', entry, re.M)
        if name:
            def number(key):
                match = re.search(r'^    \.' + key + r':\s+(\d+)', entry, re.M)
                return int(match[1]) if match else None
            metadata[name[1]] = {'vgpr': number('vgpr_count'), 'sgpr': number('sgpr_count'),
                'scratch': number('private_segment_fixed_size'),
                'lds': number('group_segment_fixed_size'),
                'kernarg_size': number('kernarg_segment_size'),
                'kernarg_align': number('kernarg_segment_align'),
                'wavefront': number('wavefront_size'),
                'max_workgroup': number('max_flat_workgroup_size'),
                'args_sha256': hashlib.sha256(entry.split('    .group_segment_fixed_size:')[0].encode()).hexdigest()}
    records = []
    for section in sections:
        if section[1] != 2:  # SHT_SYMTAB
            continue
        strings = sections[section[6]]
        strings = data[strings[4]:strings[4] + strings[5]]
        for offset in range(section[4], section[4] + section[5], section[9]):
            name, info, _, index, value, size = struct.unpack_from('<IBBHQQ', data, offset)
            if info & 15 != 2 or index != names.index('.text'):
                continue
            symbol = strings[name:].split(b'\0', 1)[0].decode()
            if not re.search(r'norm_rope|rope_table_kernel|block_pool_kernel|block_score_(?:rows_)?kernel', symbol):
                continue
            demangled = subprocess.check_output([str(llvm / 'llvm-cxxfilt'), symbol], text=True).strip()
            if 'frozen_native' in demangled:
                continue
            start = text[4] + value - text[3]
            code = data[start:start + size]
            record = {'symbol': symbol, 'name': demangled, 'size': size, 'sha256': hashlib.sha256(code).hexdigest(),
                'yarn': 'QwenRotarySource' in demangled, **metadata.get(symbol, {})}
            records.append(record)
            (out / (record['sha256'] + '.bin')).write_bytes(code)
    records.sort(key=lambda r: r['name'])
    fields = ('sha256', 'vgpr', 'sgpr', 'scratch', 'lds', 'kernarg_size',
              'kernarg_align', 'wavefront', 'max_workgroup', 'args_sha256')
    for record in records:
        if any(record.get(key) is None for key in fields):
            raise ValueError('missing required compiler metadata: ' + record['name'])
    provenance = {'source_revision': args.source_revision,
                  'source_sha256': hashlib.sha256(args.source.read_bytes()).hexdigest(),
                  'included_source_sha256s': {p.name: hashlib.sha256(p.read_bytes()).hexdigest()
                      for p in sorted(args.source.parent.glob('hip_qsa*.inc'))},
                  'cmake_cache_sha256': hashlib.sha256(args.cmake_cache.read_bytes()).hexdigest(),
                  'cmake_options': sorted(line for line in args.cmake_cache.read_text().splitlines()
                                         if line.startswith(('STRIX_', 'BUILD_TESTING:', 'CMAKE_HIP_ARCHITECTURES:', 'CMAKE_BUILD_TYPE:'))),
                  'compiler': subprocess.check_output([str(llvm / 'clang++'), '--version'], text=True),
                  'object_sha256': hashlib.sha256(obj.read_bytes()).hexdigest()}
    (out / 'provenance.json').write_text(json.dumps(provenance, indent=2) + '\n')
    (out / 'manifest.json').write_text(json.dumps(records, indent=2) + '\n')
    with (out / 'rotary.s').open('w') as stream:
        subprocess.run([str(llvm / 'llvm-objdump'), '-d',
            '--disassemble-symbols=' + ','.join(r['symbol'] for r in records), str(device)], check=True, stdout=stream)
    print(json.dumps({'kernels': len(records), 'yarn': sum(r['yarn'] for r in records),
        'text_sha256': hashlib.sha256(data[text[4]:text[4] + text[5]]).hexdigest()}))
    if args.baseline_manifest:
        baseline = json.loads(args.baseline_manifest.read_text())
        baseline_native = [r for r in baseline if not r['yarn']]
        inventory_sha = hashlib.sha256('\n'.join(sorted(r['name'] for r in baseline_native)).encode()).hexdigest()
        if len(baseline_native) != 64 or inventory_sha != '42aaad2d9c292dc50345a052ae5d1c19d2a94fe9899c2f2afa16eec3851078fc':
            raise ValueError('baseline production rotary inventory is incomplete or changed')
        baseline_provenance = json.loads((args.baseline_manifest.parent / 'provenance.json').read_text())
        if baseline_provenance['source_revision'] != '54594de6' or baseline_provenance['source_sha256'] != '80492ad91c0c521933abbaf1ec6bf949a1e5310801a2cc4cc641f1cea274ed34':
            raise ValueError('baseline must identify the pinned native source revision')
        if any(baseline_provenance[key] != provenance[key] for key in ('compiler', 'cmake_options')):
            raise ValueError('baseline compiler or CMake options differ')
        native = {r['name']: r for r in records if not r['yarn']}
        changes = []
        for old in baseline:
            if old['yarn']:
                continue
            new = native.get(old['name'])
            if new is None or any(old.get(key) is None or old.get(key) != new.get(key) for key in fields):
                changes.append({'name': old['name'], 'before': old, 'after': new})
        scratch = [r['name'] for r in records if r['yarn'] and r.get('scratch') != 0]
        additions = [r['name'] for r in records if r['name'] not in {b['name'] for b in baseline}]
        native_additions = sorted(name for name in additions if name in native)
        fixture_inventory = hashlib.sha256('\n'.join(native_additions).encode()).hexdigest()
        if len(native_additions) != 8 or fixture_inventory != '0e03a5fc39e0010482df15944d24466aad232d9f1bd8b60602f39c3daed9f7ad':
            raise ValueError('unexpected native production/fixture symbol additions')
        comparison = {'native_changes': changes, 'yarn_nonzero_or_unknown_scratch': scratch,
                      'native_fixture_additions': native_additions,
                      'additional_instances': additions, 'compared_fields': fields}
        (out / 'comparison.json').write_text(json.dumps(comparison, indent=2) + '\n')
        print(json.dumps({'native_changes': len(changes), 'yarn_scratch_failures': len(scratch)}))
        if changes or scratch:
            raise SystemExit(1)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--object', type=Path, required=True)
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--llvm', type=Path, default=Path('/opt/rocm/lib/llvm/bin'))
    parser.add_argument('--baseline-manifest', type=Path)
    parser.add_argument('--source-revision', required=True)
    parser.add_argument('--source', type=Path, required=True)
    parser.add_argument('--cmake-cache', type=Path, required=True)
    extract(parser.parse_args())
