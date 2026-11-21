"""Syntax-check the real capability header with synthetic target/builtin macros; no HIP/GPU."""
import argparse
from pathlib import Path
import subprocess
import tempfile


def check_dpp_wrappers(compiler, msvc, header, folder):
    cases = [('host', None, None, True), ('host_zero', 0, None, True)]
    cases += [(target, 1, target, True) for target in ('gfx1030', 'gfx1151', 'gfx1200', 'gfx1201')]
    cases += [(target, 1, target, False) for target in ('gfx942', 'gfx950', 'gfx9999')]
    flags = ['/nologo', '/EP'] if msvc else ['-E', '-P']
    for name, device, target, native in cases:
        source = ''
        if device is not None:
            source += f'#define __HIP_DEVICE_COMPILE__ {device}\n'
        if target:
            source += f'#define __{target}__ 1\n'
        source += f'#include "{header.as_posix()}"\n'
        source += 'strix_dpp_update STRIX_HIP_UPDATE_DPP_ROW_SHARE_XMASK(0, 1, 0x158, 0xf, 0xf, true)\n'
        source += 'strix_dpp_move STRIX_HIP_MOV_DPP_ROW_SHARE_XMASK(1, 0x168, 0xf, 0xf, false)\n'
        fixture = folder / f'dpp_{name}.cpp'
        fixture.write_text(source, encoding='utf-8')
        result = subprocess.run([compiler, *flags, str(fixture)], capture_output=True, text=True)
        trap = '::strix::hip::arch::unavailable<int>'
        update = '__builtin_amdgcn_update_dpp' if native else trap
        move = '__builtin_amdgcn_mov_dpp' if native else trap
        expected = [
            f'strix_dpp_update{update}(0,1,0x158,0xf,0xf,true)',
            f'strix_dpp_move{move}(1,0x168,0xf,0xf,false)',
        ]
        actual = [''.join(line.split()) for line in result.stdout.splitlines() if line.startswith('strix_dpp_')]
        if result.returncode or actual != expected:
            print(f'FAIL: DPP wrappers for {name}: {actual}')
            print('\n'.join(result.stderr.splitlines()[-20:]))
            return 1
    print(f'PASS: {len(cases)} DPP wrapper cases')
    return 0


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--compiler', required=True)
    args = parser.parse_args()
    header = Path(__file__).resolve().parents[1] / 'include/strix/hip_arch_caps.hpp'
    builtins = ['wmma_i32_16x16x16_iu4_w32', 'wmma_i32_16x16x16_iu8_w32',
                'wmma_f32_16x16x16_bf16_w32', 'update_dpp', 'sudot4']
    targets = ['gfx1100', 'gfx1101', 'gfx1102', 'gfx1103', 'gfx1150', 'gfx1151', 'gfx1152', 'gfx1153',
               'gfx900', 'gfx1200', 'gfx1201', 'gfx942', 'gfx950', 'gfx11510', None, 'gfx1010', 'gfx1036']
    msvc = Path(args.compiler).name.lower() == 'cl.exe'
    # Every builtin availability subset for every architecture, including host/unknown.
    # Each case is a separate TU so pragma-once and macro evaluation are real.
    with tempfile.TemporaryDirectory(prefix='strix-caps-', dir=Path.cwd()) as folder:
        fixtures = []
        for target in targets:
            for mask in range(32):
                values = [(mask >> bit) & 1 for bit in range(5)]
                rdna35 = target in targets[4:8]
                family = target in targets[:8]
                rdna4 = target in ('gfx1200', 'gfx1201')
                cdna = target in ('gfx942', 'gfx950')
                expected = [rdna35 and values[0] and values[1], family and values[2],
                            family and values[3], family and values[4], family and values[0],
                            family and values[1], family and values[2], family and values[3], family and values[4]]
                names = ['RDNA35_WMMA', 'WMMA_BF16_W32', 'DPP_ROW', 'SUDOT4', 'WMMA_IU4_W32',
                         'WMMA_IU8_W32', 'WMMA_F16_W32', 'PERMLANEX16', 'UDOT4']
                text = '#define STRIX_HAS_BUILTIN(x) STRIX_TEST_##x\n'
                text += ''.join(f'#define STRIX_TEST___builtin_amdgcn_{name} {value}\n'
                                for name, value in zip(builtins, values))
                for name, value in [('wmma_f32_16x16x16_f16_w32', values[2]), ('permlanex16', values[3]),
                                    ('udot4', values[4])]:
                    text += f'#define STRIX_TEST___builtin_amdgcn_{name} {value}\n'
                if target:
                    text += f'#define __{target}__ 1\n'
                text += f'#include "{header.as_posix()}"\n'
                for name, value in zip(names, expected):
                    text += f'static_assert(bool(STRIX_HAS_{name}) == {int(bool(value))}, "{name}");\n'
                text += f'static_assert(STRIX_HIP_ARCH_RDNA4 == {int(rdna4)});\n'
                text += f'static_assert(STRIX_HIP_ARCH_CDNA == {int(cdna)});\n'
                text += f'static_assert(STRIX_HIP_TARGET_WAVE_SIZE == {32 if family or rdna4 else 64 if cdna else 0});\n'
                row_share_xmask = family or rdna4 or target in ('gfx1010', 'gfx1036')
                text += f'static_assert(STRIX_HAS_DPP_ROW_SHARE_XMASK == {int(row_share_xmask)});\n'
                fixture = Path(folder) / f'{target or "host"}_{mask}.cpp'
                fixture.write_text(text, encoding='utf-8')
                fixtures.append(str(fixture))
        # Batch files to stay below Windows command-line limits.
        for start in range(0, len(fixtures), 24):
            flags = ['/nologo', '/std:c++20', '/Zs', '/W4', '/WX'] if msvc else [
                '-std=c++20', '-fsyntax-only', '-Wall', '-Wextra', '-Werror']
            result = subprocess.run([args.compiler, *flags, *fixtures[start:start + 24]],
                                    capture_output=True, text=True)
            if result.returncode:
                print('\n'.join((result.stdout + result.stderr).splitlines()[-20:]))
                return result.returncode
        print(f'PASS: {len(fixtures)} target/builtin cases')
        return check_dpp_wrappers(args.compiler, msvc, header.parent / 'hip/legacy_intrinsics.hpp', Path(folder))


if __name__ == '__main__':
    raise SystemExit(main())
