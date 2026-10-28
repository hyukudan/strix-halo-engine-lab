#!/usr/bin/env python3
"""Independent stdlib fp64 YaRN check of C++ TSV: f, p, i, w, cos_bits, sin_bits.

Use --sample FILE for an existing sample, or --emitter TEST_BINARY to produce
one in a temporary directory. Python math uses the host libm; this checks the
named mathematics and binary32 rounding, not cross-libm bit reproducibility.
"""

import argparse
import math
from pathlib import Path
import struct
import subprocess
import tempfile


POSITIONS = (0, 1, 8191, 8192, 32767, 65535, 131071, 196607,
             262143, 262144, 524287, 524288, 999999, 1048575)


def f32(value):
    return struct.unpack('<f', struct.pack('<f', value))[0]


def check(path):
    seen = set()
    for line in Path(path).read_text(encoding='utf-8').splitlines():
        factor, position, pair, frequency, cosine, sine = line.split('\t')
        factor, position, pair = int(factor), int(position), int(pair)
        assert factor in (2, 4) and 0 <= pair < 32
        assert 0 <= position < 262144 * factor
        key = factor, position, pair
        assert key not in seen, ('duplicate', key)
        seen.add(key)
        # Independent algebra: decimal-base exponent and one ramp multiplier.
        ramp = min(1.0, max(0.0, (pair - 14) / 8.0))
        omega = 10.0 ** (-7.0 * pair / 32.0) * (1.0 - (1.0 - 1.0 / factor) * ramp)
        assert math.isclose(float(frequency), omega, rel_tol=1e-15, abs_tol=0), key
        amplitude = 1.0 + math.log(factor) / 10.0
        for actual_bits, trig in ((cosine, math.cos), (sine, math.sin)):
            actual = struct.unpack('<f', struct.pack('<I', int(actual_bits)))[0]
            expected = f32(amplitude * trig(position * omega))
            # Different fp64 pow algebra/libm can perturb phase at very high p.
            # Require at most one binary32 ULP, and the absolute math guard.
            bits = struct.unpack('<I', struct.pack('<f', expected))[0]
            assert abs(int(actual_bits) - bits) <= 1, (key, actual, expected)
            assert math.isfinite(actual) and abs(actual - expected) <= 2e-6, key
    expected_keys = {(f, p, i) for f in (2, 4) for p in POSITIONS
                     if p < 262144 * f for i in range(32)}
    assert seen == expected_keys, 'missing or unexpected sample rows'
    print(f'Independent Python YaRN cross-check passed: {len(seen)} frequency/coefficient pairs')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument('--sample')
    source.add_argument('--emitter')
    args = parser.parse_args()
    if args.sample:
        check(args.sample)
    else:
        with tempfile.TemporaryDirectory(prefix='qwen-yarn-', dir=Path.cwd()) as directory:
            sample = str(Path(directory) / 'sample.tsv')
            subprocess.run([args.emitter, '--emit-samples', sample], check=True)
            check(sample)


if __name__ == '__main__':
    main()
