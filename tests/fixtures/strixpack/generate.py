"""Build the checked-in Strix Pack v1 fixture bundle using a wire-only oracle."""
import argparse
import hashlib
import json
import math
from pathlib import Path
import struct

TABLES = {
    2: (1, 3),
    3: (3, 10, 18, 30),
    4: (1, 3, 5, 8, 10, 14, 18, 24),
}
ALIGN = 4096
BOUND = hashlib.sha256(b"Strix Pack conformance checkpoint string").hexdigest()


def f32(x):
    return struct.unpack("<f", struct.pack("<f", x))[0]


def words(values):
    return struct.pack(f"<{len(values)}f", *values)


def sha(data):
    return hashlib.sha256(data).hexdigest()


def scale(byte, base):
    exponent, fraction = divmod(byte, 8)
    return math.ldexp(1 + fraction / 8, exponent + base)


def code_at(block, bits, j):
    if bits == 4:
        return (block[j // 2] >> (4 * (j % 2))) % 16
    low = (block[j // 4] >> (2 * (j % 4))) % 4
    return low if bits == 2 else low + 4 * ((block[32 + j // 8] >> (j % 8)) % 2)


def case(name, bits, rows, k, base, scales):
    levels = TABLES[bits]
    # These bytes spell code 0,1,... in sign/index order.
    basic = {
        2: b"\xe4" * 32,
        3: b"\xe4" * 32 + b"\xf0" * 16,
        4: b"\x10\x32\x54\x76\x98\xba\xdc\xfe" * 8,
    }[bits]
    assert [code_at(basic, bits, j) for j in range(1 << bits)] == list(range(1 << bits))
    flags = (1 if bits == 2 else 0) | (4 if base != -3 else 0)
    header = bytes([1, bits, 5, flags, base % 256, 0, 0, 0, *levels]).ljust(16, b"\0")
    payload = bytearray()
    decoded, groups = [], []
    for r in range(rows):
        row_groups = []
        for b in range(k // 128):
            # First block is the hand vector. Later blocks and rows vary every byte.
            plane = basic if r == b == 0 else bytes(
                v ^ ((r * 33 + b * 213 + i * 17) % 256) for i, v in enumerate(basic))
            sb = bytes(scales[(g + r + b) % 4] for g in range(4))
            block = plane + sb
            payload.extend(block)
            for g in range(4):
                signed = []
                s = scale(sb[g], base)
                for j in range(32 * g, 32 * g + 32):
                    c = code_at(block, bits, j)
                    level = levels[c % len(levels)] * (1 if c < len(levels) else -1)
                    signed.append(level)
                    decoded.append(level * s)
                row_groups.append((signed, sb[g]))
        groups.append(row_groups)
    # A sparse activation probes both sides of group and superblock boundaries.
    # The dense activations include fp32 inputs that round to 1.0 in binary16.
    x = [0.0] * k
    for j in (0, 31, 32, 63, 64, 95, 96, 127, 128, 129, 255):
        if j < k:
            x[j] = (-1 if j % 2 else 1) / 1024
    x += [(1 if (j * 23) % 31 < 15 else -1) *
          (1 + (2**-12 if name == "sp4t" else 0)) / 1024 for j in range(k)]
    y, head = [], []
    for a in (x[:k], x[k:]):
        for r in range(rows):
            acc = 0.0
            for j, value in enumerate(decoded[r * k:(r + 1) * k]):
                # The fixture recipe's operands have <= 13 and <= 8 significant
                # bits. Around an FP32 midpoint the exact sum fits in FP64.
                acc = f32(a[j] * value + acc)
            y.append(acc)
            if name == "sp4t":
                acc = 0.0
                for g, (signed, s) in enumerate(groups[r]):
                    # These fixtures have exact products and all FP32 partial
                    # sums within each WMMA group are exact in any reduction order.
                    act = [struct.unpack("<e", struct.pack("<e", t))[0] for t in a[g * 32:(g + 1) * 32]]
                    acc = f32(sum(v * w for v, w in zip(act, signed)) * scale(s, 0) + acc)
                head.append(math.ldexp(acc, base))
    return name, rows, k, header, bytes(payload), words(decoded), words(x), words(y), words(head)


def source_vectors():
    bf = range(65536)
    mx_levels = (0.0, 0.5, 1, 1.5, 2, 3, 4, 6)
    mx_scales = bytes(range(253))
    mx_codes = b"\x10\x32\x54\x76\x98\xba\xdc\xfe" * (2 * len(mx_scales))
    mx_values = [math.ldexp(math.copysign(mx_levels[j % 8], 1 if j % 16 < 8 else -1), e - 127)
                 for e in mx_scales for j in range(32)]
    fp_codes = bytes(i % 256 for i in range(129 * 129))
    fp_scales = (0.5, -2, 4, 0.25)
    fp_values = []
    for i, c in enumerate(fp_codes):
        r, j = divmod(i, 129)
        e, m = divmod(c % 128, 8)
        if (e, m) == (15, 7):
            fp_values.append(math.nan)
        else:
            v = math.ldexp(m if e == 0 else 8 + m, -9 if e == 0 else e - 10)
            fp_values.append(math.copysign(v, 1 if c < 128 else -1) * fp_scales[2 * (r // 128) + j // 128])
    return {
        "bf16.codes.bin": struct.pack("<65536H", *bf),
        "bf16.decoded.f32le": struct.pack("<65536I", *(b << 16 for b in bf)),
        "mxfp4.codes.bin": mx_codes,
        "mxfp4.scales.bin": mx_scales,
        "mxfp4.decoded.f32le": words(mx_values),
        "fp8.codes.bin": fp_codes,
        "fp8.scales.bin": words(fp_scales),
        "fp8.decoded.f32le": words(fp_values),
    }


def bundle():
    cases = [
        case("sp2", 2, 1, 128, -24, (0, 1, 8, 16)),
        case("sp3", 3, 9, 256, 90, (255, 247, 239, 231)),
        case("sp4", 4, 17, 256, -120, (0, 1, 9, 16)),
        case("sp4t", 4, 17, 256, -3, (0, 1, 10, 24)),
    ]
    header = f"strix-lowbit-pack/v2 {len(cases)}\ncheckpoint {BOUND}\n"
    data = bytearray(ALIGN)
    files = {}
    entries = []
    identity = hashlib.sha256()
    payloads = []
    for name, rows, k, h, payload, dq, x, y, head in cases:
        offset = (len(data) + ALIGN - 1) // ALIGN * ALIGN
        data.extend(b"\0" * (offset - len(data)))
        data.extend(payload)
        source = sha(name.encode("ascii"))
        header += f"tensor {name} strix.{name} 1 {rows} {k} {offset} {len(payload)} {source} {h.hex()}\n"
        payloads.append(payload)
        entry = {"name": name, "rows": rows, "columns": k, "header_hex": h.hex(),
                 "offset": offset, "bytes": len(payload), "payload_sha256": sha(payload)}
        for suffix, contents in (("decoded", dq), ("activations", x), ("project", y), ("head", head)):
            if contents:
                path = f"{name}.{suffix}.f32le"
                files[path] = contents
                entry[suffix] = path
        entries.append(entry)
    encoded_header = (header + "end\n").encode("ascii")
    data[:ALIGN] = encoded_header.ljust(ALIGN, b"\n")
    identity.update(encoded_header)
    for payload in payloads:
        identity.update(payload)
    files["vectors.lp"] = bytes(data)
    files["zero.f32le"] = words([-0.0] * 128)
    files["nonfinite.f32le"] = struct.pack("<384I", *(
        word for bad in (0x7fc00000, 0x7f800000, 0xff800000) for word in [0] * 127 + [bad]))
    for bits in (2, 3, 4):
        files[f"zero.sp{bits}"] = bytes(16 * bits + 4)
    files.update(source_vectors())
    obj = {"schema": "strixpack-fixtures/v1", "checkpoint_sha256": BOUND,
           "pack_identity": identity.hexdigest(), "activation_rows": 2, "entries": entries,
           "encoder": {"geometry": [1, 128], "zero_input": "zero.f32le", "zero_base": 0,
                       "zero_payloads": {str(b): f"zero.sp{b}" for b in TABLES},
                       "nonfinite_input": "nonfinite.f32le", "nonfinite_cases": 3,
                       "nonfinite_status": "reject with no output writes"},
           "geometry": {"invalid": [[0, 128], [1, 0], [1, 127], [1, 129]],
                        "large_host_test": [2049, 2048], "uint64_bound": "(2^64 - 1) / block_bytes for K=128"},
           "sources": {"bf16": [256, 256], "mxfp4": [253, 32], "fp8": [129, 129],
                       "fp8_nan": "canonicalize all output NaNs to 0x7fc00000 before hashing"},
           "sha256": {key: sha(value) for key, value in sorted(files.items())}}
    files["manifest.json"] = (json.dumps(obj, indent=2) + "\n").encode("ascii")
    return files


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("directory", type=Path)
    parser.add_argument("--check", action="store_true")
    args = parser.parse_args()
    if not args.check:
        args.directory.mkdir(parents=True, exist_ok=True)
    for name, data in bundle().items():
        path = args.directory / name
        if args.check:
            if path.read_bytes() != data:
                raise SystemExit(f"fixture differs: {path}")
        else:
            path.write_bytes(data)


if __name__ == "__main__":
    main()
