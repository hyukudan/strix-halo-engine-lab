"""Independent small packed-checkpoint reference; no torch or downloaded assets."""
import argparse
import json
from pathlib import Path
import struct
import subprocess
import tempfile

try:
    import numpy as np
except ImportError:
    np = None

PREFIX = "blocks.0.attn.qkv_proj."


def f32(value):
    return struct.unpack("<f", struct.pack("<f", value))[0]


def fp8(u):
    # Independent numerical E4M3FN definition, including the finite exponent 15.
    sign = -1 if u & 128 else 1
    exponent, mantissa = (u >> 3) & 15, u & 7
    if exponent == 15 and mantissa == 7:
        return float("nan")
    return sign * (mantissa * 2**-9 if exponent == 0 else (1 + mantissa / 8) * 2**(exponent - 7))


def save(path, tensors):
    header, data = {}, bytearray()
    for key, (dtype, shape, raw) in tensors.items():
        header[key] = {"dtype": dtype, "shape": shape, "data_offsets": [len(data), len(data) + len(raw)]}
        data.extend(raw)
    raw = json.dumps(header, separators=(",", ":")).encode()
    path.write_bytes(struct.pack("<Q", len(raw)) + raw + data)


def packed_case(group, scale_kind, codebook, wrapper=""):
    n, k = 3, 256
    # Every code and parity, including negative I8 storage and changing row scales.
    codes = [[(c * 7 + r * 3) % 16 for c in range(k)] for r in range(n)]
    raw = bytes(row[c] | row[c + 1] << 4 for row in codes for c in range(0, k, 2))
    sr = [0.0, 0.5, 1.5, 2.5, 16, 448, 1 / 512, 1.0]
    scales = [sr[(r * (k // group) + g) % len(sr)] for r in range(n) for g in range(k // group)]
    cb = [f32((c - 7.5) / 2) for c in range(16)] if codebook else None
    channels = [f32(v) for v in (0.25, 0.013, 2)]
    conf = {"format": "asym_w4a8_int8", "params": {"group_size": group, "convrot_groupsize": 256}}
    meta = json.dumps(conf).encode()
    entries = {"weight": ("I8", [n, k // 2], raw),
               "weight_s_channel": ("F32", [n], struct.pack("<3f", *channels)),
               "comfy_quant": ("U8", [len(meta)], meta)}
    if scale_kind == "F32":
        entries["weight_s_rel"] = ("F32", [n, k // group], struct.pack(f"<{len(scales)}f", *scales))
    else:
        enc = {fp8(u): u for u in range(127)}
        entries["weight_s_rel"] = (scale_kind, [n, k // group], bytes(enc[v] for v in scales))
    if cb:
        entries["weight_codebook"] = ("F32", [16], struct.pack("<16f", *cb))
    expected = []
    for r, row in enumerate(codes):
        for c, code in enumerate(row):
            value = cb[code] if cb else code - 8
            expected.append(max(-127, min(127, round(f32(value * scales[r * (k // group) + c // group])))))
    if np is not None:
        levels = np.array(cb if cb else range(-8, 8), dtype=np.float32)[np.array(codes)]
        vector = np.clip(np.rint(levels * np.array(scales, dtype=np.float32).reshape(n, k // group).repeat(group, axis=1)), -127, 127).astype(np.int8)
        assert vector.ravel().tolist() == expected
    expected_bytes = struct.pack(f"<{len(expected)}b", *expected)
    expected_bytes += b"".join(struct.pack("<f", f32(q * channels[i // k])) for i, q in enumerate(expected))
    return {wrapper + PREFIX + key: val for key, val in entries.items()}, expected_bytes


def run(binary):
    with tempfile.TemporaryDirectory(prefix="h3-checkpoints-") as tmp:
        root = Path(tmp)
        passed = 0
        for group in (4, 8, 16, 32, 64, 256):
            for kind in ("F32", "F8_E4M3", "U8"):
                for cb in (False, True):
                    entries, expected = packed_case(group, kind, cb, "model.diffusion_model.")
                    case, out = root / "packed.safetensors", root / "decoded.bin"
                    save(case, entries)
                    result = subprocess.run([str(binary), "--decode", str(case), str(out)], capture_output=True, text=True)
                    if result.returncode:
                        raise AssertionError(result.stderr[:1000])
                    assert out.read_bytes() == expected, (group, kind, cb)
                    passed += 1
        base, _ = packed_case(16, "F32", True)
        mutations = [
            ("weight", ("U8", [3, 128], bytes(384))),
            ("weight_s_rel", ("F32", [3, 15], bytes(3 * 15 * 4))),
            ("weight_s_rel", ("F32", [3, 16], struct.pack("<48f", *([-1] * 48)))),
            ("weight_s_rel", ("F8_E4M3", [3, 16], bytes([127] * 48))),
            ("weight_s_channel", ("F32", [3], struct.pack("<3f", 1, float("nan"), 1))),
            ("weight_codebook", ("F32", [15], bytes(60))),
            ("weight_codebook", ("F32", [16], struct.pack("<16f", *([float("nan")] * 16)))),
            ("weight_correction", ("F32", [16, 3], bytes(192))),
            ("bias", ("F32", [3], bytes(12))),
            ("weight_s_channel", None),
        ]
        for conf in ({"format": "w6a8_int8"}, {"format": "asym_w4a8_int8", "group_size": 3},
                     {"format": "asym_w4a8_int8", "group_size": 48},
                     {"format": "asym_w4a8_int8", "convrot_groupsize": 128}):
            data = json.dumps(conf).encode()
            mutations.append(("comfy_quant", ("U8", [len(data)], data)))
        for key, entry in mutations:
            bad = dict(base)
            if entry is None:
                del bad[PREFIX + key]
            else:
                bad[PREFIX + key] = entry
            path = root / "invalid.safetensors"
            save(path, bad)
            result = subprocess.run([str(binary), "--decode", str(path), str(root / "bad.bin")], capture_output=True)
            assert result.returncode != 0, key
        mapped = {"model.diffusion_model.rope.inv_freq": ("F32", [2], struct.pack("<2f", 1, 2))}
        save(root / "mapped.safetensors", mapped)
        collision = dict(mapped, **{"rope.inv_freq": ("F32", [2], struct.pack("<2f", 3, 4))})
        save(root / "collision.safetensors", collision)
        (root / "shards").mkdir()
        save(root / "shards/one.safetensors", mapped)
        save(root / "shards/two.safetensors", {"transformer.condition_proj.bias": ("F32", [1], bytes(4))})
        save(root / "vae.safetensors", {"first_stage_model.latents_mean": ("F16", [4], struct.pack("<4e", 1, -2, .5, 65504))})
        conf = json.dumps({"format": "int8_tensorwise", "convrot": True, "convrot_groupsize": 256}).encode()
        save(root / "int8.safetensors", {PREFIX + "weight": ("I8", [2, 256], bytes(i % 256 for i in range(512))),
             PREFIX + "weight_scale": ("F32", [2, 1], struct.pack("<2f", .1, .2)),
             PREFIX + "comfy_quant": ("U8", [len(conf)], conf)})
        curve = {"adaln_t_table": ("F32", [2, 2], struct.pack("<4f", 1, 2, 3, 4))}
        curve.update({f"blocks.{i}.norm1.weight": ("BF16", [1], b"\x80\x3f") for i in range(50)})
        save(root / "curve.safetensors", curve)
        result = subprocess.run([str(binary), str(root)], capture_output=True, text=True)
        if result.returncode:
            raise AssertionError(result.stderr[:1000])
        print(result.stdout.strip())
        print(f"W4A8 Python reference: {passed} byte-identical unpack/dequant cases; {len(mutations)} invalid layouts rejected; backend={'numpy + scalar' if np is not None else 'pure Python'}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--binary", required=True, type=Path)
    run(parser.parse_args().binary.resolve())
