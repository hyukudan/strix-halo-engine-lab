"""Offline only: immutable RoPE tables and an independent PyTorch FP32 semantic fixture.

Runtime tests read the small committed files and need no Python, torch or model assets.
The semantic fixture uses ordinary torch reductions/exp, so its comparison is tolerance
based. It does not establish g4-dflash-v1 device equality or full-model fidelity.
"""
from pathlib import Path
import hashlib
import json
import struct

import mpmath as mp
import numpy as np
import torch

OUT = Path(__file__).parent / "fixtures"
OUT.mkdir(exist_ok=True)


def rn(value):
    """Nearest binary32, verified against exact adjacent float values, ties to even."""
    if value == 0:
        return np.float32(0)
    sign = -1 if value < 0 else 1
    value = abs(value)
    bits = struct.unpack("<I", struct.pack("<f", float(value)))[0]
    candidates = []
    for b in range(max(0, bits - 1), bits + 2):
        f = struct.unpack("<f", struct.pack("<I", b))[0]
        candidates.append((abs(mp.mpf(f) - value), b & 1, f))
    return np.float32(sign * min(candidates)[2])


def rope(positions, dim):
    cs, sn = [], []
    frequencies = []
    for i in range(dim // 2):
        exponent = np.float32(np.float32(2 * i) / np.float32(dim))
        power = rn(mp.power(mp.mpf(1000000), mp.mpf(float(exponent))))
        frequencies.append(np.float32(np.float32(1) / power))
    for p in range(positions):
        for frequency in frequencies:
            angle = np.float32(np.float32(p) * frequency)
            cs.append(rn(mp.cos(mp.mpf(float(angle)))))
            sn.append(rn(mp.sin(mp.mpf(float(angle)))))
    return np.asarray(cs, dtype="<f4"), np.asarray(sn, dtype="<f4")


def save_rope(name, positions, dim):
    mp.mp.dps = 100
    cs, sn = rope(positions, dim)
    mp.mp.dps = 180
    cs_check, sn_check = rope(positions, dim)
    assert cs.tobytes() == cs_check.tobytes() and sn.tobytes() == sn_check.tobytes()
    data = struct.pack("<4sIII", b"G4DR", 1, positions, dim) + cs.tobytes() + sn.tobytes()
    (OUT / name).write_bytes(data)
    return torch.from_numpy(cs.copy()).reshape(positions, dim // 2), torch.from_numpy(sn.copy()).reshape(positions, dim // 2)


def matrix(outputs, width, seed):
    a = (torch.arange(outputs * width, dtype=torch.int64) * 7 + seed) % 17 - 8
    return a.float().reshape(outputs, width) / 64


def norm(x, weight):
    return (x * torch.pow(x.square().mean(-1, keepdim=True) + 1e-6, -0.5)) * weight


def rotate(x, positions, cs, sn):
    a, b = x[..., :4], x[..., 4:]
    c, s = cs[positions, None, :], sn[positions, None, :]
    return torch.cat((a * c - b * s, b * c + a * s), -1)


def semantic_fixture(cs, sn):
    h, inter, qheads, kvheads, dim, ctx = 16, 32, 4, 1, 8, 3
    taps = ((torch.arange(ctx * 5 * h) * 3) % 29 - 14).float().reshape(ctx, 5 * h) / 16
    embedded = ((torch.arange(8 * h) * 5) % 23 - 11).float().reshape(8, h) / 8
    c = norm(taps @ matrix(h, 5 * h, 1).T, 1.0)
    residual = embedded
    ctx_pos = torch.arange(ctx)
    block_pos = torch.arange(ctx, ctx + 8)
    keys, values = [], []
    for layer in range(2):
        x = norm(residual, 0.875)
        q = norm((x @ matrix(qheads * dim, h, 2 + layer).T).reshape(8, qheads, dim), 0.75)
        kctx = norm((c @ matrix(kvheads * dim, h, 3 + layer).T).reshape(ctx, kvheads, dim), 1.125)
        kb = norm((x @ matrix(kvheads * dim, h, 3 + layer).T).reshape(8, kvheads, dim), 1.125)
        vctx = (c @ matrix(kvheads * dim, h, 4 + layer).T).reshape(ctx, kvheads, dim)
        vb = (x @ matrix(kvheads * dim, h, 4 + layer).T).reshape(8, kvheads, dim)
        q = rotate(q, block_pos, cs, sn)
        kctx = rotate(kctx, ctx_pos, cs, sn)
        kb = rotate(kb, block_pos, cs, sn)
        keys.append(kctx); values.append(vctx)
        k = torch.cat((kctx, kb), 0).repeat_interleave(4, dim=1)
        v = torch.cat((vctx, vb), 0).repeat_interleave(4, dim=1)
        scores = torch.einsum("rhd,jhd->rhj", q, k) * np.float32(128 ** -0.5)
        att = torch.einsum("rhj,jhd->rhd", torch.softmax(scores, -1), v).reshape(8, qheads * dim)
        residual = residual + att @ matrix(h, qheads * dim, 5 + layer).T
        x = norm(residual, 0.625)
        gate = x @ matrix(inter, h, 6 + layer).T
        up = x @ matrix(inter, h, 7 + layer).T
        residual = residual + (torch.nn.functional.silu(gate) * up) @ matrix(h, inter, 8 + layer).T
    result = norm(residual, 0.75)
    all_values = torch.cat((result.flatten(), torch.stack(keys).flatten(), torch.stack(values).flatten()))
    (OUT / "torch-forward.f32").write_bytes(all_values.numpy().astype("<f4").tobytes())


torch.set_num_threads(1)
save_rope("rope128-p0-p8.g4dr", 9, 128)
cs, sn = save_rope("rope8-p0-p2063.g4dr", 2064, 8)
semantic_fixture(cs, sn)
manifest = {
    "purpose": "small host semantics and immutable rotary fixtures; no device/full-model qualification",
    "torch_version": torch.__version__, "mpmath_version": mp.__version__,
    "rounding_precision_decimal_digits": [100, 180],
    "rotary": {"theta": 1000000, "layout": "NeoX", "recipe": "FP32 exponent, rounded real pow, FP32 reciprocal/angle, rounded real sin/cos"},
    "forward": {"hidden": 16, "layers": 2, "q_heads": 4, "kv_heads": 1, "head_dim": 8, "inter": 32,
                "context_rows": 3, "block_rows": 8, "attention_scale": float(np.float32(128 ** -0.5)),
                "layout": "final normalized block, context K all layers, context V all layers", "absolute_tolerance": 0.00002},
    "sha256": {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted(OUT.iterdir()) if p.suffix != ".json"},
}
(OUT / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
print(json.dumps(manifest, indent=2))
