"""Generate small codec/linear fixtures; NumPy and the independent decoder only here."""
import argparse
import json
from pathlib import Path
import sys
import numpy as np

parser = argparse.ArgumentParser()
parser.add_argument("--decoder-root", type=Path, required=True)
args = parser.parse_args()
sys.path.insert(0, str(args.decoder_root / "python"))
from strix_trellis.codec import dequantize_rotated, decode_states
from strix_trellis.linear import input_stages_fp32, output_stages_fp32

root = Path(__file__).parent / "fixtures"
root.mkdir(exist_ok=True)
rng = np.random.default_rng(871)
x = (rng.standard_normal((8, 256)) * 0.5).astype("<f4")
x[0] = 0.0
x[1] = np.where(np.arange(256) % 2, np.float32(-0.0), np.float32(0.0))
x[2] = np.where(np.arange(256) % 2, -1, 1) * np.float32(2.0**-22)
x[3] = np.where(np.arange(256) % 3, -1, 1) * np.float32(2.0**-23)
x[4] = np.where(np.arange(256) % 2, -256, 256)
x[5] = np.resize(np.array([1.00048828125, 1.00146484375, 2.0009765625, -1.00048828125], dtype=np.float32), 256)
x[6] *= 100
x.tofile(root / "input.f32")
decode_states(np.arange(65536, dtype=np.uint32)).astype("<f2").tofile(root / "codebook.f16")
for k in (4, 5, 6, 8):
    packed = rng.integers(-32768, 32768, (16, 8, 16 * k), dtype="<i2")
    su = rng.choice(np.array([-0.25, -0.125, 0.125, 0.25], dtype="<f2"), 256)
    sv = rng.choice(np.array([-0.25, -0.125, 0.125, 0.25], dtype="<f2"), 128)
    with (root / f"k{k}.bin").open("wb") as f:
        f.write(su.tobytes()); f.write(sv.tobytes()); f.write(packed.tobytes())
    rotated = dequantize_rotated(packed, k)
    rotated.astype("<f2").tofile(root / f"k{k}-rotated.f16")
    transformed = input_stages_fp32(x, su)["core_input_fp16"].astype(np.float32)
    acc = np.zeros((32, 8, 128), dtype=np.float32)
    for lane in range(32):
        for chunk in range(lane, 16, 32):
            for element in range(16):
                i = chunk * 16 + element
                product = transformed[:, i, None] * rotated[i].astype(np.float32)
                acc[lane] = acc[lane] + product
    for distance in (16, 8, 4, 2, 1):
        acc[:distance] = acc[:distance] + acc[distance:2 * distance]
    out = output_stages_fp32(acc[0], sv)["output_fp32"]
    out.astype("<f4").tofile(root / f"k{k}-output.f32")
(root / "manifest.json").write_text(json.dumps(dict(seed=871, inputs=256, outputs=128, rows=8,
    reference="Independent mul1 ring decoder and ascending-stride FP32 transforms; chunk16 wave dots",
    description="Synthetic signed zeros, half subnormals/underflow ties, large and RNE-boundary inputs; no model weights"), indent=2) + "\n")
