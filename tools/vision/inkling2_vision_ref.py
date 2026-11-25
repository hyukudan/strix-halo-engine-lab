#!/usr/bin/env python3
"""Inkling 2 vision reference: the model's own definition (transformers >= 5.24 Inkling2VisionModel and
Inkling2ImageProcessor) run on the checkpoint's vision tensors (CPU). Needs torch, torchvision, transformers, safetensors,
numpy and pillow. Writes, per image, little-endian float32 dumps for strix-inkling2-vision-gate:

  NAME.pixels.f32    processor output pixel_values [tokens][40][40][3] in [-1, 1]
  NAME.png           the image (synthetic images are written here)
  NAME.grid.json     {"rows": token rows, "cols": token cols, "tokens": N, "width": W, "height": H, "dtype": ...}
  NAME.hidden.f32    encoder output (CLS rows included, CLS first) [1 + 4N][1024]
  NAME.rows.f32      projected image rows (the language model embeddings) [N][2304]

usage: inkling2_vision_ref.py --checkpoint DIR --out DIR [--dtype float32|bfloat16] IMAGE...  (IMAGE may be synth:WxH[:seed])
"""
import argparse, io, json, os, re, stat, sys
from contextlib import contextmanager

import numpy as np
import torch
from PIL import Image
from safetensors import safe_open
from transformers.models.inkling2.configuration_inkling2 import Inkling2VisionConfig
from transformers.models.inkling2.image_processing_inkling2 import Inkling2ImageProcessor
from transformers.models.inkling2.modeling_inkling2 import Inkling2VisionModel


def rename(key):
    # conversion_mapping.py "inkling2" rules, vision part only
    k = key.replace("model.visual.", "")
    k = k.replace("backbone.embeddings.position_embedding", "embeddings.position_embedding.weight")
    k = k.replace("backbone.embeddings.", "embeddings.")
    k = k.replace("backbone.ln_pre.", "pre_layernorm.")
    k = k.replace("backbone.layers.", "layers.")
    k = k.replace("proj_in.", "merger.linear_1.").replace("proj_out.", "merger.linear_2.")
    for a, b in ((".ln_1.", ".layernorm_before."), (".ln_2.", ".layernorm_after."), (".mlp.c_fc.", ".mlp.fc1."),
                 (".mlp.c_proj.", ".mlp.fc2."), (".ls_1.gamma", ".lambda_1"), (".ls_2.gamma", ".lambda_2")):
        k = k.replace(a, b)
    return k


def load_model(checkpoint, dtype):
    cfg_all = json.loads(read_bounded(checkpoint, "config.json", MAX_CONFIG_BYTES))
    vc = dict(cfg_all["vision_config"])
    if vc.pop("model_type", None) != "inkling2_vision":
        sys.exit("not an Inkling 2 vision config")
    config = Inkling2VisionConfig(**vc)
    for key, expected in {
        "hidden_size": 1024, "num_hidden_layers": 23, "num_attention_heads": 16,
        "mlp_ratio": 4, "native_patch_size": 20, "adapter_merge_size": 2,
        "posemb_grid_size": 32, "projection_hidden_dim": 4096,
        "text_hidden_size": 2304, "projector_bias": False,
    }.items():
        if getattr(config, key, None) != expected:
            sys.exit(f"unsupported vision_config.{key}")
    model = Inkling2VisionModel(config).eval()
    state = {}
    # safe_open needs a path for its mmap. The procfs descriptor alias names the
    # verified inode while the original file handle stays live.
    with regular_input(checkpoint, "model-00001-of-00001.safetensors", MAX_SHARD_BYTES) as (shard, size):
        header = shard.read(8)
        if len(header) != 8 or not 0 < int.from_bytes(header, "little") <= min(size - 8, 64 << 20):
            sys.exit("safetensors header outside bounds")
        with safe_open(f"/proc/self/fd/{shard.fileno()}", "pt") as f:
            for key in f.keys():
                if key.startswith("model.visual."):
                    state[rename(key)] = f.get_tensor(key)
    missing, unexpected = model.load_state_dict(state, strict=False)
    missing = [m for m in missing if "rotary_emb" not in m]
    if missing or unexpected:
        sys.exit(f"weight mismatch: missing {missing} unexpected {unexpected}")
    return model.to(dtype), config


MAX_CONFIG_BYTES = 1 << 20
MAX_SHARD_BYTES = 64 << 30
MAX_IMAGE_BYTES = 20 << 20      # the gate's decoder limit
MAX_IMAGE_PIXELS = 40_000_000
MAX_SYNTH_SIDE = 4096
MAX_IMAGE_TOKENS = 4096
CELL = 40
SYNTH = re.compile(r"synth:([0-9]{1,5})x([0-9]{1,5})(?::([0-9]{1,9}))?")
SIMPLE_NAME = re.compile(r"[A-Za-z0-9_-][A-Za-z0-9._-]{0,127}")


@contextmanager
def directory(path):
    fd = os.open(path, os.O_RDONLY | os.O_DIRECTORY | os.O_CLOEXEC)
    try:
        yield fd
    finally:
        os.close(fd)


@contextmanager
def output_directory(path):
    # The output leaf is created (if missing) and opened relative to its held parent without following a link, so a
    # symlink at the leaf, or one swapped in after creation, is refused instead of redirecting the case files.
    parent_path, leaf = os.path.split(os.path.normpath(path))
    if leaf in ("", ".", ".."):
        sys.exit(f"{path}: name the output directory itself")
    with directory(parent_path or ".") as parent:
        try:
            os.mkdir(leaf, 0o700, dir_fd=parent)
        except FileExistsError:
            pass
        try:
            fd = os.open(leaf, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW | os.O_CLOEXEC, dir_fd=parent)
        except OSError as error:
            sys.exit(f"{path}: the output must be a real directory, not a link ({error.strerror})")
        try:
            yield fd
        finally:
            os.close(fd)


@contextmanager
def regular_input(parent, name, cap):
    flags = os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK | os.O_CLOEXEC
    with os.fdopen(os.open(name, flags, dir_fd=parent), "rb") as stream:
        st = os.fstat(stream.fileno())
        if not stat.S_ISREG(st.st_mode) or not 0 <= st.st_size <= cap:
            sys.exit(f"{name}: not a regular file of at most {cap} bytes")
        yield stream, st.st_size


def read_bounded(parent, name, cap):
    with regular_input(parent, name, cap) as (stream, size):
        data = stream.read(size)
        if len(data) != size or stream.read(1):
            sys.exit(f"{name}: file length changed during read")
        return data


@contextmanager
def new_output(parent, name):
    flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW | os.O_CLOEXEC
    with os.fdopen(os.open(name, flags, 0o600, dir_fd=parent), "wb") as stream:
        yield stream


def checked_grid(width, height):
    rows, cols = (height + CELL - 1) // CELL, (width + CELL - 1) // CELL
    if not (0 < width * height <= MAX_IMAGE_PIXELS and 0 < rows * cols <= MAX_IMAGE_TOKENS):
        sys.exit(f"{width}x{height}: raster exceeds the reference pixel or token limit")
    return rows, cols


def synth_fields(spec):
    """synth:WxH[:seed], 1 <= W, H <= 4096; anything else is refused."""
    m = SYNTH.fullmatch(spec)
    if not m:
        sys.exit(f"{spec!r}: expected synth:WxH[:seed]")
    w, h, seed = int(m.group(1)), int(m.group(2)), int(m.group(3) or 0)
    if not (1 <= w <= MAX_SYNTH_SIDE and 1 <= h <= MAX_SYNTH_SIDE):
        sys.exit(f"{spec!r}: each side must be 1..{MAX_SYNTH_SIDE}")
    return w, h, seed


def synth(w, h, seed):
    rng = np.random.default_rng(seed)
    # smooth gradients plus noise and a few flat blocks: exercises padding, edges and flat areas
    y, x = np.mgrid[0:h, 0:w]
    img = np.stack([(x * 255 / max(w - 1, 1)), (y * 255 / max(h - 1, 1)), ((x + y) % 64) * 4], -1)
    img = img + rng.normal(0, 12, img.shape)
    img[h // 4: h // 2, w // 4: w // 2] = [200, 30, 30]
    return Image.fromarray(np.clip(img, 0, 255).astype(np.uint8), "RGB")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--dtype", default="float32", choices=["float32", "bfloat16"])
    ap.add_argument("images", nargs="+")
    a = ap.parse_args()
    if not sys.platform.startswith("linux") or not os.path.isdir("/proc/self/fd"):
        sys.exit("reference generation requires Linux with procfs")
    dtype = getattr(torch, a.dtype)
    torch.set_num_threads(os.cpu_count() or 8)
    with directory(a.checkpoint) as checkpoint, output_directory(a.out) as output:
        write_cases(a, checkpoint, output, dtype)


def write_cases(a, checkpoint, output, dtype):
    proc = model = None
    for spec in a.images:
        png = None
        if spec.startswith("synth:"):
            w, h, seed = synth_fields(spec)
            expected_grid = checked_grid(w, h)
            image, name, source = synth(w, h, seed), f"synth_{w}x{h}_{seed}", f"synth:{w}x{h}:{seed}"
        else:
            name, source = os.path.splitext(os.path.basename(spec))[0], os.path.basename(spec)
            if not SIMPLE_NAME.fullmatch(name):
                sys.exit(f"{spec}: the file name must be a simple name ([A-Za-z0-9._-])")
            with directory(os.path.dirname(spec) or ".") as parent:
                png = read_bounded(parent, source, MAX_IMAGE_BYTES)
            with Image.open(io.BytesIO(png), formats=["PNG"]) as opened:
                expected_grid = checked_grid(opened.width, opened.height)
                image = opened.copy()
        if model is None:
            model, _ = load_model(checkpoint, dtype)
            proc = Inkling2ImageProcessor()
        feats = proc(images=[image], return_tensors="pt")
        pixels, grid = feats["pixel_values"], feats["image_grid_thw"]
        t, gr, gc = (int(v) for v in grid[0])
        if t != 1 or (gr, gc) != expected_grid:
            sys.exit(f"{name}: processor grid differs from the admitted raster")
        with torch.no_grad():
            out = model(pixel_values=pixels.to(dtype), image_grid_thw=grid)
        rows = out.pooler_output.float().numpy()
        hidden = out.last_hidden_state.float().numpy()
        with new_output(output, name + ".png") as target:
            if png is None:
                image.save(target, format="PNG")
            else:
                target.write(png)
        for suffix, array in [(".pixels.f32", pixels.float().numpy()), (".hidden.f32", hidden), (".rows.f32", rows)]:
            with new_output(output, name + suffix) as target:
                array.astype("<f4").tofile(target)
        meta = {"rows": gr, "cols": gc, "tokens": gr * gc, "width": image.width, "height": image.height,
                "dtype": a.dtype, "source": source}
        with new_output(output, name + ".grid.json") as target:
            target.write(json.dumps(meta).encode("utf-8"))
        print(f"{name}: {image.width}x{image.height} -> {gr}x{gc} = {gr * gc} tokens, rows {rows.shape}, "
              f"|rows| mean {np.abs(rows).mean():.4f}", flush=True)


if __name__ == "__main__":
    main()
