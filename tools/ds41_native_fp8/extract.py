#!/usr/bin/env python3
"""Extracts the dense FP8 tensors of the original deepseek41 checkpoint for the native FP8 sites (read-only on the source).

For every layer: attn.{wq_a, wq_b, wkv, wo_a, wo_b} and ffn.shared_experts.{w1, w3, w2}, each as its F8_E4M3 codes
("<name>.weight") and F8_E8M0 32 x 32 block exponents ("<name>.scale"), copied byte for byte into one safetensors file
per layer (dense-fp8-LL.safetensors) plus a model.safetensors.index.json, so DeviceForward's native FP8 option
(STRIX_DS41_NATIVE_FP8=OUT_DIR) reads the extract as it reads the full checkpoint. About 162 MB per layer, 6.5 GB for 40.
Only headers and the selected tensors are read; nothing in the source directory is executed or written.

usage: extract.py SRC_DIR OUT_DIR [--layers 0-39] [--check]
  --check   verify an existing extract against the source (bytes, dtypes, shapes) instead of writing
"""
import argparse, hashlib, json, os, struct, sys

SITES = ["attn.wq_a", "attn.wq_b", "attn.wkv", "attn.wo_a", "attn.wo_b",
         "ffn.shared_experts.w1", "ffn.shared_experts.w3", "ffn.shared_experts.w2"]


def header(path):
    with open(path, "rb") as f:
        n = struct.unpack("<Q", f.read(8))[0]
        return 8 + n, json.loads(f.read(n))


def layers_arg(text):
    out = []
    for part in text.split(","):
        lo, _, hi = part.partition("-")
        out += list(range(int(lo), int(hi or lo) + 1))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("src")
    ap.add_argument("out")
    ap.add_argument("--layers", default="0-39")
    ap.add_argument("--check", action="store_true")
    a = ap.parse_args()
    wm = json.load(open(os.path.join(a.src, "model.safetensors.index.json")))["weight_map"]
    headers = {}
    os.makedirs(a.out, exist_ok=True)
    index = {}
    total = 0
    for L in layers_arg(a.layers):
        names = [f"layers.{L}.{s}.{k}" for s in SITES for k in ("weight", "scale")]
        entries, chunks, at = {}, [], 0
        for name in names:
            shard = wm[name]
            if shard not in headers:
                p = os.path.join(a.src, shard)
                if not os.path.exists(p):
                    sys.exit(f"missing shard {shard} for {name} (download incomplete)")
                headers[shard] = header(p)
            base, h = headers[shard]
            e = h[name]
            want = "F8_E4M3" if name.endswith(".weight") else "F8_E8M0"
            if e["dtype"] != want:
                sys.exit(f"{name}: dtype {e['dtype']}, expected {want}")
            b0, b1 = e["data_offsets"]
            with open(os.path.join(a.src, shard), "rb") as f:
                f.seek(base + b0)
                data = f.read(b1 - b0)
            entries[name] = {"dtype": e["dtype"], "shape": e["shape"], "data_offsets": [at, at + len(data)]}
            chunks.append(data)
            at += len(data)
        if any(not entries[n]["shape"] or entries[n.replace(".weight", ".scale")]["shape"] !=
               [entries[n]["shape"][0] // 32, entries[n]["shape"][1] // 32] for n in names if n.endswith(".weight")):
            sys.exit(f"layer {L}: scale shapes are not 32 x 32 blocks")
        fname = f"dense-fp8-{L:02d}.safetensors"
        js = json.dumps({"__metadata__": {"format": "pt", "source": "original checkpoint, dense FP8 sites"}, **entries},
                        separators=(",", ":")).encode()
        js += b" " * (-len(js) % 8)
        path = os.path.join(a.out, fname)
        digest = hashlib.sha256(b"".join(chunks)).hexdigest()[:16]
        if a.check:
            base, h = header(path)
            with open(path, "rb") as f:
                for name, data in zip(names, chunks):
                    e = h[name]
                    f.seek(base + e["data_offsets"][0])
                    if e["dtype"] != entries[name]["dtype"] or e["shape"] != entries[name]["shape"] or f.read(len(data)) != data:
                        sys.exit(f"{fname}: {name} differs from the source")
            print(f"{fname} matches the source ({at} bytes, sha256 {digest})")
        else:
            with open(path + ".part", "wb") as f:
                f.write(struct.pack("<Q", len(js)))
                f.write(js)
                for c in chunks:
                    f.write(c)
            os.replace(path + ".part", path)
            print(f"{fname} {at} bytes sha256 {digest}", flush=True)
        for n in names:
            index[n] = fname
        total += at
    if not a.check:
        with open(os.path.join(a.out, "model.safetensors.index.json"), "w") as f:
            json.dump({"metadata": {"total_size": total}, "weight_map": index}, f, indent=1)
    print(f"total {total} bytes in {len(set(index.values()))} files")


if __name__ == "__main__":
    main()
