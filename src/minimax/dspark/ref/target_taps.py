"""Real DSpark inputs from the MiniMax target on the PC CPU: the CPU production chain of the G4 oracle (
cpu_layer_reference_fast.py, imported read-only, the same arithmetic) over one prompt, keeping what the drafter needs:

  tokens.u32           [T]
  taps.f32             [T][5][6144]  residual after target layers 3, 17, 31, 45, 59 (prod chain, FP32)
  greedy.u32           [T]           target argmax after each row (final norm + EXL3 K8 head, prod numerics)
  margin.f32           [T]           top-1 minus top-2 logit of that row
  meta.json

usage: python target_taps.py --model path/to/fixture --trellis-python path/to/fixture
           --tokens 2048 --text FILE [FILE ...] --out DIR
"""
import argparse
import json
import os
from pathlib import Path
import sys
import time

import numpy as np
import torch

ORACLE_DIR = os.environ.get("MINIMAX_ORACLE_DIR", "minimax-oracle")
sys.path.insert(0, ORACLE_DIR)
import cpu_layer_reference_fast as F  # noqa: E402

R = F.R
TAPS = (3, 17, 31, 45, 59)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--trellis-python", required=True)
    ap.add_argument("--tokens", type=int, default=2048)
    ap.add_argument("--text", nargs="+", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--smoke-layers", type=int, default=0, help="plumbing check: first N layers only")
    args = ap.parse_args()
    args.dump_layers, args.full_layers = None, args.smoke_layers
    torch.set_num_threads(max(1, torch.get_num_threads()))
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    pack = R.Pack(args.model, args.trellis_python)
    F.install_fast_hadamards(pack.st)
    ids, x = F.load_prompt(pack, args)
    T = len(ids)
    np.asarray(ids, dtype="<u4").tofile(out / "tokens.u32")
    taps = np.lib.format.open_memmap(out / "taps.npy", mode="w+", dtype=np.float32, shape=(T, len(TAPS), 6144))
    t_start = time.time()

    def emit(L, h, t0, loader):
        if L in TAPS:
            taps[:, TAPS.index(L), :] = h.astype(np.float32)
            taps.flush()
        print(json.dumps({"layer": L, "seconds": round(time.time() - t0, 1)}), flush=True)

    xp, loader = F.run_chain(pack, pack.cfg, x, args, "prod", emit)
    del taps
    np.load(out / "taps.npy", mmap_mode="r").tofile(out / "taps.f32")
    os.remove(out / "taps.npy")
    # Final norm + head on every row, prod numerics (head_logits' prod half), in row chunks.
    head = loader.get("language_model.lm_head", casts=False)
    st = pack.st
    wn = pack.plain(f"{R.PFX}norm.weight")
    greedy = np.empty(T, np.uint32)
    margin = np.empty(T, np.float32)
    V = head.rot.shape[1]
    for r0 in range(0, T, 256):
        r1 = min(T, r0 + 256)
        hp = R.gemma_rmsnorm(xp[r0:r1], wn, pack.cfg["rms_norm_eps"], np.float32)
        ap_ = torch.from_numpy(st.input_stages_fp32(hp, head.suh)["core_input_fp16"].astype(np.float32))
        lp = np.empty((r1 - r0, V), np.float32)
        for c0 in range(0, V, 16384):
            c1 = min(V, c0 + 16384)
            corep = (ap_ @ torch.from_numpy(head.rot[:, c0:c1].astype(np.float32))).numpy()
            lp[:, c0:c1] = st.output_stages_fp32(corep, head.svh[c0:c1])["output_fp32"]
        top2 = np.argpartition(-lp, 1, axis=1)[:, :2]
        v2 = np.take_along_axis(lp, top2, axis=1)
        order = np.argsort(-v2, axis=1, kind="stable")
        greedy[r0:r1] = lp.argmax(1)
        margin[r0:r1] = np.take_along_axis(v2, order, 1)[:, 0] - np.take_along_axis(v2, order, 1)[:, 1]
    greedy.astype("<u4").tofile(out / "greedy.u32")
    margin.astype("<f4").tofile(out / "margin.f32")
    agree = float((greedy[:-1] == np.asarray(ids[1:])).mean())
    meta = {"tokens": T, "taps": TAPS, "text": [str(p) for p in args.text], "greedy_matches_text": agree,
            "seconds": round(time.time() - t_start, 1), "chain": "cpuprod (fast oracle)"}
    (out / "meta.json").write_text(json.dumps(meta, indent=1))
    print(json.dumps(meta), flush=True)


if __name__ == "__main__":
    main()
