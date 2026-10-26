"""Export one real MSA decode case (oracle values) for strix-minimax-msa-tests (STRIX_MINIMAX_MSA_CASE=DIR).

Runs the float64 oracle attention of cpu_layer_reference.py for one layer over a real prompt and writes, for a
few query rows, the normed/roped q, the full k/v/index-k caches up to the last row, the index q, the oracle's
selected blocks per index head and the oracle attention output (before o_proj). Plus the raw q and the q norm
weight of the first row to check the norm + RoPE helper.
"""
import argparse
from pathlib import Path
import sys

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import cpu_layer_reference as ref  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--trellis-python", required=True)
    ap.add_argument("--text", nargs="+", required=True)
    ap.add_argument("--tokens", type=int, default=4096)
    ap.add_argument("--layer", type=int, default=0)
    ap.add_argument("--rows", default="100,2047,2048,3001,4095")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    pack = ref.Pack(args.model, args.trellis_python)
    from tokenizers import Tokenizer
    tok = Tokenizer.from_file(str(Path(args.model) / "dspark" / "tokenizer.json"))
    text = "\n\n".join(Path(p).read_text(encoding="utf-8") for p in args.text)
    ids = tok.encode(text).ids[: args.tokens]
    T = len(ids)
    x = pack.embed_rows(ids)
    oracle = ref.Oracle(pack, pack.cfg)
    for L in range(args.layer + 1):  # real input of the requested layer
        a = f"{ref.PFX}layers.{L}.self_attn."
        names = [a + n for n in ("q_proj", "k_proj", "v_proj", "o_proj")]
        lin = ref.load_linears(pack, names, 8)
        for n in names:
            lin[n].W = lin[n].original_weight()
        xa, info = oracle.attention(L, x, np.arange(T), lin)
        if L == args.layer:
            break
        mlp = ([f"{ref.PFX}layers.{L}.mlp.{n}" for n in ("gate_proj", "up_proj", "down_proj")])
        lin = ref.load_linears(pack, mlp, 8)
        for n in mlp:
            lin[n].W = lin[n].original_weight()
        x, _ = oracle.ffn(L, xa, lin)
    rows = [int(r) for r in args.rows.split(",")]
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    f32 = lambda a: np.ascontiguousarray(a, dtype="<f4")
    f32(info["q"][rows]).tofile(out / "q.f32")                  # [rows][64][128]
    f32(info["k"]).tofile(out / "k.f32")                         # [T][4][128]
    f32(info["v"]).tofile(out / "v.f32")                         # [T][4][128]
    f32(info["iq"][rows]).tofile(out / "iq.f32")                 # [rows][4][128]
    f32(info["ik"][:, 0, :]).tofile(out / "ik.f32")              # [T][128]
    f32(info["attn"][rows]).tofile(out / "attn.f32")             # [rows][64][128]
    f32(info["q_raw"][rows[0]]).tofile(out / "q_raw0.f32")       # [64][128]
    f32(pack.plain(f"{ref.PFX}layers.{args.layer}.self_attn.q_norm.weight")).tofile(out / "q_norm_w.f32")
    sel = info["sel"]  # [4][T][blocks]
    with open(out / "sel.txt", "w") as f:
        for r in rows:
            for g in range(sel.shape[0]):
                f.write(" ".join(str(b) for b in np.nonzero(sel[g, r])[0]) + "\n")
    theta = pack.cfg["rope_theta"]
    (out / "meta.txt").write_text(f"{T} {len(rows)} {theta} {args.layer}\n" + " ".join(map(str, rows)) + "\n")
    print("exported", out, "T", T, "rows", rows)


if __name__ == "__main__":
    main()
