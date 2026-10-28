"""EXL3 requant of the MiniMax DSpark draft with exllamav3's quantizer (MIT, used as a tool, the trellis venv), from the
exact dequantization of the shipped MXFP8 linears, with the Hessians of accept_eval.py --hessians (real target taps).

Every draft linear becomes EXL3 tensors `<name>.{trellis,suh,svh,mul1}` (the target pack's naming, so the engine's trellis
loader reads them); fc optionally too; norms, Markov and confidence heads are copied as they are (BF16). Writes
model-dspark-exl3.safetensors + model.safetensors.index.json + config.json (+ quantization note) and proxy_err.tsv.

Hessian sites (accept_eval.py): fc (taps), L{l}.qkv (block rows into q/k/v), L{l}.kv (context rows into k/v), L{l}.o,
L{l}.gu (gate/up), L{l}.down. k/v see both block and context rows: H = H_qkv + H_kv.

usage (trellis venv, RTX PRO; a host ledger line first):
  python quant_draft.py --draft DIR --hessians H.npz --K 4 [--k-map down=5,fc=6] [--fc] --out DIR
"""
import argparse
import json
import os
from pathlib import Path
import shutil
import sys
import time

import numpy as np
import torch

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import dspark_reference as D  # noqa: E402

P = D.P
SITE = {"q_proj": "qkv", "k_proj": "qkv+kv", "v_proj": "qkv+kv", "o_proj": "o", "gate_proj": "gu", "up_proj": "gu",
        "down_proj": "down"}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--draft", required=True)
    ap.add_argument("--hessians", required=True)
    ap.add_argument("--K", type=float, required=True)
    ap.add_argument("--k-map", default="", help="per-kind override: q=5,k=6,v=6,o=5,gate=4,up=4,down=5,fc=6")
    ap.add_argument("--fc", action="store_true", help="also quantize fc (else BF16 as shipped)")
    ap.add_argument("--sigma-reg", type=float, default=0.025)
    ap.add_argument("--no-scaled-hessian", action="store_true",
                    help="turn off scaled_hessian (on by default: LDLQ uses the Hessian with the input-channel scales folded in)")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    from exllamav3.modules.quant.exl3_lib.quantize import quantize_exl3
    from safetensors.torch import save_file

    dev = torch.device("cuda:0")
    kmap = {k: float(v) for k, v in (p.split("=") for p in a.k_map.split(",") if p)}
    hz = np.load(a.hessians)
    dr = D.Draft(a.draft)
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    store, tsv = {}, open(out / "proxy_err.tsv", "w")

    def hessian(site):
        H, n = None, 0
        for s in site.split("+"):
            for key in [s]:
                h = torch.from_numpy(hz[key]).double()
                H = h if H is None else H + h
                n += int(hz["n." + key])
        return H, n

    def quant(name, W_oi, site, K):
        H, n = hessian(site)
        H_data = {"H": H.float().to(dev), "count": n, "finalized": False, "device": dev, "first_key": name}
        qa = {"K": int(K) if float(K).is_integer() else K, "seed": 787, "sigma_reg": a.sigma_reg, "devices": [dev],
              "mul1": True, "apply_out_scales": None,
              "scaled_hessian": not a.no_scaled_hessian}
        _, perr, packed = quantize_exl3(W_oi.T.contiguous().float().to(dev), H_data, qa, return_weight_q=False)
        for k, v in packed.items():
            store[f"{name}.{k}"] = v.contiguous().cpu()
        tsv.write(f"{name}\t{K}\t{float(perr):.6g}\n")
        tsv.flush()
        print(time.strftime("%H:%M:%S"), name, K, f"proxy_err {float(perr):.5g}", flush=True)

    t0 = time.time()
    for li in range(D.L):
        for mod, kinds in (("self_attn", ("q_proj", "k_proj", "v_proj", "o_proj")), ("mlp", ("gate_proj", "up_proj", "down_proj"))):
            for kind in kinds:
                rel = f"layers.{li}.decoder_layer.{mod}.{kind}"
                W = dr.linear(rel)  # exact float64 dequantization of the MXFP8 tensor
                K = kmap.get(kind.split("_")[0], a.K)
                site = "+".join(f"L{li}.{s}" for s in SITE[kind].split("+"))
                quant(P + rel, W, site, K)
    for name, (dt, shape, raw) in dr.t.items():
        if dt == "BF16" and not (name.endswith("fc.weight") and a.fc):
            store[name] = D.bf16(raw, shape).to(torch.bfloat16).contiguous()
    if a.fc:
        quant(P + "fc", dr.plain("fc.weight"), "fc", kmap.get("fc", a.K))
    save_file(store, str(out / "model-dspark-exl3.safetensors"), metadata={"format": "pt"})
    json.dump({"weight_map": {k: "model-dspark-exl3.safetensors" for k in store}}, open(out / "model.safetensors.index.json", "w"))
    cfg = json.loads((Path(a.draft) / "config.json").read_text())
    cfg["quantization_config"] = {"quant_method": "exl3", "codebook": "mul1", "K": a.K, "k_map": kmap, "fc": a.fc,
                                  "source": "dequantized MXFP8 dspark", "hessians": os.path.basename(a.hessians),
                                  "scaled_hessian": not a.no_scaled_hessian}
    (out / "config.json").write_text(json.dumps(cfg, indent=1))
    print(json.dumps({"out": str(out), "tensors": len(store), "seconds": round(time.time() - t0)}), flush=True)


if __name__ == "__main__":
    main()
