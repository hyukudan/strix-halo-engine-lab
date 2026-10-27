"""EXL3 requant of the MiniMax target's lm_head for a DSpark draft-only head (item 6 of the drafter plan): the draft reads
the K8 head (614 MB per rank half, ~86 % of the read ceiling already), a narrower private copy reads fewer bytes. The
target's own head is untouched, so verification keeps the output exact; only the draft's acceptance can change
(accept_eval.py --head-alt measures it).

Source = the exact float64 reconstruction of the pack's K8 head (accept_eval.AltHead), quantized with exllamav3's
quantizer (MIT, used as a tool, the trellis venv) under the head-input Hessian of accept_eval.py --head-hessian (draft
rows y after the final norm, real target taps). One output dir per K: model-head-exl3.safetensors with the pack's tensor
names (language_model.lm_head.{trellis,suh,svh,mul1}) + model.safetensors.index.json + proxy_err.tsv.

usage (trellis venv, a CUDA GPU):
  python quant_head.py --hessian HEAD.npz --K 4 6 --out-prefix DRAFTS/head-K
"""
import argparse
import json
from pathlib import Path
import sys
import time

import numpy as np
import torch

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from accept_eval import AltHead  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--trellis-python", required=True)
    ap.add_argument("--hessian", required=True)
    ap.add_argument("--K", type=float, nargs="+", required=True)
    ap.add_argument("--sigma-reg", type=float, default=0.025)
    ap.add_argument("--out-prefix", required=True, help="output dir = <prefix><K>")
    ap.add_argument("--rows", type=int, default=0, help="smoke test: only the first ROWS outputs (multiple of 128)")
    a = ap.parse_args()
    from exllamav3.modules.quant.exl3_lib.quantize import quantize_exl3
    from safetensors.torch import save_file

    dev = torch.device("cuda:0")
    t0 = time.time()
    W = AltHead("K8", a.model, a.trellis_python).W  # [vocab, 6144] float32 of the float64 reconstruction
    if a.rows:
        W = W[:a.rows].clone()
    hz = np.load(a.hessian)
    H, n = torch.from_numpy(hz["head"]).double(), int(hz["n.head"])
    print(time.strftime("%H:%M:%S"), f"head {tuple(W.shape)} loaded, Hessian over {n} rows, {time.time() - t0:.0f} s", flush=True)
    for K in a.K:
        out = Path(f"{a.out_prefix}{int(K) if float(K).is_integer() else K}")
        out.mkdir(parents=True, exist_ok=True)
        H_data = {"H": H.float().to(dev), "count": n, "finalized": False, "device": dev, "first_key": AltHead.NAME}
        qa = {"K": int(K) if float(K).is_integer() else K, "seed": 787, "sigma_reg": a.sigma_reg, "devices": [dev],
              "mul1": True, "apply_out_scales": None}
        _, perr, packed = quantize_exl3(W.T.contiguous().to(dev), H_data, qa, return_weight_q=False)
        store = {f"{AltHead.NAME}.{k}": v.contiguous().cpu() for k, v in packed.items()}
        save_file(store, str(out / "model-head-exl3.safetensors"), metadata={"format": "pt"})
        json.dump({"weight_map": {k: "model-head-exl3.safetensors" for k in store}}, open(out / "model.safetensors.index.json", "w"))
        (out / "proxy_err.tsv").write_text(f"{AltHead.NAME}\t{K}\t{float(perr):.6g}\n")
        (out / "config.json").write_text(json.dumps({"quantization_config": {
            "quant_method": "exl3", "codebook": "mul1", "K": K, "tensor": AltHead.NAME,
            "source": "float64 reconstruction of the pack's K8 head", "hessian": Path(a.hessian).name, "rows": n}}, indent=1))
        print(time.strftime("%H:%M:%S"), f"K{K} proxy_err {float(perr):.5g} -> {out}", flush=True)
        del packed, store
        torch.cuda.empty_cache()
    print(json.dumps({"seconds": round(time.time() - t0)}), flush=True)


if __name__ == "__main__":
    main()
