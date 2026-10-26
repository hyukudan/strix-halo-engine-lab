"""Cross-check the vectorized EXL3 state decoder on real MiniMax tiles at every rate.

For one tensor per rate (K3 expert, K5 shared/dense, K6 attention, K8 head) it samples
tiles and compares strix-trellis `unpack_states_vectorized` with the independent bit
oracle of strix-trellis tests/test_reference.py (byte-by-byte shift register).
"""
import argparse
import importlib.util
import json
from pathlib import Path
import sys

import numpy as np

TENSORS = {
    3: "language_model.model.layers.3.block_sparse_moe.experts.0.w2",
    5: "language_model.model.layers.3.block_sparse_moe.shared_experts.down_proj",
    6: "language_model.model.layers.0.self_attn.o_proj",
    8: "language_model.lm_head",
}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("model")
    ap.add_argument("--trellis-repo", required=True)
    ap.add_argument("--tiles", type=int, default=96)
    args = ap.parse_args()
    repo = Path(args.trellis_repo)
    sys.path.insert(0, str(repo / "python"))
    import strix_trellis as st
    spec = importlib.util.spec_from_file_location("ref_tests", repo / "tests" / "test_reference.py")
    ref = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(ref)
    root = Path(args.model)
    index = json.loads((root / "model.safetensors.index.json").read_text())["weight_map"]
    rng = np.random.default_rng(2026)
    ok = True
    for k, prefix in TENSORS.items():
        w = st.load_exl3(root / index[prefix + ".trellis"], prefix)
        flat = np.asarray(w.trellis).reshape(-1, w.trellis.shape[-1])
        picks = np.unique(np.concatenate([[0, len(flat) - 1], rng.integers(0, len(flat), args.tiles)]))
        tiles = flat[picks]
        fast = st.unpack_states_vectorized(tiles, k)
        oracle = np.stack([ref.bit_oracle(t, k) for t in tiles])
        scalar = st.unpack_states(tiles, k)
        same = bool((fast == oracle).all() and (scalar == oracle).all())
        ok &= same
        print(json.dumps({"K": k, "tensor": prefix, "tiles": int(len(tiles)), "states_equal": same}))
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
