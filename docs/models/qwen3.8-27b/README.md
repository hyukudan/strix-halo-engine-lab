# Qwen3.8-27B (qwen35 dense) - CUDA notes

## Dense arithmetic identity (lead decision, 2026-11-16)
The qwen35 CUDA exact identity is `qwen35-cuda-exact-mma3-v1`: BF16 tensor-core `mma.sync m16n8k16` with FP32 accumulate,
the FP32 activation split exactly into three bf16 terms (hi + mid + lo, products with bf16 weights exact), accumulated
lo, mid, hi. One fixed K order for every row count 1..32 (rows ride the N dimension, so each row is bitwise identical to
rows = 1): 32-element K blocks, warp w of 8 owns groups of U consecutive blocks (U = 4 bf16, 8 q8_0), the 8 warp partials are
summed in ascending warp order. q8_0: int8 values are exact in bf16; each 32-block partial starts at zero and is folded as
`tot = tot + fl(d * part)`. Source: src/cuda_core/dense/dense_mma.cu.
- `--dense-arith fp32` keeps the CUDA-core FP32 kernels as the reference / off switch (documented order in dense.hpp).
- FAST policies: `mma2` (hi + mid), `mma1` (hi only; max |dlogit| 8x larger, see RESULT). Not the default.
- Shapes not covered (in % 32, out % 16, f32 weights) fall back to fp32 kernels.
