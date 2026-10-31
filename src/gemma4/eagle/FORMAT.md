# e4b-eagle drafter file format (GGUF v3), version 1

This is the export target of the drafter training contract (v1). A future export script writes it with
gguf-py. The engine loader is `eagle::Weights::load` in `host.cpp`. It refuses unknown or missing tensors and keys.
All shapes below are listed in GGUF dimension order, `[inputs, outputs]`, so `ne[0]` is the contiguous row.

## Metadata

| key | type | value (production) | notes |
|---|---|---|---|
| `general.architecture` | string | `e4b-eagle` | required |
| `eagle.version` | u32 | 1 | required |
| `eagle.embedding_length` | u32 | 2560 | drafter width H (= target hidden) |
| `eagle.feed_forward_length` | u32 | 5120 | GeGLU width |
| `eagle.attention.head_count` | u32 | 8 | |
| `eagle.attention.head_count_kv` | u32 | 2 | heads / kv = 4 (device attention group) |
| `eagle.attention.key_length` | u32 | 256 | must equal value_length; 256 = target sliding RoPE table |
| `eagle.attention.value_length` | u32 | 256 | |
| `eagle.attention.sliding_window` | u32 | 512 | q - p < 512 |
| `eagle.attention.scale` | f32 | 0.0625 | 1/sqrt(256) (`attn_scale=rsqrt`) or 1.0 (`one`); must be a power of two |
| `eagle.attention.layer_norm_rms_epsilon` | f32 | 1e-6 | |
| `eagle.rope.freq_base` | f32 | 10000 | NEOX over all 256 dims = target sliding table |
| `eagle.vocab_size` | u32 | 65536 | draft vocabulary rows Vd |
| `eagle.final_logit_softcapping` | f32 | 30 | 30*tanh(x/30) |
| `eagle.target_layers` | i32[] | `[11, 23, -1]` | tap order of `fc`; 0-based target layer outputs from {5,11,17,23,35}; -1 = final RMSNorm output |
| `eagle.combine` | string | `concat` | `concat` (in_proj on [h;e]) or `sum` (h+e, no in_proj) |
| `eagle.aux` | bool | false | the PLE aux adapter is not supported by engine v1 (load refuses true) |
| `eagle.target_embedding_length` | u32 | 2560 | checked against the target |
| `eagle.target_vocab_size` | u32 | 262144 | d2t range |
| `eagle.embedding_scale` | f32 | 50.5 | bf16(sqrt(2560)); checked |
| `general.license*` | string | optional | carried to receipts (OpenMDW-1.1 + notices, see ledger) |

## Tensors

Matrices are Q8_0 in production (blocks of 32 along the input dimension, fp16 scale). For fixtures, the loader also accepts
BF16 and F32. Norm vectors stay F32 and act as direct multipliers, with no +1. `d2t` is I32.

| GGUF name | PyTorch (`train/drafter.py`) | type | shape `[in, out]` |
|---|---|---|---|
| `tap_norm.{j}.weight` (j = 0..n-1) | `tap_norms.{j}.weight` | F32 | `[2560]` |
| `fc.weight` | `fc.weight` | Q8_0 | `[n*2560, 2560]` |
| `h_norm.weight` | `h_norm.weight` | F32 | `[2560]` |
| `e_norm.weight` | `e_norm.weight` | F32 | `[2560]` |
| `in_proj.weight` (concat only) | `in_proj.weight` | Q8_0 | `[5120, 2560]` |
| `blk.0.attn_norm.weight` | `layer.input_norm.weight` | F32 | `[2560]` |
| `blk.0.attn_q.weight` | `layer.q_proj.weight` | Q8_0 | `[2560, 2048]` |
| `blk.0.attn_k.weight` | `layer.k_proj.weight` | Q8_0 | `[2560, 512]` |
| `blk.0.attn_v.weight` | `layer.v_proj.weight` | Q8_0 | `[2560, 512]` |
| `blk.0.attn_q_norm.weight` | `layer.q_norm.weight` | F32 | `[256]` |
| `blk.0.attn_k_norm.weight` | `layer.k_norm.weight` | F32 | `[256]` |
| `blk.0.attn_output.weight` | `layer.o_proj.weight` | Q8_0 | `[2048, 2560]` |
| `blk.0.post_attention_norm.weight` | `layer.post_attn_norm.weight` | F32 | `[2560]` |
| `blk.0.ffn_norm.weight` | `layer.pre_ffn_norm.weight` | F32 | `[2560]` |
| `blk.0.ffn_gate.weight` | `layer.gate.weight` | Q8_0 | `[2560, 5120]` |
| `blk.0.ffn_up.weight` | `layer.up.weight` | Q8_0 | `[2560, 5120]` |
| `blk.0.ffn_down.weight` | `layer.down.weight` | Q8_0 | `[5120, 2560]` |
| `blk.0.post_ffw_norm.weight` | `layer.post_ffn_norm.weight` | F32 | `[2560]` |
| `output_norm.weight` | `final_norm.weight` | F32 | `[2560]` |
| `output.weight` | `head.weight` | Q8_0 | `[2560, 65536]` |
| `d2t` | buffer `d2t` | I32 | `[65536]`, unique ids in [0, 262144) |

There is no V-norm weight (V RMSNorm has no scale). There is no embedding: e(i) is read at runtime from the target's resident
`token_embd` (Q8_0 in the deployed Q4_K_M), as bf16(bf16(w) * 50.5), the same as the target's own input embedding.

Q8_0 rounding must match `drafter.fake_q8_` / ggml `quantize_row_q8_0`: d = amax/127 stored as fp16, q = round(x/d).
`eval_accept.py --q8` evaluates exactly these weights.

## Engine-side derived values (not stored)

- `q_norm_eff = q_norm * attention_scale`. Folding a power-of-two scale into the Q-norm multiplier is exact in FP32 (barring
  subnormals), because it scales every product of the RoPE and of the QK dot by the same 2^-4. The result is therefore
  bit-identical to scaling the scores. The g4-v1 attention kernels have scale 1.
- The fused device QKV matrix is the row concatenation q|k|v of the original packed rows (blocks never cross rows),
  so each output keeps its own dot and bytes.

## Size (production)
253.0M parameters. Q8_0 = 1.0625 B/weight: head 178.3 MB, body ~89 MB, ~268 MB total.
