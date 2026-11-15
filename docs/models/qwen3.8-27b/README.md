# Qwen3.8-27B (dense) - family `qwen35`

HF `Qwen3_5ForConditionalGeneration`, text model `qwen3_5_text`: 64 layers, hidden 5120, FFN 17408, vocab 248320 (untied head),
48 gated-delta-net (GDN) layers + 16 gated full-attention layers (layer l is attention iff `(l+1) % 4 == 0`), 1 MTP layer.
Vision tower is not converted. Host contract: `include/strix/qwen35/config.hpp`, `src/qwen35/`, registry row `qwen3_8_27b`.

## Semantics (reference: `tools/qwen35/ref_forward.py`)
- Layer: `x += Mixer(RMS(x)); x += MLP(RMS(x))`. RMSNorm `y = x*rsqrt(mean(x^2)+1e-6)*(1+w)`; the GGUF stores `1+w` (F32).
  The GDN gated norm is `x*rsqrt(..)*w*silu(z)` with plain `w`.
- Attention: `q_proj` rows are per head `[q 256 | gate 256]` (24 heads), K/V 4 heads x 256, per-head q/k RMSNorm, NeoX rotate-half
  RoPE on the first 64 dims (theta 1e7; text mrope == plain RoPE), GQA group 6, scale 1/16, `out = attn*sigmoid(gate)`.
- GDN: `[q 2048|k 2048|v 6144] = in_proj_qkv(x)`; causal depthwise conv k=4 (state = last 3 inputs, no bias) + SiLU;
  `beta = sigmoid(b)`, `g = ssm_a*softplus(a + dt_bias)` with `ssm_a = -exp(A_log)`; q,k l2-normalised, q scaled 1/sqrt(128);
  `S = S*exp(g); d = (v - S^T k)*beta; S += k (x) d; o = S^T q`; gated norm with `z`; `out_proj`. State f32 [48][128][128].
- MTP: `h' = eh_proj([RMS_e(emb(t_{i+1})) | RMS_h(h_i)])`, one attention layer + MLP with its own KV (RoPE position i),
  `shared_head_norm`, shared `output`. The reference takes `h_i` after the final norm (`--mtp-hidden pre` = residual before it).

## GGUF contract
`general.architecture = qwen35`; `qwen35.{block_count 64, context_length, embedding_length, feed_forward_length,
attention.{head_count, head_count_kv, key_length, value_length, layer_norm_rms_epsilon}, rope.{freq_base, dimension_count},
ssm.{conv_kernel, state_size, group_count, time_step_rank, inner_size}, full_attention_interval, nextn_predict_layers}`;
`tokenizer.ggml.*` (gpt2 BPE, pre `qwen35`). Tensors (full list: `expected_tensors()`): `token_embd, output_norm, output`;
`blk.N.{attn_norm, post_attention_norm, ffn_gate, ffn_up, ffn_down}`; GDN `{attn_qkv, attn_gate, ssm_conv1d, ssm_beta, ssm_alpha,
ssm_dt.bias, ssm_a, ssm_norm, ssm_out}`; attention `{attn_q, attn_k, attn_v, attn_output, attn_q_norm, attn_k_norm}`;
MTP `blk.64.{nextn.enorm, nextn.hnorm, nextn.eh_proj, nextn.shared_head_norm}` plus the attention/FFN set.
Matrices are BF16 or Q8_0, everything else F32. **V-head order is TILED** (engine convention): GGUF value head t holds HF head
`(t%16)*3 + t/16`; applied to v rows of `attn_qkv` and its conv channels, `attn_gate` rows, `ssm_beta/alpha` rows, `ssm_dt`,
`ssm_a`, and the `ssm_out` input columns (blocks of 128).

## Tools
```
python tools/qwen35/convert.py HF_DIR OUT.gguf --pack bf16|q8_0        # streams tensor by tensor
python tools/qwen35/convert.py --synthetic-hf DIR                        # tiny random HF model (real per-head dims)
python tools/qwen35/ref_forward.py M.gguf --ids 3,100,400 --n 32 --out DIR   # ref.npz + summary.txt (f64 accumulation)
python tools/qwen35/xcheck_transformers.py HF_DIR DIR/ref.npz            # one-shot Qwen3Next cross-check (torch)
```
Cross-check on the synthetic model (GGUF bf16 -> ref_forward vs transformers Qwen3NextForCausalLM, f32): max |dlogit| 1.1e-5,
top-1 8/8. Q8_0 uses ggml reference rounding (blocks of 32, d = amax/127 as f16, round half away from zero).

## Sizes (parameters: trunk 24.35 B + head 1.27 B; vision skipped)
BF16 pack ~51.2 GB per decode pass; Q8_0 pack ~27.2 GB (34/32 bytes per weight). GDN state 151 MB, KV 64 KiB/position (bf16).
