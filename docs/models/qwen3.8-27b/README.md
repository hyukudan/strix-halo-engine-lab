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
Real pack (2026-11-16): CUDA exact q8_0 vs ref_forward f64 on a 15-token chat prompt + 31 greedy tokens: max |dlogit| 0.032
(0.06 % of the logit range), top-1 46/46, greedy 31/31.

Measurement kits (`scripts/bench/qwen35/`, parametrised; hosts/paths are arguments):
`run_classes.sh BIN GGUF OUT [N]` exact drafting per prompt class (code/json/prose, EXACT gate inside);
`rows_table.sh BIN "MODEL_ARGS" "ROWS"` verify-window ms per rows; `nsys_step.sh NSYS BIN OUT "ARGS"` per-kernel step table;
`turbo_quality.py REF.bin TURBO.bin VOCAB [ALIGNED_ROWS]` TURBO gate (KL, top-1, error clustering);
`dense_pair.sh selfcut|identity|timing --bin DIR --model GGUF --ids FILE --out DIR [--ref DIR]` layer split over ranks
(`strix-hip-qwen35-pair`): one-process cut test, two-rank identity against the one-node reference, two-rank timing
(second rank on `KIT_PEER`).

## Sizes (parameters: trunk 24.35 B + head 1.27 B; vision skipped)
BF16 pack ~51.2 GB per decode pass; Q8_0 pack ~27.2 GB (34/32 bytes per weight). GDN state 151 MB, KV 64 KiB/position (bf16).

## CUDA backend notes

### TURBO candidates

`--profile turbo` on the CUDA decode and speculative CLIs inherits the admitted FAST options and
prints a quality warning. Four model TOML defaults are pending. Each needs a final
`tools/strix-gate/gate_v2.py qual --tier turbo` verdict and a positive workload receipt against FAST
for its device, weight format and route. The drafter also needs complete token identity against the
same target. The HIP CLIs decline the profile and name the unavailable FP8 attention arm.

| Candidate | Explicit qualification switch | Off switch |
|---|---|---|
| E4M3 KV read | `--kv fp8` | `--kv native` |
| E4M3 QK and PV on linear CUDA sm_120 decode and verify | `--decode-attn8 on` | `--decode-attn8 off` |
| FP8 attention output and FP4 recurrent output and FFN down projections on compatible NVFP4 trunk windows | `--dense-activation on` | `--dense-activation off` |
| MTP chain depth up to 15, subject to the arbiter and the target window | `--chain-k 15` on the speculative CLI | `--chain-k 7` |

The current KV and QK/PV defaults are admitted together. CUDA reads an E4M3 shadow alongside BF16
backing storage. `--kv fp8` and `--decode-attn8` are alternative spellings for the decode setting.
The qualified FAST prefill row retains its own admission. Compare the complete TURBO composition
against FAST with the same checkpoint, route and state capacity before enabling it automatically.
For the CUDA teacher-forced gate, `--qual-null-order` on the reference command changes only the
order of the last two warp partials of output RMSNorm. The gate checks that the null moves on each window.

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

## FAST prefill window (HIP decode and speculative apps)

FAST and TURBO prefill with the 512-row IU4 window (`--wide 512` on the decode app, `--prefill-rows 512`
on the speculative app, A4W4 off) by default when the prompt is inside a gated band: one gfx1151
device, the whole model prefilled in the process (one rank, or each rank of the tensor-parallel
decode: `--tp-rank` ranks prefill the whole prompt alone and split only the decode), one lane, the
IU4 g64 pack and a known execution. Both apps resolve the same rows through the same resolver
(`include/strix/profiles/qwen_dense_hip.hpp`: `kQualified`, `kBandRows` / `kSpecBandRows`, one band
ladder spelled per window lever); each app's own execution spelling binds the band row as that run's
reference. Outside the bands, and for EXACT, other devices or packs, the previous route stays (64-row
EXACT windows). Off switches: `--profile exact`, an explicit `--wide N` / `--wide-exact N` /
`--prefill-rows N` / `--prefill-exact`, or `--rows N`.
`--dump-logits FILE --dump-window` dumps every row of the last window of the route the profile
resolves; `--dump-logits` alone keeps the one-row reference path.

FAST gate v2 per content class (`tools/strix-gate/gate_v2.py qual`, limits KL <= 0.02, top-1 >= 0.93,
dNLL <= +0.01, >= 800 rows per class): held-out texts (prose: Perl pod and licence texts; code: the
Python standard library; structured: ISO code JSON and the freedesktop MIME XML), 4 windows per class at
2,048 and 8,192 tokens and one text per class cut at 24,576 and 24,064 tokens (32,768 and 32,256 for the 32K band),
511 scored rows per window, reference EXACT `--wide-exact 512` (`--tf-rows`; at 32K the 64-row EXACT windows, the
same bits), null arm `STRIX_DENSE_NULL=0:100:1234` (32K: `0:26724:1234` for prose and code; for structured that step
was absorbed before the scored rows, null == reference, so it ran at `0:100:1234`).

| Band (tokens) | Arm | prose KL / top-1 / dNLL | code KL / top-1 / dNLL | structured KL / top-1 / dNLL | Verdict |
|---|---|---|---|---|---|
| 2,048 | 512-row window | 0.00008 / 0.9961 / -0.0008 | 0.00002 / 0.9971 / -0.0001 | 0.00001 / 1.0000 / +0.0001 | PASS |
| 2,048 | window + A4W4 48-63 | 0.2095 / 0.8454 / +0.1157 | 0.0669 / 0.9310 / +0.0518 | 0.0707 / 0.9677 / +0.0464 | FAIL |
| 8,192 | 512-row window | 0.00007 / 0.9961 / +0.0002 | 0.00003 / 0.9976 / -0.0003 | 0.00001 / 1.0000 / +0.00001 | PASS |
| 8,192 | window + A4W4 48-63 | 0.1792 / 0.8704 / +0.1099 | 0.0649 / 0.9178 / +0.0632 | 0.0097 / 0.9902 / +0.0004 | FAIL |
| 24,576 (cuts 24,576 + 24,064) | 512-row window | 0.00006 / 0.9951 / -0.0004 | 0.00001 / 1.0000 / -0.0000 | 0.00001 / 1.0000 / -0.0003 | PASS |
| 32,768 (cuts 32,768 + 32,256) | 512-row window | 0.00006 / 0.9971 / -0.0007 | 0.00004 / 0.9961 / -0.0001 | 0.00001 / 1.0000 / -0.00003 | PASS |

The nominal timing bands are 1..2,048, 2,049..8,192, 8,193..24,576 and 24,577..32,768 tokens. The timed 512-row window
wins at every timed point from 64 tokens up: 64, 256, 1,024, 2,048, 2,049, 4,096, 8,192, 8,193 and 24,576 (at 4 and
16 tokens it ties the EXACT windows, no gain) and 32,768. FAST/TURBO apply it on 64..2,048, 2,049..8,192, 8,193..24,576
and 24,577..32,768 tokens.
Other prompts use the 64-row EXACT windows. See the [timing inventory](../../band-edges.md) for each gate cut.

Timing (one gfx1151 node, the IU4 pack, `--no-mtp --gen 1`, meas_nodes prompt ids, mirrored order,
EXACT one cold prefill per round, the window warm of `--repeat 2`, kfd evictions 0, prompt seconds):

| Prompt tokens | EXACT 64 rows | 512-row window | window + A4W4 48-63 | `--profile fast` (default) |
|---|---:|---:|---:|---:|
| 2,048 | 10.24 / 10.30 | 2.98 / 3.00 | 2.81 / 2.80 | 2.82 / 2.81 |
| 8,192 | 67.57 / 67.63 | 12.91 / 12.94 | 12.09 / 12.11 | 12.11 / 12.10 |
| 24,576 | 534.39 | 45.45 / 45.42 | not run | 45.32 / 45.76 |
| 32,768 | 996.5 (four segments) | 62.52 / 62.35 | not run | not run |

After the row-share change (EXACT 64-row windows on the 3-pair attention form, same bits: a whole 32,768-token EXACT
prefill reproduces the segmented reference bit for bit) the EXACT column reads 8,192: 46.74 s (68.07 s with
`STRIX_DECODE_ATTN_ROW_SHARE=0`, same binary), 24,576: 197.56 s, 32,768: 299.5-302.4 s (three texts); the window still
wins x3.7 / x4.6 / x4.8. 65,536 tokens: the window 147.66 s (warm; no EXACT reference, no band, explicit `--wide 512`
only). Decode (`--no-mtp`, 64 tokens after the prompt): 8,192 13.21 tok/s, 24,576 11.47, 32,768 10.74, 65,536 8.65.

2,048 and 8,192 ran on one node in one turn; 24,576 ran on the other node, the window and default arms in one
turn and the EXACT run (one cold prefill, 534 s) in its own turn, same binary and ids.

The `--profile fast` column above ran the provisional data (window + A4W4); with the gated data the
default dumps are bitwise the 512-row window arm (2,048 and 8,192 tokens, 3 cases each).

## A4W4 layer table (HIP, FAST)

The 512-row window and A4W4 layers 48-63 formed one candidate. It failed the per-class gate above
at 2,048 and 8,192 tokens (prose and code), so its FAST cell stays pending. The single-text
8,192-token result below did not predict the per-class result. Other devices, multi-rank or
multi-lane execution, other packs, other window widths and different execution modes keep their
previous routes. There is no separately qualified HIP TURBO lever.

For an explicit experiment, set `STRIX_DENSE_A4W4_LAYERS=48-63` with `--wide 512`
(decode) or `--prefill-rows 512` (spec). Layers 48-63 use `dense-iu4a4-rot-v1`
(Hadamard-128 rotated A4 g32 activations and rotated W4 per-128 weights on IU4 WMMA).
Other layers use `dense-iu4a16-f16-v1`. Use `STRIX_DENSE_A4W4_LAYERS=none` to disable
the layer table. The all-layer arm failed the gate. Explicit EXACT windows and decode
row diagnostics use their own routes. A plain `--dump-logits` retains every prompt row
when no explicit width is supplied.

Window timing on one gfx1151 node used revision `ca2de6ae8`, the Qwen 27B IU4 pack and
identical prompt IDs. The command was `strix run qwen3.8-27b --profile exact --asset MODEL -- --prompt-ids @IDS --gen 64 --repeat 2`;
the candidate used `--profile fast` and appended `--wide 512` on that revision.
Both arms ran with MTP enabled. The 32,500-token EXACT run used `--repeat 1`.
Warm GPU clocks were 2,612-2,726 MHz for completed EXACT runs and 2,473-2,554 MHz
for the candidate. Runs started below 70 °C via the same cooldown procedure.
The power cap was unrecorded.

| Prompt tokens | EXACT 64 rows (warm tok/s) | Candidate 512 + A4W4 48-63 (warm tok/s) |
|---|---:|---:|
| 2048 | 187.1 | 612.2 |
| 8192 | 115.7 | 547.8 |
| 32500 | timeout at 470 s | 436.8 |

FAST gate v2 (`tools/strix-gate`): 8,192-token prompts, 256 teacher-forced rows per window, 4 windows (1,024 rows), null
arm `STRIX_DENSE_NULL=0:100:1234` (one FP16 ulp of one residual element before layer 0; per-window null KL 2-3e-5).
Set A = 2 texts x 2 cut points (used to choose the layers); set B = 4 slices of a third text (held out).

| A4W4 layers | reference | windows | KL | top-1 | dNLL | verdict |
|---|---|---|---|---|---|---|
| 0-63 | wide IU4 | A | 0.0763 | 0.906 | +0.0090 | FAIL |
| 48-63 | wide IU4 | A | 0.0053 | 0.976 | +0.0026 | PASS |
| 8-15, 48-63 | wide IU4 | A | 0.0095 | 0.960 | +0.0073 | FAIL (one window dNLL) |
| 48-63 | EXACT (`--wide-exact 512`) | B | 0.0069 | 0.971 | -0.0104 | PASS on B |
| none (wide IU4 alone) | EXACT (`--wide-exact 512`) | B | 0.00003 | 0.999 | -0.0003 | PASS |

Choice: one arm per group of 8 layers (set A, reference wide IU4). Pooled KL by group 0-7 .0233, 8-15 .0047,
16-23 .0082, 24-31 .0101, 32-39 .0166, 40-47 .0116, 48-55 .0034, 56-63 .0018; the sum (.0797) matches the all-layer
KL (.0763). Adding any further group to 48-63 breaks one window's KL or dNLL limit.

The rotated copy of the 16 layers uses about 3 GiB of device memory.

## Multi-session batched decode (CUDA, exact class, default path for slots)

`CudaEngine::configure_slots(n)` gives n sessions (own KV, GDN/conv state, parity, length) that share the weights.
`forward_slots(groups, tokens)` runs one dense pass per layer over the rows of all groups (a group = consecutive rows of one
slot: 1 row = a decode lane, more = a verify/prefill window) with per-session attention and GDN: `cuda_core::RowMap`
(per-row position + cache slot for `kv_store`/`rope_apply`/`attention_decode`) and `gdn_window_slots`/`gdn_commit_slots`
(device slot tables, so one CUDA graph per (rows, groups) serves any set of sessions). Every row is bitwise equal to the
session run alone (`qwen35_cuda_engine_test` section 4; `strix-cuda-qwen35-decode --slots N --slots-verify` on real packs).
Windows commit with `commit_slots(accept[])` (partial accept supported). Bench: `--slots N --arms 8,4,1 --ab-pairs 3 --n-gen S
--ctx-fill C` (strict ABAB, host-timed steps incl. token read-back, SM clock). `StepPlanner`/`plan_rows` lane rows map 1:1 to groups.

## Bounded prefill submission (CUDA, scheduling only)
A long prefill is queued chunk by chunk through `cuda_core::SubmitWindow` (`include/strix/cuda_core/submit_window.hpp`):
before the host queues chunk i it waits for chunk i - N to finish, so at most N chunks are queued ahead of the GPU instead
of the whole prompt before one sync. Same kernels, same per-stream order and event dependencies: output is bitwise
identical (logits hashes equal at 8K and 32K, throughput neutral). `strix-cuda-qwen35-decode` flags:
- `--prefill-inflight N` (default 3): chunks queued ahead of the GPU; `0` = unbounded (the previous form, off switch).
- `--prefill-ramp R` (default 0 = off): the first chunks grow R, 2R, 4R, ... rows (multiples of 128, below
  `--prefill-chunk`) so the card steps up from idle. It changes chunk boundaries, so it is exact only where the prefill
  is chunk-size invariant (measured for this model); it stays off by default.
Both values are printed by `--print-effective-config` and in the prefill line (`chunks`, `inflight`).
