# Inkling 2 Mini native bring-up

The text decoder can run a serial CPU reference directly from the original
checkpoint. The native HIP and CUDA text decoders compose all 48 visits
using resident weights, a device route plan, and a persistent state slice per
visit. CPU compilers and both device toolchains can build the bring-up
fixtures. Full-device behavior still needs qualification on hardware.

## Checkpoint inspection

The inspector checks the original single-shard text tensor schema at
[checkpoint revision ed40e7d](https://huggingface.co/thinkingmachines/Inkling-2-Mini/tree/ed40e7da625f594fc9c4990f5f66be24873c22d7).
It reads tensor metadata and validates the optional index against the shard.
The vision tensors are counted. Their model-specific geometry and the separate
drafter checkpoint require their own admission checks.

```sh
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build --target strix-inkling2-mini-inspect
build/strix-inkling2-mini-inspect MODEL_DIR/model-00001-of-00001.safetensors \
    MODEL_DIR/model.safetensors.index.json
```

Expert matrices in physical layers 0â€“19 pack two E2M1 values in each byte.
Each block of 16 has an E4M3 scale and each expert has an outer scale of
`amax / outer_scale_denominator`. Layers 20â€“23 use E4M3 weights with E8M0
scales for blocks of 32. This schema validates their metadata. Scale values
and device weight decoding are separate numerical checks.

## CPU text reference

```sh
cmake --build build --target strix-inkling2-mini-cpu
build/strix-inkling2-mini-cpu MODEL_DIR/model-00001-of-00001.safetensors \
    TOKEN_ID TOKEN_ID --generate 2 --logits next-token.f32
```

Pass the input token IDs in sequence. The reference evaluates prompt rows
serially and projects logits only for the last prompt row and requested decode
steps. The optional logits file contains the next-token row as 200058
little-endian FP32 values. Generation uses greedy selection and stops on EOS.
The loader reads only selected expert matrices during each MoE call. The CPU
cache stores BF16 operands widened to FP32, so the BF16 cache budget below
applies to the planned device layout. CPU output still needs numerical
qualification against an independent reference for the full checkpoint.

## Native text entry points

```sh
cmake -S . -B build-hip -DSTRIX_ENABLE_HIP_RUNTIME=ON
cmake --build build-hip --target strix-inkling2-mini-hip

cmake -S . -B build-cuda -DSTRIX_CUDA_CORE=ON -DCMAKE_CUDA_ARCHITECTURES=120
cmake --build build-cuda --target strix-inkling2-mini-cuda
```

Both commands accept token IDs, raw text via `--prompt`, or chat via
`--messages FILE`. The chat file can be a JSON message array or an object
with `messages` and optional `tools` and `reasoning_effort` fields. The
official `tokenizer.json` defaults to the checkpoint directory. Supply
`--tokenizer FILE` to select another copy of the same tokenizer. Use
`--tokenize FILE` to print raw-text token IDs without loading model weights.
Chat files may carry images as base64 `image_url` parts; see
[Inkling 2 Mini image input](../../engine/vision-adapters.md#inkling-2-mini).

Generation defaults to greedy selection and can emit one FP32 prompt logits row
with `--logits FILE`. CPU and device generation accept `--temperature`,
`--top-k` and `--seed` through the shared sampler.

```sh
build-hip/strix-inkling2-mini-hip MODEL.safetensors \
    --prompt 'Write one line about Ibiza.' --generate 24
build-cuda/strix-inkling2-mini-cuda MODEL.safetensors \
    TOKEN_ID TOKEN_ID --prefill-chunk 128 --generate 24
```

HIP defaults to the one-row packed expert GEMV for decode. CUDA defaults to
grouped GEMM. Use `--packed-decode` or `--grouped-decode` to select the
route explicitly. The packed route uses one kernel for gate/up with BF16
activation and one for down with ordered expert combine.

## Execution and reuse

| Stage | Shared component | Mini binding |
| --- | --- | --- |
| Weights | `strix-weight` safetensors indexer and packed K16 NVFP4/MXFP8 operand view | 822 text tensor names, extents and dtypes, including quantization sidecars. The loader exposes contiguous code and scale ranges by physical layer and projection for device residency. CPU and device expert tiles decode compact weights into BF16 operands with the checkpoint's gate/up row order. |
| Tied layers | `src/core/looped_layers.hpp` | 24 physical layers, with layers 6â€“17 visited three times for 48 visits |
| Residual and cross-pass messages | `src/core/residual_reference.hpp` | Save the prelude output and block bank. Before each repeat, capture the previous pass's second block input and final output, then restore the prelude. Read four messages from the captured bank and apply rank-64 gates. |
| Attention | `src/attn_core` causal GQA reference, prepared BF16 cache binding, dense HIP prefill and CUDA Flash | 24 query heads, 8 KV heads and dimension 128. Both backends append prepared K/V together and dispatch attention with relative logits and dynamic sinks on the same stream. |
| Short convolutions | `src/core/short_convolution.hpp` and the shared HIP/CUDA device body | Four histories per visit for K, V, post-attention and post-MoE. Full attention uses dynamic K/V taps. Local K/V and both output convolutions use static depthwise taps. The device primitive updates history after the output rows have consumed the previous history. |
| MoE | `strix::moe::RoutePolicy`, activation traits, expert-major plans and grouped BF16 WMMA/MMA | Sigmoid plus correction bias for selection of 8 of 128 experts. Normalize unbiased scores with scale 8. Softcap the two linear factors in SwiGLU and evaluate sigmoid on the raw gate. Both backends select routes, build expert-major descriptors, gather BF16 rows, project around softcapped SwiGLU and combine into token order on a single stream. The original host plan remains a fixture reference. |
| Embeddings, tokenizer and sampling | Shared embedding rows, tokenizer and logit selection | Learned embedding RMS norm, separate output matrix, final tanh softcap 30 and output vocabulary of 200058 tokens |
| Sessions and placement | `include/strix/session`, `include/strix/pair` and `include/strix/dist` | Own the weight binding by physical layer and the persistent state by logical visit. Preserve the two message-bank rows as dependencies at pass boundaries. |
| Vision and drafter | Shared image ops, serving and speculative verification | Vision tensor map and projector. Bind drafter taps and qualify rollback across all visits before enabling speculation. |

The Mini expert layout binds the checkpoint's physical-layer format and
gate/up order. The common Mini MoE dispatch binds model dimensions, BF16
scratch and softcaps, with HIP and CUDA adapters selecting backend tile sizes.
Each attention adapter binds the Mini query/KV head counts, relative logits,
sinks and window to its backend. The session plan fixes cache and history
offsets by logical visit at the largest admitted chunk so single-token decode
uses the same layout as prefill.

The routed MoE stage reuses each of its two temporary buffers after its last
reader. The gather and activation share a BF16 buffer, and the gate/up and
down projections share an FP32 buffer. This requires 13,824 bytes per routed
row, or 108 KiB per token at top-8, plus routing metadata. Choose a bounded
prefill chunk when allocating scratch.

The cache plan uses eight full-context KV allocations and 40 sliding KV rings.
At 1,048,576 positions, BF16 payload is 32 GiB for full attention and 160 MiB
for the sliding rings with one-row writes. For prefill that writes keys before
attention, size each sliding ring to hold the preceding 1023 keys and the
entire chunk. A later staged implementation can keep a 1024-row persistent
ring and expose the new chunk separately. Test prefill/decode parity across
the ring wrap before changing the storage strategy.

The residual controller shares two normalized bank rows across the repeat.
The lag projection reads the original K-by-N checkpoint layout in one launch
and keeps its FP32 accumulator and BF16 result. CUDA also reuses one
large-M activation split for compatible same-input projections.


## Verification

```sh
cmake -S tests/inkling2 -B build/inkling2-host -DCMAKE_BUILD_TYPE=Release
cmake --build build/inkling2-host
ctest --test-dir build/inkling2-host --output-on-failure
```

The root host CTest also registers the Inkling fixtures and includes a test for
the checkpoint inspector's schema and shared reader. The standalone build
runs the header-only fixtures.

`strix-hip-inkling2-mini-attention-test` is built by HIP offline CI. On a
compatible device it exercises the optional relative bias with and without
Split-K, contiguous and wrapped K/V, and Mini's cache-append path.
`strix-hip-inkling2-mini-short-conv-test` checks static and dynamic taps,
including history continuity from prefill into decode.
`strix-hip-inkling2-mini-grouped-test` exercises both expert formats,
interleaved gate/up rows, uneven route groups and Mini's down projection.
The grouped-row and expert-activation fixtures isolate ordering and rounding.
`strix-hip-inkling2-mini-moe-test` compares the full routed stage with a scalar
calculation using synthetic packed weights in both Mini formats. The CUDA
fixture uses the same calculation and packed weights. The CUDA short
convolution fixture likewise shares its data with HIP. The CUDA attention test
covers prepared full/wrapped KV, bias and sinks with both Flash layouts.

```sh
cmake -S . -B build-cuda -DSTRIX_CUDA_CORE=ON -DBUILD_TESTING=ON \
    -DCMAKE_CUDA_ARCHITECTURES=120
cmake --build build-cuda --target strix-cuda-inkling2-mini-attention-test \
    strix-cuda-inkling2-mini-short-conv-test strix-cuda-inkling2-mini-moe-test
ctest --test-dir build-cuda -L inkling2 -R '^strix-cuda'
```

- [x] Pin the checkpoint and verify the original text tensor inventory.
- [x] Exercise pass restarts, message gates, state identities, score and router policies in host fixtures.
- [x] Execute the full CPU token path, including greedy generation, with a synthetic checkpoint in the published layout.
- [x] Compare CPU next-token logits with an independent calculation on a single-token prompt using the original checkpoint.
- [ ] Expand CPU parity coverage across prompt lengths, decode positions and the sliding-window boundary.
- [x] Build native text entry points and share resident execution between HIP and CUDA.
- [ ] Compare native operators with pinned reference intermediate tensors on device.
- [ ] Validate text prefill, multi-token decode and logits on a single Strix Halo with the official weights.
- [ ] Validate cache rollover, state restore and rollback for all visits.
- [ ] Validate image and text prefill, followed by DSpark.
- [ ] Measure TTFT, prefill, decode and peak memory at 1K, 8K and 32K before enabling any faster profiles or USB4 partition plan.

Tracking #909 for Mini and #908 for Small.

## Prefill schedule (CLI and server)

The CLI resolves the prefill chunk and the tail rule from `mini_prefill_schedule`
(`src/inkling2/inkling2_mini_schedule.hpp`): HIP 8192-row chunks with a remainder of up to chunk / 8 rows absorbed into
the final chunk, CUDA 2048-row greedy chunks. `--prefill-chunk N` overrides the chunk, `STRIX_INKLING_PREFILL_TAIL=0`
restores greedy chunks. The server (`strix-serve --model inkling2-mini`) takes its served prefill from the model data
`[serve_prefill]` in `models/inkling2-mini.toml`: on HIP the CLI's chunk and tail rule; on CUDA the incumbent 128-row
chunks, because the CLI's 2048-row chunk beside a decoding request is not yet qualified there (`--prefill-rows 2048`
opts in). `--prefill-rows N` (alias `--prefill-chunk`) overrides the served chunk; the decode slot cap (`--slots`,
default 128) does not follow the chunk. The startup line `Inkling served prefill ...` states the chunk, the pieces, the
gap target, the share, the device class and the decode route that ran.

Bits. A served request prefills its whole prompt and takes its first token from the prefill's last row, as the CLI
does, and each prefill call absorbs only the remainder the decoder admits (more than 16 rows; 1..16 rows run as their
own chunk). A prefill row's expert body does not depend on the decode route: a 1-row chunk (a 1-token prompt, or
8192 + 1 rows) takes the prefill body in the CLI as in the server. So, at the same chunk, the served prefill rows take
the CLI's chunks and routes at every prompt length (host test over the schedule and the route choice,
`tests/inkling2/mini_schedule_test.cpp`). The server decodes with the grouped expert route: its decode rows equal CLI
runs with `--grouped-decode`, while the CLI default (packed decode) runs decode rows on another expert body. Receipts of
the earlier server default (128-row slices, the final prompt token advanced through the decode path) differ from the
CLI for prompts whose last slice had 1..16 rows (for example 4101 and 8209 rows at a 4096-row chunk).

The server has two regimes:
- no other request decoding: the prompt runs whole chunks (8K in one pass on HIP);
- other requests decoding (HIP): the prompt runs pieces of the same chunks, cut only at 16-row wave-tile offsets and
  never shorter than 17 rows, so every row keeps its route form and the prompt keeps the bits of the unbounded
  schedule. A piece is `--prefill-rows-decoding` rows (data: 512) or, on a device class with a measured row in the data
  (gfx1151), the most rows whose predicted pass (240 ms fixed, every pass streams all expert weights, plus the previous
  pass's cost per row) and one decode round fit the planning target `--prefill-gap-ms` (gfx1151 data: 1000 ms). The
  target is a prediction used for sizing, not a guaranteed bound: pieces predicted above it (the floor, an indivisible
  piece) are recorded by the scheduler, and one call beside a decoding request runs at most one backend chunk. Between
  pieces the decoding requests are owed `--prompt-share` ms of decode per prompt ms (data: 0 = one decode round per
  piece). Every piece also computes the score row of its last position (the served adapter cannot yet tell the final
  piece apart); that cost is inside the numbers below. The CUDA decoder declares no cut rule, so CUDA runs whole
  chunks. Values the scheduler would refuse are refused before any device work.

Measured on one gfx1151 node (receipts srvmd_fix3 t1 / t2: binary 7323047fc plus workspace changes, the predecessor of
this tree with the schedule given as flags; EXACT, KV BF16, one request streaming while one prose prompt arrives, one
run per cell; first token of the prompt / longest wait of the streaming request):

| prompt | alone | gap 1000 ms, share 0 (gfx1151 data) | 1024-row pieces, share 0 | 512-row pieces, share 0 | 512-row pieces, share 1 (previous) | whole chunks |
|---:|---:|---:|---:|---:|---:|---:|
| 8K | 4.7 s | 6.0 s / 1.03 s | 6.6 s / 0.82 s | 8.8 s / 0.56 s | 18.5 s / 0.65 s | 4.8 s / 4.75 s |
| 32K | 21.2 s | 26.7 s / 1.03 s | 27.6 s / 0.95 s | 36.7 s / 0.62 s | 60.5 s / 0.69 s | 36.1 s / 5.90 s |

With share 1 the streaming request gets about five times more tokens while the prompt arrives (121 against 26 at 8K)
but loses about the same decode time overall, and the arriving first token doubles (gap 1000 ms, share 1: 10.5 s at 8K).
The earlier server default (128-row slices, no tail) waited 0.36 s with a first token of 32.8 s at 8K. Ids in that run
(gap 1000, share 0, pieces of varying size): 2053, 8209 and 8236-row prompts served alone == beside the streaming
request == the CLI with `--grouped-decode` (64 ids each). These are single runs on one node; other device classes,
several decoding requests and consecutive arrivals are not measured.

This tree (receipts srvmd_fix4 n1 / n2, the same gfx1151 node, EXACT, KV BF16, the served data defaults, one run per
cell): served ids (16) == the CLI with `--grouped-decode` at P = 1, 17, 4101, 8192, 8193, 8208, 8209, 8292, 9216 and
16389; the CLI's prefill score row with its default packed decode == with `--grouped-decode` at each P; 8193 and 8209
beside a streaming request == alone. Gap row (1000 ms, share 0): 8K 5.7 s / 1.03 s, 32K 25.6 s / 1.03 s. In n1 the
first piece waited 2.05 s (8K) and 2.10 s (32K) while every later piece held about 1 s: the streaming request's own
short prompt had set the row cost the first piece was sized with; the scheduler now learns it from passes of at least
the decoding floor only (n2).
