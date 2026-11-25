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

Decode runs the experts on the one-row packed form by default on the command
line (both backends; `--grouped-decode` selects the grouped tile kernel there,
printed as `moe_decode` in `strix-decode-route/v1`). `strix-serve` keeps its
existing route in every profile and takes no decode form flag: the last
prompt token is held back and decoded, prompts run in 128-row chunks
(`--prefill-rows`, clamped to the context, which without `--slots` also caps
the lanes), and a cohort decodes with the grouped form in one batched forward
(CUDA: its width programs), whose lane bits depend on the cohort width as
before. The served route switch `STRIX_INK_SERVED_CLI_ROUTE=1`
(`inkling2-mini-served-cli-route@1`, off in every profile column) makes a HIP
server run the command line's route; a CUDA server declines it. The startup
line reports it as `served_route off`, `declined`, or `on qualification
unqualified` with its form, chunk and tail; the command line prints
`strix-served-route/v1 ... effective n/a`. On that route decode is packed and
a served lane keeps the bits of its own lone decode at any cohort width:
cohorts run in passes of at most 16 lanes, the row limit of the dense one-row
form, with one head launch per pass; lanes default to 128 without `--slots`.
A cohort decode on that route checks every lane before any pass appends and
runs its passes through `serve::run_cohort_passes`: the input ids are copied
before the first pass, the picks are published after the last one, and every
lane of the cohort is retired if a pass fails. The packed route uses one kernel for gate/up with BF16
activation and one for down with ordered expert combine.

On HIP, the command line prefills in chunks of 8192 rows (`--prefill-chunk`),
and a remainder of up to chunk / 8 rows (and more than 16) rides in the final
chunk (`STRIX_INKLING_PREFILL_TAIL=0` turns that off). A HIP server on the
command line's route uses the same schedule (`--prefill-rows` sets the chunk),
reads the whole prompt by prefill and takes the first id from its final row, as
the command line does. While another lane decodes it runs a prompt in 128-row
pieces cut inside those chunks only at 16-row wave tiles and never shorter than
17 rows, where every row keeps its route forms, so the answer does not depend
on other requests. The CUDA command line prefills in 2048-row chunks. An
explicit `--prefill-rows` without `--slots` also caps the served lanes at that
many rows, as before. The server prints `served_route`, `prefill_chunk`,
`prefill_tail` and `first_id` (`prefill` on the command line's route, `decode`
otherwise) in its startup line; the command line prints
`strix-prefill-route/v1`.

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
