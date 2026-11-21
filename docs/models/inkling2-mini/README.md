# Inkling 2 Mini native bring-up

The current implementation provides a checkpoint metadata inspector, the loop and
state layout, and host operator fixtures. The native prefill and decode entry
points still need to be wired to the shared kernels.

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

Expert matrices in physical layers 0–19 pack two E2M1 values in each byte.
Each block of 16 has an E4M3 scale and each expert has an outer scale of
`amax / outer_scale_denominator`. Layers 20–23 use E4M3 weights with E8M0
scales for blocks of 32. This schema validates their metadata. Scale values
and device weight decoding are separate numerical checks.

## Execution and reuse

| Stage | Shared component | Mini binding |
| --- | --- | --- |
| Weights | `strix-weight` safetensors indexer and shared NVFP4/MXFP8 row decoders | 822 text tensor names, extents and dtypes, including quantization sidecars. Device consumption of the decoded formats is pending. |
| Tied layers | `src/core/looped_layers.hpp` | 24 physical layers, with layers 6–17 visited three times for 48 visits |
| Residual and cross-pass messages | `src/core/residual_reference.hpp` | Save the prelude output and block bank. Before each repeat, capture the previous pass's second block input and final output, then restore the prelude. Read four messages from the captured bank and apply rank-64 gates. |
| Attention | `src/attn_core` KV row storage, dense prefill, online softmax and relative bias views | 24 query heads, 8 KV heads and dimension 128. The decoder uses learned relative logits, query-dependent sinks and output gates. The device path needs a relative-bias hook before softmax. |
| Short convolutions | `src/core/short_convolution.hpp` and shared session state contracts | Four histories per visit for K, V, post-attention and post-MoE. Full attention uses dynamic K/V taps. Local K/V and both output convolutions use static depthwise taps. |
| MoE | `strix::moe::RoutePolicy`, activation traits and shared grouped kernels | Sigmoid plus correction bias for selection of 8 of 128 experts. Normalize unbiased scores with scale 8. Softcap the two linear factors in SwiGLU and evaluate sigmoid on the raw gate. |
| Embeddings, tokenizer and sampling | Shared embedding rows, tokenizer and logit selection | Learned embedding RMS norm, separate output matrix, final tanh softcap 30 and output vocabulary of 200058 tokens |
| Sessions and placement | `include/strix/session`, `include/strix/pair` and `include/strix/dist` | Own the weight binding by physical layer and the persistent state by logical visit. Preserve the two message-bank rows as dependencies at pass boundaries. |
| Vision and drafter | Shared image ops, serving and speculative verification | Vision tensor map and projector. Bind drafter taps and qualify rollback across all visits before enabling speculation. |

`src/inkling2` owns the text checkpoint schema, the Mini loop and cache geometry,
and the Mini MoE policy constants. Generic scheduling and arithmetic live in the
shared components above.

The cache plan uses eight full-context KV allocations and 40 sliding KV rings.
At 1,048,576 positions, BF16 payload is 32 GiB for full attention and 160 MiB
for the sliding rings with one-row writes. For prefill that writes keys before
attention, size each sliding ring to hold the preceding 1023 keys and the
entire chunk. A later staged implementation can keep a 1024-row persistent
ring and expose the new chunk separately. Test prefill/decode parity across
the ring wrap before changing the storage strategy.

Two useful targets for the device work are batching the 48 learned message
queries at each repeat boundary over the same two normalized bank rows, and
feeding relative logits into the attention tile without retaining a full
query-by-key image. Qualify both against the pinned reference before enabling
them in a performance profile.

## Verification

```sh
cmake -S tests/inkling2 -B build/inkling2-host -DCMAKE_BUILD_TYPE=Release
cmake --build build/inkling2-host
ctest --test-dir build/inkling2-host --output-on-failure
```

The root host CTest also registers the Inkling fixtures and includes a test for
the checkpoint inspector's schema and shared reader. The standalone build
runs the header-only fixtures.

- [x] Pin the checkpoint and verify the original text tensor inventory.
- [x] Exercise pass restarts, message gates, state identities, score and router policies in host fixtures.
- [ ] Compare native operators with pinned reference intermediate tensors.
- [ ] Validate text prefill, multi-token decode and logits on a single Strix Halo with the official weights.
- [ ] Validate cache rollover, state restore and rollback for all visits.
- [ ] Validate image and text prefill, followed by DSpark.
- [ ] Measure TTFT, prefill, decode and peak memory at 1K, 8K and 32K before enabling any faster profiles or USB4 partition plan.

Tracking #909 for Mini and #908 for Small.
