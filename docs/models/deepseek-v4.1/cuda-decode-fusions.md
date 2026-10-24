# Exact CUDA decode launch fusions

Decode and pipeline FRONT accept `--fuse-hc` and `--fuse-shared`. Both are off by
default. The corresponding `CudaPrefillOptions` fields are `decode_fuse_hc` and
`decode_fuse_shared`. They affect `step()` only; prefill and the numeric identity
selection are unchanged. The shared flag selects the two-launch plan at R1;
wider windows retain the existing plan under the decode selection policy. `describe()` records both choices. Select the intended
weights and routed-expert identity separately, for example `--expert-tp --overlay
OVERLAY.gguf`.

## HC and norm

`enqueue_hc_mixes_norm` replaces three launches with two. The first computes the
original 640-element chunk partials. In the second, one block per row performs
the original ascending chunk fold and Sinkhorn rounds; other blocks perform the
original collapse, RMSNorm and A8 parts. Norm uses the preceding site's pre
coefficients. It does **not** consume the mix being produced by the same launch.
The two block families write disjoint outputs and require no grid barrier.

The norm and Sinkhorn bodies are shared with the unfused entry points. Chunk
boundaries, lane assignments, ascending sums, butterfly levels, scalar operations
and A8 rounding are preserved. `HcExpandView::pre_carry` optionally writes the
four pre coefficients during HC expansion, removing FFN post's separate device
copy. Stream expansion still supports an in-place residual update.

## Shared expert

`enqueue_shared_expert_fused` executes two launches on the existing tiled Q8_0
weights. Gate/up uses the original projection. Each down-output tile then
computes the same SwiGLU values and complete 32-value A8 groups in CTA-local
memory before running the original integer MMA, FP32 lane-class sums and
butterfly. The activation calculation is repeated independently by down tiles;
there is no cross-block publication, atomic accumulation or changed reduction.

`enqueue_swiglu_a8_down` exposes the second launch for inspection, with an
optional A8 tap written only by output tile zero. It supports intermediate widths
that are positive multiples of 32 up to 4096. Rows are processed in independent
passes of eight. The existing shared-expert scratch allocation remains valid.

## Verification and measurement

`strix-cuda-ds41-decode-fusions-tests` compares device bytes and prints digests
for mix, collapse, norm, A8 codes/deltas/sums, post stream/pre-carry and shared
output. Real-weight layers 0/1/2/20 cover R1/2/4/6/8/9/16 with normal, zero and
large finite activation patterns. Graph replay gates check the same outputs and
assert node counts: HC+norm 3 to 2, FFN post 2 to 1, shared expert 3 to 2. A
three-layer chained `LayerCore` gate also compares every existing per-op tap,
including embedding, Engram, router and head. The captured host-reference gate
can use `STRIX_DS41_CUDA_TAP_FUSE=1` with `--layer-core`.

Set `STRIX_DS41_FIXTURE_GGUF` to the dense fixture or a complete model. With a
complete overlay, use the complete model's metadata because overlay validation
requires all replacement tensors to exist. `--overlay FILE` selects that overlay.
`--bench` adds a short per-site graph microbench: 24 alternating samples after
three warmups, with timing events embedded in each graph and a 256 MiB L2 flush outside each
event interval. Times cover one site and its internal graph edges; they do not establish a
complete-workload speedup. Final end-to-end timing belongs to the integration
campaign, with each fusion flag tested independently and together.

Native qualification on the Windows sm_120 build passed 933 byte comparisons with
original weights and 933 with the c-u6 overlay, including chained taps and graph
replays. The c-u6 expert-TP identity gate matched 170/170 greedy steps and full
logits digests from the imported 2048-token state. Separate resident three-layer
FRONT runs matched complete serialized boundary bytes with graph sections at
R1/R4/R8 (seven captured steps after one warmup at each shape). These gates cover
the declared c-u6 identity; they do not qualify a different overlay or establish
end-to-end performance.
