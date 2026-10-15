# Opt-in Q8 indexed-attention input prefetch

The `--prefetch-q8-qsa-input` experiment changes only the loading schedule of Q8_0
query, key, value, index-query and index-key projections. It is off by default.
All new kernels live in the original indexed-attention translation unit, with its
existing floating-point compiler options. The original kernels remain unchanged.

Each physical wave lane owns eight original T256 leaves. Features remain in
ascending order within each leaf. The five shuffle stages reproduce original
strides128..8 before the original4/2/1 operand order. Weight decoding remains
`half_scale * signed_code` before multiplication by the BF16 activation, with
ordinary compiler contraction. The final conversion is the original BF16 RNE
rounding stored as FP32. An independent fixed256 FMA/BF16 oracle and a decode-
association underflow witness distinguish this contract from reordered sums.
This arithmetic argument is not a substitute for actual linked-ISA qualification.

A pure manifest plan admits each type8, width-divisible-by32 matrix independently.
It validates complete projection families, geometry, checked byte extents and
16-byte tensor base offsets. Each layer with any eligible input binds the distinct
`.prefetch8.virtual256.q8-qsa-in` operator ID; layers with none bind the original ID.
Within a mixed layer, ineligible matrices use the unchanged legacy projection.
Explicit JSON receipts distinguish eligible/fallback operator steps from eligible/
fallback projections and identify every projection choice. Projection indices0..4
mean query, key, value, index-query, index-key. No eligible layer is an option error.

Catalog and binding requests carry the same ordered eligible-layer list. The
bundle accepts only its exact base/prefetch ID pair, with distinct launcher and
compiler identities. New-mode preflight validates writable/read-only owner ranges,
aliases and actual Q8 base alignment before the first enqueue, including graph
control and state updates. No additional scratch allocation is required.
Speculative verification explicitly rejects the new runtime mode in this increment;
the existing indexed-attention launcher registers C1 one-row variants only.

Host tests cover pure planning, mixed tensor/layer fallbacks, kernel ID binding,
default identity stability, zero eligibility, missing/duplicate tensors, alignment,
extent and overflow rejections. Device fixtures exercise the actual legacy,
virtual256 control and prefetch kernels against an independent oracle, with complete
output poisoning, surrounding guards, changing inputs and three captured replays.
Shapes include32-wide and non256 tails,2560x12288, signed codes including-128,
finite half extremes, signed zeros, BF16 subnormals and the association witness.

Qualification status: source implementation and fixtures exist. Native compiler,
linked legacy-function identity, candidate arithmetic ISA/descriptors, device
fixtures,554 full-array equality,512-token equality and measured full-workload
benefit remain required. No performance gain or production qualification is claimed.
