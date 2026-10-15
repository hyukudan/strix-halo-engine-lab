# Opt-in Q8 indexed-attention input prefetch

The `--prefetch-q8-qsa-input` experiment changes only the loading schedule of Q8_0
query, key, value, index-query and index-key projections. The low-level and manual
defaults are off. The [automatic profile candidate](decode-optimization-profile.md)
selects this mode only after its content and execution admission; that policy
still requires its own automatic-workload qualification before adoption.
All new kernels live in the original indexed-attention translation unit, with its
existing floating-point compiler options. The original kernels remain unchanged.

Each physical wave lane owns eight original T256 leaves. Features remain in
ascending order within each leaf. The five shuffle stages reproduce original
strides 128..8 before the original 4/2/1 operand order. Weight decoding remains
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
fallback projections and identify every projection choice. Projection indices 0..4
mean query, key, value, index-query, index-key. No eligible layer is an option error.

Catalog and binding requests carry the same ordered eligible-layer list. The
bundle accepts only its exact base/prefetch ID pair, with distinct launcher and
compiler identities. New-mode preflight validates writable/read-only owner ranges,
aliases and actual Q8 base alignment before the first enqueue, including graph
control and state updates. After model residency, all eligible actual weight views are admitted before the
first token. The manifest plan is checked against the compiled tensor-access
slots at setup. No additional scratch allocation is required.
Speculative verification explicitly rejects the new runtime mode in this increment;
the existing indexed-attention launcher registers C1 one-row variants only.

Host tests cover pure planning, mixed tensor/layer fallbacks, kernel ID binding,
default identity stability, zero eligibility, missing/duplicate tensors, alignment,
extent and overflow rejections. Device fixtures exercise the actual legacy,
virtual256 control and prefetch kernels against an independent oracle, with complete
output poisoning, surrounding guards, changing inputs and three captured replays.
Shapes include 32-wide and non256 tails, 2560x12288, signed codes including -128,
finite half extremes, signed zeros, BF16 subnormals and the association witness.

Qualification status: source implementation and fixtures exist. Native compiler,
linked legacy-function identity, candidate arithmetic ISA/descriptors, device
fixtures, 554 full-array equality, 512-token equality and measured full-workload
benefit remain required. No performance gain or production qualification is claimed.

The projection receipt uses `projection_kernel` for a projection schedule label
and includes `tensor_name`; operator registry IDs remain in compiled bindings.
OFF emits only `prefetch_q8_qsa_input: false`. ON also emits
`prefetch_q8_qsa_input_dispatched_projections`, a host atomic submission count:
eligible projections times eager operator calls, plus projections submitted during
graph capture. Device graph replay does not increment it. Derive expected counts
from the actual manifest rather than assuming indexer inputs are Q8. Misaligned
eligible tensors are rejected, rather than silently choosing fallback.

Full-operator fixtures compare complete output, FP32 workspace, KV and raw-index
bytes across OFF/ON eager and captured replay, including a BF16 indexer fallback.
They inspect five or four block-32 candidate nodes and reject misaligned weights
and mutable-owner overlaps with zero nodes and unchanged owners. Mixed catalog
fixtures require eleven candidate layers, one base layer, and an unchanged base
launcher identity. Native execution of this follow-up remains a separate gate.
