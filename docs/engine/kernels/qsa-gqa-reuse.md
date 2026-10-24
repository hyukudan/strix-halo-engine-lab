# Selected-attention GQA reuse diagnostic

The shared backend groups two or four query heads sharing the same KV
head in one workgroup. The reference arm invokes the original score, softmax
and value/gate kernels in their original translation unit. The candidate replaces
only score and value/gate. The explicit production option `--qsa-gqa-heads-per-block 2|4` now selects these
same kernels. The default remains one head; the KV state format is unchanged.

For QK, every feature loads one BF16 key value and uses it for each local query
head. Each head still stores 256 individual FP32 products and reduces exactly
the incumbent feature pairs with strides 128, 64, 32, 16, 8, 4, 2 and 1. Each head
has independent LDS scratch. The divide by `sqrtf(head_dim)` stays after that
tree. Grouping adds no sum across query heads and does not fuse softmax.

For V, every feature loads one BF16 value per selected slot and updates two or
four independent FP32 accumulators. Each accumulator sees the original selected
slots in the original chronological order with the same `score * value` operand
order. The attention result, sigmoid gate and product retain their original
BF16 rounding points. The output remains FP32 storage containing BF16-rounded
values, exactly as the original stage's output buffer.

For the 24-query/2-KV geometry, the reference has 24 score CTAs per selected
slot and 24 value CTAs. The two-head arm has 12 and 12; the four-head arm has 6
and 6. All CTAs retain 256 threads. Score LDS grows from 1024 bytes to 2048 or
4096 bytes. The source-level K/V loads shared across grouped heads shrink by a
factor of two or four for this divisible geometry. This is a logical access
count, not a measured DRAM transaction ratio. Query and score traffic does not
receive that reduction. Fewer CTAs, more live accumulators, LDS use, cache
behavior and compiler lowering can change occupancy and latency in either
direction. No performance or compiled-FP equivalence result exists yet.

The bounded interface is C1, head dimension 256, up to 64 query heads divisible
by KV heads, and capacity from 1 through the current model limit of 262144.
Head groups smaller than the candidate width use masked local head tails and
never cross KV-head boundaries. Capacity is distinct from occupied position.
The maximum selected allocation is `min(capacity, 2051)`, including the largest
possible three-token remainder before a capacity divisible by four.

Both eager submission and captured replay read the position from a caller-owned
device uint64. The wrapper passes the existing dynamic selection ABI with
compression four and top-512 blocks. The actual count is
`min((position+1)/4,512)*4+(position+1)%4`. The caller provides the selected table
in strictly increasing token order; no sorting or selection is performed here.
Only active slots are read. KV remains the original interleaved BF16 cache;
there is no query-head-expanded K/V copy.

The wrapper checks exact host view sizes, alignment, disjoint ranges, shape and
owned non-default stream before enqueue. A common device precheck rejects
out-of-capacity positions, invalid selected indices and nonfinite source values
before attention writes. All arms use the existing graph arithmetic-status path;
a common final kernel checks output finiteness. Status is sticky and never
cleared by the wrapper. Source/control rejection suppresses score/output writes;
arithmetic failure invalidates potentially partial scores/output. These rejection
rules bound the diagnostic. They do not establish equivalence to invalid inputs
on the legacy eager decoder path.

The five submitted kernels are source validation, QK, original softmax, V/gate
and output validation. The common validation cost must remain explicit in any
future comparison. A return proves submission only. The caller retains the
stream, all allocation owners and producer ordering through completion, including
enqueue failure and graph lifetime. No allocation, synchronization, host status
publication or thread-local launch-context modification occurs in `enqueue`.
The separate `enqueue_scores` test seam submits only validation and QK, leaves
output untouched and exposes the unnormalized score workspace for bitwise
comparison before softmax can hide differences.

The new declarations and test source are available through
`STRIX_BUILD_QSA_GQA_REUSE_DIAGNOSTIC` in testing builds. Test source covers literal
reference versus candidate bitwise comparisons of raw QK, the entire normalized
score workspace and output; repeated poisoned eager and captured submissions; changing occupied
positions and selected counts; sparse chronology and compression tails; canonical
GQA and incomplete local head groups; finite cancellation, signed zeros,
subnormals and saturated gates; read-only cache/input preservation; outer guards;
and invalid admission with zero captured nodes. Nonfinite, bad-index, stale-status,
position and arithmetic failures exercise diagnostic containment separately.

The boundary starts after Q/K normalization and RoPE and ends before the final
output projection. It does not test KV append, a complete QSA layer, batching,
session restore or full-model fidelity. Source, compiler/ISA, native bitwise,
complete-operator, model-quality and full-workload performance gates remain
pending. No build, test or benchmark of these additions has been run.

Production source integration requires eager C1 with head dimension 256,
compression four, budget 2048 and the caller's owned deferred-completion stream.
Whole-graph, integer-head and prefix-roundtrip modes reject candidate grouping.
A caller-owned eight-byte position and four-byte status live in appended arena
storage; a small same-stream control kernel writes position without borrowing
an asynchronous host buffer. With hierarchical selection enabled, its rejection
status is preserved. Otherwise only this local status is initialized. No helper
clears the separate deferred first-error record.

The shared backend's source and output validation surrounds the same score,
softmax and value/gate computation used by the component diagnostic. A publisher
uses the distinct `qsa_attention` domain and preserves any earlier selection or
MoE error. The existing token completion fence rejects a failed token before
host publication. Metadata admission checks all source, writable and record
ranges before the first projection is submitted; actual capture is rejected.

Candidate IDs carry `.gqa-heads2-v1` or `.gqa-heads4-v1`, with explicit request,
catalog, compiler/implementation and workspace binding. Receipts report actual
host submissions and per-layer counters after the loop, labelled `not_assessed`.
These counters are not device-completion or performance measurements. Added
production fixture source compares full outputs, state and chronological indices
against the incumbent, including selector composition and masked group tails.
No compilation, fixture execution or benchmark has been performed for this source
integration; joint review and final validation remain pending.
