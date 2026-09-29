# 05 — Kernel Design

## Decision

Kernel design is shape-specific, numerically explicit and measured end to end.

Occupancy is a diagnostic metric, not the optimization target.

## Registry

Every production kernel variant MUST have a registry entry describing:

- operator family;
- phase;
- supported shapes;
- input and output layout;
- precision;
- wave size;
- workgroup geometry;
- LDS usage;
- register pressure;
- scratch usage;
- compiler identity;
- numerical constraints;
- qualification status;
- measured cost.

Kernel selection MUST come from an offline-qualified plan.

Production inference MUST NOT run an uncontrolled autotuner.

## Specialization

Template or generated specialization SHOULD be driven by observed shape histograms.

The engine MUST avoid accumulating variants that are never selected.

Variants that lose consistently SHOULD be removed from the production registry.

## Fusion

Fusion SHOULD be considered when it removes memory traffic, state round-trips or launch overhead.

Fusion MUST NOT be accepted merely because it reduces kernel count.

A fused kernel must be compared against the complete unfused sequence for:

- latency;
- bandwidth;
- scratch;
- occupancy;
- numerical output;
- rollback complexity.

## Matrix operations

Matrix-vector and small-row matrix operations are critical for decode.

The design SHOULD have dedicated low-row-count kernels rather than forcing decode through large-GEMM assumptions.

Prefill MAY use separate matrix kernels, precision or tiling because its arithmetic and shape priorities differ.

### Low-row packed projection

Decode and speculative verification SHOULD have explicit low-row shape classes, beginning with one row and adding wider variants only when traces justify them.

Logical row count and request concurrency are independent dimensions. A single active request may verify multiple candidate rows, while multiple active requests may each contribute one or more rows to a shared round. Kernel identities and benchmarks MUST record both dimensions where both affect the shape.

A multi-row projection SHOULD seek to reuse packed-weight loads, scale metadata and dequantization work while preserving the declared arithmetic contract of each logical row.

Vector/VALU execution and architecture-native matrix instructions MUST be treated as competing implementation candidates. Tile padding, layout conversion, accumulator precision and masked-row cost are part of the comparison.

An integer low-precision matrix instruction MUST NOT be used to justify silently quantizing activations or changing the checkpoint's declared quantization semantics.

## MoE

Mixture-of-experts execution SHOULD:

- keep routing deterministic;
- use stable tie-breaking;
- group rows by expert where beneficial;
- preserve output row order;
- avoid widening packed expert weights globally;
- provide dedicated low-row-count expert paths.

Host-visible routing in the critical path SHOULD be avoided when it forces synchronization.

MoE qualification SHOULD record:

- distinct experts touched per round;
- rows assigned per expert;
- overlap of expert sets across adjacent or speculative rows;
- routed weight bytes;
- sorting, grouping and padding overhead;
- shared-expert cost where applicable.

The engine SHOULD maintain at least a direct tiny-row expert path and a grouped path where route reuse is sufficient. Total row count alone is not enough to choose between them.


### Per-projection expert formats

MoE expert storage MUST be planned per projection, not per expert as one monolithic dtype choice.

For the pinned target geometry:

- hidden size is 2560;
- expert intermediate size is 640;
- there are 512 routed experts;
- top-k is 10.

A quantization block that divides the 2560-wide `gate_up` input may still be incompatible with the 640-wide `down` input.

Therefore:

- `gate_up` and `down` MAY use different storage formats or layouts;
- the planner MUST validate each input dimension against the selected physical block geometry;
- an incompatible tensor MUST NOT be assigned a synthetic padded byte count unless a lossless padding/repack format is explicitly defined and qualified;
- grouped execution MAY share packed weight reads only within a projection whose physical layout is valid.

IQ4_XS grouped execution is currently an experimental weight-reuse vehicle, not a declaration that all routed-expert tensors in the canonical checkpoint should use IQ4_XS.

## Recurrent kernels

Recurrent or scan-style layers SHOULD keep state on chip across multiple logical updates when this reduces repeated global-memory traffic and preserves exact update order.

State tiling, register retention, DPP exchange and LDS exchange SHOULD all be evaluated as alternatives.

The best variant is the one with the lowest full-step cost under the required numerical contract.

Recurrent qualification MUST count persistent-state bytes read and written, not only arithmetic throughput.

A kernel that keeps state on chip but materializes every speculative intermediate state can still lose to a design that writes less state and replays accepted updates.

## Attention

Attention kernels MUST preserve explicitly defined:

- score precision;
- reduction order constraints;
- softmax precision;
- positional behavior;
- sparse-selection semantics;
- tie-breaking.

Sparse selection MUST be evaluated as a pipeline, including index computation, selection, gathers and attention, not as an isolated top-k benchmark.

## Resource pressure

High VGPR or LDS use is acceptable when it improves full-step performance.

A kernel MUST NOT be rejected solely because it reduces theoretical occupancy.

Spills, however, MUST be measured under realistic shapes because they can turn register pressure into hidden memory traffic.

## Compatible projection fusion

Projections sharing the same input MAY be stacked, packed or fused when the transformation is lossless and the complete operator sequence improves.

The plan MUST account for extra persistent repacked weight bytes, peak repack workspace, changed cache behavior, launch savings, dequantization reuse and graph compatibility.

Source weights SHOULD remain available when needed for a reference path or when the execution cache is rebuildable rather than authoritative.
