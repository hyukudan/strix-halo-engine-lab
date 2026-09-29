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

## MoE

Mixture-of-experts execution SHOULD:

- keep routing deterministic;
- use stable tie-breaking;
- group rows by expert where beneficial;
- preserve output row order;
- avoid widening packed expert weights globally;
- provide dedicated low-row-count expert paths.

Host-visible routing in the critical path SHOULD be avoided when it forces synchronization.

## Recurrent kernels

Recurrent or scan-style layers SHOULD keep state on chip across multiple logical updates when this reduces repeated global-memory traffic and preserves exact update order.

State tiling, register retention, DPP exchange and LDS exchange SHOULD all be evaluated as alternatives.

The best variant is the one with the lowest full-step cost under the required numerical contract.

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
