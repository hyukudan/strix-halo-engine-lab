# 03 — Model Loading and Weight Layout

## Decision

Model loading is a validated planning stage, not a file-copy operation.

The loader MUST separate logical model interpretation from physical storage and execution layout.

## Stages

A model load SHOULD proceed through:

1. parse metadata;
2. validate architecture;
3. validate tensor inventory;
4. validate shapes and dtypes;
5. calculate physical byte ranges;
6. verify non-overlap and file bounds;
7. build a logical tensor map;
8. choose execution layouts;
9. build a memory plan;
10. admit or reject;
11. establish mappings and residency;
12. create execution objects.

No GPU kernel should run before the model has passed structural and memory validation.

## Storage format

The input format is an interchange/storage representation.

The engine MUST NOT couple kernel layout directly to file layout when a different lossless in-memory arrangement is materially faster.

A tensor MAY use one of three paths:

- direct read/use from stored layout;
- temporary repack at load;
- persistent lossless execution cache.

## Internal cache

A persistent internal cache MAY be introduced only when:

- it is lossless;
- it is versioned;
- it is rebuildable;
- its fingerprint includes source weight identity;
- its layout identity includes compiler/runtime assumptions;
- startup or runtime benefit is measured.

The cache MUST NOT silently requantize weights.

## Tensor inventory

The engine SHOULD generate a machine-readable inventory containing:

- tensor name;
- logical role;
- dimensions;
- source dtype;
- source byte range;
- execution dtype;
- execution layout;
- resident or file-backed status;
- selected kernel family;
- any repack cache identity.

This inventory should be inspectable without running inference.

## Quantized weights

Quantized formats SHOULD remain packed as long as possible.

The default policy is:

**decode packed values inside the compute kernel rather than expanding an entire checkpoint into wider intermediate storage.**

Expansion MAY be used only for a bounded cache or when the performance benefit outweighs the physical-memory cost.

## Sparse tables

Very large sparse lookup tables SHOULD be treated separately from dense model weights.

The loader MAY keep them file-backed and use bounded host-side row staging when:

- access locality is poor;
- keeping them resident would materially reduce context capacity;
- I/O latency can be overlapped or bounded.

## Placement

Placement is explicit in the logical manifest from the first implementation.

Every tensor MUST carry a placement record even when all tensors are local. The single-node default is local node zero.

A future partitioner MAY change a tensor or operator group to a remote node, but storage format, logical role and numerical semantics MUST remain independent of transport choice.

Placement metadata MUST NOT imply that a remote tensor can be accessed transparently. Remote execution requires an explicit execution boundary and transport plan.

Residency and placement are separate decisions:

- a local tensor can be resident or file-backed;
- a remote tensor can be resident or file-backed on its owning node;
- transport staging is not model residency.

The initial manifest MUST remain conservative for memory accounting. A tensor is considered resident unless there is a specific, architecture-supported reason to classify it as a file-backed candidate.

## Integrity

The runtime SHOULD support model identity based on hashes of all execution-relevant files.

Snapshots, tuning plans and persistent caches MUST be invalidated when execution-relevant model identity changes.
