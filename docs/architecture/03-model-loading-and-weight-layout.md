# 03 — Model Loading and Weight Layout

## Decision

Model loading is a validated planning stage, not a file-copy operation.

The loader MUST separate logical model interpretation from physical storage, execution layout, memory residency and execution placement.

## Stages

A model load SHOULD proceed through:

1. parse metadata;
2. validate architecture;
3. validate tensor inventory;
4. validate shapes and dtypes;
5. calculate physical byte ranges;
6. verify non-overlap and file bounds;
7. build a logical tensor map;
8. assign default tensor ownership and execution placement;
9. choose execution layouts;
10. build a memory plan;
11. build a transport plan if any tensor boundary is remote;
12. admit or reject;
13. establish mappings and residency;
14. create execution objects.

No GPU kernel should run before the model has passed structural and memory validation.

## Model manifest

The loader SHOULD produce a machine-readable `ModelManifest` that is independent of the execution backend.

The manifest SHOULD describe at least:

- model identity;
- architecture revision;
- tensor inventory;
- logical tensor roles;
- persistent state classes;
- execution placement;
- memory-residency policy;
- transport boundaries;
- quantization metadata;
- execution-layout identity.

The manifest is the boundary between model interpretation and runtime execution.


## Runtime tensor binding

The manifest and the loaded tensor addresses are separate objects.

Each runtime tensor binding MUST identify:

- manifest tensor index;
- storage mode;
- local address when locally addressable;
- bound bytes;
- allocation generation;
- owner node.

Supported storage identities SHOULD distinguish at least:

- resident local tensor;
- file-mapped local tensor;
- lossless repacked cache;
- remote tensor.

A remote tensor MUST NOT expose a fake local address.

The runtime MUST validate manifest layout identity before a tensor binding can be used.

The loader MUST additionally supply a non-zero content identity derived from hashes of all execution-relevant model files. Manifest layout identity alone is insufficient because two checkpoints can share the same tensor geometry.

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
- owner node;
- execution node;
- partition descriptor if applicable;
- selected kernel family;
- any repack cache identity;
- explicit execution-boundary descriptors when placement can become remote.

A remote-capable execution boundary SHOULD identify the exact tensor shape, dtype, logical stream structure, byte range, auxiliary state and producer/consumer operators. Distributed planning MUST NOT estimate boundary traffic from hidden size alone when the actual representation is wider or carries additional state.

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

For multi-node execution, sparse tables MAY be replicated or partitioned independently from dense weights when that reduces communication on the decode path.

## Placement

Placement is explicit in the logical manifest from the first implementation.

Every tensor MUST carry a placement record even when all tensors are local. The single-node default is local node zero.

The manifest SHOULD be able to represent concepts equivalent to:

```text
Local(node=0, device=gfx1151)
Remote(owner_node=N)
Replicated(nodes=[...])
Partitioned(axis=..., ranges=[...])
```

A future partitioner MAY change a tensor or operator group to a remote node, but storage format, logical role and numerical semantics MUST remain independent of transport choice.

Placement metadata MUST NOT imply that a remote tensor can be accessed transparently. Remote execution requires an explicit execution boundary and transport plan.

Residency and placement are separate decisions:

- a local tensor can be resident or file-backed;
- a remote tensor can be resident or file-backed on its owning node;
- transport staging is not model residency.

The initial manifest MUST remain conservative for memory accounting. A tensor is considered resident unless there is a specific, architecture-supported reason to classify it as a file-backed candidate.

Supporting placement metadata MUST NOT impose a runtime cost on the single-node executor when every tensor is local.

## Partition planning

A distributed placement plan MUST be evaluated by bytes crossing node boundaries per inference step, not only by parameter count.

The planner SHOULD estimate:

- bytes transferred per generated token;
- transfers per layer or stage;
- synchronization points;
- persistent state owned by each node;
- expected compute imbalance;
- expected transport overlap;
- memory saved per node.

A placement MUST be evaluated by total critical-path cost, not synchronization count alone.

For a latency objective, a candidate SHOULD be compared using a model equivalent to:

```text
critical_path = concurrent_compute
              + uncovered_communication
              + synchronization_and_control
```

Additional synchronization does not by itself disqualify a placement if parallel compute or reduced local memory traffic more than pays for it.

A placement optimized for memory capacity MAY remain useful even when it does not improve single-request latency, but that objective MUST be identified explicitly.

## Integrity

The runtime SHOULD support model identity based on hashes of all execution-relevant files.

Snapshots, tuning plans, placement plans and persistent caches MUST be invalidated when execution-relevant model identity changes.
