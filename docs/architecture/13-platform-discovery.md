# 13 — Platform Discovery and Capability Negotiation

## Decision

Hardware and operating-system capabilities MUST be discovered rather than inferred from a product name.

The engine MUST separate static host discovery from active qualification. Static discovery may identify available interfaces and topology; active qualification is required before a capability changes execution policy.

## Static discovery

The platform probe SHOULD collect, without requiring the inference runtime to start:

- operating-system and kernel identity;
- physical memory;
- GPU nodes exposed by KFD;
- `gfx_target_version` and derived GPU architecture;
- SIMD and compute-unit geometry when exposed;
- ROCm installation identity;
- availability of the HIP compiler/toolchain;
- Thunderbolt/USB4 domains;
- host routers visible in sysfs;
- configured USB4 stream character devices;
- presence of the USB4 streaming driver.

The probe MUST be read-only.

## GPU identity

`gfx1151` MUST be established from the runtime/kernel topology rather than assumed from the host model name.

Kernel plans, tuning data and persistent execution caches MUST include the discovered GPU architecture in their identity.

If the expected GPU architecture is not present, architecture-specific kernels MUST NOT be selected automatically.

## USB4 topology

Discovery MUST distinguish between:

- a physical USB4-capable host controller;
- a visible host router;
- a connected peer;
- a configured data stream;
- a qualified transport link.

These are not equivalent states.

Finding two host routers does not prove that two links provide independent bandwidth. Independence requires an active simultaneous-transfer test.

## Dual-link qualification

For a pair of hosts with two candidate USB4 links, active qualification SHOULD measure:

1. each link alone in both directions;
2. both links transmitting in the same direction;
3. both links transmitting in opposite directions;
4. latency at queue depth one;
5. sustained bandwidth;
6. CPU and interrupt cost;
7. performance while the GPU is under inference load.

The result SHOULD record whether the links appear independent, partially shared, or fully bottlenecked by a common upstream resource.

## Capability levels

A capability SHOULD move through explicit states:

1. **absent** — not discovered;
2. **present** — interface or device exists;
3. **usable** — a basic functional test succeeds;
4. **qualified** — performance, correctness and recovery gates pass;
5. **preferred** — it is selected by policy for a defined workload.

Presence MUST NOT be treated as qualification.

## Transport negotiation

When multiple nodes connect, each side SHOULD exchange a compact capability record containing:

- engine protocol version;
- GPU architecture;
- supported tensor dtypes;
- transport types;
- number of qualified links;
- measured payload limits;
- graph-safe or mapped-buffer capabilities relevant to transport;
- model/partition identity.

A negotiated session MUST choose the intersection of compatible capabilities.

## Probe versus benchmark

The static probe answers what appears to exist.

The qualification harness answers whether it is fast, correct and safe enough to use.

Execution policy MUST depend on qualification results where performance or synchronization semantics matter.

## Principle

Never infer a fast path from branding.

Discover the topology, prove the capability, measure the real workload, then select the path.
