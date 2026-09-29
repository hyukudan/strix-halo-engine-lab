# 13 — Platform Discovery and Capability Negotiation

## Decision

Hardware and operating-system capabilities MUST be discovered rather than inferred from a product name.

The engine MUST separate static host discovery from active qualification. Static discovery may identify available interfaces and topology; active qualification is required before a capability changes execution policy.

The platform profile is a first-class input to admission, tuning, transport selection and distributed placement.

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
- configured direct-stream character devices;
- presence of direct USB4 streaming support.

The probe MUST be read-only.

## GPU identity

`gfx1151` MUST be established from the runtime/kernel topology rather than assumed from the host model name.

Kernel plans, tuning data and persistent execution caches MUST include the discovered GPU architecture in their identity.

If the expected GPU architecture is not present, architecture-specific kernels MUST NOT be selected automatically.

## Unified-memory inventory

The probe SHOULD also report, where discoverable:

- physical RAM;
- current host baseline;
- GPU-visible mapping limit;
- current mapped usage;
- registered or pinned usage;
- graphics carve-out;
- page size;
- huge-page configuration;
- IOMMU state;
- relevant memory-management kernel parameters.

The probe MUST distinguish a mapping limit from physically available memory.

## USB4 topology

Discovery MUST distinguish between:

- a physical USB4-capable host controller;
- a visible host router;
- a physical port;
- a connected peer;
- a configured data stream;
- a qualified transport link.

These are not equivalent states.

For each candidate port or link the probe SHOULD attempt to record:

- stable local identifier;
- controller and host-router identity;
- port identity;
- peer identity;
- connection state;
- negotiated generation or rate;
- lane count when observable;
- domain/path identity;
- interrupt vector and affinity;
- associated direct-stream device if present.

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

The result SHOULD record one of:

```text
independent
partially_shared
strongly_shared
unknown
```

This classification is a measured property, not a product-specification property.

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
- mapped-buffer capabilities relevant to transport;
- model/partition identity.

A negotiated session MUST choose the intersection of compatible capabilities.

## Peer topology graph

For multi-node operation the runtime SHOULD construct a graph:

```text
Node
  -> physical link
    -> peer node
      -> qualified transport capabilities
```

Each edge SHOULD retain measured attributes such as:

- payload bandwidth;
- one-way latency;
- round-trip latency;
- full-duplex behavior;
- direct-stream support;
- mapped-buffer capability;
- health state.

Placement planning MAY consume this graph, but only qualified attributes may affect production selection.

## JSON contract

The platform probe output SHOULD be stable enough to feed other tools.

A conceptual shape is:

```json
{
  "host": {},
  "memory": {},
  "gpu": {},
  "usb4": {
    "routers": [],
    "ports": [],
    "links": []
  },
  "capabilities": {},
  "qualification": {}
}
```

The exact schema may evolve, but fields used by admission, tuning or placement MUST be versioned.

## Probe versus benchmark

The static probe answers what appears to exist.

The qualification harness answers whether it is fast, correct and safe enough to use.

Execution policy MUST depend on qualification results where performance or synchronization semantics matter.

## Startup use

Normal inference startup SHOULD consume a previously qualified profile and perform a lightweight verification.

A full topology benchmark MUST NOT run automatically on every server start.

If hardware identity or topology changed, affected tuning and transport profiles MUST be invalidated.

## Principle

Never infer a fast path from branding.

Discover the topology, prove the capability, measure the real workload, then select the path.
