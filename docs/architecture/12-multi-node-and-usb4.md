# 12 — Multi-Node and USB4 Interconnect

## Decision

The engine MUST be designed so that two or more Strix Halo hosts can cooperate without changing the single-node execution contract.

The first multi-node target is a small cluster of directly connected Strix Halo systems using native USB4 links.

Multi-node support is an extension of the engine, not a dependency of the single-node fast path.

## Goals

The distributed design SHOULD support:

- model partitioning across hosts;
- tensor-parallel or pipeline-style execution where profitable;
- memory-capacity aggregation;
- two-node low-latency execution as the primary distributed case;
- three-node direct-link topologies where available;
- multiple physical USB4 links between the same two nodes;
- graceful fallback to a simpler transport.

The primary optimization target remains low-latency local inference, not large-cluster throughput.

## Transport abstraction

The engine MUST expose a transport interface independent of any one kernel or model family.

A transport instance SHOULD report capabilities such as:

- usable payload bandwidth;
- one-way and round-trip latency;
- full-duplex capability;
- maximum practical transfer size;
- number of physical links;
- whether links are independently schedulable;
- whether links share an upstream bottleneck;
- host-copy requirements;
- GPU-visible mapped-buffer capability;
- completion semantics;
- disconnect and recovery behavior.

Model code MUST NOT call sockets, character devices, or kernel-specific APIs directly.

## Transport API shape

The transport abstraction SHOULD be able to express operations equivalent to:

```text
capabilities(peer)
prepare_send(buffer, bytes, metadata)
prepare_recv(buffer, bytes, metadata)
submit()
poll_or_wait()
cancel()
health()
```

The exact API may differ, but ownership and completion MUST remain explicit.

## Control plane and data plane

Control traffic and bulk tensor traffic SHOULD be separable.

The control plane MAY use:

- ordinary Ethernet;
- IP over USB4;
- another reliable network path.

The bulk data plane SHOULD prefer the lowest-overhead qualified direct transport.

Control messages include:

- capability negotiation;
- model and partition identity;
- tensor shape and dtype;
- sequence and generation identifiers;
- health checks;
- recovery coordination.

Bulk data includes:

- hidden states;
- partial reductions;
- routed expert data where partitioning requires it;
- logits or partial logits;
- other explicitly planned tensor boundaries.

## USB4 direct streaming

Where the deployed Linux kernel provides direct USB4 streaming support, it SHOULD be evaluated as the preferred direct host-to-host bulk transport.

The architecture SHOULD treat it as a byte-stream/message transport over USB4 rather than as conventional IP networking.

The first implementation MAY use ordinary userspace reads and writes.

More advanced modes MAY use:

- fixed mmap-able slot pools;
- registered host memory;
- GPU-visible mapped buffers;
- DMA-BUF-backed pools;
- asynchronous completion;
- direct GPU production or consumption of transport buffers.

Every more direct mode MUST retain a simpler fallback.

## One link first

A single USB4 link MUST be qualified before link aggregation.

The baseline qualification MUST measure:

- queue-depth-one one-way latency;
- round-trip latency;
- sustained one-way bandwidth;
- simultaneous bidirectional bandwidth;
- CPU utilization;
- interrupt rate;
- copies per transfer;
- latency under GPU load;
- end-to-end token impact.

A transport microbenchmark is not sufficient; the same link MUST be measured inside model execution.

## Dual-link operation

Systems with two usable USB4 ports MAY connect both ports between the same two hosts.

Dual-link operation is an explicit target.

The engine SHOULD expose each physical link separately to the transport scheduler.

Two links MUST NOT be treated as twice the bandwidth until topology and simultaneous-transfer measurements prove it.

The platform probe MUST attempt to identify:

- physical USB4 host routers;
- port-to-router mapping;
- negotiated link rate per port;
- peer mapping;
- shared upstream topology;
- interrupt affinity;
- whether each link remains fully active under simultaneous traffic.

## Dual-link strategies

The transport layer SHOULD support several policies.

### Independent directional links

One physical link MAY primarily carry A-to-B traffic and the other B-to-A traffic.

This is attractive when the workload is naturally full duplex and can reduce contention without requiring message striping.

### Message striping

Large tensors MAY be divided into chunks and transferred over both links concurrently.

Striping MUST preserve:

- tensor ordering at reassembly;
- generation and sequence identity;
- per-link error detection;
- bounded buffering.

Small latency-sensitive tensors SHOULD NOT be striped unless measurements show a benefit, because synchronization and reassembly overhead can exceed transfer time.

### Flow-level distribution

Independent tensor transfers MAY be assigned to different links.

This is often preferable to striping because it avoids reassembly and can overlap different pipeline stages.

### Role specialization

One link MAY be reserved for a latency-critical class while the other carries larger or background transfers.

This MAY be useful when speculative, expert, prefill and decode traffic have different size distributions.

## Transport scheduler

The distributed scheduler SHOULD know:

- tensor byte size;
- destination;
- deadline or criticality;
- available links;
- recent per-link latency;
- recent per-link bandwidth;
- directional utilization;
- current in-flight bytes;
- reassembly cost;
- whether another independent tensor can use the second link.

It MAY choose the link policy per tensor.

Static round-robin SHOULD be only a baseline.

## Buffer ownership

Transport buffers require explicit ownership states.

A buffer SHOULD move through states similar to:

1. producer owns;
2. producer release complete;
3. transport owns;
4. receive completion;
5. consumer acquire complete;
6. consumer owns;
7. reusable.

GPU and external-DMA synchronization MUST be proven for the exact buffer type in use.

A device synchronization primitive that only orders GPU queues MUST NOT be assumed to establish visibility for writes performed by an external DMA engine.

## Zero-copy policy

Zero-copy is a candidate optimization, not a requirement.

A mapped or direct transport path MUST beat a copy path end to end before it becomes the default.

On a unified-memory architecture, writing directly into transport-visible system pages may reduce copies but also reduce GPU write performance or cache efficiency.

The engine MUST therefore measure:

- direct mapped output;
- GPU-to-staging copy;
- CPU-assisted staging;
- network-stack baseline where relevant.

## Model partitioning

The first distributed partition SHOULD minimize transfers on the decode critical path.

Partition decisions MUST consider:

- bytes transferred per generated token;
- synchronization points per token;
- compute balance between nodes;
- recurrent-state placement;
- KV ownership;
- MoE routing locality;
- speculative verification traffic.

A mathematically balanced partition can still be slower if it crosses the interconnect too often.

The engine SHOULD prefer partitions with fewer, larger, predictable boundary transfers over many small synchronization-heavy transfers.

## Pipelining

The runtime SHOULD permit communication/computation overlap when correctness allows it.

Candidates include:

- double or multi-buffered boundary tensors;
- pre-posted receive slots;
- next-layer or next-token preparation while the previous transfer completes;
- concurrent transfer of independent tensors on separate links.

Overlap MUST NOT change the deterministic single-node semantic contract.

## Topologies

### Two nodes

Two nodes connected directly are the primary distributed target.

Where both systems expose two qualified USB4 ports, dual-link direct connection SHOULD be tested and retained as a first-class topology.

### Three nodes

With two ports per node, a three-node direct ring or triangle MAY be supported.

Topology-aware placement SHOULD prefer a direct neighbor for latency-critical communication.

### More nodes

Larger clusters are secondary.

They MAY use:

- mixed USB4 and Ethernet topology;
- host-specific high-speed NICs;
- external switches or adapters.

The engine SHOULD not assume all peers have equal link cost.

## Failure and fallback

Each data-plane link MUST have health state.

On link failure, the engine SHOULD:

1. stop assigning new transfers to that link;
2. preserve or abort in-flight work deterministically;
3. fall back to another qualified direct link when available;
4. fall back to the control/network transport when safe;
5. fail the request cleanly if semantic continuity cannot be guaranteed.

A failed second link MUST NOT force failure if the workload can continue correctly on one link.

## Qualification gates

Multi-node execution is qualified only when it passes:

- output equivalence against the single-node contract where the same numerical mode applies;
- long decode tests;
- long prefill tests;
- C1 and small-concurrency tests;
- reconnect tests;
- peer restart tests;
- one-link and dual-link tests;
- directional, flow-level and striped dual-link policies;
- simultaneous bidirectional stress;
- memory-pressure tests on all hosts;
- repeated model load/unload;
- sustained soak tests.

## Principle

The interconnect must justify the partition.

Adding a second host is useful only when the extra memory or compute gained is worth the communication and synchronization introduced.

Adding a second USB4 link is useful only when it reduces end-to-end latency or increases useful throughput under the real model workload.
