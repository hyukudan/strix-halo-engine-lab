# 16 — Direct GPU Transport over USB4

## Decision

The engine SHOULD support a direct GPU-owned transport tier for qualified USB4 links when the platform can import GPU-exported DMA buffers into the USB4 data path with a proven ownership and synchronization contract.

This transport tier is optional.

It MUST coexist with a portable control/network path and MUST NOT become a dependency of the single-node runtime.

The first distributed target remains two directly connected Strix Halo hosts.

## Scope

The direct path is two-sided message passing.

It is not remote virtual memory, one-sided RDMA, remote atomics or a shared allocation spanning hosts.

Each peer owns local TX and RX pools and participates in receive posting, completion, validation, consumption and reposting.

The direct path may remove CPU payload copies while still requiring:

- GPU pack or production into the TX pool;
- host submission/progress work;
- USB4 controller DMA;
- GPU validation/consumption from the RX pool;
- completion accounting;
- RX reposting.

“Zero-copy” MUST therefore be qualified by the exact copies removed rather than used as a generic performance claim.

## Control plane before data plane

The portable link MUST become healthy before a direct stream is armed.

A direct-link lifecycle SHOULD follow this order:

1. establish the portable control link;
2. identify the physical USB4 link and peer;
3. reconcile both peer reports;
4. verify exclusive direct-link lease availability;
5. configure/import TX and RX pools;
6. prime RX storage;
7. arm the direct data path;
8. re-inspect both endpoints;
9. mark the link qualified only after reciprocal readiness is confirmed.

One prepared endpoint is not a ready pair.

Partial setup MUST be cleaned on both hosts before portable fallback is selected.

The runtime MUST NOT silently leave one half armed.

## Per-link ownership

Every physical direct link owns its own:

- connection epoch;
- endpoint identity;
- TX pool;
- RX pool;
- transport ring state;
- completion counters;
- receive-repost state;
- lease;
- health state;
- measured latency/bandwidth profile.

Two physical links between the same hosts MUST NOT share one implicit epoch or one aggregate ownership state.

A failure or reset on link A MUST invalidate link A traffic without automatically invalidating already-qualified link B traffic.

## GPU-owned pools

A future highest transport tier SHOULD evaluate dedicated GPU allocations that can be exported as DMA buffers and imported by the direct USB4 stream driver.

The preferred shape is:

```text
GPU producer
  -> dedicated TX pool
  -> external USB4 DMA
  -> dedicated RX pool
  -> GPU validation / consumer
```

TX and RX SHOULD use separate ownership domains.

A single allocation MUST NOT be imported simultaneously for conflicting DMA directions unless the exact driver/runtime contract explicitly permits it.

The engine SHOULD prefer dedicated transport pools over reusing model weights, KV or general scratch allocations.

Transport pools SHOULD have independent allocation generations so graph captures and in-flight descriptors can detect relocation.

## Buffer lifecycle

The receive lifecycle is stricter than “DMA arrived”.

A direct RX slot SHOULD move through states equivalent to:

```text
transport owns
  -> receive DMA complete
  -> consumer acquire complete
  -> consumer owns
  -> consumer release complete
  -> repost pending
  -> reposted / reusable
```

An RX slot MUST NOT become reusable merely because the USB4 DMA completion fired.

Reuse requires both:

- consumer completion;
- transport/ring accounting that makes the slot safe to repost.

The sender-side TX lifecycle likewise distinguishes payload production, release to the external DMA engine and local reuse after transport completion.

## Terminal commit record

Direct GPU-visible messages SHOULD reserve a small terminal metadata region that is owned by the protocol rather than by model payload.

The terminal record SHOULD bind at least:

- protocol version;
- connection epoch;
- message sequence;
- payload length;
- stripe index;
- stripe count;
- commit marker.

The producer MUST publish the commit marker after payload and metadata are visible to the external DMA domain.

The consumer MUST observe the commit marker with an acquire-equivalent operation before reading the message as valid.

A commit record with a stale epoch, unexpected sequence, invalid stripe geometry or impossible payload length MUST invalidate the message before model state can advance.

The exact GPU memory-ordering primitive is backend-specific and MUST be qualified on the target ROCm/kernel combination.

## Terminal-region rule

The protocol SHOULD avoid assuming that “the last payload write observed” proves all preceding DMA writes are visible.

A dedicated terminal region is preferable because it gives producer and consumer one explicit ownership/visibility boundary.

This region MAY occupy its own transport frame when required by the qualified hardware/driver behavior.

## Circular frame accounting

Ring addressing MUST advance by physical frames, not by an assumed rounded number of logical messages.

Messages may wrap across the end of the frame pool.

For a wrapped message the runtime MUST preserve:

- payload ordering;
- terminal-record ownership;
- per-frame completion accounting;
- exact frame reclamation.

A ring slot MUST NOT be reused from a message-level cursor if the underlying physical frame cursor has not advanced safely.

## RX priming

Receive storage SHOULD be posted before the peer is allowed to transmit.

A direct endpoint MUST NOT be marked usable until its receive ring/pool is primed.

This is part of the link qualification state, not a one-time installation assumption.

## Completion events

The transport completion model SHOULD distinguish:

- local TX buffer reusable;
- RX DMA complete;
- peer transport acknowledgement;
- peer GPU consumer complete;
- RX repost complete;
- failure.

A peer-consumer credit is stronger than receive completion and MAY be useful for pipeline scheduling because it proves the remote GPU has consumed the payload.

Transport completion MUST NOT be confused with semantic model commit.

## Deferred retirement

The synchronous baseline SHOULD retire and repost completed receives before returning.

A later optimization MAY defer retirement of a bounded number of completed messages to overlap host event accounting with subsequent GPU work.

Deferred retirement MUST be bounded by:

- pool capacity;
- explicit in-flight accounting;
- consumer completion;
- deterministic cleanup on error.

It is an optimization tier, not the baseline correctness contract.

## Progress modes

At least the following progress strategies SHOULD be benchmarked independently:

- interrupt-driven completion;
- busy polling;
- dedicated polling thread;
- bounded deferred retirement.

The fastest microbenchmark mode does not automatically qualify for inference.

CPU usage, interrupt pressure and interference with inference threads MUST be measured.

## Exclusive lease

A direct imported-pool endpoint SHOULD be treated as an exclusive resource unless the kernel/runtime proves safe multi-owner semantics.

The runtime SHOULD expose a lease state per physical link.

Launching a second owner while a lease is active MUST fail rather than multiplexing implicitly.

Cleanup MUST identify holders and SHOULD refuse destructive cleanup while another process owns the endpoint.

## Link re-arm and stale completion state

Reusing the same physical rings, interrupt vectors or endpoint identifiers can expose stale completion state on some hardware/driver combinations.

Qualification MUST include:

- stop;
- cleanup;
- re-arm;
- first transfer after re-arm;
- repeated endpoint reuse.

A kernel version string alone MUST NOT be treated as proof that ring re-arm reliability is fixed.

The platform profile SHOULD record the qualified direct-stream ABI/driver identity.

## Dual-link architecture

Two physical USB4 links are modeled as two independent transports first.

Each link has its own:

```text
physical_link_id
connection_epoch
lease
TX pool
RX pool
ring/frame counters
completion queue
health
performance profile
```

Only after simultaneous qualification MAY a scheduler expose a dual-link policy.

The aggregate is therefore:

```text
MultiLinkTransport
  -> PhysicalLink A
  -> PhysicalLink B
```

not one opaque “80 Gbps” device.

## Dual-link policies

### Flow distribution

Independent messages are assigned to different links.

This SHOULD be the first dual-link production experiment because it needs no single-message reassembly.

### Direction specialization

One link MAY preferentially carry A-to-B traffic and the other B-to-A traffic.

This SHOULD be benchmarked even when each physical link is full duplex because controller/ring contention may differ under real traffic.

### Role specialization

One link MAY serve latency-critical decode/verification traffic while another serves large prefill or background transfers.

### Single-message striping

Large messages MAY be striped across both physical links only after link independence is proven.

Every stripe MUST carry independent epoch/sequence/stripe validation.

The message becomes consumable only after all required stripes are valid.

Failure of one stripe invalidates the whole logical message.

The runtime MUST NOT publish a partially reassembled tensor.

Small messages SHOULD remain on one link unless complete-operation latency proves otherwise.

## Dual-link pool policy

The initial dual-link direct-DMA implementation SHOULD use separate TX/RX pools per physical link.

Sharing one pool across two external DMA engines SHOULD be deferred until ownership and synchronization are independently proven.

Separate pools simplify:

- failure isolation;
- per-link epochs;
- frame accounting;
- graph/binding invalidation;
- lease cleanup;
- performance attribution.

## Control/data coexistence

The portable USB4 network path SHOULD remain active for:

- capability negotiation;
- health;
- leases;
- fallback;
- non-accelerated messages;
- diagnostics.

The direct data path SHOULD carry only explicitly qualified tensor shapes or message classes.

An unsupported shape MUST fall back cleanly rather than forcing a generic direct-DMA path.

## Direct-operation adapters

The fastest direct path MAY be operation-specific.

Examples include:

- activation transfer;
- partial reduction;
- exact-shape all-reduce;
- pipeline-stage boundary tensor.

An operation-specific adapter is acceptable when:

- the exact shape/dtype contract is explicit;
- a general fallback exists;
- numerical output is qualified;
- the adapter does not leak transport semantics into model code.

## Compression

Transport compression is a separate numerical mode.

It MUST NOT be enabled merely because a larger payload benchmark benefits.

Qualification MUST be shape-specific and include model-level accuracy/equivalence requirements.

For very small decode payloads, codec overhead may dominate the bytes saved.

## Benchmark boundary

The primary latency metric is a completed useful GPU operation.

For a receive-and-consume path this includes:

```text
GPU producer work
+ release to transport
+ USB4 transfer
+ receive validation
+ GPU consumer work
+ required consumer credit / completion
```

Bare cable latency, TCP echo latency and raw DMA bandwidth remain diagnostic metrics.

## Mandatory direct-DMA qualification

For one link:

- small decode-sized payload;
- 64-KiB-class payload;
- multi-megabyte prefill-sized payload;
- simultaneous bidirectional traffic;
- interrupt-driven and polling modes;
- GPU load during transfer;
- repeated stop/re-arm;
- error and timeout injection;
- RX repost pressure;
- exact numerical validation of any fused consumer operation.

For two links:

- each link alone;
- both links same direction;
- both links opposite directions;
- independent flows;
- role specialization;
- large-message striping;
- one-link failure during independent flows;
- one-link failure during a striped message;
- degrade to one link;
- recovery/requalification of the failed link.

## Promotion rule

A direct GPU transport tier becomes preferred only if:

1. its synchronization contract is proven;
2. completed GPU-operation latency improves;
3. the model workload improves;
4. CPU/interrupt cost is acceptable;
5. cleanup and re-arm are reliable;
6. fallback remains available.

A direct DMA path that wins only a transport microbenchmark MUST remain experimental.
