# 10 — Benchmarking and Qualification

## Decision

No optimization becomes a default without reproducible end-to-end qualification.

Microbenchmarks are diagnostic tools, not release gates.

## Environment fingerprint

Every benchmark result MUST record:

- CPU and GPU identity;
- firmware/BIOS profile;
- physical RAM;
- kernel version;
- ROCm version;
- compiler version;
- GPU target;
- power mode;
- relevant kernel parameters;
- model identity;
- weight format;
- context;
- concurrency;
- speculative settings;
- cache state;
- engine commit;
- numerical-plan identity;
- kernel-plan identity;
- graph/allocation generation policy where applicable.

For multi-node results it MUST additionally record:

- node count;
- placement plan identity;
- transport implementation;
- physical link count;
- per-link negotiated rate;
- physical USB4 router and port topology;
- link-to-node mapping;
- interrupt affinity;
- transport-buffer mode;
- transport scheduler policy.

## Required benchmark classes

At minimum:

1. cold startup;
2. warm startup;
3. short prefill;
4. long prefill;
5. C1 decode;
6. C2 decode;
7. C4 decode;
8. speculative decode;
9. fresh long conversation;
10. resumed long conversation;
11. memory pressure;
12. teardown and reload.

For a multi-node-qualified build, also:

13. one-link transport latency and bandwidth;
14. one-link simultaneous bidirectional traffic;
15. dual-link independent traffic;
16. dual-link directional split;
17. dual-link flow distribution;
18. dual-link striping for large messages;
19. distributed C1 decode;
20. distributed long prefill;
21. distributed failure and fallback;
22. peer restart and reconnect;
23. sustained multi-node soak.

## Transport message matrix

Transport microbenchmarks SHOULD cover message sizes representative of actual execution boundaries rather than only powers of two.

The matrix SHOULD include:

- small control messages;
- decode hidden-state boundaries;
- partial reductions;
- routed-expert payloads;
- large prefill boundaries;
- logits or partial logits where applicable.

For each class measure queue depth one and pipelined operation.

## Single-link versus dual-link qualification

Two physical links qualify as useful only if they improve the real model workload.

The mandatory comparison is:

```text
single link
vs dual link, directional
vs dual link, flow-level
vs dual link, striped
```

Report:

- effective payload bandwidth;
- p50/p95/p99 transfer latency;
- synchronization overhead;
- reassembly overhead;
- CPU usage;
- interrupts;
- end-to-end tokens/s;
- time/token;
- time to first token.

A dual-link mode that increases aggregate bandwidth but worsens decode latency SHOULD NOT become the default for interactive serving.

## Latency metrics

Measure distributions, not only averages.

The engine SHOULD record:

- time to first token;
- decode inter-token latency;
- p50/p95/p99 request latency;
- queue time;
- prefill time;
- verify-cycle time;
- page-fault stalls;
- per-boundary transport wait;
- link-queue delay.

## Throughput metrics

Report separately:

- prompt tokens/s;
- generated tokens/s;
- accepted speculative tokens/s;
- aggregate throughput under concurrency;
- transport payload GiB/s.

Never mix prompt and decode throughput into one headline number.

## Operator traffic diagnostics

When qualifying low-row, recurrent or MoE fast paths, the benchmark SHOULD also capture:

- effective packed-weight bytes consumed;
- repack or dequantization bytes and time;
- persistent-state bytes read and written;
- distinct experts touched per round;
- rows per expert and grouping/padding overhead;
- launch count and exposed synchronization;
- replay or rollback work per committed token.

These diagnostics explain why a faster microkernel can still lose in complete inference.

## Memory metrics

Capture:

- resident process memory;
- reclaimable file-backed pages;
- GPU-visible mapped bytes;
- registered/pinned bytes;
- KV bytes;
- recurrent-state bytes;
- scratch peaks;
- transport-buffer bytes;
- driver/runtime overhead.

Memory measurements MUST be taken during realistic peak phases, not only after steady state.

## Correctness gates

A performance result is invalid if it fails any required correctness test.

Qualification SHOULD include:

- serial versus optimized;
- solo versus batched;
- fresh versus resumed;
- speculative versus serial;
- deterministic tie cases;
- long-context boundary cases;
- repeated load/unload;
- local versus distributed execution where numerical semantics are intended to match;
- one-link versus dual-link equivalence;
- reconnect and fallback without stale tensor reuse.


## Multi-node qualification

A multi-node benchmark MUST additionally record:

- node count;
- placement-plan identity;
- transport implementation;
- physical link count;
- negotiated rate per link;
- USB4 router and port topology;
- link-to-peer mapping;
- interrupt affinity;
- transport-buffer mode;
- transport scheduler policy.

The required transport matrix is:

```text
single link
dual link, same direction
dual link, opposite directions
dual link, flow-level distribution
dual link, large-message striping
```

For each mode measure:

- queue-depth-one one-way latency when a defensible cross-host clock method is available;
- round-trip latency;
- p50/p95/p99 transfer latency;
- effective payload bandwidth;
- simultaneous bidirectional bandwidth;
- CPU utilization;
- interrupt rate;
- synchronization and reassembly overhead;
- end-to-end tokens/s;
- time/token;
- time to first token.

If one-way latency is estimated from round-trip time rather than measured with synchronized clocks, the result MUST be labelled as an estimate.

Distributed results MUST state whether the objective is latency, throughput or memory capacity. A capacity-improving placement MUST NOT be presented as a latency speedup without evidence.

Two links qualify as useful only if they improve the real model workload. A mode that raises aggregate bandwidth but worsens interactive decode latency SHOULD NOT become the default.

Transport message sizes SHOULD be derived from real execution boundaries such as hidden states, partial reductions, expert payloads, prefill boundaries and logits rather than only synthetic powers of two.

## A/B discipline

Comparisons MUST change one material factor at a time where practical.

Order SHOULD be alternated to reduce thermal, cache and background-load bias.

Results that cannot be reproduced across repeated runs MUST be treated as noise until explained.

Ablations SHOULD isolate high-level mechanisms as well as kernel choices. At minimum, relevant comparisons SHOULD include serial versus speculative, full-state snapshot versus replay-oriented rollback, direct versus grouped experts, eager versus graph replay, and one-link versus each qualified dual-link policy.
