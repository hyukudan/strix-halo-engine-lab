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
- engine commit.

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

## Latency metrics

Measure distributions, not only averages.

The engine SHOULD record:

- time to first token;
- decode inter-token latency;
- p50/p95/p99 request latency;
- queue time;
- prefill time;
- verify-cycle time;
- page-fault stalls.

## Throughput metrics

Report separately:

- prompt tokens/s;
- generated tokens/s;
- accepted speculative tokens/s;
- aggregate throughput under concurrency.

Never mix prompt and decode throughput into one headline number.

## Memory metrics

Capture:

- resident process memory;
- reclaimable file-backed pages;
- GPU-visible mapped bytes;
- registered/pinned bytes;
- KV bytes;
- recurrent-state bytes;
- scratch peaks;
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
- repeated load/unload.

## A/B discipline

Comparisons MUST change one material factor at a time where practical.

Order SHOULD be alternated to reduce thermal, cache and background-load bias.

Results that cannot be reproduced across repeated runs MUST be treated as noise until explained.
