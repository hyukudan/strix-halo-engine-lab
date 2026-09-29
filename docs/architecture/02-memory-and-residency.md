# 02 — Memory and Residency

## Decision

Unified memory is the central systems constraint.

The engine MUST manage physical RAM, GPU mappings, file-backed pages, caches and temporary workspaces as one coordinated budget.

A model that can be mapped is not necessarily a model that can safely run.

## Memory classes

Every allocation or mapping MUST belong to a visible category:

1. resident model weights;
2. file-backed model mappings;
3. sparse or lookup tables;
4. KV cache;
5. recurrent state;
6. speculative state and rollback;
7. execution scratch;
8. graph buffers;
9. runtime and library overhead;
10. bounded reusable caches;
11. host I/O buffers;
12. operating-system reserve.

No significant pool may remain "miscellaneous".

## Admission model

Admission MUST be based on physical-memory safety, not only on nominal GPU allocation.

The admission controller MUST account for:

- physical RAM;
- current baseline usage;
- configured host reserve;
- resident model bytes;
- expected file-backed residency;
- per-session persistent state;
- reply growth;
- peak prefill workspace;
- graph and capture buffers;
- temporary conversion or repack buffers;
- recovery margin.

The engine MUST reject a request or model load before entering an unsafe memory state.

## Host reserve

A fixed recovery reserve MUST remain outside normal admission.

The reserve exists for:

- kernel and system services;
- page cache churn;
- process teardown;
- driver recovery;
- short-lived peaks;
- administrative access.

The GPU mapping limit MUST NOT be interpreted as an allocation target.

## File-backed memory

File-backed mappings MAY be used for weights or large lookup tables when they reduce duplication or enable demand-driven access.

For each mapped region the engine SHOULD know:

- total bytes;
- expected active fraction;
- whether access is sequential or random;
- whether pages should be prefetched;
- whether readahead is desirable;
- whether pages are reclaimable;
- whether the GPU directly accesses the region.

Random-access tables SHOULD use access hints that avoid destructive readahead.

## Prefault and prefetch

Page preparation MUST be bounded.

Large synchronous prefault operations MUST NOT block request service unpredictably.

Prefetch SHOULD be:

- asynchronous where possible;
- capped by a memory budget;
- measurable;
- cancellable;
- aware of whether pages are already resident.

## Registered and pinned memory

Pinned or registered memory is especially dangerous on a unified-memory host because it reduces reclaimability.

The engine MUST track registered bytes explicitly.

Registration SHOULD be limited to hot regions whose benefit is demonstrated.

Whole-model registration MUST NOT be a default until measured against alternatives.

## Huge pages

Huge pages MAY be tested for copied or anonymous hot buffers.

They MUST NOT be assumed beneficial for sparse random file-backed access.

Any huge-page mode MUST be qualified for:

- startup time;
- page-fault behavior;
- fragmentation;
- decode latency;
- prefill throughput;
- teardown.

## Pressure response

The runtime MUST expose memory pressure before failure.

At increasing pressure it SHOULD:

1. stop admitting new requests;
2. reduce optional caches;
3. release reclaimable retained state;
4. drain work;
5. unload the model if necessary;
6. refuse automatic reload if the driver or mapping state is inconsistent.

Out-of-memory recovery MUST be designed, not delegated to the operating system.
