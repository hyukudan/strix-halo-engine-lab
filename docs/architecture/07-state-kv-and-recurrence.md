# 07 — KV, Recurrent State and Snapshots

## Decision

Every request owns logically independent execution state.

Concurrency, speculative decoding and prefix reuse MUST preserve that independence.

## State classes

A session may contain:

- attention KV;
- recurrent state;
- convolution or short-history state;
- sparse-attention index state;
- n-gram history;
- RNG state;
- speculative checkpoints;
- accepted position;
- prompt-cache metadata.

Each class MUST have explicit lifetime and byte accounting.

## Allocation

Persistent session state SHOULD be allocated from capacity-known pools created after model admission.

Unbounded per-token allocation is forbidden.

The engine SHOULD know the maximum number of positions and sessions it can sustain before accepting traffic.

## KV layout

KV layout SHOULD optimize the dominant decode access pattern while keeping append operations simple.

The layout MUST support:

- deterministic indexing;
- prefix retention;
- rollback or truncation;
- bounded context growth;
- per-session isolation.

## Recurrent state

Recurrent state MUST have a canonical update order.

Speculative execution MUST NOT commit recurrent state until the corresponding tokens are accepted.

Where full snapshots are expensive, the preferred strategy is:

1. retain the minimal pre-update state needed;
2. verify candidate tokens;
3. replay or recompute accepted updates;
4. restore rejected tails without cloning the entire state per draft.

## Snapshots

Snapshots SHOULD store only valid state, not full allocated capacity.

Persistent snapshots MUST include enough identity to reject incompatible reuse, including:

- model identity;
- execution-layout identity;
- kernel-plan identity;
- context/chunk-plan identity;
- state-schema version.

## Prefix reuse

A retained prefix MUST correspond to a real semantic boundary in the rendered token stream.

Reuse MUST stop at the longest prefix for which all required state classes are valid.

A cached prefix that matches token IDs but not recurrent, sparse or auxiliary state MUST NOT be reused.

## Eviction

Prefix state is a bounded cache.

Eviction policy SHOULD consider:

- bytes held;
- rebuild cost;
- recency;
- session activity;
- current memory pressure.

Admission takes priority over retaining old prefixes.
