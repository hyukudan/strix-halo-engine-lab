# Experimental prepared eager submission

`--prepared-eager` selects an immutable prepared copy of the compiled serial
execution, IO, arena and host submission order. Preparation runs the full existing
preflight and caches persistent-state write counts and the first PLE step.
The ordinary eager executor remains the default and the diagnostic reference.

The prepared runner keeps dynamic token/shape/position checks, required state
flags, generation-overflow checks, resource validation and compatibility/binding
identities. A changed physical generation or binding requires new preparation.
The state layout is compared with its prepared snapshot so mutation under a stale
identity is rejected. A deeply owned tensor-view index snapshots the validated manifest, physical
bindings and canonical step/slot access plan. Actual allocation owners remain
strongly retained. Request token spans are never retained.

Each run creates only its lightweight result/state metadata. Its submission order
is a view of immutable prepared storage, avoiding a newly allocated completed
vector and result-order vector. The view remains valid while the prepared object
lives. Tensor helpers use the immutable views: index lookup is constant time,
name lookup uses a prebuilt transparent tree and step/slot access uses the existing
binary search. This avoids rescanning and validating every model tensor for each
individual lookup. Before publishing the cache in any dispatch context, admission
still compares all source metadata and physical identities; source descriptor
handles are also checked at lookup. GDN and shared-expert launchers consume
precomputed typed tensor roles from that same snapshot. Discovery runs during
preparation, including shared projection geometry; prepared dispatch does not
repeat their name/role scans. Launcher-specific dynamic shape, numerical,
extent and generation checks remain. Dynamic entry admission is still linear
in model metadata and is not claimed to meet every hot-path goal.

Prepared scratch lookups use a private per-step allocation-index table over
that same owned compiled snapshot. The table stores logical allocation indices,
with full buffer class, slot and generation checks; it stores no device addresses
or physical generations. Each resolution still validates the current memory
binding and address bounds. Another arena handle, a changed dispatch step or a
mismatched use is rejected. Preparation rejects duplicate uses and invalid
allocation extents. The ordinary eager resolver is unchanged.

The prepared session adapter preserves the existing session's readiness, PLE
prefetch, before-commit hook, poison/taint boundary, state generations, lineage
and consumed-position publication. The decoder pins the real experiment owner,
or its deferred completion owner and shadow state, until prepared work is drained.
A host launcher return is not device completion; existing completion owners and
drain paths remain responsible for it.

The option supports eager full-output execution and explicit scalar greedy
composition. Graphs, integer heads and prefix roundtrip are rejected in this first
slice. RAW/reference execution retains complete arrays. The kernel bundle and
automatic profile admission are unchanged; the submission mode is separately
identified and has no inherited native qualification.

The allocation-index extension has source fixtures; its review, build, native
and model checks are pending. Fixtures compare complete resolved views, dispatch
order and state generations with the ordinary executor, including invalid
contexts, current physical bindings and fork/join aliases. No speedup is
established by the source change.
