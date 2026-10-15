# Experimental prepared eager submission

`--prepared-eager` selects an immutable prepared copy of the compiled serial
execution, IO, arena and host submission order. Preparation runs the full existing
preflight and caches persistent-state write counts and the first PLE step.
The ordinary eager executor remains the default and the diagnostic reference.

The prepared runner keeps dynamic token/shape/position checks, required state
flags, generation-overflow checks, resource validation and compatibility/binding
identities. A changed physical generation or binding requires new preparation.
The state layout is compared with its prepared snapshot so mutation under a stale
identity is rejected. No tensor pointers or request token spans are cached.

Each run creates only its lightweight result/state metadata. Its submission order
is a view of immutable prepared storage, avoiding a newly allocated completed
vector and result-order vector. The view remains valid while the prepared object
lives. Launcher-specific validation is unchanged. This does not remove tensor
searches inside launchers or all metadata work: those remain subsequent prepared
operator increments. Dynamic resource validation is deliberately retained.

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

Status: source implementation, compilation/tests/measurements deferred. Source
fixtures compare dispatch order and state generations with the original executor
and exercise invalid requests, stale resource generations and failures. Native
model/state equality, transactional failure injection, performance and full
prepared-operator coverage are still required. No speedup is established.
