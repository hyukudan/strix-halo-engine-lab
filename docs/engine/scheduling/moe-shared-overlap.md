# Experimental shared-expert fork/join

The opt-in eager dispatcher keeps the existing route, shared-expert, routed-expert
and combine host submission order. It submits the shared-expert launcher on one
persistent nonblocking sibling stream. A main-stream event makes its input ready;
a side-stream event is joined on main before combine. Route has already been
submitted when this first implementation forks. Other launchers and their
arithmetic are unchanged.

## Arena and identities

`MoeForkJoinArenaPlan` conservatively relocates every shared output and temporary
into a dedicated appended region. It retains the serial offsets for other uses.
Only one layer is active, and a join precedes reuse of that region. The complete
contiguous arena, including serial holes and alignment, must fit the explicit
budget. This is a scratch budget, not a claim about total device memory capacity.

The plan rebinds a deeply checked source execution into a new layout/execution
identity. Allocation and fresh launcher compilation follow rebinding. The adapter
validates the expanded execution once during construction, retains an immutable
snapshot and checks dynamic scratch identity/base/extent/generation plus each
accessed arena mapping at dispatch. An equivalent prepared eager arena copy is
accepted; its prepared tensor-view pointer is forwarded unchanged.

## Ownership and publication

Main is borrowed from the D1 owner; the adapter owns its side stream and two
persistent events. Strong D1/resource and launcher-catalog leases and the original
device guard remain alive until completion. The caller must retain the compiled
execution, runtime-resource and telemetry objects until adapter destruction.

The side scope copies the active launch context, changes its stream and removes
its deferred error record. It restores the exact original TLS pointer through
RAII. This is an explicit override, not a nested `HipLaunchScope`. Main remains
the single publisher of the eight-byte D1 record.

The enclosing transaction must call `begin_token`, dispatch all admitted steps,
`verify_before_commit`, D1 completion/publication, and finally `mark_committed`
after the actual session commit. Verification drains main, independently queries
side completion and checks the original device. This conservative increment adds
a main drain before the existing D1 completion boundary; its overhead is unmeasured.
Any exception, including an admission failure, requires enclosing `abort`. Abort
and destruction recover/drain the original device before releasing owners; an
unproved drain terminates the process rather than freeing possibly active memory.

State-only prefix composition explicitly allows omission of exactly the terminal,
state-free kernel LM head. The adapter receives that output intent at verification;
it advances only logical traversal, never a launcher or GPU-work counter. Missing
other steps, layers, joins or an unexpected head cannot publish completion.

## Scope and pending validation

This implementation admits eager, single-row, single-request execution with D1.
Graph capture, integer-head execution and prefix-cache roundtrip are excluded.
Defaults remain unchanged. The native fixture source covers actual producer/fork/
join ordering with both branch delay directions, TLS restoration, main-only D1
record preservation, incomplete protocols and pre-enqueue graph rejection. A second fixture constructs the canonical compiled DAG, a real expanded GPU
arena and synthetic resource metadata. Its registered toy callbacks exercise the
actual adapter through serial and prepared session transactions, both terminal
head intents, and routed/completion failures. Per-layer GPU ledgers and guarded
allocations check join/data integrity. Its publication callback is a stub: neither
fixture proves model arithmetic or actual D1 shadow publication.

No compilation, test execution, model fidelity or performance measurement is
reported for this source increment. Required subsequent validation includes the
arena host tests, native event/error paths, prepared and state-only composition,
unchanged model outputs and session state, and paired complete-workload timing.
Concurrent submission alone does not establish overlap or a speed improvement.
