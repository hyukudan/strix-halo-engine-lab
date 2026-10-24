# Deferred eager MoE status

Current status: this mode is included in the adopted canonical automatic v3
profile at source `265fd78762884d18eb32aa60294408f080088d52`. Its combined
native, complete554-array, generated512 and descriptive workload gates passed;
see the [qualified adoption](optimization-profile.md#qualified-adoption-2026-10-15)
for scope and limits. The source-increment and fixture observations below retain
their historical scope; their earlier pending gates are closed only for this
qualified content/context/execution slice. Low-level manual selection stays explicit.

`strix-hip-decode-token --deferred-moe-status` selects an experimental eager C1
completion policy. Low-level and manual defaults are disabled. The
[automatic profile](optimization-profile.md) includes this policy
after content and execution admission, within its qualified workload slice.
The completion policy rejects graph execution,
speculative verification and the separate integer LM-head stream. Each affected
MoE kernel ID includes `.deferred-status-v1`; implementation and compiled execution
identities differ from synchronous eager execution. Scratch admission and existing
arithmetic kernel bodies are unchanged. JSON receipts include the selected flag
and `deferred_moe_status_completed_tokens`, read from the actual request owner
at each of the three receipt boundaries. It counts successful owner completions;
no layer or status-execution counter is inferred.

A request owns a nonblocking stream, an eight-byte device record and a persistent
host completion record. After each layer, one ordered device work item retains
the first nonzero local status as `{domain, bits}`. Legacy and Q4K A8 domains are
distinct. A later local reset or successful layer cannot erase the first failure.
The existing exception class, full A8 bit value and message are preserved. The
record is separate from graph status, so eager GDN and PLE keep eager semantics.

All token kernels use the owned stream. PLE keeps its existing explicit stream
fences around synchronous host staging and history updates. The mode removes
per-layer MoE status reads/fences; it does not remove these PLE fences. Runtime
owners, the source resources, host D2H storage and shadow histories survive actual
completion or `recover_or_fail_stop()` before destruction.

The session executes against an aligned shadow of its complete host history.
Position preparation, PLE hashing/prefetch and n-gram updates use this shadow.
Device KV/GDN allocations remain in-place. The completion hook drains the token
stream and checks status inside the session's exception boundary, before copying
host history back or publishing generations, lineage, next position and committed
token count. A dispatch, completion or status error preserves published host
history and metadata, taints lineage and poisons the session. Reuse requires
recovery. Deterministic preflight failure before dispatch leaves the session ready.
The shadow changes binding identity while retaining layout compatibility.

Portable Release checks use GNU 13.3 under WSL with HIP disabled: 152/152 CTest
entries pass. They cover a failure in MoE layer 17 followed by all remaining
layers, host dispatch failure, completion failure, status exception reporting,
unpublished host histories and metadata, successful publication, poisoned reuse,
mode identities and graph/verification rejection. Native fixtures extend existing
K-quant launcher tests with invalid routes/weights, precise A8 activation/delta/
pipeline bits and preservation across a later successful layer.

Native fixture execution, HIP compiler equivalence, review, full-model arrays,
tokens/state fidelity and performance remain pending. No native speedup or model
qualification follows from the portable checks.


The follow-up first-error fixture calls `merge_deferred_moe_first_error`, the same
host/device helper used by the actual device publisher. A zero/A8-5/legacy-1/zero
sequence must retain the original A8 domain and the complete bits value 5. Session fixtures
also cover an error recorded before a later host exception: the host exception
takes precedence, bypasses completion and cancels shadow publication. Abort drains
for owner safety and discards the device record; it does not replace or nest the
original host exception.

The owner still clears the record in `begin()`, before serial preflight. Therefore
a deterministic serial preflight rejection can require a recovery drain even
though the session remains ready and publishes nothing. Removing this enqueue via
a constructor zero plus dirty-state protocol is pending owner-level failure and
recovery injection coverage. The conservative reset semantics remain in place.

New admission fixtures use matching row buffers and exact exception messages,
including multirow and C2 rejection. Host launch-scope fixtures check exact record
size, alignment, overflow, identities/generations and exclusion from graph scopes.
Native fixtures check deferred-without-scope, synchronous-with-deferred-scope and
record/arena alias rejections before scratch or semantic output mutation; the two
owned-scope cases must capture zero nodes. A bounded host callback holds only the
fixture stream while invoking the deferred launcher: it must return with the
stream still pending. The callback has a five-second escape and is released on
exceptions before recovery, so a regression to a layer drain produces failure
rather than an intentional infinite wait. This is a liveness gate, not a latency
measurement. New native fixtures and their API behavior remain pending execution.

Result-publication follow-up: the mode boolean is static prefix metadata, while
`deferred_moe_status_completed_tokens` is read from the actual completion owner
only when restore/greedy results are ready or after the raw token loop. The host
regression compares a retained zero prefix snapshot with a fresh completed reader
after the real serial-session commit path advances to one. It does not execute
the native HIP owner; full-model post-loop counts remain a separate native gate.
