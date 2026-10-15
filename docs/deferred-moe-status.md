# Deferred eager MoE status

`strix-hip-decode-token --deferred-moe-status` selects an experimental eager C1
completion policy. It is disabled by default and rejects graph execution,
speculative verification and the separate integer LM-head stream. Each affected
MoE kernel ID includes `.deferred-status-v1`; implementation and compiled execution
identities differ from synchronous eager execution. Scratch admission and existing
arithmetic kernel bodies are unchanged. JSON receipts include the selected flag.

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
layers, host dispatch failure, completion failure, precise first-error reporting,
unpublished host histories and metadata, successful publication, poisoned reuse,
mode identities and graph/verification rejection. Native fixtures extend existing
K-quant launcher tests with invalid routes/weights, precise A8 activation/delta/
pipeline bits and preservation across a later successful layer.

Native fixture execution, HIP compiler equivalence, review, full-model arrays,
tokens/state fidelity and performance remain pending. No native speedup or model
qualification follows from the portable checks.
