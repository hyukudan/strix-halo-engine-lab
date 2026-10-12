# Reusable execution components

## Attention position/state stage

`GraphAttentionStatePlan` describes a C1 post-projection stage, not complete attention
or a whole-model graph. It binds the execution identity, FP32 query/key/value workspaces,
resident F32/F16/BF16 normalization weights, device argument/status allocations and
one serial BF16 KV layer. Geometry, checked context bytes, allocation generations,
alignment and aliases are validated before launching. Query stride is one or two head
widths; power-of-two heads up to 1024 features use the existing reduction. Exactly
shared read-only norm views are counted once. Counts describe retained views, not a
physical memory reservation or a native graph executable memory observation.

The optional HIP consumer reuses the existing norm/RoPE and KV-write kernels. A graph
reads absolute position from stable device data; the separate scalar control rejects
capture. FP32 tree RMS, BF16 rounding, `(1 + norm_weight)` and split-half RoPE arithmetic
remain the existing operator semantics. This is not a generic interchangeable RMSNorm
definition. Plan identity binds all execution-relevant geometry, scalars, mode and
resource bindings; changing metadata or allocation generations invalidates an old plan.

Status initialization precedes finite-source validation, normalization and the KV
write on the caller's explicitly owned stream. Status 2 rejects an out-of-context
position, 4 rejects nonfinite source/norm or V conversion, and 8 rejects nonfinite
normalization/RoPE arithmetic. A head broadcasts a single atomic status observation
before its workgroup barriers so concurrent head failures cannot cause divergent
early return. The later KV writer observes all prior kernel status updates and skips
the write on failure. Scratch Q/K may be partly modified on arithmetic failure and
cannot be reused as a successful projection; caller recovery must restore/recompute
them. A runtime submission error can leave state completion unknown and requires
the enclosing session's poison/ownership rules.

Validation checks only the live query normalization features. Query gate padding is
preserved and ignored consistently for every head; its later consumer has separate
validation responsibility. Arithmetic checks include the shared RMS reduction,
mean/inverse and final BF16 conversion, since finite FP32 alone cannot guarantee
a finite stored state.

The launcher is asynchronous. Allocation owners and input values must remain stable
through actual stream completion; no return from launch establishes device completion.
Reading a zero status after completion establishes only this stage's result, not
accepted-token commit, prefix lineage, numerical qualification or whole-model replay.
Existing whole-model graph-safe flags remain unchanged. Sparse selection, attention
score/output, PLE host lookup and other phase/state operators still need their own
capture-compatible implementation and target qualification.

The native fixture captures this real stage, changes device positions on successive
replays, compares query/KV bits to the scalar control and checks an independent small
RMS/RoPE/V reference. Invalid U64 positions, nonfinite source and arithmetic overflow
must preserve persistent KV; query-gate padding and KV allocation guards remain intact.
Later valid replay must reset status. These fixtures are compiled offline; execution
and state/model quality remain mandatory target gates.

## Integer projection execution

The standalone integer projection lab now consumes the shared opt-in
`HipIntegerProjectionLauncher`; its preparation and Wave32 IU8/IU4 kernels live in
`src/hip_integer_projection.hip`. The portable execution contract binds the phase,
numerical identity, fixed geometry, device, allocation extents/generations and
per-access epochs. Unspecified phases, overlapping writable views, stale bindings
and excessive retained allocations are rejected. There is no default runtime dispatch.

An explicitly recorded producer event and allocation owners establish input
readiness. The blocking launcher prepares activations, computes into scratch and
validates device status before publishing the destination. Completion retains its
destination owner and binds the operation and access epochs; changing contents
invalidates older completion evidence. Failed work keeps its owners until actual
stream quiescence or process-lifetime quarantine. Event readiness must also match
the recorded resource epochs, not merely the same owner objects.

Portable tests validate contracts and completion rejection. Offline compilation
checks gfx1151 instruction emission and linkage, without executing GPU work.
Native fragment mapping, numerical accuracy, preparation-inclusive cost and
full-model quality remain separate target gates.

## Copied boundary payload transport

`CopiedStreamTransportBackend` transfers bounded opaque host tensor bytes through
an explicitly owned FD byte stream. Its payload/ACK envelope v1 nests the existing
v2 boundary metadata. A dedicated payload connection is separate from the control
connection. Exact logical and physical identities, peer epochs and monotonically
increasing sequences are checked before exposing inbound data. Connection replay
history survives backend reopening within the same epoch.

Typed producer and consumer completion proofs retain owned staging through delay,
failure and cancellation. Receipt and consumed ACKs have distinct meanings; neither
commits model state. Cancellation poisons the local epoch, and later cleanup proof
releases memory without sending an ACK or restoring the failed connection.
Unproved work retains its allocations rather than freeing memory still in use.

`CopiedStreamDistributedBackend` connects this payload path to the existing queued
v2 boundary control and local execution interfaces. Payload bounds are checked
before allocating host storage or invoking preparation. Socketpair and injected
IO fixtures check actual copies, replay rejection, delayed completion, callback
errors and cleanup. The C1 path has one serialized progress owner and a bounded
poll budget; peers must make independent progress. It performs no numeric
conversion, GPU dereference, DMA or link striping. Actual two-host character-device
behavior, topology, throughput and full-model state continuity remain target gates.
