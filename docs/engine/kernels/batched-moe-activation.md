# Batched routed-expert activation

The explicit `--batched-moe-activation` option replaces the A8 routed pipeline's
per-slot SiLU submissions with one two-dimensional kernel. Its x grid is
`ceil(intermediate_size/256)` and its y grid is `top_k`; each block still has 256
threads. Each y block addresses the same existing FP32 `[gate, up]` slot. The
kernel uses the original feature index, bounds guard and expression
`(gate / (1 + expf(-gate))) * up`. It writes only the gate plane. It performs no
cross-slot arithmetic and changes no projection, quantizer or down reduction.

Admission requires eager execution and the existing Q4K A8 slot workspace. All existing owner, range,
alignment and runtime workspace checks finish before the initial clear or any
other enqueue. A Q5_K layer in a mixed checkpoint keeps its original per-expert
fallback. With the option off, the A8 consumer also retains its original loop.
The default, automatic optimization profile and original kernel bodies stay
unchanged. The selected kernel ID, compiled execution and launcher implementation
identities distinguish the explicit scheduling overlay; the numerical contract
does not acquire a different precision or reduction policy.

The status word is read-only throughout the activation phase. A prior nonzero
status suppresses every activation write. The pre-activation scan, complete
post-activation finite scan, downstream validation and semantic output commit
remain in their original order. The new kernel neither clears status nor
publishes a token, session generation or output commit. Completion and deferred
error handling remain owned by the original runtime/session boundary.

For ten active slots, the change replaces ten activation kernel submissions
with one while preserving the number of activation elements and their logical
workspace traffic. It does not reduce weight bytes, arithmetic work or model
precision. The receipt counters `batched_moe_activation_launches` and
`batched_moe_activation_slots` record successful submission of the new launch and
its slot count; they do not count graph replays,
completed GPU work, valid activations or committed tokens. The legacy fallback
does not increment them. No latency or physical traffic improvement is measured.

The diagnostic seam is built only with
`STRIX_BUILD_MOE_BATCHED_ACTIVATION_DIAGNOSTIC`. It calls the actual original or
batched kernel on an explicitly owned stream, allocates nothing, synchronizes
nothing and keeps the caller-owned sticky status untouched. It deliberately
exposes raw activation behavior: full-pipeline finite scans remain outside this
single-stage seam. The caller retains every owner through completion, including
after enqueue exceptions and across graph lifetime.

The new test source compares the complete workspace bitwise for finite inputs,
checks every untouched up plane and outer guard, repeats evolving in-place
activation in eager and captured execution, and exercises intermediate and slot
tails. Nonfinite raw inputs check the incumbent propagation class without
asserting NaN payload identity; sticky rejection checks every bit unchanged.
Captured graph nodes verify ten original launches versus one batched launch and
the actual two-dimensional geometry. Invalid shape, view, alias and stream
admission must create no captured work. Production A8 workspace admission remains
stricter than the diagnostic tail fixtures.

This source has not yet been compiled or executed. Compiler/ISA correspondence,
native diagnostic results, full-model logits/state/token fidelity and complete
paired workload timing remain pending before any qualification or adoption.
