# Experimental wave router selection

The low-level `--wave-moe-router-topk` defaults to OFF in manual selection,
requires `--parallel-moe-router`, and
excludes `--ranked-moe-router-topk`. It selects a distinct `.wave.parallel`
kernel ID, launcher tag and `hip-gfx1151-moe-route-v6-wave-parallel` compiler
identity. Catalog, serial, speculative and bundle admission check this mode.
Existing modes and their device function bodies remain unchanged.
With no low-level flags, the content-bound [automatic profile](decode-optimization-profile.md) selects wave
and lane-chain HC output for its qualified checkpoint/mode/context. Explicit ranked
routing is retained. Low-level flags without an explicit auto request retain their
legacy configuration. The combined automatic profile still requires its own validation.

The parallel projection is reused. The new selection kernel copies the current
256-thread softmax expression sequence: per-lane ascending input scan, maximum
tree 128..1, retained maximum barrier, exponential evaluation, ascending local
sum, sum tree 128..1, and per-probability division. Its final lane-zero selected
sum and division retain slot order. No new compiler FP flags are introduced.

Only selection bookkeeping changes. Wave zero owns 16 candidate slots per lane
for 512 experts. Non-NaN probabilities form a strict total order under descending
float value and ascending expert index. Signed zeros compare equal and tie by
index. Each lane finds its best remaining candidate and XOR shuffles combine
these maxima across all 32 lanes. Removing the unique winning index produces the
same ordered list as literal serial insertion. Shuffles transport value bits;
they do not reorder floating sums or compare integer encodings of floats.
Missing candidates retain the negative-infinity/expert-count sentinel, including
the original all-NaN row behavior in the lane-zero division epilogue.

All 256 threads reach both additional uniform barriers: one publishes normalized
probabilities to wave zero; the other publishes selected slots to lane zero.
The nominal selection LDS storage is 3328 bytes, below the unchanged conservative
router resource declaration 9472 bytes. Compiled LDS, register count, scratch,
instruction sequence and FP descriptor modes still require actual target review.
This is not an occupancy or performance measurement.

Wave selection accepts 1..512 experts and 1..min(32,E) selected slots at the
launcher boundary. Existing default/ranked kernels keep their 16-slot arrays.
K32 qualification is launcher-only: downstream expert/whole-model limits have
not been expanded. All layers share the model router shape, so unsupported
shapes reject rather than silently fall back. JSON reports the boolean mode,
eligible and fallback route steps counted from the actual compiled bindings in
all three app result branches. Active wave mode requires every route binding to
end in `.wave.parallel`; disabled mode counts its existing route bindings as
fallbacks.

The shape-only registration API cannot express expert groups or selection bias.
Raw GGUF intake rejects group-count metadata and `*.exp_probs_b*` tensors when
wave mode is requested. Before GPU setup the app also validates the complete
manifest, verifies its actual layout hash, and creates an owned immutable deep
copy. Catalog/configuration retain this admission object, and execution resources
reference that exact snapshot. Dispatch checks pointer identity in constant time;
it never rescans tensor names or trusts a cache of a mutable declared layout ID.
A changed caller manifest, even with the same declared ID, cannot mutate the
snapshot or pass the dispatch pointer check. Direct callers must obtain the same
admission and bind its const manifest before enqueue, and must run the raw-input
validator before reducing metadata to `ModelContract`.

Wave selection adds no router-weight alignment requirement: it does not read
weights, and the existing projection remains unchanged. Activation/output natural
alignment and the existing aligned FP32 logits workspace contract are preserved.

Local validation: WSL GNU 13.3 Release, HIP-OFF complete build and 152/152 CTest
cases passed. The independent host wave emulator compares 6,864 cases against
literal insertion, including K1..32, signed zeros, selectable negative infinity,
NaN holes/payloads, subnormals, ownership ties and sentinel slots. It snapshots
lanes before each simulated shuffle and checks uniqueness, gaps and selected
value bits. Duplicate/hole mutants and unsupported geometry/group/bias inputs
are rejected. Serial/speculative tests verify distinct identities and wrong-mode
binding rejection without changing arena sizes.

Device fixtures are supplied but not executed here. They compare wave with
existing default/parallel/ranked results over the existing typed matrix, plus
an independent fixture-local 32-slot literal insertion kernel. F32 one-hot
projection injects ties, underflow, infinities and partial NaN logits. K32 covers
tails 33/255/257/500/511/512, both activation types and normalization modes;
four-row cases cover all four supported weight types. Injected K<=16 states
also run the actual production default, parallel and ranked launchers in poisoned
eager and captured executions, with raw selected bits compared. A mutable bias
manifest reusing an admitted ID must reject without changing poisoned outputs. Eager and captured output
buffers and logits workspace are poisoned, raw selected bits are compared, and
each captured router must have exactly projection+selection kernel nodes.

Pending: actual Core10 HIP compile, compiled softmax/epilogue and descriptor
review, preservation of old device functions, device fixtures, full 554 arrays
and 512 generated IDs against the pinned baseline, and isolated/whole-route
measurement. No target qualification, model exactness or speed benefit is claimed.
