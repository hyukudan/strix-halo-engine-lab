# Shared-expert row reuse diagnostic

This standalone component evaluates R2, R4 and R8 BF16 input/output rows with
Q8_0 gate, up and down matrices and an F32 scalar gate. It does not change the
decoder, catalog, numerical defaults or the complete-network one-row guard.
Other matrix formats remain on the existing operator path.

The reference invokes the existing prefetch8 Q8 kernels separately for each row.
The candidate assigns one output feature to a wave, loads eight Q8 codes and one
scale per lane/chunk, and reuses those coefficients across independent row
accumulators. Each row retains the virtual T256 leaf mapping and ordered reduction
helper. No reduction crosses rows. The eight waves of a block handle eight
output features; this grouping does not assign a different session's state.

Per-row workspace keeps gate[M] followed by up[M]. The existing activation kernel
consumes up before the existing scalar projection overwrites up[0]. The down
projection multiplies the scalar gate first and uses the existing BF16 rounding
helper. The reference submits 5R kernels; the candidate submits three matrix
projections and 2R existing activation/scalar kernels. These are submission counts,
not measurements of memory traffic or speed.

The caller supplies exact buffer extents, aligned byte strides, disjoint writable
views and an explicit owned nondefault stream. Read-only aliases are allowed.
Validation precedes stale-error inspection and every enqueue. No allocation,
clear, synchronization or session-state publication occurs in the API. Owners
must survive completion even if a later enqueue fails.

Source fixtures cover padded strides, independently distinct rows, signed zeros,
subnormals, cancellation, extreme finite values, Inf/NaN payloads, read-only data,
poison/canaries, input-row isolation, detectable Q8 coefficient mutation, exact
zero-pipeline output/workspace, graph identities/counts and three graph replays.
Invalid views/strides/rows/arms/default streams are checked for zero captured
nodes and unchanged writable storage.

Status: implementation source only. Compilation, tests, linked ISA/resource and
operand-priority checks, actual Q8 load-reuse confirmation, native fidelity and
performance are deferred. Source expressions do not establish hardware bitwise
identity, including nonfinite payload selection or contraction. No multirow
decoder, prefill throughput gain or complete-network qualification is claimed.
