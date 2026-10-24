# Experimental parallel hyper-mix

`strix-hip-decode-token --parallel-hyper-mix` selects an optional implementation
that separates per-stream normalization, low-rank projection, injection and
output mixing. Independent projection outputs run across several workgroups
instead of sharing one block for the entire row. The default implementation
remains available for comparisons.

The candidate preserves ascending per-output dot-product traversal and the
256-thread per-stream RMS reduction. One thread computes the inverse RMS and
broadcasts it through shared memory. Normalized values and low-rank outputs are
FP32. Changing kernel boundaries can still change compiler contraction and
rounding; fixtures do not establish full-model bitwise equivalence.

Execution IO has a generation-owned `hyper_workspace` temporary at each
hyper-mix step. Its admitted size is
`rows * (hidden_size * hc_count + hc_lowrank) * sizeof(float)`.
The arena accounts its lifetime and peak, and the launcher rejects a missing,
wrongly typed, undersized or unaligned workspace. No per-step allocation or
host synchronization is introduced by the candidate. Kernel and implementation
identities distinguish it from the default path.

Native FP32/BF16 fixtures cover attention, MLP and final mixing, original packed
weight loading, independent mathematical results and exact paired fixture
outputs/injection weights. The full integration suite passed on gfx1151 before
the additional BF16 cases, which also passed natively. Real-checkpoint operator
replay, numerical fidelity, graph replay, multiple rows and workload performance
remain qualification gates. The candidate remains experimental and opt-in.
