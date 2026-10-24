# Experimental HC projection prefetch

`--prefetch-hc-projection-next-tile` anticipates the next 128-feature tile's raw
input and weights while consuming the current tile. It uses the original Q8_0
down weights and F32 injection weights. Each output retains its ascending FMA
chain, division, activation and output format. The existing 128-thread block,
eight output rows and padded shared-memory layout remain in use.

The option requires the parallel, staged, fused, typed, row-prefetch HC path,
BF16 activations, hidden size 2560, four streams, rank 320 and 48 layers. It is
limited to eager execution with one row and one request; graph and split-K
composition are rejected. It preserves automatic base-profile resolution and
stays disabled by default. An active receipt declares it as `not_assessed`.

Setup validates every one of the 97 mixers, including the final mixer without
injection. File admission precedes residency. Device-view admission precedes
persistent-state initialization and checks the selected tensors, actual scratch
slices, retained owners, generations, alignment, extents and writable aliases.
Launchers repeat constant-time checks before normalization or projection can
write anything. There is no repacked weight owner.

The marked kernel and implementation identities are distinct. The original
kernel bodies and selection remain unchanged when the option is disabled.
Source fixtures cover full words, state, replay, partial tiles, partial output
groups, NaN operand roles and rejection without enqueuing work. Compiler, native,
complete-model fidelity and paired performance checks for this implementation
are pending. No speed or default-adoption claim follows from this source change.
