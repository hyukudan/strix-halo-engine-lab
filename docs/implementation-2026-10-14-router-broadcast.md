# Router reduction broadcast ordering

The router computes the softmax maximum using a shared reduction array. Every
wave must retain that maximum before lane zero reuses the same shared location
for its contribution to the denominator sum. A workgroup barrier now orders
these reads before reuse. The dot products, exponentiation, reduction trees,
selected-weight normalization and deterministic tie order are unchanged.

The launcher and compiler identities advance to distinguish this correction.
The fixtures exercise F32 and Q8 weights with FP32 and BF16 activations at
expert counts around wave and workgroup boundaries, including partially
occupied waves and a second expert iteration.

The GDN resource declaration is also corrected to include the immutable delta
broadcast and graph control beside its reduction workspace. It declares a
conservative, 256-byte-aligned shared-memory bound; this is admission metadata,
not a measurement of compiler allocation or achieved occupancy.

These checks preserve error propagation and generation-owned memory. They do
not establish full-checkpoint numerical fidelity or workload performance.
