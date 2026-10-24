# Experimental virtual-leaf Q8 LM head

This opt-in candidate changes only the Q8_0 LM-head projection. Enable it with
`--virtual-q8-lm-head`. The existing projection remains the default. The option
rejects other weight formats and cannot be combined with an integer LM head.

Each output row uses one 32-thread workgroup. Each thread retains eight separate
FP32 accumulators representing virtual leaves `8 * lane + local`. Each leaf
visits its original features in ascending steps of 256, using the existing Q8
decoder and multiply-add source expression. Five cross-lane stages reproduce
the original strides 128, 64, 32, 16 and 8, followed by register stages 4, 2 and 1
in lane zero. Reducing registers before the cross-lane stages would change the
tree and is prohibited.

The candidate adds no dynamic allocation or admitted scratch. It reads the same
hidden and weight bindings and writes the same FP32 logits. Registry metadata
declares 32 workgroup threads and zero LDS, with distinct kernel, compiler and
launcher implementation identities. Default identities are preserved.

Implemented device fixtures cover FP32 and BF16 input, varied Q8 values and
block scales, widths 32, 96, 256, 288, 640 and 2560, poisoned input tails and a
NaN-scale block outside the exact weight binding. They compare full output bits
against the original device path and an independent fused CPU oracle. Capture
introspection distinguishes the 32-thread candidate from the 256-thread default
native function. Rejection checks poison the destination first. Static capture
does not establish complete model graph qualification.

A non-HIP CPU target executes the independent 256-leaf fused oracle and eight
mutation controls. These detect premature register reduction, reversed stages,
wrong mapping under the existing tree, descending leaf updates, negative-zero
or first-product initialization and unfused updates. Signed-zero and finite Q8
decode extremes have device fixtures. Subnormal inputs retain strict candidate
versus default comparison while reporting any CPU-oracle difference explicitly.

The candidate is experimental. Strict host checks, including the new CPU oracle
target, pass. Those host checks do not compile the changed HIP/application paths.
The Core10.1 gfx1151 device-enabled build passes all176 tests on one device,
including the projection, capture, rejection and CPU-oracle fixtures.
Candidate instruction inspection, second-device fixtures and complete-workload
fidelity and timing are required before promotion. No measured speedup is asserted.
