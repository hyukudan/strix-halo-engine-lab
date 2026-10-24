# Opt-in exact grouped Q5_1 down runtime

`--grouped-exact-q51-down` requires `--q4k-a8g32-gate-up` and a build with
`STRIX_ENABLE_EXPERIMENTAL_Q4K_A8_RUNTIME=ON`. Both flags default to false.
This mode groups the selected experts' Q5_1 down projections only when the
layer uses the existing Q4_K A8G32 gate/up path. Q8 down projections and the
Q5_K gate/up fallback keep their existing launch path.

The grouped kernel consumes the original FP32 post-SiLU slot workspace and
original packed Q5_1 weights. It keeps 256 virtual leaves, ascending feature
updates within each leaf, the original binary reduction tree, and ascending
expert-slot fused accumulation from positive zero. It introduces no additional
activation quantization or weight conversion. The Q4_K gate/up numerical policy
is still approximate; grouping does not claim equivalence to the earlier
unquantized gate/up model.

One 256-thread workgroup computes eight output rows, using one complete
32-lane wave per row. The existing typed workspace is reused without extra
scratch. Shape, alignment, owner aliasing, routes and route-weight admission,
all-slot activation finite checks, final finite/BF16 overflow checks, graph
aggregate status, and semantic output commit retain the existing contracts.
All host admission completes before any new GPU work is submitted.

Execution, implementation and compiler identities change only when the option
is enabled. The compact RAW identity is `q51g-v1:v256:slot-fma:+0`; the runtime
JSON also reports the complete numerical identity, eligible layers and host
submission count. Counts describe host enqueue/capture calls, not graph replay
kernel executions. The option and its dependency must match across catalog,
binding request, buffer admission and selected kernel suffix.

## Qualification status

The standalone kernel has passed its native device oracle, eager/captured
failure guards and instruction/resource audit. The integrated runtime has
strict host contract coverage. Integrated native fixtures, complete-model
FP32-array equality against the same Q4_K A8-only trajectory, and paired
performance measurements remain required before claiming runtime qualification
or speed improvement. Original pre-quantization model hashes are not the
numerical reference for this grouping change.

Native tests exercise FP32 and BF16 output, distinct and duplicate expert
routes, the Q5_K fallback, FP32 accumulation equality before BF16 rounding,
semantic output equality, graph geometry, and inherited invalid-input/status
publication guards. The standalone tests retain independent packed-weight and
256-leaf CPU oracles, signed-zero/subnormal cases and reassociation mutants.
