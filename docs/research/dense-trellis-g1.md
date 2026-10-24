# Experimental selected dense projections

This C1 eager experiment requires a build with
`STRIX_ENABLE_EXPERIMENTAL_DENSE_TRELLIS=ON` (default OFF), an explicit
`--dense-trellis-pack PATH`, and the decode numerical profile
`hc-splitk-f32+q51-q80-a8-down+q5k-a8-gateup+dense-a8-trellis-g1-fp32-s4-v1`.
The profile and pack must be selected together. HTTP, cohort execution and a
simultaneous dense Q6 overlay are rejected. The pack is pinned to one complete
file digest and the source checkpoint's file and content identities.

The selected assets replace the input gate projection in 36 recurrent layers
and the query projection in 12 attention layers. Each input width is 2560;
output widths are 6144 and 12288 respectively. Other projections retain their
parent profile. The pack contains circular K5 tiles and signed F16 input/output
scales, with source asset digests in its metadata. Packaging does not quantize
or change the source assets. Preserve the model's existing provenance and
license notices when distributing an asset pack.

For a column input vector, the mathematical projection is
`diag(sv) H Q^T H diag(su) x`, where H is the orthonormal block Hadamard
transform of width 128 and Q is the exactly decoded F16 codebook matrix.
The executable recipe uses BF16 inputs, FP32 signed scaling and transforms,
four fixed input partitions, explicit FP32 FMA chains, a fixed lane reduction,
then an ascending sum of the four partitions. Native projection row views
round the final value to BF16 and store it in F32, matching their output ABI.
Changing the row count preserves the per-row arithmetic. This recipe differs
from multiplying an F16 materialization of the original-basis weight matrix.

Only R1/R2/R4/R8 decode projections use the compact assets. Wide prompt blocks
retain the named prefill recipe. The admitted fast-prefill pairing therefore
describes a hybrid numerical profile and has its own identity. Calibration
rows and serial prompt suffixes use the decode recipe. Existing calibration
artifacts from another profile cannot qualify this one.

All immutable weights and per-stream workspaces are reserved before execution;
captured projections do not allocate device memory. The pack owner retains the
registered mappings until the request and its streams have drained. The usage
mask in receipts records host entry coverage, including capture; it is not a
count of replayed projections or measured memory traffic.

The source Q8 mappings remain retained for wide prefill. This overlay therefore
adds residency; its smaller selected payload describes logical decode reads,
not freed RAM. Window projection-launch savings retain the compiler's legacy
nominal basis. The compact adjustment is relative to that basis; the field is
not a measurement of dispatched GPU kernels. Use a trace for dispatch counts.

Status: not adopted. The selected asset mix did not pass its predeclared model
quality gate. Keep the build option OFF. The experimental implementation and
its separate identities are retained for research, without a quality or
performance qualification. Native codec, scalar-organization, row-count and
graph-replay checks cover kernel arithmetic only. Small layer error against
materialized weights does not establish model quality. Prefix qualification
must separately prove
tokens, every physical logit row, committed state, physical state across
matched schedules, and rejection/owner transitions within this numerical
profile. Adoption also requires a deployed-profile quality gate and the full
workload campaign. Leave this experiment disabled until those gates pass.
