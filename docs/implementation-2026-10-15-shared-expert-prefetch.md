# Opt-in Q8 shared-expert projection prefetch

`--prefetch-q8-shared-expert` is an experimental, default-off loading schedule for
eligible Q8_0 gate, up and down matrices. Every original device function remains
literal in the same HIP translation unit. The scalar sigmoid gate and the SiLU
activation kernel are unchanged. Gate/up results retain raw FP32 precision; down
retains the original FP32 intermediate input and multiplies the final sum by the
same scalar gate before the original BF16 or FP32 store.

Each physical wave lane owns eight original T256 leaves. Each leaf accumulates
features in its original order, with `activation * (half_scale * signed_code)`.
Five shuffle stages reproduce strides 128 through 8, followed by the original
4/2/1 operand order in lane zero. Eight independent waves process eight rows per
256-thread workgroup. A partial final workgroup returns only complete inactive
waves; no LDS or block barrier is introduced. Compiler contraction and FP modes
are inherited from the existing translation unit and need linked ISA inspection.

A manifest plan selects each type-8 matrix with width divisible by 32, validating
complete matrix families, geometry, checked byte sizes and 16-byte weight offsets.
Misaligned eligible weights are rejected. Ineligible types or widths retain the
original projection. Layers with any eligible matrix bind a distinct
`.prefetch8.virtual256.q8-shexp` operator ID; wholly ineligible layers retain the
base ID and launcher identity. Catalog configuration and binding requests must
carry the same ordered eligible-layer list. This increment rejects speculative
verify because the actual launcher registers only the one-row C1 path.

Setup checks the plan against compiled tensor-access slots and admits actual
eligible resident addresses and extents before token execution. New-mode operator
preflight checks all mutable/read-only owners, overflow, alignment and aliases
before any queue submission. Existing scratch is reused. Projection receipts use
`projection_kernel` as a schedule label, with `tensor_name`, per-layer and
per-projection eligibility/fallback counts. OFF emits only the false option key.
The `prefetch_q8_shared_expert_dispatched_projections` atomic counter records actual
host eager submissions and graph-capture submissions; device graph replay does
not increment it. Expected counts must come from the actual manifest.

The host fixture covers 48 layers and 144 matrices, mixed layer/tensor fallbacks,
scalar-gate exclusion, access-slot admission, distinct binding identities, OFF
identity stability and exact-message rejections. A mixed catalog fixture requires
47 candidate layers and one base layer, both compiler identities, two launcher IDs,
base identity stability and configuration mismatch rejections.

Standalone device fixtures compare legacy, virtual control and candidate kernels
against an independent fixed256 FMA/store oracle. They cover BF16 and FP32 input
and output, gate/up and down roles, canonical 2560x640 and 640x2560 matrices,
non256 widths, partial row groups, code/scale extremes, signed zeros, subnormals,
a decode-association witness, semantic output poisoning and surrounding guards.
Three graph replays vary inputs. Invalid exact views must preserve destinations.
Full-operator fixtures compare all output and FP32 intermediate bytes across
OFF/ON eager and captured replay for both activation types. They test all-Q8 and
mixed Q8/F32 matrices, retain all five original launches, inspect candidate row
group dimensions and assert real dispatch counts. Alias and misaligned-weight
failures must capture zero nodes and preserve the complete arena and counter.

Status: source and fixtures implemented. The limited host suite passes in an
external Windows portability harness excluding the baseline POSIX stream source.
This does not qualify HIP compilation or execution. Linked original-function ISA,
new kernel arithmetic/FP descriptors, native adversarial fixtures, full-array/token
fidelity and paired complete-workload measurements remain mandatory. No measured
speedup or production qualification is claimed.


The catalog retains and forwards the same statistics owner into its native
launcher bundle. Its mixed catalog fixture checks this owner directly, in addition
to checking that fallback launchers cannot select prefetch. Tensor-name receipts
escape quotes, backslashes and JSON control bytes, with host fixtures for embedded
nulls and other controls. A nonfinite diagnostic flavor supplies distinct NaN
payloads, infinities and nonfinite half scales; it compares the three actual device
modes directly and deliberately does not use host FMA as an AMD payload oracle.
Native execution of these additional cases remains pending.

Static mode and projection-plan fields may be emitted before token execution.
The dynamic dispatch receipt retains the same statistics owner as the launcher
and reads its live host enqueue/capture count only at completed receipt emission.
Teacher-forced output emits that count after the token array; greedy generation
and prefix-restoration output emit it after their workloads complete. Graph replay
does not manufacture host dispatch counts from committed tokens. Portable tests
check zero before owner updates, later live values, owner lifetime and isolation,
and disabled-field omission; they do not claim GPU dispatch execution.
