# Experimental tiled integer LM-head consumer

`RuntimePlan` and `HipIntegerProjectionRuntimeBundle` connect the shared integer
projection implementation to the existing compiled eager dispatch registry. This
is an explicitly selected alternate decode target; it does not authorize another
phase, authoritative speculative verification, graph capture, a default numerical
mode or native/model qualification. The default floating LM-head bundle rejects
another kernel identity instead of attaching its arithmetic to an integer profile.

The plan checks the actual compiled LM-head step and its hidden/logits IO views,
model content/layout and tensor-access identities, original local GGUF output
tensor, exact rank/dimensions/type/bytes, runtime allocation generations and owning
source/arena allocations. Source storage is exact BF16 or FP32; logical IO sizes
must agree with that selection. Generic repacked caches, unknown/nonlinear raw IU4
codes, unspecified approvals, incompatible K storage blocks, aliased model/arena
pools and insufficient full-allocation admission are rejected. Q4_0 has 32-element
storage blocks and supports K640; IQ4_XS requires exact 256-element blocks and
rejects K640. Neither model codes nor checkpoint scales are requantized.

The blocking HIP path widens the source exactly to binary64, losslessly repacks
original checkpoint columns into 32-element groups, and invokes the shared Wave32
IU8/IU4 preparation/projection launcher. IQ4_XS subgroup scale bits and signed
scale offsets remain separate from its nonlinear codebook lookup. Each group's
checked integer sum is scaled in the shared binary64 canonical order. A separate
integer IEEE conversion defines round-to-nearest/ties-even binary32 results,
gradual subnormals and signed zero without relying on a host rounding mode or a
device float32 flush mode. Nonfinite source/scales/arithmetic and nonfinite binary32
output fail. Source BF16/FP32 storage, row-local A8/A4 quantization, integer operands,
group binary64 arithmetic and final binary32 conversion have one exact approval
identity; merely requesting “INT4” or “FP32” cannot select it.

Columns are tiled, beginning with 1024 and capped at 4096 per tile. For N153600,
K2560, the default plan has 150 tiles, reads 208896000 original IQ4_XS bytes
(221184000 Q4_0 bytes), and avoids allocating a full binary64 scale/output/group
expansion. The final short tile retains its actual column count. Activation
preparation is repeated once per tile in this implementation and is counted;
no cross-tile preparation reuse or speed benefit is claimed. All tiles convert
into a full FP32 logits scratch buffer. Only after every tile and conversion
passes does one final copy publish the complete logits vector. A late tile error
preserves all prior destination logits. An error during the final copy poisons
the destination and the enclosing eager/session consumer receives an exception.

There are twelve internal allocations: the ten shared projection roles, full
logits staging and conversion status. Their actual HIP-reported capacities must
fit the explicit internal ceiling, which retains the core's 256 MiB maximum.
The separate full retained ceiling includes each entire external model/arena
allocation once, including unused tails and other weights within a shared model
pool. `RuntimeAccounting` is a checked requested-byte plan; native outcomes report
actual internal and retained capacities. Runtime/driver internal allocations,
other session/model pools and device context overhead remain separate admission
responsibilities rather than being silently included in these counts.

Strong allocation pins hold actual allocator/model leases. The provided native
pin factory discovers device-local allocation bases/extents, so resident-copy
model bindings can be connected without guessing tensor-sized ownership.
Registered-file-mapped sources additionally need a caller-owned pin describing
the complete registered mapping; the factory deliberately does not infer that
extent from a tensor pointer. Runtime tensor bytes/content are trusted outputs of
the loader, not authenticated by a new per-token model hash. One scheduler must
exclude conflicting writes and advance source/arena access epochs before reuse.
The consumer snapshots and rereads those epochs throughout execution. Failed
work retains source, destination, scratch and stream owners until actual stream
quiescence; unproved cleanup quarantines the bounded owner through process teardown.

An explicit legacy stream works with the existing unscoped eager path. A nondefault
owned stream requires the matching eager `HipLaunchScope`; readiness is established
by actual producer ordering and the shared recorded event. Both a graph scope and
actual active HIP capture are rejected because this implementation synchronizes
and reads status. No supplied boolean declares inputs or output ready.

Portable tests cover the full target tiling/accounting, exact storage eligibility,
model/binding/epoch tampering, owner exceptions, byte admission and an independent
IEEE conversion comparison across exponent boundaries and subnormals. The native
fixture calls the registered consumer through `DirectLaunchDispatcher`, compares
logit bits against the existing independent scalar projection oracle, and covers
negative scales, partial columns, K640/K2560, BF16/FP32, IU8/IU4, explicit streams,
capture/default-profile rejection and failure before complete-logits publication.
The HIP code and fixtures can be compiled and disassembled offline. They have not
run on gfx1151 hardware. Fragment ABI correctness, complete projection cost,
quality/tie/state effects and full-model workloads still require target execution.
