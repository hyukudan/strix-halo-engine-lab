# Experimental exact lane-chain HC output

Current status: this mode is included in the adopted canonical automatic v3
profile at source `265fd78762884d18eb32aa60294408f080088d52`. Its combined
native, complete554-array, generated512 and descriptive workload gates passed;
see the [qualified adoption](../../engine/decode/optimization-profile.md#qualified-adoption-2026-10-15)
for scope and limits. The source-increment and fixture observations below retain
their historical scope; their earlier pending gates are closed only for this
qualified content/context/execution slice. Low-level manual selection stays explicit.

The low-level `--lane-chain-hyper-output` flag remains OFF in
`--optimization-profile manual` unless explicitly requested, and requires `--prefetch-q8-hyper-output` and its typed fused staged parallel prerequisites. It replaces only eligible HC output steps. Original normalization, low projection and injection remain unchanged. The content-bound [automatic decoder policy](../../engine/decode/optimization-profile.md) now
selects it with no low-level flags for the admitted eager/register checkpoint
and context as part of the complete v3 bundle described in that policy.
Automatic workload qualification is recorded in the adoption note above. Other checkpoints and
incompatible modes retain their explicit selection.
Supplying low-level optimization flags also preserves that explicit configuration
unless `--optimization-profile auto` is requested.
This is an exact arithmetic optimization candidate; any complete-logit difference is a correctness failure, not an approximation allowance.

The output kernel remains in the original HC translation unit with its compiler settings. A wave owns one stream and each lane owns one feature. For each gate it decodes the original signed Q8 coefficient as a separate FP32 half-scale times signed-code multiplication, then accumulates features in ascending order from positive zero. After the original sigmoid, wave zero combines streams in ascending order and applies the original division and activation storage conversion. Compiler ISA must establish the same fusion, operand ordering and FP32 denormal behavior as the original output kernel before numerical qualification.

Eligibility requires hidden and rank divisible by32, streams2 through8, rank at most512, original local Q8 up shape and exact metadata extent, and `streams*(34*rank+128)` bytes at most65536. For registered owners the tensor file offset must be16-aligned. Copied owners use their actual device allocation alignment. Runtime and bundle preflight reassert an aligned, sufficiently large nonoverflowing up owner before any normalization enqueue. Noneligible steps retain the exact base Q8 output identity. A request with no eligible steps fails. Explicit eligible step lists are checked for duplicates and non-mixer entries. The receipt reports the opt-in boolean and eligible/fallback step counts. Serial decode binding supports the new pair of identities; speculative verification rejects this opt-in.

Each stream stages32 consecutive Q8 rows, totaling `34*rank` bytes, using aligned16-byte vectors. Because rank is divisible by32, both the whole tile and every stream base are16-aligned; no individual34-byte block is assumed16-aligned. At hidden2560/streams4/rank320 the dynamic LDS is44032 bytes and the output workgroup has128 threads. The variant conservatively retains the maximum256-thread requirement of the full mixer, and existing global scratch remains unchanged.

For rank320 an output row spans340 bytes, or85 dwords. Scalar scale/code reads at a fixed block and coefficient map the32 lanes bijectively onto32 four-byte banks. This algebra does not prove general bank freedom: rank32 rows span34 bytes and can collide. Vector staging and actual bank behavior require ISA/profiling evidence. No throughput claim follows from the layout.

Host tests cover checked metadata eligibility, exact LDS bounds, alignment policy, no-eligible rejection, bank-index algebra, and mixed eligible/base binding. Device fixtures are supplied for same-input bit comparisons, distinct output poison and guard bytes, extreme signed Q8 scales/codes and FP32 inputs, eager and three graph replays, complete mixer output/injection preservation, and rejection before scratch/semantic writes. These source fixtures are not evidence of GPU execution.

Qualification requires independent source review, actual HIP/ISA and device fixtures, complete model arrays and generated IDs, and paired workload measurements. The isolated canonical HC-output increment has passed its source/native/fidelity and paired-workload checks. The automatic v3 combination (wave, QSA, HC output, shared-expert prefetch and deferred status) passed its own native/model/workload gates in the slice above; separate gains are not added to establish its performance. Manual selection remains explicit. No model-quality relaxation or approximate HC mode is included.

Historical host qualification checkpoint: Ubuntu24.04 under WSL, GCC13.3, HIP-OFF with warnings as errors: build succeeded and152/152 host tests passed. This does not compile the HIP translation unit. At that historical host-only checkpoint the supplied32 same-input and12 complete-mixer cases had not yet been executed on target; later isolated native qualification is described above.

Copy residency eligibility uses the exact shared coalesced allocation range and the tensor offset within it; registered residency uses the file offset. Runtime admission checks the actual address with the same residue contract. Copy allocation layout is unchanged. Same-input device fixtures include non-finite FP16 scales and signed FP32 NaNs; complete-pipeline fixtures include two independent rows.
