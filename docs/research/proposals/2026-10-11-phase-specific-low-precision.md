# Phase specific packed weight and integer execution proposal

Status: Draft. Requirement IDs `LPE`. Complements architecture 05, 06, 09 and 15 without promoting any new native kernel, quantization or performance result. No new model file format is required for the first experiments.

## Decision under investigation

Keep supported weights compact and select execution by phase, actual rows per operation, encoding, numerical contract and native measurements. Compare vector execution and matrix execution; do not make integer activation precision or matrix instructions mandatory across the model.

LPE-001: Distinguish checkpoint storage bits, decoded operand representation, activation encoding, accumulator, scale/offset application, output dtype, persistent-state precision and wire encoding. A four-bit file code is not necessarily a signed integer of four bits.

LPE-002: Native `gfx1151` compiler/instruction capability is a gate, not an inference speed claim. Retain exact compiler/runtime/ISA target and disassembly proof. Hardware documentation establishes IU4/IU8 candidates; offline compilation does not establish native correctness, useful throughput or model quality.

## Candidate matrix

| Phase and operation | Conservative control | Integer candidates | Selection evidence |
| --- | --- | --- | --- |
| One-row decode | Packed weight interpretation with local floating conversion | VALU integer dot, IU8 or IU4 only where profitable | Actual weight traffic, exposed conversion cost, complete projection time |
| Dense/shared multi-row verification | Qualified floating low-row kernel | W4A8/IU8 and explicit W4A4/IU4 | Row reuse, padding, prepare/scale/commit costs and authoritative target equivalence |
| Routed experts in verification | Dedicated low-row packed kernel | Histogram-qualified vector or matrix route | Per-expert rows, grouping/scatter, not total request rows |
| Prefill dense/shared projections | Qualified floating GEMM | IU8 and opt-in IU4 | Complete preparation/compute/output path and model fidelity |
| Prefill routed experts | Packed expert route | IU8/IU4 when actual expert rows justify tiles | Per-expert occupancy histogram and complete MoE round |

LPE-003: W4A16 names a storage/activation class, not one arithmetic implementation or a guarantee of row invariance. Preserve the exact floating reduction/conversion contract of the control. W4A8 and W4A4 introduce activation quantization and require separate numerical identities and quality gates.

LPE-004: Prioritize actual bottlenecks. One-row decode may be dominated by weight traffic; additional activation reduction/quantization can lose even when the matrix operation is faster. Maintain a VALU/packed floating control and use native measurements before allocating a production variant.

LPE-005: Use actual MoE expert row histograms as a feasibility gate. A verification window of several tokens does not give every expert that many rows. Dense/shared projections and long-prefill experts can be better initial matrix targets than sparsely populated verify experts.

## Hardware lowering and tile utilization

LPE-006: The IU4/IU8 experiment shall use explicitly verified wave32 builtins and the actual lane/register fragment ABI. Check mirrored inputs, accumulator mapping, signedness and saturation policy against an independent scalar oracle. Wave64 candidates remain separate paths and cannot invoke a wave32 builtin under an incompatible execution mode.

LPE-007: Record physical versus useful tile work. For a tile with 16 output rows, one logical row uses only one row unless qualified grouping supplies more. Measure all packing, padding, masks, LDS/VGPR traffic, spills and preparation. Compare instruction issue throughput and dependent latency separately from complete kernel time.

LPE-008: Do not infer library support from ISA support. Validate a custom intrinsic candidate and any selected library API independently. Architecture labels from another device/backend do not qualify this target.

## Encoding and mathematical fidelity

LPE-009: First allow lossless repacking of existing codes/scales into bounded execution tiles. Any re-quantization, changed codebook/group size, rotation or changed scales produces a distinct model artifact or numerical plan with explicit approval and qualification.

LPE-010: Direct IU4 requires a mathematically compatible integer representation, including signed/unsigned mode and offsets. Nonlinear codebooks cannot be fed as their nibble indices. For the existing IQ4_XS codebook, test a nibble-to-INT8 lookup followed by IU8/dot accumulation as an alternative to floating dequantization. The integer codebook mapping can be exact while activation quantization remains a separate approximation.

LPE-011: Apply scales at their declared K-group boundaries. For affine codes, the scalar oracle shall cover:

```text
W = sw * (qw - zw), A = sa * (qa - za)
integer_group = sum(qw*qa) - zw*sum(qa) - za*sum(qw) + Kgroup*zw*za
output = sum_groups_in_declared_order(sw * sa * integer_group)
```

Do not accumulate groups with different scales into one unscaled sum. A format using a separate minimum/bias needs its exact formula. Reuse group activation sums only for the same input/version, quantization grouping and rounding contract.

LPE-012: Keep storage blocks, quantization groups and matrix K-tiles distinct. The target expert dimensions 2560 and 640 divide 16; 640 does not divide a 256-element storage block. Audit each actual tensor encoding and shape. Select a valid existing encoding or an explicitly qualified artifact; never invent silent tail codes or re-quantize a tensor to make a kernel fit.

LPE-013: Padding shall contribute exact mathematical zero at the operand/accumulator level. A stored nibble zero is not necessarily a dequantized zero; some codebooks have no zero entry. Mask invalid rows/columns/K elements without out-of-range access or nonzero padding contributions.

## Activation definition and sensitive operators

LPE-014: Specify activation scales/granularity, finite-input policy, zero-row behavior, clipping range, tie rounding, reciprocal/divide behavior, FTZ/denormals, signedness and output conversion. Hash the contract. Do not use an implicit generic 'INT4 round' rule.

LPE-015: Row-local quantization must not depend on unrelated batch rows, slot number, speculative rejection or transport epoch. Compare the same row alone, duplicated, reordered and inside a verify window. Group floating scale combination uses a declared canonical order where exact invariance is required.

LPE-016: Quantize an input once and reuse it only for exact shared input content/version and the same quantization contract. Fuse preparation into a producer operation only if its numerical output, ownership and completion semantics remain qualified; include the fused path in identity and timing.

LPE-017: Router scores, normalization, output head, KV and recurrent state remain under their existing contracts by default. Altering projection activations can still change downstream state even when its storage dtype is unchanged. Quality and snapshot/replay tests must include that accumulated effect.

LPE-018: Compare bounded mixed-precision outlier handling and selective operator exemptions before making A4 global. A fixed calibrated outlier set, its content/calibration identity and side-kernel costs belong to the plan and resource budget. Dynamic selection has its own sorting/gather cost and requires deterministic semantics. A rotation must be paired with compatible transformed weights; it is not a lossless repack of arbitrary existing quantized tensors.

## Authoritative speculative verification

LPE-019: A fast verifier cannot silently change the target. Exact speculation requires the same declared target probabilities/decisions and committed state as the corresponding serial target. Floating tolerance alone does not establish exact argmax near ties or an identical sampled distribution.

Allowed experiments:

1. Qualified W4A16 serial/verify kernels with the required row invariance.
2. An explicitly selected W4A8 or W4A4 target with its own serial/verify/state equivalence and model quality qualification.
3. A lower-precision proposal or screening path whose results never replace final authoritative target verification.
4. A separately named approximate hybrid mode, only with explicit semantic/quality policy; it cannot claim exact equivalence to the conservative target.

LPE-020: For GDN and other recurrence, accepted state comes from the authoritative target; rejected candidate state is rolled back or discarded. Proposal/screening state cannot contaminate the target cache. Prefill, decode and verify choices, their input history and compatible state identities are part of the full numerical plan.

LPE-021: Do not assume that a same-size lower-precision screening model is a useful drafter. It may read the same weight bytes and add extra full-model work. Keep screening optional unless complete measured acceptance/verification savings exceed its cost.

## Secondary two-IU4 experiment

LPE-022: For already quantized signed INT8 q, define h=floor(q/16), l=q-16*h, with h in [-8,7] and l in [0,15]. Test `Wq = 16*Wh + Wl` using independently signed high and unsigned low operands. Reconstruction uses checked wide multiplication/addition, not left-shifting a negative signed integer.

LPE-023: Prove all intermediate bounds, including zero-point corrections and group lengths, before narrowing to INT32. Explicitly disable or model saturation; it cannot hide overflow. Two IU4 products do not inherit a twofold arithmetic benefit over one IU8 product. Compare preparation savings across output columns with all additional instructions and scale work.

## Experiment and acceptance gates

| Gate | Required artifact | Promotion condition |
| --- | --- | --- |
| LP0 | Checkpoint tensor encodings/shapes and observed row histograms | Candidate is mathematically representable and relevant |
| LP1 | Scalar code/scale/offset/activation oracle | All sign, edge, rounding, zero and overflow cases pass |
| LP2 | Pinned offline gfx1151 compile/disassembly | Intended builtin/ISA and resource report; no native claim |
| LP3 | Native synthetic correctness and complete projection benchmarks | Independent oracle, actual overhead and useful tile work |
| LP4 | Model numerical quality and state replay | Predetermined quality limits, long-context and task coverage |
| LP5 | Serial/verify/fresh/resumed/solo/shared equivalence | Exactness claim matches declared target and state |
| LP6 | Complete model and distributed workloads | Better committed-token/TTFT objective under budgets and quality |

LPE-024: Projection matrix includes actual N/K tensors and M=1/2/4/8/16 plus qualified prefill sizes; expert distributions include empty, one-row, repeated-hot and dispersed routing. Compare warm/cold cache, register/LDS pressure and transient memory. Report both kernel-only and complete phase costs.

LPE-025: Model checks include teacher-forced logits/KL/perplexity, greedy ties, sampling where supported, code, JSON/tools where qualified, long context and recurrent resume. Define tolerances and quality limits before inspecting outcomes. A perplexity delta is not a percentage of intelligence or task accuracy.

LPE-026: Record draft/verify/accept/replay counts and committed tokens. Runtime selection changes only among qualified variants with compatible semantics and measured applicability. No instruction-throughput or third-party benchmark becomes a model speed guarantee.

LPE-027: Local integer execution does not authorize compressed network activations. Wire quantization, GPU DMA visibility and mixed-backend boundaries have their own contracts. A CUDA implementation or another GPU architecture is qualified independently.

## Initial implementation order

1. Extend manifests and collect operator/expert row relevance.
2. Define independent arithmetic/quantization oracles and integer lookup candidates.
3. Build intrinsic probes and disassembly/resource audit.
4. Compare packed VALU/floating, IU8 and IU4 on representative dense/shared prefill and verify shapes.
5. Add routed expert candidates only when histograms justify them.
6. Evaluate model quality and authoritative serial/verify state; measure complete local workload.
7. Compose with the qualified two-host executor and measure full useful workload.

Bounded probes can proceed alongside serial model development. Default activation still requires all relevant native/model gates. Existing correct paths remain available.
