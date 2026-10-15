# Standalone Q4_K gate/up A8G32 probe

This isolated, opt-in experiment quantizes one activation row. It changes numerical behavior and is not an exact replacement for the default expert projection. It is not connected to model execution, admission catalogs, sessions, or workspace publication. Existing runtime source and default kernel identities remain unchanged.

## Contract and arithmetic

`moe_q4k_a8_contract.hpp` names the numerical policy. Admission requires explicit enablement and exact identity approval; one logical row; hidden width256..2560 in multiples256; output rows32..640 in multiples32;1..512 experts; top-k1..10; separate Q4_K type12 gate/up; BF16 or FP32 activation input. All nine caller-owned views require exact byte lengths,16-byte alignment and checked address ends. Writable views are disjoint from every other view; gate and up are disjoint. The API allocates nothing and retains no pointer.

Per32 activations, delta is max(abs(x))/127 rounded to binary32 nearest-even, or1 for an all-zero group. Each code is nearest-even integer of the **binary32-rounded** quotient x/delta, then clamped to[-127,127]. Nonfinite input and underflow to zero delta reject the entire quantization before any external code/delta/sum or output is written. BF16 input expands exactly to FP32. Gradual FP32 subnormals are required, not silently ignored. A maximum of64 minimum FP32 subnormals maps to64, not127. A maximum of63 underflows its delta and is rejected.

The kernel calls the declared `__ocml_div_rte_f32` directly: Core10.1's `__fdiv_rn` uses plain division unless a separate macro is enabled. Actual linked device code, FP modes and device division oracles must qualify this call; a source declaration or unresolved assembly call is insufficient.

For each Q4_K group, R=sum(q*a) and S=sum(a) are integer-exact. The combine is separate FP32 u=d*(s*R), v=dmin*(m*S), t=u-v, contribution=delta*t. Two consecutive groups are added, then accumulated in ascending256-element block order into four leaves. Only lane c0 stores ((leaf0+leaf1)+(leaf2+leaf3)). No FP contraction, weight conversion, activation fallback or precision mode is implicit. Eight native signed-activation/unsigned-weight dot4 operations process each lane's two16-element halves. There is no emulation fallback.

Projection equality is qualified only when the weight headers and **every** combine/accumulation intermediate are finite. `evaluate_row` executes this predicate and rejects unqualified cases. The GPU propagates exceptional arithmetic; fixtures record unqualified output bits and verify bounds/status, without asserting universal NaN payload or signed-zero commutativity. Half conversion covers all65536 encodings: finite/Inf/signed-zero bits are exact-oracle gates; NaNs have classification-only scope.

## Status and lifetimes

The caller supplies a standalone sticky status word. Route bit1, nonfinite activation bit2 and invalid delta bit4 are independent of existing graph/session statuses. The API never clears status. Route validation precedes quantization; one256-thread quantizer stages the whole row in LDS and publishes external workspace only after a CTA-wide error check. Projection follows on the same stream. Status and all other bindings must remain alive until the original device stream is known to have quiesced, including after a launch error. Enqueue success alone proves no completion.

Tests use `HipSetupQuiescenceGuard` to drain the original observed device before owner cleanup. This experiment does not implement session poison/publication contracts; integration must add them explicitly. It must never silently replace an exact path after a failed probe.

## Build and gates

The independent host contract/oracle test builds under ordinary BUILD_TESTING. `--exhaustive` additionally checks every24-bit packed factor byte triple at all four offsets, both group branches. The independent quantizer oracle implements binary32 division using integer significands and RN ties-even before integer rounding. Executed negative controls distinguish direct-rational integer rounding, activation threshold changes, packed factors, dmin removal, FMA contraction, reversed block order and the wrong final tree.

For a real HIP compiler, configure with `STRIX_ENABLE_HIP_RUNTIME=ON`, `STRIX_BUILD_EXPERIMENTAL_Q4K_A8_PROBE=ON`, gfx1151, strict warnings. First build targets `strix_q4k_a8_dot4_compile_probe`, `strix_q4k_a8_hip_probe`, and `strix-hip-q4k-a8-probe-tests`, but **do not execute** until linked gfx1151 object and ISA are reviewed. These are independent targets; no new TU or compiler option enters strix_hip_runtime.

Required linked-ISA gates: native mixed-sign dot4 and correct literal signs/no clamp; actual rounded OCML division at both quantizer sites; FP32/FP16 denormal and rounding modes; separate FP32 combine operations, writer tree/order; bounded LDS/private/register resources and spills; no unsafe reassociation. The fact that the tiny root compile probe emits a native instruction is only the first gate.

After that review, the standalone device fixture exercises signed-dot integer oracles, randomized and adversarial exact-division oracles, both input storage types, H256/512/2560, canonical640 rows/top-k10, zero/subnormal/FLT_MAX/NaN cases, full workspace/output guards, route/status/input failure publication, and three static graph replays with stream-ordered poison before each replay. Source tests do not claim actual GPU results before those receipts exist.

Full-model integration is a separate future change. Preserve default behavior independently, report this candidate's complete-logit divergence and quality on fixed held-out tasks, and require acceptable operator error/quality gates before performance comparisons. The old554-array exact reference is not this candidate's success criterion. No speed improvement is claimed by this standalone probe.
