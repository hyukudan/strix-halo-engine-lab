# Experimental two-phase MoE router

`strix-hip-decode-token MODEL.gguf TOKEN_ID --parallel-moe-router` opts into
an independently implemented projection and selection pipeline. The flag
defaults to disabled and duplicate use is rejected. It can be combined with
the separate hyper-mix options. The JSON receipt reports `parallel_moe_router`.
This implementation is experimental; no speed or complete-model equivalence
claim follows from its source or host tests.

The projection grid uses eight experts per 128-thread workgroup, with one grid
dimension for the expert groups and another for logical rows. Each tile covers
256 ascending input features. Loader lanes fetch adjacent coefficients into
registers, then publish them to padded FP32 LDS rows. Eight consumer lanes keep
one accumulator each. Producer/consumer barriers protect every tile, including
inactive experts and feature tails; padding contributes no arithmetic terms.
The accumulator starts at positive zero and uses one explicit fused FP32
multiply-add per feature in ascending order. The baseline gfx1151 router's
compiled dot chains use fused accumulation. Preserving the operation and order
does not replace device comparisons of dequantization, exceptional values or
compiler behavior.

The first kernel writes an FP32 logit for every active `(row, expert)` to admitted
workspace. The second kernel uses one 256-thread workgroup per row and loads
those logits before executing the original router's selection tail. The maximum
and sum reduction trees, maximum broadcast fence, division, deterministic
top-k insertion, tie ordering, selected-sum order and normalization switch
retain the original source expressions. The original default `moe_route_kernel`
source text and two-parameter device symbol remain intact.

Projection LDS contains eight coefficient rows with a 260-float pitch and one
256-float input tile: 9,344 bytes. Its conservative 256-byte admission bound is
9,472 bytes; the selection kernel needs 3,072 bytes. The operator declares the
larger bound and a maximum workgroup size of 256. These are source admission
bounds, not compiler occupancy measurements. There are two kernel launches per
router dispatch, and no implicit allocation, reset counter or cross-workgroup
completion protocol.

`router_workspace` is appended to the buffer-class enumeration. Existing enum
values and aggregate fields keep their positions; new request fields default
to disabled. The workspace admits `rows * experts * 4` bytes, rounded to 256
bytes. Arena admission requires an alignment divisible by 16. Every layer uses
its own IO generation and the corresponding arena allocation generation.
The two kernels share its lifetime within one dispatch and run in stream order.
The launcher rejects missing, extra, wrong-class, stale-use, undersized or
unaligned workspace bindings before launching. Logical row grids above 65,535
are rejected by the opt-in launcher.

The binding request, buffer request and complete HIP catalog must agree on the
opt-in mode. Serial and speculative compilation reject a selected router ID
that disagrees with admitted IO. Kernel IDs gain `.parallel`; launcher identity
and compiler identity (`hip-gfx1151-moe-route-v4-parallel`) distinguish the
candidate. Both compiled execution identities include an opt-in domain tag and
the admitted IO/arena. The absent flag adds no hash input, workspace allocation
or temporary use. Fixed host fixture receipts check exact pre-change default
serial and speculative identities and arena sizes.

There is no fallback after a launch failure. Existing stream, graph lifetime,
session poisoning and downstream rejection rules continue to apply. In
particular, all-NaN router probabilities produce sentinel expert indices and
NaN selected weights when normalization is enabled, or negative infinity
weights when disabled; the experts operator rejects those routes. This router
does not add a new status word or promise physical rollback.

Device fixtures pair candidate outputs against the default on identical inputs
and retain an independent CPU route oracle within its qualified scope. They cover
FP32/BF16 activations, F32/F16/BF16/Q8_0 weights, expert tails, feature tails,
two rows, equal-logit ties, all-NaN logits and both normalization modes. Full
512-expert, 2,560-feature BF16/F32 fixtures use non-trivial inputs and seeded
signed coefficients with mixed exponents, zeros and subnormals. The seeded
fixture must distinguish fused and separately rounded accumulation on the CPU.
Eager execution and each of three static-operator graph replays poison both
destinations and the whole workspace before dispatch; verification requires
every active entry to be overwritten. Alignment padding may remain poisoned.
Negative tests cover workspace admission and use-generation failures.

Launcher-bundle registration preflights the complete router binding and IO
metadata against the bundle mode before publishing any launcher. Both mode
mismatch directions, workspace-slot mismatch, wrong binding IDs and wrong
temporary counts are rejected before token execution. Runtime checks also
require the admitted workspace allocation to belong to the current router step;
another layer's otherwise valid generation cannot substitute for it.

Expanded device fixtures cover all eight activation/format pairs at full
512-by-2,560 geometry with two distinguishable rows. Multi-tile feature tails
include 257, 523 and 2,627 elements, and Q8_0 widths 288 and 2,592. Signed Q8_0
codes include -128, zero and 127; per-block scales include signed zeros and
half subnormals. F16 coefficients explicitly include subnormals. Row-local
NaN, positive/negative infinity and zero-times-infinity coefficients verify
that the projection does not contaminate unrelated experts. Negative cases
require the intended rejection message, including a default launcher supplied
with an extra temporary.

Every active logit is compared with an owned legacy-style device projection
reference using the original 256-lane ownership and ascending expression.
Finite values, infinities and signed zeros compare by bits; NaNs compare by
class. The actual production default router separately supplies the paired
selected indices and weights; the test-only projection reference does not
replace that comparison. A CPU oracle uses ascending `std::fma`; finite
bit checks and CPU route checks conservatively cover chains without subnormal
inputs, coefficients, intermediate products/sums or raw half components.
Exceptional NaNs retain CPU class checks. Extreme-chain entries are counted
explicitly and use the device reference, avoiding an assumed CPU FTZ policy.
Runtime probes print FP32 input/output-underflow and F16 decode observations;
they neither impose IEEE gradual underflow nor change compiler/runtime flags.
Compiled per-kernel denormal descriptors remain part of device qualification.

Qualification still requires strict HIP compilation and fixtures on both
devices, compiled LDS/scratch receipts, default code-object comparison,
all-call real-checkpoint shadows, full logits/token equality with and without
profiling, and paired complete-workload performance measurements. Operator
graph replays do not qualify whole-model graph capture or distributed execution.
The host-only suite passes 144 tests, including admitted workspace accounting,
request mismatch rejection, serial/speculative identity separation and the
fixed default receipts. The actual Core10.1 gfx1151 device-enabled build passes
all174 tests on one device, including the typed projection fixtures and static
operator replay. Second-device tests and complete-workload qualification remain
pending; no measured generation improvement is asserted.
