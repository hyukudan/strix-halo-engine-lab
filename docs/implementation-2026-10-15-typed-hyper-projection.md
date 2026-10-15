# Typed fused hyper-connection projection

Status: opt-in experimental implementation. Host regression checks pass;
native HIP compilation, device qualification, full-model fidelity and timing
remain required before promotion. This document makes no performance claim.

## Selection and admission

`--typed-fused-hyper-projections` requires `--parallel-hyper-mix`,
`--staged-hyper-projections` and `--fused-staged-hyper-mix`. Duplicate options
and incomplete combinations are rejected. The selected variant has a separate
kernel ID, implementation identity and compiler revision
`hip-gfx1151-hyper-mix-v6-typed-projection`. Existing v2–v5 identities remain
unchanged and are pinned by fixture goldens.

Registration requires the complete execution tensor access plan and manifest.
Before the first token or capture, every compiled attention, MLP and final
mixer must bind a Q8_0 down projection (GGML type 8). Non-final mixers must bind
an F32 injection projection (GGML type 0). The launcher repeats these exact
format checks before it enqueues normalization or projection work. An invalid
format throws; there is no per-launch fallback under the typed identity.

The final mixer has no injection blocks. Its null injection weight and output
pointers are never accessed. Up projection, normalization and output kernels
remain generic and retain their original format admission and arithmetic.

## Arithmetic and memory contract

The projection retains 128 threads, 128 feature values per tile, eight output
chains and a 129-element coefficient pitch. Each block takes one uniform role
branch into a complete down or injection loop. Producer and consumer barriers
remain outside lane predicates; every thread reaches both barriers in every
tile of its role.

Q8 coefficients use the original expression: convert the binary16 scale to
FP32, multiply by the signed code converted to FP32, and materialize the
coefficient in shared memory. F32 injection coefficients are direct loads.
Each output starts with positive zero and visits all features in ascending
order through one FP32 accumulation chain. Scaling and the role activation
are unchanged. No scale factoring, partial accumulator reduction, tile-size
change, launch bounds or global floating-point option is introduced.

Shared storage remains 4,640 bytes of projection values and coefficients;
the existing declared rounded bound is 4,864 bytes. The admitted global
workspace layout is unchanged. These are memory bounds, not occupancy or
throughput estimates.

## Qualification gates

The operator fixtures cover widths 288, 320 and 384, lowrank 320,
attention/MLP/final mixers, and BF16/F32 activation storage. Q8 down and up
coefficients vary within and between blocks; down scales include signed zero,
binary16 subnormals and 65,504. F32 injection values include adjacent
representable coefficients. A last-block code mutation must change the
generic control output, including with BF16 storage.

The complete admitted workspace and outputs are poisoned before eager runs
and captured replays. A no-op control must fail comparison. Typed eager and
three graph replays must match the generic result bitwise. Wrong down or
injection formats must preserve all poisoned buffers byte for byte. Model
preflight tests include a wrong format in the final mixer binding.

Before promotion, actual compiler output must show compile-time coefficient
types, unchanged coefficient materialization and ascending accumulator order,
uniform barriers, unchanged default functions and matching denormal modes.
Device fixtures must pass, followed by complete-model logit and token checks
with poisoned shadow destinations in profiled and unprofiled execution.
Timing requires alternating same-binary complete-workload runs after those
gates pass. NaN/Inf scale payload behavior is not established by the finite
operator fixtures and must not be inferred from them.
