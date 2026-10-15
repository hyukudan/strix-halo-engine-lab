# Experimental ranked router selection

`--ranked-moe-router-topk` requires `--parallel-moe-router` and defaults to OFF.
It replaces only the final selection bookkeeping of the parallel router.
Projection, the 256-thread maximum and denominator trees, exponential evaluation
and probability division retain their existing operation order. Existing default
and parallel-only functions remain available without this option.

Each non-NaN probability receives an integer rank by counting values that precede
it under the existing descending-value, ascending-expert-index comparator. Equal
signed zeros tie by index, a CPU-only fixture state. NaNs never select. Each selected rank has exactly one
writer; unused slots retain the original negative-infinity/expert-count sentinel.
Placeholder initialization precedes the existing probability producer barrier.
One additional uniform barrier completes rank writes. Lane zero then performs
the original FP32 sum and division in slot order. When no expert selects,
normalizing the negative-infinity sentinel sum produces the original NaN weights.

The selection kernel adds 128 bytes of shared slot storage, for 3200 bytes total.
The existing 9472-byte conservative router maximum remains sufficient. No new
global workspace, precision conversion, reduction tree or fast-math flag is added.
Kernel IDs end in `.ranked.parallel`; compiler and launcher identities distinguish
the candidate. Catalog, serial, speculative and bundle admission check the exact
mode. Workspace admission remains the same as parallel-only selection.

Independent host tests compare a literal legacy insertion oracle with ranks,
including ownership boundaries, all K from 1 through min(16,E), signed zeros,
partial/all NaNs, infinities, ties and missing slots. They also reject rank gaps
or duplicate writes. CPU NaN outputs are compared by class; other weights by bits.
Device fixtures pair complete selected indices and weight bits against actual
default-kernel outputs, also verified by the parallel-only path, including NaN
payloads, with outputs and logits workspace poisoned before
each eager execution and graph replay. Reachable device classes include all-NaN,
ties, exponential underflow to zero, and subnormal-or-flushed probabilities.
Spread fixtures use the actual default device output as the oracle, since the
normal-chain FMA probe does not establish the device exponential's FTZ policy.
Captured graphs must contain two kernel nodes, with a common root projection
function and a distinct dependent selection function; no production test accessor
is introduced. Device tests are supplied for qualification, not reported as run.

Integer ranking has quadratic worst-case comparison work. This experimental
source does not establish a speedup. HIP compilation, device fidelity, real-model
router shadows and complete generation fidelity must precede alternating warmed
measurements on both devices. Resource declarations are bounds, not occupancy or
compiled scratch measurements.
