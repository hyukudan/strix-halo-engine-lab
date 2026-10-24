# Experimental Q8 hyper-connection output producer

Status: source and compiler probe only. This opt-in variant has no candidate GPU,
model-fidelity or timing receipt. No speed or occupancy result is claimed.

`--prefetch-q8-hyper-output` requires the typed, fused, staged and parallel
hyper-connection options. It can be selected with or without the independent
row-prefetch projection option. It requires every selected up projection to be
Q8_0, with a positive lowrank width divisible by32. Whole-model admission checks
every compiled mixer before the hyper bundle's first launcher registration or
graph capture. Earlier unrelated catalog bundles may already be registered;
failed hyper admission does not claim transactionality across the whole catalog.
The per-launch type check rejects an unsupported up projection before any
normalization, projection or output work is enqueued. It never falls back to a
generic kernel under the specialized identity.

The new output kernel has a unique ID suffix `.prefetch-q8-output-v1`, conditional
implementation tag and compiler identity
`hip-gfx1151-hyper-mix-v9-q8-output-prefetch8` (with a distinct
`-with-projection-prefetch` compiler identity when down rows are prefetched).
Existing flags remain false by
default and all previous device definitions are unchanged in source. The existing
LDS/resource declaration, scratch ownership,128-thread geometry,8 output chains
and129-float coefficient pitch are retained.

For each output group, a block-uniform choice selects a complete-row or partial
row producer before the full stream/tile loops. Each valid producer lane loads
one FP32 low value and eight Q8 coefficients into local registers through the
unchanged typed decoder helper, then stores to shared memory. Partial rows repeat
the last valid source and destination row with identical same-thread values;
invalid logical rows are never addressed. The grid ensures at least one valid
row. The existing lowrank-tail predicate guards the low and coefficient loads.

Each active output lane still accumulates one gate in ascending lowrank order,
uses the original sigmoid, accumulates normalized streams in their original
order, divides by the same stream count and uses the original activation store.
Both barriers remain outside active-lane predicates. There is no reassociated
reduction, integer activation quantization, new launch bound, cache transform or
floating-point compiler option.

Source order does not guarantee compiler load grouping. Before any GPU execution,
actual target code must prove overlapping low/coefficient loads before dependent
waits and LDS stores on both full and partial producer paths, preserved Q8
materialization and ascending consumer/epilogue operations, uniform barriers,
matching descriptor modes, no private memory or spills, and unchanged previous
function instructions and constant-pool targets. Register declarations and grid
counts alone establish no occupancy or speed conclusion.

Numerical qualification must include finite varied-code and extreme controls,
unsaturated output mutations, poisoned eager/captured replays, byte-preserving
wrong-format rejection, and full-model array/token comparison before matched
performance measurements. The typed-down constraint requires expanded input
width divisible by32; with at most8 streams, legal partial output groups have
4 valid rows. Other partial-group counts are not claimed as admitted fixture
shapes. Non32 lowrank and F32 up projections remain supported by prior paths;
this opt-in variant must reject them without mutating its destinations.

The supplemental `strix-hip-hyper-output-prefetch-tests` target runs the
`output-prefetch` fixture mode. Its 48 cases cover F32/BF16 storage, attention,
MLP and final mixers, hidden widths with complete and admitted partial-four
blocks, and lowrank widths 32, 96, 128, 160, 288, 320 and 512. Together these
cover all four tile-tail classes and multiple tiles. One shape retains the
checkpoint dimensions 2560 hidden features, four streams and lowrank 320.
Signed finite Q8 scales include zeros, subnormals and maximum magnitude; codes
include -128 and 127. The independent CPU oracle and unchanged generic/typed
kernels provide separate reference controls. These fixtures use one logical
row; they do not establish multi-row or nonfinite-payload qualification.

Each candidate begins from poisoned complete scratch and destinations, checks
full normalized/low/output/injection state, and replays captured work three
times. Separate last-K changes in first/full/last up rows must change reference
output bits. Wrong F32-up admission must report the exact format error and
preserve every destination byte. Model preflight tests positive registration
and malformed final up bindings before any specialized launcher is registered.
Device fixture execution, checkpoint equality and timing remain pending for
this supplemental source snapshot.
