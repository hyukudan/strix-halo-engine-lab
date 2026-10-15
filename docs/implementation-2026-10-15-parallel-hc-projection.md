# Parallel HC projection diagnostic

Status: diagnostic prototype. Source revision `db3740f` compiled successfully and
passed 227 native tests. A single captured-input screening experiment completed;
complete-workload qualification remains pending. No production launcher or decoder
option selects these kernels. The existing ordered projection is unchanged.

The diagnostic consumes already normalized FP32 activations, original Q8_0 down
weights and optional FP32 injection weights. It supports width 10240, 2–8 streams,
1–512 low-rank outputs and 1–65535 rows. All input bases require 16-byte alignment.
Outputs require 4-byte alignment and cannot overlap inputs, each other or scratch.
Admission completes before either kernel is enqueued.

The primary mapping has sixteen 32-thread waves per block and four outputs per
block. Each wave owns 640 features; each lane accumulates twenty products in a
fixed order. Input registers are reused across the output rows. Five XOR-shuffle
additions reduce each wave, followed by one block barrier and an ascending sum
of the sixteen wave partials. The final division and FP32 SiLU/injection sigmoid
expressions retain the existing producer epilogue.

`hc-down-splitk16x640-r40-fp32-v1` is a distinct numerical identity: summation is
reassociated and is not bitwise equivalent to the ordered projection. A two-output
mapping is the only alternative geometry. A two-kernel global-partial comparator
uses the same arithmetic and must produce identical finite FP32 outputs. A double
accumulation arm is diagnostic evidence, not a production precision mode.

The two-kernel comparator requires 16-byte-aligned caller-owned scratch of
`rows * 16 * (rank + (injection_present ? streams : 0)) * sizeof(float)` bytes.
The single-kernel variants require an empty scratch span. All owners must remain
valid through stream completion; the API performs no allocation or synchronization.

The independent CPU oracle decodes the original payload, materializes coefficients
in FP32 and accumulates products in FP64. Candidate raw dots use a predetermined
gamma-40 sum-of-absolute-products bound plus FP64 and underflow terms. Postactivation
bounds propagate that error through the division and SiLU/sigmoid derivatives;
the double-accumulation arm separately checks the epilogue error allowance.
Exact-reference bit differences are reported rather than treated as a correctness
criterion for this numerical identity. NaN payloads and signs are unspecified;
nonfinite classification is checked separately from finite error metrics.
In particular, infinite injection dots can produce finite 0 or 2 sigmoid outputs.

Native fixtures must check rejected-owner atomicity, output and scratch canaries,
read-only inputs, tails, cancellation, nonfinite values, repeated determinism and
eager/graph agreement. One captured-input microbenchmark is only a screening
experiment. Production integration requires broader layer/position coverage,
fixed teacher-forced and held-out quality gates, and complete-workload timing.
No model quality, speedup or default-selection claim follows from this prototype.

The capture qualifier accepts a fixed 165-case inventory: eight selected layers
and the final projection at positions 0, 42, 43, 52 and 61, together with all 97
projections at position 61. Every captured ordered low/injection output must be
reproduced bitwise before the numerical candidate is qualified. The final projection
has no injection weights. Its complete 97-call graph reads 353,443,840 logical
weight bytes, with one actual binding and captured input per call.

Timing uses eager hot, graph hot and the complete 97-call graph, with five warmup
ABBA blocks and 100 measured ABBA blocks (200 matched pairs). The four-output
single-kernel mapping has priority if its paired graph speedup reaches 2 and no
mode falls below 1.5; otherwise the two permitted alternatives are considered.
If no variant reaches 2, the best of all three may qualify as marginal when its
graph ratio and every other mode reach 1.5. This gate permits subsequent
experimental model integration; it does not establish model quality or adoption.
