# Parallel HC projection diagnostic

Status: source prototype; native compilation, numerical execution and complete
workload qualification are pending. No production launcher or decoder option
selects these kernels. The existing ordered projection is unchanged.

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

The two-kernel comparator requires caller-owned scratch of
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
