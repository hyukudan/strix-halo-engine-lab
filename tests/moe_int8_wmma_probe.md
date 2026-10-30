# Routed INT8 WMMA research probe

This executable is isolated from production dispatch. Its numerical identity is
`int8-wmma-routed-research-v1`, pending compiler, native execution, model fidelity
and complete-workload qualification. A microbenchmark win does not qualify it.

Build and run on a native gfx1151 HIP host:

```sh
cmake -S . -B build-hip -DCMAKE_BUILD_TYPE=Release -DBUILD_TESTING=ON \
  -DSTRIX_ENABLE_HIP_RUNTIME=ON -DSTRIX_BUILD_EXPERIMENTAL_Q4K_A8_PROBE=ON
cmake --build build-hip --target strix-hip-moe-int8-wmma-probe -j
timeout --signal=KILL 400s ./build-hip/strix-hip-moe-int8-wmma-probe \
  MODEL.gguf --layer 0 --seconds 390 --rounds 5
```

The CPU fixture is `strix-moe-int8-wmma-fixture-tests`, also included in ordinary
CPU CTest. The GPU benchmark is deliberately not an automatic CTest: it needs a
model and a bounded native measurement turn. All output goes to stdout/stderr;
the executable writes no artifacts. Keep native receipts outside Git.

Weights are the original `blk.N.ffn_{gate,up,down}_exps.weight` tensors, read by
the repository GGUF loader. Gate/up must be Q4_K, down Q5_1 or Q8_0. The dimensions
must be hidden 2560, intermediate 640, 512 experts. Unsupported shapes/formats
fail instead of substituting synthetic model weights. Only the small exhaustive
packed-format fixtures use synthetic weights, including negative/subnormal/zero
scales, packed scale/min boundaries and Q5 high bits.

Inputs are deterministic BF16, with zeros, tiny groups and outliers. Top-k is ten
distinct experts per row, with deterministic rank^-1.5 sampling without replacement;
T8192 and T16384 have identical prefixes. T256 uses balanced routes with partial
expert buckets. The fixture rejects NaN, infinity and invalid routes before timing.
This is a generated workload, not a captured model activation distribution.

Four candidates compile: M64/M128 assignments crossed with N128/N256 columns.
N is the combined gate/up width; down uses N output columns. Each K64 byte stage
has two independently scaled K32 dots, each implemented by two signed-A iu8
WMMAs. Q4_K and Q5_1 positive codes fit signed bytes; Q8_0 stays signed. Original
G32 quantizers and packed decoders are reused. Integer fragments are converted
immediately, with explicit FP32 FMA scale/min correction into one accumulator
chain. The single XOR-permuted LDS stage uses next-K64 register prefetch.
No exact reduction-order contract is claimed. The fused gate/up epilogue
uses FP32 SwiGLU and publishes A8 down codes, scales and exact sums. Down writes
FP32 raw dots, followed by the existing ordered slot-weight FMA merge semantics.
Route BF16 storage is off.

The incumbent launches the production FP16 DA16/header-reuse kernels with tune
82 (80 for the small shape, following production admission), route policy 14,
and FP32 raw dots. Its complete raw and merged bytes are checked against the
production `enqueue_prefill` entry point before timing. Capture specializations
are untimed; they expose the production gate/up body to the CPU oracle. Measured
incumbent kernels have no capture stores or capture branch.

Production host admission and static pools stop at T8192. The research executable
owns CSR, FP16 images, counters and multi-CTA CSR scratch sized through T16384;
it does not widen production admission. At T16384 the production-entry witness
uses two admitted T8192 calls, which must equal the whole-T direct production
kernel launches byte for byte. The old M16 integer WMMA diagnostic also uses two
admitted calls at T16384 and is labelled accordingly. Candidate input quantization
uses the admitted quantizer on two disjoint row ranges, once per row.

CPU FP64 references independently decode packed weights. The unquantized reference
propagates the original BF16 input through both projections, SwiGLU, all intermediate
channels, down and merge. The quantized chain reference uses reconstructed G32
input values and independently quantized reference SwiGLU. A separate local down
reference consumes the candidate's actual reconstructed down image. Three rows
(first, middle, last), all intermediate channels, all slots and eight deterministic
hidden output channels are checked. Input/down codes, deltas and sums must match
the CPU quantizer exactly. Small GPU fixtures check K32 integer WMMA dots and
packed decoders exactly. Any nonfinite, decode mismatch or local arithmetic
normalized L2 error >=2e-5 stops the turn. Mathematical/model-quality errors are
reported, not declared qualified. Metrics include max absolute error, RMSE,
normalized L2, cosine error, bias and absolute-error quantiles, with denominator
floor 1e-12.

Warm timing uses A0/A1/A1/A0 controls and alternating ABBA/BAAB rounds. Every chain
includes input conversion/quantization, GPU CSR, fused gate/up plus down image,
down, and merge. Stage medians and full-chain medians include ranges. At least
three complete counterbalanced rounds are required. One JSON summary per shape
and candidate contains:

- `chain_ms_incumbent`, `chain_ms_candidate`: full-chain medians.
- `gain_pct`: 100 * (incumbent - candidate) / incumbent.
- `aa_noise_pct`: the larger of the maximum within-control range percentage and
  the median A0/A1 difference percentage, a conservative noise envelope.
- `kill`: true for gain below 5%, gain within the noise envelope, or insufficient
  rounds. `complete` distinguishes enough rounds from an inconclusive turn.

A variant must clear both long shapes and the tail case to warrant further work.
An incomplete turn or a missing summary is inconclusive, never a continue verdict.
The soft budget is at most 390 seconds with an unconditional 400-second process
alarm. Budget exhaustion exits nonzero with `turn_complete:false`. Kernel, PerfOpt,
compiler/runtime identity, binary and packed-tensor SHA-256, and available read-only clock/power/temperature telemetry
are printed. Missing telemetry is explicitly `unavailable`; no host settings change.

Single-stage LDS expectations including metadata and row/slot tables are 15,872,
26,112, 21,504 and 31,744 bytes for M64/N128, M64/N256, M128/N128 and M128/N256.
The epilogue reuses that allocation sixteen assignments at a time. Each thread
holds M FP32 accumulators plus an eight-word temporary integer fragment, operand
fragments and metadata. M128 is at particular risk of high VGPR use or spills;
no occupancy gain is assumed. Actual registers, LDS and private bytes are printed
from `hipFuncGetAttributes`. The model tensors occupy about 1.5 GiB for Q5_1 down,
on both host and device; allow roughly 12 GiB total transient memory at T16384,
including sampled captures, the full-byte witness and production static pools.
