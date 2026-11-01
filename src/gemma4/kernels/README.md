# g4-v1 device path

This package implements the declared FP32 identity on gfx1151 wave32. It is an
opt-in backend, with no automatic production admission. Packed BF16,
Q8_0, Q4_K and Q6_K matrices retain their original blocks. Scalar arithmetic
comes from `contracts/g4_identity.hpp`; target graph and import contracts come
from the manifest/reference package. No hardware reciprocal square root,
runtime rotary generation, contraction or fast math is admitted.

`gpu::DeviceTarget` copies validated `HostWeights` and immutable `RopeTables`
once into device allocations. Host mappings may then be released. Its
`forward_rows` accepts up to eight active rows in a `RowPlan`, including
multiple streams and multiple contiguous candidate positions per stream.
Output rows follow `output_row`, independent of execution slots. Tap order is
blocks 5/11/17/23/35 after all residuals and the layer scalar.

`gpu::DeviceStream(config, admitted_context)` owns FP32 producer KV and absolute
tags, and implements `begin()`/`frontier()`/`digest()`. The returned `KvTxn`
owns unpublished candidate buffers. A forward uses immutable live rings plus
that overlay. Consumers alias their configured producer, without writing KV.
`commit(prefix_count)` publishes only that processed prefix, in position order,
once. Destruction discards all unpublished rows. Calls are serialized at the
scheduling boundary; a hardware failure during publication invalidates the
stream. Admission includes all candidate positions, with no position wrap.
The logical digest matches `HostStream`, including tags and normalized/rotated K.
Discarded candidate allocations are retained privately per stream for reuse,
with capacity for eight rows. They are never live KV and descriptors admit only
the current candidate prefix. `STRIX_G4_KV_POOL=0` restores per-transaction
allocation. Published transactions may outlive a subsequent transaction safely.
One publication launch covers all producers, with each producer still copying
the accepted prefix in position order. The frontier changes only after device
completion. `STRIX_G4_KV_PUBLISH=0` restores individual producer launches.

One wave handles one output and independent accumulators for all active rows.
The explicit scheduling experiment `STRIX_G4_OUTPUT_TILE=1` handles four
plain-linear outputs at C1 and two gate/up or C2..C4 outputs per wave, preloading
their chunks before consumption. It retains independent accumulators for every
output/row and the same arithmetic. It is disabled by default after a throughput
regression; unset or `0` uses one output per wave. C5..C8 and incomplete
16-element input chunks always use the original kernel.

`linear_rows`/`gate_up_rows` (`rows_linear.hip`) evaluate any number of rows,
for prompt chunks. Their bytes equal `linear`/`gate_up` called on consecutive
slices of at most eight rows: lane L of a wave still owns chunks L, L+32, ...,
each accumulator starts at +0 and adds RN(w * x) over its chunks and their
elements in ascending order, and the 32 lane values meet in `wave_sum`. A
workgroup of eight waves covers 32 outputs and 16 rows (Q6_K: 16 outputs and 32
rows); each wave decodes its lane chunk once (the shared `WeightChunk` decoder) and applies
it to every row, while the rows' FP32 inputs are staged in LDS per 512-element K
block, half a block per barrier. Gate/up runs as one fused tile that reads x
once for both matrices; after the two wave sums of each pair, pair p of a wave
is kept by lane p % 32, which evaluates `gelu_tanh(gate) * up`, so the GELU
epilogue runs on all 32 lanes instead of lane 0 (same scalar operations, same
bytes). Exact prefill, Q4_K_M, en_chat, interleaved A/B: 470/448 -> 490/470
tok/s at 512/2K, 427 -> 444 at 8K (+4-5 %). `STRIX_G4_ROWS_GATE_SPLIT=1` restores two linears
plus the separate gelu product.
`STRIX_G4_ROWS_TILE=0..7` forces one tile shape (same bytes);
`STRIX_G4_PREFILL_EXACT_MANYROW=0` uses the eight-row slices. Calls of at most
eight rows, unaligned widths and mixed gate/up formats always use the slices.
`strix-gemma4-device-rows-linear-tests` compares every variant with the slices
by memcmp (`bench` prints synthetic production-shape timings).
Coefficients are decoded outside the row loop, which gives an
actual shared weight pass. `STRIX_G4_SHARED_WEIGHTS=0` selects separate launches
with the same arithmetic. `STRIX_G4_FUSION=0` separates the FFN gate/up linears
and their GELU product, and separates branch norm/residual epilogues. The fused
attention epilogue also performs the following pre-FFN norm, retaining both
reduction orders and checking every intermediate. Typed 16-value loads decode each packed chunk's metadata
once; `STRIX_G4_CHUNK_DECODE=0` restores scalar coefficient lookup. The dot's
lane visits, accumulation and reduction remain unchanged.
BF16 matrices preload two successive lane chunks while retaining the one-output
grid, for R1 and full input chunks with width at least 1024. Consumption and
decoded-weight reuse across rows are unchanged. `STRIX_G4_CHUNK_PREFETCH=0`
restores the original typed chunk path for every format. Explicit `1` extends
paired loading to other formats as an experiment. Wider row buckets retain
ordinary chunks; explicit output tiling takes precedence. Receipts identify
when the default BF16-only policy applies, including mixed-format models.
Q4_K loads the 16-byte block header, and Q6_K loads the 16 signed subscales,
with wide reads. `STRIX_G4_PACKED_HEADER=0` restores the byte metadata path.
Metadata reconstruction happens in registers; FP16 expansion and coefficient
arithmetic still use the shared scalar routines. This changes neither packed
blocks nor decoded coefficients.
Packed weight vector loads default to non-temporal reads for BF16 or passes
sharing weights across at least four rows. Quantized C1/C2 retain normal reads.
`STRIX_G4_STREAMING_WEIGHTS=0` disables streaming; `1` requests it for all
eligible formats and row counts. Payload alignment and activation cache policy
remain unchanged. Receipts identify automatic and BF16-only policy selections.
Kernel specialization keeps the policy choice outside the lane loop.
No arithmetic or row scheduling changes.
Linear stores flag nonfinite results directly; `STRIX_G4_LINEAR_FINITE=0`
restores a separate output scan, with identical request failure semantics.
Norm, head/RoPE and PLE stores do the same under `STRIX_G4_NORM_FINITE`;
setting it to `0` restores their separate scans.
Norms use one row workgroup with a first-wave reduction;
`STRIX_G4_NORM_GROUP=0` restores the original one-wave kernel.
Norm loads prefetch eight consecutive lane visits before consuming them in
the original order; `STRIX_G4_NORM_PREFETCH=0` restores one load at a time.
`STRIX_G4_INPUT_CACHE=1` experiments with a transposed shared activation row
for C1 plain projections: full input chunks, width 256..4096 and at most 4096
outputs. Sixteen output waves share each copy, packed in four-value LDS vectors.
Output tiling takes precedence; wider row buckets and
other shapes retain the ordinary path. Unset/0 disables this experiment.
Cooperative shared-memory loads are enabled for
whole-row norms and residual epilogues up to width 4096. Only the first wave
reduces, with identical lane visits and arithmetic. `STRIX_G4_NORM_STAGE=0` keeps global loads;
larger rows always retain that path. Shared storage is private to each row.
Attention prefetch caches each lane's Q components and batches K loads in
attention scores by default. Visible V pointers are validated once per
absolute chunk, then value loads are batched eight keys at a time. Numerator
and denominator accumulators retain their original key order and chunk merges;
missing keys retain their failure flags and omitted contributions.
`STRIX_G4_ATTN_PREFETCH=0` restores the original score/value kernels.
No approximate exponential is used.
`STRIX_G4_ONE_CHUNK_ATTN=1` independently fuses the maximum and final divide into
the value kernel when the score stride admits one absolute 64-position chunk.
The score kernel remains separate. Explicit initial +0 additions retain both
ordered chunk merges. Unset/0, `STRIX_G4_FUSION=0`, disabled target fusion and
larger strides retain the four-kernel schedule. Prefetching is selected separately.
Head norms default to register-cached
inputs, norm weights and rotary coefficients for sliding head-256 and global
head-512 norms. A lane retains both coordinates of each active rotary pair,
so the intermediate normalized vector needs no global reread or barrier.
Lane-order sums, exact reciprocal-root search and coordinate operations are
unchanged. `STRIX_G4_HEAD_REGISTERS=0` and other head dimensions retain the original path.
The PLE residual kernel's following norm stage prepares the next block's input,
or the final head input, by default (`STRIX_G4_LAYER_NORM_FUSION=0` restores
separate input/final norm launches). The rounded residual remains stored before
that norm; drafter taps read the same residual. Disabled target fusion or
ungrouped norms also retain the separate launches. No norm reduction or scalar
arithmetic changes.
The wave searches the reciprocal-square-root result
with shared exact integer comparisons and the same midpoint rule as the scalar
reference; `STRIX_G4_NORM_SEARCH=0` selects the scalar search. The same wave
search covers Q/K/V head norms and model PLE slices. The original one-wave
whole-row norm retains scalar search when `STRIX_G4_NORM_GROUP=0`. No approximate
floating seed is used. These switches are cached at startup.
Exact exponent bounds narrow the integer search to one output exponent interval,
including subnormal inputs and exact powers of two. `STRIX_G4_NORM_BOUNDS=0`
restores the full-range wave search; both use the shared exact comparisons and
midpoint/tie rule. The host layout gate checks the bounds with exact inequalities.
`STRIX_G4_NORM_INTERVALS=1` further refines each exponent interval with 256
normalized-mantissa bins per exponent parity. Every integer endpoint is certified
against the shared exact product comparator; subnormal inputs are normalized with
integer shifts. Search and midpoint rounding remain unchanged. This opt-in needs
both parallel search and exponent bounds; unset/0 retains the exponent-only path.
Regenerate the committed integer table with
`python tests/gemma4_gpu/generate_norm_intervals.py`; native tests need no Python.

Decode schedule (each default exact and switchable; values equal the separate
kernels byte for byte):
- `STRIX_G4_FUSED_ATTN` (default on): one launch per layer computes the query
  norm/rotation, scores, ordered chunk maxima, chunk partials and their ordered
  merge when the visible span fits `STRIX_G4_FUSED_ATTN_CHUNKS` (default 2,
  at most 9) 64-position chunks. Longer spans use two
  chunk-parallel launches: scores with the query norm recomputed per chunk
  block, then partials that merge the chunk maxima in order, with the last
  chunk block of each (row, head) merging the partials in chunk order. Producer
  K norm/rotation and V norm are written straight into the overlay. `0`
  restores head-norm, overlay and four-kernel attention launches.
- `STRIX_G4_ATTN_KV_SHARED` (default on): the two long-span launches run on a
  (row x head, chunk) grid, so every row and query head reading one key/value
  chunk runs back to back and the chunk leaves memory once per pass; passes of
  more than one row use one block per (row, key/value head, chunk) that loads
  each key and value once for the group's four query heads. The merge block
  forms the maximum of the chunk maxima with all its threads (order free except
  +0/-0, which give the same exponentials). `1` keeps one block per query head,
  `2` always groups, `0` restores the chunk-fastest grid.
  `strix-gemma4-device-attention-shared-tests` compares the modes byte for byte.
- `STRIX_G4_STORE_FUSION` (default on): the PLE gate store applies the PLE
  activation and the head store applies the softcap.
- `STRIX_G4_MULTI_LINEAR` (default on): producer Q/K/V projections of one
  input share a grid when their packed formats match; every output keeps its
  own wave, visits and store.
- `STRIX_G4_FAST_ARGMAX` (default on): capped selection by a reduction tree.
  (value, id) under the larger-value/lower-id rule is a total order and NaN
  never wins, so any tree selects the ordered scan's candidate.
- `STRIX_G4_PAIR_TILE` (default on for shared passes of eight rows; measured
  faster at C8 and slower at C4, `1` selects it for every row count):
  two outputs per wave share every input load; each (output, row, lane)
  accumulator keeps its chunk and element order. `STRIX_G4_QUAD_TILE=1`
  (opt-in) uses four outputs per wave for single-matrix passes.
- `STRIX_G4_ROW_PREFETCH=1` (opt-in) loads each lane's next chunk before
  accumulating the current one.
- `STRIX_G4_HARDWARE_HALF` (default on): packed half scales use the hardware
  half-to-single conversion, exact for every non-NaN half because all target
  kernels preserve half denormals; model tensors are validated finite.
- `STRIX_G4_EPILOGUE_EARLY_LOADS=1` (opt-in) issues the norm/residual
  epilogue's branch, residual and weight loads before its first reduction.
- `STRIX_G4_FAST_EPILOGUE` (default on): the norm/residual epilogue for widths
  2048/2560/3072/4096 with f32 or bf16 norm weights (and layer scalar) and
  non-overlapping rows: compile-time width, no per-element format dispatch,
  every operand requested in one batch and the scalar through a scalar load.
  Reductions, rsqrt and elementwise order are norm_residual_kernel's.
- `STRIX_G4_FAST_ATTN` (default on): the single-launch fused attention for spans
  of at most two 64-position chunks requests all keys of a wave in one batch,
  each thread's first-chunk value column before the softmax, and reads each
  chunk's scores into registers before the same ordered maximum/denominator
  scans. Every score, scan and accumulation keeps its order.
- `STRIX_G4_ATTN_KV_STORE` (default on, needs `FAST_ATTN`): when every row of a
  pass is its transaction's only new position, the producer K/V epilogue runs
  inside that attention launch (waves 1-2, kv_store's arithmetic); the first
  block of each KV group writes the overlay and every block reads its current
  position from the shared copy. The single-row bit is part of the graph key.
- `STRIX_G4_FFN_PLE_GATE` (default on, one or two rows, Q4_K/Q6_K gate): the
  FFN norm/residual epilogue and the PLE gate linear with its activation store
  share one launch. Every block recomputes the epilogue rows exactly into
  shared memory and dots against them; the last block to take a ticket stores
  the residual.

Fixed-row HIP graphs bind stable scratch and device plan descriptors. Absolute
64-position buckets determine storage and launch bounds; visibility and all
reductions still use each row's absolute range. Each pass uploads current
snapshot/overlay descriptors before launch. Scratch growth invalidates captures;
the cache retains at most eight graphs. `STRIX_G4_GRAPHS=0` selects plain launches.
Profiling uses plain launches. WMMA is not implemented or admitted.
Metadata and complete result transfers use reusable pinned host buffers on the
same queue as the graph, followed by one completion boundary. Returned-vector
allocation overlaps device execution. `STRIX_G4_PINNED_IO=0` restores synchronous
pageable transfers. No pointer or bytes from a prior pass enter returned rows.

Attention scores retain the lane reduction order. The four stages are scores
and chunk maxima, ordered maximum merge, exponent/value chunk partials, then
ordered partial merge and FP32 division. Chunks follow absolute multiples of
64, including a partial first sliding chunk. Every GQA group has four queries.
Nonfinite outputs, overflowing norm reductions and missing KV tags fail the
transaction before publication.

Build with `STRIX_FAMILY_GEMMA4=ON`. HIP OFF supplies
`strix-gemma4-kv-layout-tests`; HIP ON adds `strix_gemma4_device`,
`strix-gemma4-device-ops-tests`, `strix-gemma4-device-rows-tests` and
`strix-gemma4-bench`. GPU tests require exclusive externally scheduled device
access. Each ctest is bounded by 60 seconds. Full-model checks are explicit CLI
work, never ctest dependencies.

```sh
strix-gemma4-bench --model MODEL_DIR --route bf16 --rope TABLE_DIR \
    --rows 1 --steps 8 --context 1024 --check-host --check-serial
strix-gemma4-bench --model MODEL.gguf --route q4_k_m --rope TABLE_DIR \
    --rows 4 --steps 8 --context 1024
strix-gemma4-bench --model MODEL.gguf --route q4_k_m --rope TABLE_DIR \
    --verify-rows all --context 1024
```

Use `--rows all` for C1/C2/C4/C8 diagnostics with one weight upload, or select an
individual row count. Timing receipts include the running kernel and the
read-only `amdgpu.iommu_perfopt` parameter when present. `--verify-rows` compares
a causal candidate pass to both host and serial device execution, including
complete logits, taps and canonical KV digests. `all` checks every bucket from
one to eight using the same serial reference prefixes. Full-model checks need
an exclusively scheduled machine with enough host and device memory.
`--profile` records HIP event
times by operation family, and marks the throughput receipt as profiled because
event synchronization changes scheduling.
`--time-graph` instead times the complete queued pass with two events outside
capture, retaining graph execution. It also separates forward and commit wall
time; these are diagnostic means, while pass throughput uses its stated median.
It cannot be combined with per-family profiling. Unprofiled throughput covers forward,
result transfer and commit, excluding optional reference comparisons. Decode
uses the returned device capped-logit selection; verification also checks its
ID and value against the complete logits. Receipts
state `NO-REFERENCE` when a comparison was not requested. Weight bytes/token and
effective weight GB/s count packed linear reads and embedding/table gathers;
they exclude KV, activations, norm/scalar weights and cache effects, and are
neither hardware memory counters nor a complete bandwidth measurement.

`strix-gemma4-device-ops-tests --bench-linear` is a separate synthetic diagnostic
for production linear shapes at C1/C4/C8. It reports medians of repeated HIP
event samples and checks sample output bytes against the host arithmetic.
`--bench-linear-unfused` compares separate gate/up projections and their
activation/product against the same sampled host bytes.
`--bench-linear-small` instead diagnoses attention and layer-PLE projections,
including the shapes admitted by the C1 activation-cache experiment.
Run separate processes with `STRIX_G4_OUTPUT_TILE=0` for scheduling comparisons;
the diagnostic is excluded from ctest and never loads model assets.

Synthetic byte gates do not establish model fidelity, compiler/ISA qualification,
long-context behavior or complete-workload throughput. Those are separate release
requirements. The family is disabled by default in CMake;
the internal scheduling switches preserve the g4-v1 numerical identity.

## Opt-in identity g4-a8-v1

`g4-a8-v1` is a separately named numerical identity, never selected implicitly:
`DeviceOptions::identity = "g4-a8-v1"`, or `STRIX_G4_IDENTITY=g4-a8-v1` when the
option is unset; unknown names are refused and the default stays `g4-v1`. It
changes only the dot products of the Q8_0, Q4_K and Q6_K linears (including the
tied Q8_0 head and the PLE gate/projection). Each input row is quantized per
32-element group (`delta = amax / 127`, codes `rint(x / delta)` clamped to
+-127, zero/underflowing groups `delta = 1` and codes 0); weight codes and
activation codes meet in exact int32 dot products; each group contributes one
FP32 term with the declared scale/minimum order; sixteen slots accumulate the
groups in ascending order and an 8/4/2/1 tree combines them. Norms, attention,
PLE, softcap, residuals, selection and the BF16 model-PLE projection keep the
`g4-v1` recipe. `a8.hpp` is the normative declaration and host oracle;
`a8_host.hpp` installs it into the host graph through `HostTarget`'s
`LinearRecipe` hook.

Row invariance is structural as in `g4-v1`: a row's quantization, integer dots,
slots and tree never depend on peers, row count or tiling, so R causal rows
equal R serial passes and DFlash verification equals serial decoding. The
producers of quantized inputs (input/final norm, the three branch epilogues,
attention, gate/up and the PLE gate) write the quantized rows in the same
launch; any other schedule (switches above) falls back to a separate
quantization launch with identical bytes. `strix-gemma4-bench --identity
g4-a8-v1` runs the byte gates and timing; `strix-gemma4-a8-teacher` writes
teacher-forced fidelity receipts of both identities on the same weights.
Fidelity qualification is separate; this identity has no default admission.
