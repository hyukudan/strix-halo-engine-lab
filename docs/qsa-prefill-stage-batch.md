# QSA prefill stage batching (draft)

`STRIX_QPF2K_QSA_MASK` bit 9 (`0x200`) is OFF by default. With
`STRIX_QPF2K_QSA=1`, it requires mask bits 0 and 1 (`0x3`); otherwise it is
ignored. Bit 2 remains optional and selects the same wave norm as before.
For example, compare masks `0x7` and `0x207`, keeping all other engine settings
identical. Existing masks, including indexer batching and deferred selection,
can retain their other bits.

Admission is all-or-nothing for one layer-chunk. The per-group path runs when
any of these conditions fails:

- Normal eager prefill mode (`STRIX_QSA_PREFILL_TILES` and prefill self-check
  disabled), with 1 through 128 complete 64-row groups and no remaining tiles.
- `STRIX_QSA_NORM_GROUP=1`, BF16 KV storage, and an available pooled-key buffer.
- Index geometry: four heads of dimension 128; every query pointer aligned to
  16 bytes; every scored prefix within both the pool capacity and 65536 blocks.
- The three norm tensors share one weight format; their dimensions are powers
  of two between 2 and 256. Half the rotary dimension is at most 64 and the
  rotary dimension fits each norm dimension. These are the existing grouped
  norm conditions. The existing wave-norm conditions still determine wave/LDS.
- Rows have contiguous positions, common raw/KV owners and extents, common
  geometry, common norm owners, journal, compression ratio and selection top-k.
- Private table/maxima device buffers and pinned descriptor buffers can be
  allocated. Allocation rejection occurs before any stage launch or state write.

Windows, cohorts, graph execution, partial row groups and int8 KV retain their
existing paths. HIP execution errors still propagate; they are not retried after
partially executing a batch.

The store and pool kernels are shared with the fixtures through
`hip_qsa_prefill_stages.inc`. The batch specializations only select group
descriptors and reject padding workgroups. For pooling, each group receives
exactly its original newly completed block interval, disjoint from every other
group's interval. Pool/range grids flatten those intervals and recover the
owning group with integer bounds lookup, avoiding padding every group to the
first group's context prefix. Each block retains its ascending token sum, LDS norm tree,
RoPE expressions and BF16 conversions. Range calculation retains the same
per-key magnitude loop.

Scoring keeps each group's original maximum block count, including its last-key
clamp for incomplete 32-block tiles. Row bounds, exponent guard, fused/separate
choice, head-dot tree, head sum, division and four-block maxima tree are common
code. Each group has its own maxima slice and row stride; immediate and deferred
selectors receive that slice. A larger batch grid never changes a group's
local tile index or guard inputs.

RoPE tables are private per group. Local row/head/wave/feature indices,
positions, norm trees, weights, strides and conversions are unchanged. Stores
write disjoint positions with the original conversions. Only the final group's
last eight rows publish the shared journal; earlier publications had no consumer
before their serial overwrite. The execution stream orders table, norm, store,
pool, range, score and selection. A two-slot event-protected ring retains tables,
descriptors and maxima until the last consumer, including deferred selection.

With `STRIX_QPF2K_QSA_TRACE=1`, grep `qpf2k qsa stage_batch`. A fully populated
128-group batch emits this per-call launch witness:

```text
qpf2k qsa stage_batch groups 128 pool 1 range 1 score 1 rope 1 norm 1 store 1 fallback 0
```

Pool/range/score counts are zero when there is no corresponding work. A rejected
batch emits zero counts with `fallback 1`; existing trace hashes remain enabled.

Build the standalone fixture with the engine's compiler and arithmetic flags
(append any additional engine build flags to the following command):

```sh
mkdir -p build-qsa-flash
amdclang++ -O3 -std=gnu++20 --offload-arch=gfx1151 -x hip \
  -Wall -Wextra -Wpedantic -Wconversion -Wshadow -Werror \
  tests/qsa_flash_fixtures.hip -o build-qsa-flash/qsa_flash_fixtures
env STRIX_QPF2K_QSA=1 STRIX_QPF2K_QSA_MASK=0x207 STRIX_QSA_NORM_GROUP=1 \
  ./build-qsa-flash/qsa_flash_fixtures --stage-batch-only
```

The fixture directly invokes both schedules, independently of environment
selection. Its 48 cases cover 1, 2, 7 and 128 groups, wave and LDS norms,
FP32/FP16/BF16 norm weights, partial block tiles, masked rows, groups without
scores/index-query norms, signed zeros, subnormals, overflow, infinities and
NaN payloads. It compares complete buffers including canaries, guard counts,
private tables/maxima, raw keys, KV, journal, selected indices and post-selection
scores byte-for-byte. HIP failures and any mismatch return nonzero. Journal
contents also have an independent final-eight-row check.

Implementation validation here is limited to warning-clean host/device syntax
checks and optimized device LLVM generation for the engine and fixture using
Clang 19 and ROCm 7 headers. A host check of flattened interval ownership passed
384 cases, including empty intervals and large context prefixes. Optimized IR
has no private descriptor allocations or copies in the batched kernels. No GPU
execution was performed. Native compiler/ISA,
fixture execution, model fidelity and complete-workload qualification remain
required before this draft can be qualified.
