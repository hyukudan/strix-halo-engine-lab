# PLE grouped projection qualification

`STRIX_PREFILL_PLE_BATCH=1` opts into batching complete 64-row prefill groups.
The default remains off. The supported recipe is the default lane path: wide kernel 2,
wide stage disabled, 16 waves, lane stage 8, prefetch enabled, no lane ablation.
Both projections must pass lane admission and alignment checks. Scratch allocation
failure returns to the original loop. Eight-row tails and convolution sequencing
retain their original code. `STRIX_PREFILL_PLE_WIDE_CHECK=1` includes the batched rows.

Each key/value projection uses the same `PleRowsSite<64>` cached weight pack.
`PleRowsGroupSite<64>` adds the Y group offset to input and output rows, and the
lane kernels add the corresponding packed-row slab offset. X block mapping,
coefficient arithmetic, FMA chains and reduction trees retain their original form.
Two barrier-ordered packs are retained: the value pack waits for the key consumer.
The window owns one `rows * width * 4` byte scratch allocation (80 MiB for 8192 x 2560).

For 128 groups, each projection changes from 128 row packs and 128 projections to
one row pack and one projection. Key plus value therefore changes from 512 launches
to four, excluding cached weight-pack misses and optional diagnostic launches.

CPU compilation on gfx1151 with `amdclang++ -O3 -std=gnu++20 -Werror` reports:

| Kernel | VGPR | Scratch bytes/lane | LDS bytes/block | Compiler waves/SIMD |
|---|---:|---:|---:|---:|
| Original row pack | 13 | 0 | 0 | 16 |
| Grouped row pack | 12 | 0 | 0 | 16 |
| Original projection (16 waves, stage 8, prefetch) | 256 | 0 | 65536 | 5 |
| Grouped projection (same arguments) | 256 | 0 | 65536 | 5 |

These are compiler resource reports, not measured hardware residency. Both new
kernels have zero reported spills. Device assembly comparison against the starting
revision preserves all 38 existing `PleRowsSite<64>` kernel instruction bodies
(after normalizing local label numbers and comments). Original and grouped default
projections also have identical 192 dual-FMA, 60 dual-add and 32 scalar-add instruction
strings and operands. Independent coefficient temporaries are allocated differently.

Build the standalone GPU fixture and engine object, and execute CPU-only gates:

```sh
# ROCM_IMAGE: any ROCm image with hipcc for gfx1151
docker run --rm --network none -v "$PWD":/src -w /src ROCM_IMAGE bash tests/ple_batch_build.sh
```

The script contains the complete compiler commands, CMake include/telemetry options
and resource-report flag. Outputs remain in ignored `build-ple-batch/`.
CPU gates check descriptor addresses, pack bijection, store addresses, admission
bounds and eight dispatch overrides. The fixture's default invocation requires a
GPU; it was built but **not executed** during this CPU-only work:

```bash
build-ple-batch/ple_batch_fixture
```

Its exact matrix covers groups 1/2/3/64/128, 512 -> 65/33 and model-default
2560 -> 10240/2560 shapes, aligned padded and odd input strides, odd output strides,
BF16 signed zeros/subnormals/huge values/infinities/signed NaN payloads, poisoned
outputs and scratch, guard regions and a CPU packed-bit oracle. The small shapes
and odd strides directly exercise the grouped kernel while retaining production
admission rejection. After the complete exact matrix passes, it prints hipEvent
medians for warmed key+value A/A and ABBA at 128 groups; the baseline preserves
per-group key/value interleaving. Any mismatch exits nonzero before timing.

No GPU exactness, benchmark speedup, model fidelity or complete-workload qualification
is claimed. The implementation remains opt-in pending those gates.
