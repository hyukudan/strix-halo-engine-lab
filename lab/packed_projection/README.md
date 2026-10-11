# Packed Low-Row Projection Lab

This is the first A1.1 packed projection implementation.

The initial format is Q8_0. Q8_0 is used here as a transparent packed-format
reference because its block layout is simple enough to audit independently. It
is **not** a decision that the full single-node checkpoint should use Q8_0.

## Kernel structure

For one output feature, one workgroup walks the packed weight row.

Each packed weight value is decoded once and multiplied by all active logical
rows held by that kernel variant:

```text
packed weight
    |
    +-> row 0 accumulator
    +-> row 1 accumulator
    +-> row 2 accumulator
    ...
```

The source provides separate compile-time row classes:

- C1 logical row;
- 2 logical rows;
- 4 logical rows;
- 8 logical rows.

These are logical-row shapes, not request-concurrency modes.

## Why Q8_0 first

The objective is to qualify:

- packed weight consumption inside the compute kernel;
- exact physical row geometry;
- weight-load reuse across verify rows;
- reduction structure;
- Wave32 versus Wave64;
- register/LDS growth as row count increases;
- row invariance.

Once that machinery is trustworthy, the same harness can host the more compact
canonical format selected for the real checkpoint.

## Build

```bash
cmake -S lab/packed_projection -B build/packed-projection -G Ninja \
  -DCMAKE_BUILD_TYPE=Release
cmake --build build/packed-projection
```

Run:

```bash
./build/packed-projection/strix-q8-projection-w32
./build/packed-projection/strix-q8-projection-w64
```

The harness requires `gfx1151` and validates the compiled device wave size.

## Output

The JSON result includes, for each row class:

- microseconds per packed projection;
- physical packed tensor bytes;
- equivalent serial weight bytes;
- physical packed-weight GiB/s;
- serial-equivalent weight GiB/s;
- function register/shared/local-memory attributes.

The serial-equivalent number is not claimed hardware bandwidth. It expresses
how much repeated weight traffic separate single-row kernels would have needed.

## Promotion rule

The IQ4_XS lab now compares the LDS reduction with an experimental register-shuffle
tail. Both use the fixed 256-thread tree (128/64/32/16/8/4/2/1) and explicit FP32
FMA in each ascending lane K chain. The candidate retains the first three LDS
reductions and moves the final five into 32-lane shuffles, avoiding five group
barriers while preserving the arithmetic tree. Packed weights are already reused
across C1/2/4/8 rows in both variants.

Before timing, native fixtures compare both variants bit-for-bit against the
independent host lane/tree reference at K=256/512/2560/4096 for all row classes;
guarded output allocations detect inactive writes. The portable reference tests
include mixed-magnitude, signed-zero/subnormal and explicit FMA/reduction negative
controls. Arithmetic identity: `iq4_xs-f32-fma256-tree-v1`. This explicitly defined
lab identity replaces the earlier implicit contraction choice for this experiment.

The latest local check compiled both Wave32 and Wave64 binaries offline with
ROCm 7.2.4 and ran only the host reference tests. No GPU execution, bitwise native
fixture result or throughput gain has been measured. The JSON result retains
`qualified=false`; passing a synthetic kernel fixture is not full-model promotion.

The harness also captures this fixed-shape projection in an owned HIP stream and
replays it after changing input values in the same device allocation. Reference bits
and guarded writes are checked again. Timings distinguish eager from graph replay
and report host capture/instantiation time separately. Graph executable memory remains
unknown (`graph_memory_bytes=null`), so this experiment cannot establish graph memory
admission or enable whole-model replay. Captured scalar K/N are fixed for the graph's
lifetime; varying token/position arguments in other launchers still require device data.

Q8_0 remains an infrastructure/reference packed path until a real workload
justifies it.

The canonical single-node checkpoint format must be selected from full memory,
quality and runtime measurements. A later compact-format kernel receives a
different kernel ID and numerical contract.


## IQ4_XS compact candidate

The same lab now also contains an experimental IQ4_XS projection path.

IQ4_XS is represented as its exact 136-byte / 256-element packed block:

- one FP16 global scale;
- eight 6-bit local scales, one per 32 values;
- 128 packed 4-bit code bytes;
- a fixed 16-entry nonlinear signed codebook.

The host core owns an independent byte-layout/dequant contract and tests it
without HIP. The GPU lab then consumes the packed block directly and reuses each
decoded weight across compile-time logical-row classes 1/2/4/8.

Build outputs:

```text
strix-iq4-xs-projection-w32
strix-iq4-xs-projection-w64
```

As with the Q8_0 reference path, Wave32 is native and Wave64 is a separate
experimental compile.

IQ4_XS is a strong compact-format candidate because it satisfies the current
single-node memory envelope substantially better than heavier K-quants, but it
is not yet the canonical checkpoint decision. Quality, real Strix Halo
throughput, resource pressure and full-model behavior still decide that.
