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
