# Basic gfx1151 Kernel Lab

This lab is the first A1.0 GPU slice.

It intentionally begins with FP32 storage and FP32 accumulation so the first
optimized GPU paths can be compared against simple semantic references without
mixing quantization errors into kernel bring-up.

The lab is standalone and does not make HIP/ROCm a dependency of the host-side
core.

## Kernels

The first slice contains:

- centred-weight RMSNorm;
- residual add;
- partial RoPE;
- deterministic argmax;
- deterministic small-k top-k.

The top-k implementation is a correctness-oriented GPU baseline. It repeatedly
reduces one winner per row and is not yet intended as the final production
selection kernel.

## Wave builds

The same source is built twice for `gfx1151`:

```text
strix-basic-kernels-w32
strix-basic-kernels-w64
```

Wave32 uses the native RDNA compile mode. Wave64 is compiled separately with the documented experimental `-mwavefrontsize64` option. The device-side harness also checks `warpSize` at runtime so a toolchain that ignores or rejects the requested mode cannot silently qualify the binary.

## Build

On a native ROCm host:

```bash
cmake -S lab/basic_kernels -B build/basic-kernels -G Ninja \
  -DCMAKE_BUILD_TYPE=Release
cmake --build build/basic-kernels
```

Run both variants:

```bash
./build/basic-kernels/strix-basic-kernels-w32
./build/basic-kernels/strix-basic-kernels-w64
```

Each executable:

1. checks that the active device reports `gfx1151`;
2. runs tiny deterministic fixtures;
3. compares GPU output against CPU references;
4. checks one-row and two-row execution;
5. emits a JSON summary.

## Qualification

Passing this lab means only that the basic kernel implementation is a viable
candidate.

A production registry entry still requires:

- target-hardware profiling;
- compiler identity;
- workgroup/wave metadata;
- resource usage;
- graph-safety qualification;
- measured cost;
- oracle match under the intended precision mode.

## Numerical policy

The first kernel variants use FP32 input, accumulation and output.

Later BF16 or mixed-precision variants MUST receive separate kernel IDs and
numerical contracts. They do not replace the FP32 reference path implicitly.
