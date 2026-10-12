# Experimental integer projection

This opt-in laboratory implements actual gfx1151 Wave32 IU8/IU4 matrix
projection and row/group activation preparation. It is not registered for
production selection. Offline compilation is distinct from native arithmetic,
model fidelity and performance qualification; those gates remain open.

Q4_0 retains its unsigned nibble codes, FP16-derived group scales and offset 8.
IQ4_XS retains its nibbles and exact signed INT8 codebook lookup, with the
FP16 global scale times signed six-bit local scale per 32 elements. IQ4_XS
indices are rejected as raw IU4 operands. Staging is lossless, operation-local
and packed (16 bytes per scale group), not a model file format or full-model
expansion. K must be exact storage blocks: Q4_0 accepts 640 and 2560;
IQ4_XS accepts 2560 and rejects 640.

Activations use the existing explicit binary64 nearest-even A8/A4 contract,
group size 32, finite input only, gradual subnormals, symmetric ranges
[-127,127]/[-7,7], division by absmax/qmax, zero group scale 1. No automatic
precision selection occurs. Binary64 preparation, scales and output are
reference-oriented choices, not a proposed production layout. The full
versioned numerical identity is printed per case; A4/A8 are alternative
numerical plans, not equivalent to the conservative floating model path.

Each group uses two K16 WMMA panels. Both lane halves replicate input fragments;
accumulator register v belongs to row `2*v+lane/16`, column `lane%16`.
Operand packing uses unsigned shifts, explicit signed interpretation and
disabled saturation. Raw INT32 group bounds are at most 32*127*127=516128.
Q4_0 correction subtracts `8*sum(activation codes)` in INT64 before narrowing.
Scales combine as `(weight_scale*activation_scale)*corrected_dot`, then add
groups in increasing K order, binary64 without contraction. A failed device
status invalidates the entire result. Missing rows/columns load interpreted
integer zero before packing, and output bounds suppress invalid stores. No K
padding or invented storage groups are permitted.

The fragment layout is derived from the official
[AMD RDNA3/3.5 dense WMMA builtin reference](https://rocm-handbook.amd.com/projects/amd-rocm-optimization-guide/en/latest/compiler-builtins/rdna/rdna3-dense-wmma-builtins.html)
and [AMD WMMA description](https://gpuopen.com/learn/wmma_on_rdna3/).
Code is independently implemented here; host simulation does not establish
that the hardware fragment mapping works.

Build the portable checks without HIP:

```sh
cmake -S lab/integer_projection -B /tmp/strix-integer-host
cmake --build /tmp/strix-integer-host
/tmp/strix-integer-host/integer-projection-host-tests
```

Build and explicitly run the native checks only on gfx1151:

```sh
HIPCXX=/opt/rocm/llvm/bin/clang++ cmake -S lab/integer_projection \
  -B /tmp/strix-integer-native -DSTRIX_INTEGER_PROJECTION_NATIVE=ON
cmake --build /tmp/strix-integer-native
/tmp/strix-integer-native/integer-projection-native --run
```

The program rejects a nonempty `HSA_OVERRIDE_GFX_VERSION`, selects device 0
and checks exact gfx1151 plus actual Wave32.
It executes Q4_0 A8/IU8, A4/IU4 and A4/IU8, IQ4_XS A8/A4/IU8;
M=1/3/17, N=19/640/2560 and K=32/256/640/2560 where storage permits.
It compares all corrected integer groups against independent direct integer
mathematics and binary64 scale/output bits against the CPU reference. It
re-executes each native row alone after the batched run, compares preparation
codes/scales/sums and outputs, and checks guards including logical solo bounds.
Preparation controls cover zero groups, signed tie rounding, largest finite
inputs and gradual subnormals; negative controls reject NaN, infinity and
scale underflow, and report projection overflow. Weight fixtures contain
different positive, zero and negative local/global scales. Host checks also
reject unknown encodings, tampered sums/codes, shape/length overflow,
nonfinite scales, changed floating environment and nonlinear raw-IU4 requests.
The laboratory limits staging/reference working storage to 256 MiB and rejects
larger operations before allocation. Host checks require nearest-even and
gradual subnormals even for previously staged operands.

`bash lab/integer_projection/compile_offline.sh PRIVATE_OUTPUT_DIRECTORY` uses an existing ROCm installation
to compile/link the whole harness and save source hashes, compiler identity,
device bundle, extracted ELF, disassembly and metadata. It does not execute the
binary, access GPU devices or download dependencies. Keep compiler artifacts
outside Git. The script resolves output symlinks/ancestors, rejects every Git
checkout and the source checkout, and requires a new unused output directory.
All kernels explicitly use the legacy stream; allocations are
freed only when querying that stream proves quiescence, and retained on failure.

Remaining gates: native execution of this exact harness; profiler evidence for
VGPR/scratch/memory traffic and device preparation costs; representative tensor
and row histograms; integration with producer/buffer identities; model/state/
speculative fidelity; complete-workload timing against the packed floating
control. No latency/throughput measurement or qualification is claimed here.
