# Portable low-precision experiment

This host-only lab implements the scalar arithmetic gate of the
[draft proposal](../../docs/proposals/2026-10-11-phase-specific-low-precision.md).
It also contains a bounded offline instruction-emission probe. Neither changes
runtime dispatch, numerical plans, checkpoint storage, authoritative verification,
KV/recurrent state or wire encoding. Native correctness/performance and model
quality/replay remain open gates.

## Build and use

From the repository root:

```bash
cmake -S . -B /tmp/strix-precision-dev-20261011 -G Ninja -DCMAKE_BUILD_TYPE=RelWithDebInfo
cmake --build /tmp/strix-precision-dev-20261011 --target strix-low-precision-tests strix-low-precision-reference
ctest --test-dir /tmp/strix-precision-dev-20261011 -R low-precision --output-on-failure
/tmp/strix-precision-dev-20261011/strix-low-precision-reference candidates --rows 1 --k 2560 --n 2560 --encoding iq4_xs
/tmp/strix-precision-dev-20261011/strix-low-precision-reference candidates --rows 8 --k 640 --n 2560 --encoding iq4_nl
/tmp/strix-precision-dev-20261011/strix-low-precision-reference vectors > /tmp/low-precision-vectors.json
```

The lab can also build independently with `cmake -S lab/low_precision -B
/tmp/strix-low-precision-standalone`. It has no HIP dependency.

`candidates` emits five finite candidate combinations: a packed floating
control, decoded I8 with A8 or A4, direct I4 with A4, and two I4 products with
already quantized A8. `--rows` means the actual rows of one projection or one
expert. Evaluate each nonempty expert independently; total verification/request
rows are not expert occupancy. Empty experts perform no projection.

Q4_0 has 32 elements/18 bytes with `W=d*(q-8)`. Its unsigned file nibble can
be consumed with explicit offset correction or losslessly mapped to signed
`q-8` for the two-I4 reference. IQ4_NL has 32 elements/18 bytes and IQ4_XS
has 256 elements/136 bytes. Both nonlinear codebooks map to exact I8 operands,
but their indices are rejected as raw I4 operands. K=640 fails IQ4_XS storage
eligibility even though it fits 16-element matrix tiles; it fits IQ4_NL/Q4_0.
There is no synthetic storage tail or automatic re-quantization.

Tile dimensions default to 16/16/16 and may be supplied with `--tile-m`,
`--tile-n`, `--tile-k`. They are accounting hypotheses, independent of storage
blocks and 32-element scale groups. The scalar reference never pads. A later
matrix implementation must mask padding to mathematical zero after operand
interpretation; IQ4 index zero decodes to -127 and is not a zero operand.
Partial tile reductions must preserve the declared scale-group boundaries.

JSON separates useful products, issued tile products, masked outputs,
preparation elements, scale/correction groups, physical packed bytes, ideal
reused weight-read lower bound, and serial weight-read upper bound. The
two-I4 path counts two matrix products. Decoded weight scratch describes one
tile, not a full expanded model. These are arithmetic/traffic counts, not a
latency model or performance ranking. Packing, synchronization, outlier paths,
spill/LDS traffic and native output/scatter costs require measurements.

Input/output byte accounting declares four-byte floating buffers. Prepared
A8/A4 payload counts assume byte/nibble packing; host `QuantizedRow` retains
one byte per code for inspection and is not that executable layout. The
separate activation-scale field counts eight-byte binary64 *reference* scales,
not a proposed GPU scale layout. Candidate eligibility means mathematical
representability only, not a qualified instruction, kernel or model mode.

## Arithmetic contract

`include/strix/low_precision.hpp` exposes small independently testable functions:

- nibble/byte interpretation: signed I4 [-8,7], unsigned I4 [0,15], signed
  I8 [-128,127], and IQ4 index lookup via the existing IQ4_XS codebook;
- group dot: `sum(qw*qa)-zw*sum(qa)-za*sum(qw)+K*zw*za`; raw products
  accumulate with checked INT32 narrowing, every correction operation uses
  checked INT64, and final corrected INT32 narrowing rejects overflow;
- scale application: `(sw*sa)*corrected_dot`, followed by binary64 left-to-right
  group addition. Different group scales are never merged into one integer sum;
- signed-I8 split: `h=floor(q/16)`, `l=q-16*h`. Reconstruction uses wide
  multiplication/addition. Negative signed division is explicitly corrected;
  no negative signed shifts occur. High digits are signed, low digits unsigned;
- packed IQ4_XS reference: exact low/high nibble positions and signed six-bit
  local scales are consumed one 32-element group at a time. Binary64 arithmetic
  here has its own identity and does not claim to reproduce an FP32 projection.

Row quantization uses exact K groups, per-group absolute maximum, a binary64
divide by 127 (A8) or 7 (A4), signed symmetric ranges [-127,127]/[-7,7], zero
points zero, clipping to those ranges, and explicit nearest-even tie rounding.
A zero group uses scale one and zero codes, including negative zero inputs.
NaN/infinity are rejected. A positive scale that underflows to zero is rejected;
gradual subnormals are required, with a runtime check for common FTZ/DAZ modes.
`FE_TONEAREST` is required; no reciprocal approximation or contraction is used.
The quantization contract string is explicit and versioned. A stable FNV-1a
identifier also includes activation encoding and exact scale-group geometry.
This identifies semantics, not input content, and is not a security hash.
Quantization depends only on one row's content and declared groups. Reuse across
operators still requires the exact input version and the same complete contract.

Tests enumerate all 256 I8 decompositions, all 4096 signed-I4/I8 products,
all affine U4/I8 code pairs with extreme offsets, and signed/unsigned I4 pairs.
Independent direct dequantized mathematics checks corrections and the nonlinear
lookup; packed IQ4_XS checks every subgroup against both the direct formula and
the existing dequantizer, including negative local/global scales. Tests cover
unequal group scales/order, ties, zero/subnormal/extreme/nonfinite rows,
alone/duplicate/reordered rows, raw/final/wide overflow and storage/tile cases.

## Bounded offline probe

`intrinsic_probe.cpp` uses the documented Wave32 GFX11 IU8/IU4 builtin ABI from
the [Clang builtin reference](https://clang.llvm.org/docs/AMDGPUBuiltinReference.html).
It deliberately does not construct a logical matrix fragment: the artifact
proves compiler acceptance and instruction emission only. A/B replication,
lane/register mapping, overflow bounds, exact scalar correspondence and native
execution remain unresolved for an executable matrix kernel.

```bash
CLANG=/opt/rocm/llvm/bin/clang++ bash lab/low_precision/compile_probe.sh /tmp/strix-offline-gfx1151
```

The script downloads nothing and accesses no GPU. It retains compiler version,
flags, source hash, LLVM IR, assembly, object, disassembly and stderr; it checks
IU8/IU4 emission and Wave32 kernel metadata. On 2026-10-11 the locally available
ROCm 7.2.4 compiler emitted both instructions for `gfx1151`, including signed
high/unsigned low I4 mode, with saturation disabled. The compiler identity was
AMD clang 22.0.0git, `roc-7.2.4`, revision
`f58b06dce1f9c15707c5f808fd002e18c2accf7e`.
No target GPU was used and no native result or speed estimate is reported.
