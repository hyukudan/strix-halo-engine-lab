# Strix Pack v1 backend fixtures

The [wire specification](../../../docs/engine/strix-pack-v1.md) defines admission, reconstruction and
arithmetic identities. This directory holds a common bundle for CPU, HIP and CUDA tests.

## Files

- [v1/manifest.json](v1/manifest.json) lists the entries, payload geometry, 16-byte headers,
  payload SHA-256, pack identity, checkpoint digest and SHA-256 of every binary file.
- [v1/vectors.lp](v1/vectors.lp) is one complete v2 container. Its checkpoint and source
  fingerprints are synthetic. Pass the declared checkpoint digest to the host reader.
  The device fixture can stage the validated slices by offset and length.
- Each `.decoded.f32le` is a row-major `[N,K]` image. Each `.activations.f32le` is
  `[2,K]`. Each `.project.f32le` is the ordered binary32 output `[2,N]`.
  `sp4t.head.f32le` is the fixed-table grouped-scale head output `[2,N]), before softcap.
  All are tightly packed IEEE binary32, little-endian, with no prefixes.
- The encoder cases in the manifest include one all-negative-zero input with its three
  expected payloads, and three rows to submit separately with quiet NaN, +Inf or −Inf
  in the last column. The geometry cases specify invalid and large matrix dimensions.
- The source cases specify `[N,K]` for raw code and scale plane files and their expected
  FP32 output. BF16 compares all 32-bit patterns literally. For FP8, normalize any NaN
  result to the binary32 pattern `0x7fc00000` before hashing.
- [generate.py](generate.py) is an independent wire-only generator. Use
  `python -I tests/fixtures/strixpack/generate.py tests/fixtures/strixpack/v1 --check`
  to check the committed bytes. Omit `--check` to generate a new destination.

The first superblock spells code 0 through the maximum in ascending order and repeats it.
Its SP2 code bytes are `e4` repeated, SP3 is `e4` repeated plus sign bytes `f0` repeated, and
SP4 is the repeating sequence `10 32 54 76 98 ba dc fe`. SP2 starts at scale `2^-24`, the
smallest binary16 subnormal. The first SP2 weights are `[1,3,-1,-3] * 2^-24`. SP3 starts
at base +90 and scale 255, the largest allowed exponent and fraction. The first eight SP3 weights
are `[3,10,18,30,-3,-10,-18,-30] * (15/8) * 2^121`. The first SP4 weights use base −120
and byte 0, so they follow the full signed table times `2^-120`. Later blocks and rows
vary code bits independently. Sparse activations isolate the boundaries at K32 and K128.
Dense SP4T activations distinguish the FP16-input head from the FP32-input grouped projection.

## Backend checks

1. Verify the pack identity, payload bytes and input file digests before dispatch. Use the
   declared storage and header, a row pitch of `(K/128) * (16*B+4)`, the declared base, and
   the logical output length. There is no hidden leading header on the device payload.
2. Decode to binary32, download the complete logical matrix and hash the little-endian
   bits. Check all four entries against `.decoded.f32le`, including the small `1x128`
   matrix and the odd output counts 9 and 17. Record any unsupported fixture geometry.
3. Exercise the expert-major dot with one-token/assignment rows and both input rows.
   The ordered FP32 contract must match `.project.f32le` byte for byte. Exercise
   token-indexed and pre-grouped input paths and the mixed-storage launch. To build
   a mixed test with uniform dimensions, take the first rows and/or whole superblocks
   to form `1x128` experts from each entry, and compute reference outputs in
   ascending K using the sliced activations and published decoded values.
4. For the fixed-table head, compare to `sp4t.head.f32le`. Run each input row
   alone and the two rows together. Compare these bytes before softcap, then test
   the epilogue separately against softcap of those projection bytes.
   Fixture activations have FP16-representable values after narrowing, and group
   partials are exact in FP32 irrespective of native WMMA reduction order. Keep the
   existing general-input head error-envelope test as a separate check.
5. Refuse odd K such as 127 or 129, zero N or K, an overflowing byte count, and
   malformed metadata before staging. Preserve the last logical row and output.
   A sentinel after the output must survive a partial tile. Larger checks can
   repeat full fixture rows to `2049x2048` and use the independent decoder for
   the generated weights, as the host conformance target does with seeded bytes.

The host test also exercises the encoder's all-zero group, NaN and both infinities,
rounding midpoints, saturation, scale retention on lossless re-pack and uint64
geometry limits. Its source checks exhaust BF16 encodings, the E4M3 codes across
two block dimensions and the MXFP4 codes across the E8M0 exponent range with
finite products.

`ctest -R '^weight-strixpack-conformance$' --output-on-failure` checks the bundle on CPU.
HIP and CUDA launches using this bundle require device execution.
