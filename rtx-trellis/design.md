# CUDA compressed-weight linear proposal

Status: experimental, not admitted for model execution. Build option
`STRIX_ENABLE_CUDA_TRELLIS=OFF`; every linear descriptor defaults to disabled.
No existing model or serving dispatch is changed by building the kernels.

## Scope and tensor placement

The prefill candidate keeps routed gate/up/down payloads compressed through
file, pinned host tier, PCIe transfer and device slots. Decode occurs inside
the consuming tile, rather than expanding a layer before transfer or retaining
FP16 expert matrices. Dense candidates are q_a/kv, q_b, grouped o_a, o_b and
shared gate/up/down at K4, with an optional K5 vocabulary head. Embeddings,
router, norms, cache formats, Engram tables and state transport retain their
existing contracts. A grouped o_a must retain a separate H128 transform for
each group; concatenation cannot cross a logical group boundary.

For split decode, the same compressed dense tensors and routed experts would
be resident for layers [0,22). The head belongs to the tail. A larger resident
front is a separate placement experiment; it cannot be inferred from a kernel
test. Expert routing, SwiGLU, route-weight placement, TP ownership and combine
order remain caller responsibilities. The primitive does not remove the
production A8 policy implicitly. Integration must explicitly reproduce or
quality-gate its activation rounding as well as the new weight representation.

The initial implementation supplies a borrowed-device-weight linear API and
diagnostic probes. File/pinned tier replacement, model-name mapping, grouped
expert scheduling and production frontend dispatch are subsequent integration
gates, not implemented by the standalone linear API.

Stored projection inventory (whole RTX layer, excluding router/norms/Engram):

| Tensor | I | N | Copies | K | Packed + scales, bytes |
|---|---:|---:|---:|---:|---:|
| q_a | 5120 | 1280 | 1 | 4 | 3,289,600 |
| kv | 5120 | 512 | 1 | 4 | 1,321,984 |
| q_b | 1280 | 32768 | 1 | 4 | 21,039,616 |
| o_a, separate groups | 4096 | 1024 | 8 | 4 | 16,859,136 |
| o_b | 8192 | 5120 | 1 | 4 | 20,998,144 |
| shared gate/up/down | 5120 | 2304 | 3 | 4 | 17,739,264 |
| vocabulary head, tail only | 5120 | 129280 | 1 | 5 | 413,964,800 |

Down reverses I/N but has the same stored byte count. The 22-layer dense
subset above is 1,787,450,368 bytes; all routed K2 experts for those layers
are 75,118,804,992 bytes. These are serialized weight totals, not an admitted
VRAM budget: add embedding, unchanged projections, KV/Engram state, allocator
and scratch memory, and validate the real model's shapes before placement.

## Bytes and conditional performance model

For I input and N output channels, integer rate K and optional bias:

`bytes = I*N*K/8 + 2*(I+N) + (bias ? 4*N : 0)`.

Each 5120 x 2304 expert matrix at K2 occupies 2,963,968 bytes including its
two scale vectors. Three such matrices occupy 8,891,904 bytes/expert,
3.414 GB/layer for 384 experts, or 136.58 GB for 40 layers. Effective rate
is 2.01007 bpw. These counts exclude allocator alignment and index metadata.
The corresponding current three-matrix payload is 9,953,280 bytes/expert
(two IQ2_XXS gate/up matrices, one Q2_K down matrix), so the conditional byte
ratio is 0.89336. Budgeting must use actual serialized lengths.

Holding residence, overlap and compute constant, 73.72 GB transferred becomes
65.86 GB; 1.54 s exposed wait becomes about 1.376 s. With 0.78 s compute and
0.25 s capture/host tail, 2.57 s becomes about 2.406 s (6.4% lower wall,
6.8% higher request rate). This is a sensitivity calculation from an existing
project measurement, not a measured result. DMA is not synonymous with exposed
wait: increased resident capacity may improve the result; additional H128,
activation conversion, decode cost or changed slot scheduling may erase it.
Use measured pinned-copy rates, not the nominal Gen5 link ceiling.

At a fixed physical resident-byte budget there is a second benefit. For example,
152.882 GB of baseline expert weights minus 81.36 GB copied including preload
implies about 71.522 GB resident. Holding those resident bytes rather than the
resident expert count constant, 136.580 GB of K2 weights leaves about 65.057 GB
to copy, a ratio of 0.800. A crude exposed-wait scaling then predicts 1.23 s
instead of 1.54 s (about 2.26 s total). This ignores slot rounding, which layers
are retained and preload overlap, and gives no credit for additional dense
memory savings. Replan residency and measure the actual schedule; this example
cannot be used as a qualification result.

For a Q8_0 dense tensor, K4's asymptotic byte ratio is 4/8.5 = 0.4706;
for six-bit blocks or mixed baseline tensors compute the ratio tensor by
tensor. The K5 head's ratio to Q8_0 is 0.5882 before scale overhead. No
whole-front speed ratio follows without a byte inventory of its actual overlay.

For a 7 ms front plus 19 ms tail/crossing, a hypothetical 20% front reduction
gives 24.6 ms, 40.65 tok/s versus 38.46; 30% gives 23.9 ms, 41.84 tok/s.
Eliminating the front altogether caps this fixed-tail model at 52.63 tok/s.
These are unqualified estimates. A kernel's theoretical ALU budget does not
establish that decompression is free. Count transform, partial, reread and
launch traffic, and measure the complete linear operator.

## Storage and arithmetic

The compatibility contract is a circular MSB-first stream of 256*K bits per
16x16 tile, in little-endian uint16 containers paired into uint32 words. State
position p is the 16 bits ending at (p+1)*K, wrapping inside that tile. Tiles
are input-major then output-major. For p=8*lane+fragment:

```
i = 2*(lane%4) + fragment%2 + 8*((fragment/2)%2)
n = lane/4 + 8*(fragment/4)
```

The admitted procedural codebook uses unsigned modular product
`p=state*0x83dcd12d`, sums its four unsigned bytes b, and returns
`RN16(((1024+b)*1774 - 1329*2048)/262144)`. The dyadic numerator is exact
in FP32, so a single final RN16 avoids a half intermediate rounding error.
Only integer K1..K8, complete signed FP16 suh/svh, the exact multiplier and
H128-aligned positive dimensions are accepted. Other markers, half rates,
historical sign-only scales and external output scales are unsupported.

Input stages: FP32 x times FP16 suh, RN16, natural Sylvester H128 with FP32
butterflies at strides 1,2,4,8,16,32,64, one normalization by
`0x1.6a09e6p-4f`, RN16. Output stages: sum split planes in ascending order
from +0, H128 in FP32, normalization, multiply by FP16 svh, optional FP32
bias. Output overwrites Y without reading its prior contents. No alpha/beta.
No global expanded Q is allocated by the linear operator.

## Kernels and workspace

1. One 128-thread block per input H128 and real row materializes transformed A.
2. Decode R1..R8: a warp owns N16, decodes eight values per lane for each K16,
   reuses those values across real rows, accumulates two column groups, reduces
   four-lane groups by shuffle-down 2 then 1. Logical K16 split boundaries are
   floor(T*s/S), floor(T*(s+1)/S). Pair arithmetic is two exact half products,
   their FP32 sum, then FP32 accumulator add; no hidden contraction.
3. Prefill: a 128-thread CTA owns M16/N64. It decodes each K16/N16 tile once
   into shared memory, shares transformed A and uses four FP16/FP32 WMMA
   fragments. Row tails are zero-filled; no output row beyond R is written.
4. One output H128 block per row folds S split planes and applies the epilogue.

Caller-owned workspace is `align256(2*R*I) + 4*S*R*N` bytes, independent of K
and mode. Enqueue performs no allocation, synchronization or device query and
supports stream capture. Queries check shapes and overflow without CUDA.
Submission checks declared capacities, pointer arithmetic/alignment and writable
aliasing before launching. The caller must supply live allocations of the
declared capacities on the stream's device and keep them alive through completion.

## Exactness and qualification

Bitstream, procedural half values and both H128 stages have stage-level exact
CPU oracles. GEMV uses the explicit CUDA pair-sum identity above. Its A/P/Y
must match the independent CPU implementation bit for bit at every real row,
including R invariance, split boundaries, padded strides and graph replays.
MMA is a quality path with FP64 split-dot and propagated error checks, never
an inference of cross-vendor bitwise equivalence.

The HIP path's packed dot instruction and reduction choices do not have an
established CUDA bitwise replacement here. Matching H128 and shuffle structure
does not establish packed-dot bits; hardware matrix accumulation also does not
promise the same order. Exact emulation is not proven impossible in principle,
but is unavailable in this implementation. `hip_bitwise_required` is rejected
before enqueue. Consequently the strict split-FRONT identity contract is NOT
met. Do not substitute this kernel into an exact FRONT. The proposed alternative
is an explicitly selected quality-gated frontend with a separate model/state
identity, or future instruction-conformance work followed by an exact pair gate.

Prefill quality requires full-vocabulary KL mean/max at 64 positions on the same
2K/8K/32K corpus and 64–256-token AMD continuation: neither KL statistic may
increase more than 10%, nor top-1 lose more than 1 percentage point against the
frozen base. Compare arithmetic using the SAME compressed tensors first, then
compare quantization quality against the admitted baseline. An earlier CPU
quantization study cannot qualify these CUDA activations or reductions.

Strict FRONT admission requires tokens and complete-logit digests against the
pair decoding the SAME trellis weights, with its production TP/EP split and
expert fold order, including forced rejected R2/R4/R6/R8 rows. Transport-only
changes still require identical state bytes/digests. Full workload, compiler,
native execution and model gates all precede READY or any speed claim.

## References and publication boundary

The trellis/incoherence mathematical background is Albert Tseng, Qingyao Sun,
David Hou and Christopher De Sa, *QTIP: Quantization with Trellises and
Incoherence Processing*, NeurIPS 2024,
[official paper](https://proceedings.neurips.cc/paper_files/paper/2024/file/6de2e84b8da47bb2eb5e2ac96c63d2b0-Paper-Conference.pdf).
The CUDA implementation uses the public NVIDIA WMMA interface and round-to-nearest
half conversions; see [CUDA guide](https://docs.nvidia.com/cuda/cuda-c-programming-guide/)
and [CUDA half conversions](https://docs.nvidia.com/cuda/cuda-math-api/cuda_math_api/group__CUDA__MATH____HALF__MISC.html).
No external implementation source is incorporated. Private format inspection
records, source identities, fixture/model provenance and comparative material
are archived outside Git. Actual dependency/model licenses remain applicable.
