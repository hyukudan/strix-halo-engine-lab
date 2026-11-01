# Compact rounded stage images

The full-T QSA owner stores its BF16-rounded attention output in a BF16 image
before the output projection. This exact storage optimization defaults on for
eligible score-ring owners with 256-byte alignment. Set
`STRIX_QPF2K_COMPACT_IMAGES=0` before startup to retain FP32 containers while
preserving the other memory policies. The owner can also explicitly retain
legacy images with `KnownPrefillRequest::compact_stage_images=false`.

The producer still computes every expression and rounding in the preceding
order. The store extracts the upper 16 bits of the **already rounded** result;
the reader reconstructs the FP32 word with an integer left shift. No hardware
BF16 cast or second rounding replaces the existing integer rounding. Both zero
signs, subnormals, infinities and every NaN sign/payload therefore retain the
producer's bits, including the integer rounding's exceptional wrap behavior.
Finite checks and the output projection's FP16 overflow counting remain in
place. Native local CI passed the host/HIP fixtures and found 32K last-row logits
bitwise equal with compact images OFF.

The active prefill option set, its identity hash, rotary admission and
`prefill_kernel_plan_identity` are unchanged. The physical arena and
`prefill_execution_identity` change to describe the smaller allocation. Saved
states and prefix-cache numerical identities do not gain a new option.

| Edge | Representation and reader audit |
| --- | --- |
| GDN qkv | Retained FP32. Convolution writes BF16-rounded values, but Q/K L2 normalization rewrites them with unrounded FP32 products. Recurrence, fused conv/core routes, vector loads, convolution history and state writers share that in-place image. Compacting the entire image would lose those normalized bits. |
| GDN z | Retained FP32 at the current compatibility boundary. Scalar gated RMS norm, window/recurrent norm, strided prefill norm and fused core/norm readers, window projection sites and MTP workspace descriptors retain their FP32 ABI. No compact z allocation or producer is admitted. |
| QSA raw q/gate/k/v | Retained FP32. Projection sites, in-place norm/RoPE (including failure-before-store paths), range validators, KV quantizers/cache writers, pooling views, vectorized fast-attention query loads and shared decode/MTP descriptors retain their FP32 ABI. No raw compact producer is admitted. |
| QSA attention output | Compacted for full-T owners only. Ordinary value/gate, grouped and prefetched output, fused tail/quantizer, fast multi/ordered/occupancy, FP16 shadow, split/combine and int8 paths use the same rounded image view. Tiled and row output projections widen BF16 before their existing arithmetic; scalar projection fallbacks do the same. |
| Observers | The attention self-check reconstructs FP32 bytes on the host. Trace hashing reconstructs the same FP32 words on the device. Aliasing, alignment and extent validation price the actual element size. |
| Decode, MTP, closed windows, capture, segmented owners | Retain their preceding FP32 allocation and descriptor contracts. They do not borrow a compact full-T owner's image. Optional-entry callers retain FP32. The compact owner requires the existing full-T QSA entry promise. |

The QSA output image is 6,144 elements per row: 24,576 bytes becomes 12,288
bytes. This removes 100,663,296 bytes (96 MiB) from its T8192 workspace stripe;
retained tails are priced separately. A 32,768-token prompt across twelve QSA
layers saves `32768 * 12 * 6144 * 2 = 4,831,838,208` bytes of producer stores
and the same logical bytes on one consumer read, totaling 9,663,676,416 bytes.
This is calculated image traffic, not measured memory-bus traffic or latency.
Native local-CI interleaved 32K A/B was neutral within noise (OFF 19.95 s, n=2,
versus default 20.09 s); no speed gain claimed. The memory saving is calculated.
The image is reused by layers; these are not twelve simultaneous allocations.

CPU tests exhaust all 65,536 BF16 words and boundary tails of the incumbent
rounding, parse the off switch, check compact extents and stripe boundaries,
and check unchanged numerical identity versus the FP32-image plan. Full CTest
also runs the compiled-plan checks with the environment switch off.

`strix-hip-compact-stage-images-tests` directly launches production producers
and typed projection readers at T8192 and a 40-row tail. It compares expanded
image bytes, projected BF16 bytes, nonfinite status, finite FP16 overflow counts
and allocation guards. It covers multi, ordered, occupancy, shadow, split,
ordinary/grouped/prefetched/tail, int8 and special-value paths, with tiled FP16,
row FP16, wave8, scalar FP32 and dense-A8 projection readers. It must be run
natively; a skipped test is not an exactness pass. Complete default and exact
FP16 identity workloads must additionally match the preceding implementation's
logits, full generated token IDs, convolution/recurrent state and KV/index
state byte-for-byte, with all other settings fixed, to establish that broader
qualification. The native comparison of the seven state payloads remains open
and nonblocking; local-CI fixture and last-row-logit checks do not establish it.
See [bundle local CI](hc-output-wmma-qualification.md#bundle-local-ci-evidence).
