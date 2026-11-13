# Vision tower adapters

Image input is built once in shared code. A model family contributes one *adapter*: a small class that states
what is specific to its tower and runs nothing itself.

| Layer | Shared home | What the adapter supplies |
|---|---|---|
| Decode, resize, normalise, patchify | `include/strix/vision/{image,preprocess}.hpp` | resize rule, resampler, mean and standard deviation, patch size and row order |
| Weights | `include/strix/vision/gguf_weights.hpp` (`GgufWeights`, any GGUF tower) | tensor names (`bind`) |
| Tower | `include/strix/vision/encoder.hpp` (`ReferenceEncoder` on the host, `Encoder` on the device) | `EncoderConfig` |
| Projector | `DenseOps::linear` of the backend in use | which rows are merged, which linears run, the special rows |
| Rows in the prompt | `include/strix/session/embed_rows.hpp` | `SpanLayout`: one role and position per row, the token id the rows count as |
| Cache | `include/strix/vision/embed_cache.hpp` | `identity()` |

The interface is `strix::vision::VisionAdapter` (`include/strix/vision/adapter.hpp`). Shared code names no family.

## Writing an adapter

1. **Validate the file.** A strict validator in the family's model source (metadata keys and types, value ranges,
   exact tensor inventory), as `deepseek41::validate_vision` does. Admission by digest stays where the family
   already pins its files.
2. **`config()`**: fill `EncoderConfig` from the validated metadata. Check `vision::hip_supported(config)`.
3. **`bind()`**: map tensor names to `EncoderWeights` through a `WeightSource`. `GgufWeights` returns a tensor
   with GGUF dimensions `(d0, ..., dn)` as the matrix `[dn][d0 * ... * d(n-1)]`, which is the source framework's
   `[out][in...]` weight; a flattened convolution weight therefore has features in (channel, frame, y, x) order,
   the order `vision::patchify` emits. Open the file with a `vision::HostBudget`: the file length is checked
   against `max_bytes` (and against `exact_bytes`, the pinned length, when the family pins its file) before a byte
   is allocated, then the bounded header is validated, then the bytes are read.
4. **`patches()`**: `ResizePolicy::target`, `vision::resize`, `vision::normalize`, `vision::patchify`, then the
   patch coordinates. Refuse an image the rule cannot represent with `std::invalid_argument`. Call
   `vision::check_grid_axes` before the resample: a patch coordinate is 16 bits, so an axis is at most 65,536
   patches whatever the token limit allows.
5. **`project()`**: receives the tower output and the grid it came from. Build the projector input (merge,
   unshuffle, pool), call `dense.linear` for every linear so the host reference and the device share the graph,
   and write the rows in span order, special rows included. The grid is a caller's argument: start with
   `vision::checked_merged_grid(grid, merge, max_tokens)` (64-bit products, positive axes, `rows` equal to the
   product, the token limit) and size every buffer from its result. `layout()` does the same.
6. **`layout()`**: the same span as roles and positions. Row count and order must equal what `project()` writes.
7. **`device_options()`** (optional): a policy bit of the device arithmetic the tower needs, found by the gate.
8. **`identity()`**: the weights digest plus everything that changes the rows. Cached rows and prefix entries
   are keyed by it, so a changed parameter can never reuse old rows. State a float parameter by its bit
   pattern, not by a rounded decimal, and refuse a non-finite one.
9. **Tests**: a host test on a small synthetic tower with the real tensor names (projector against a direct
   computation, layout, resize table, cache key, refusal of malformed grids and files), and a gate on the real
   file. Gates compare tensors with `vision::abs_diff` (`include/strix/vision/compare.hpp`): a NaN or an infinity
   on either side is a failure, it never disappears in a maximum.

A family that needs something the interface does not carry extends the shared interface; it does not work around
it inside the adapter.

Open before a tower is served: the cache key ends with `EncoderBackend::arithmetic()`, one constant string per
backend class. It does not state the options a device backend was built with (`EncoderOptions`, for example split
activations), so two device backends with different options share keys today. The effective options have to become
part of that string, and the backend has to be built from `device_options()` in one place, before any served path
shares a cache between differently built backends.

## DeepSeek V4.1

`src/deepseek41/vision_adapter.{hpp,cpp}`, validator `deepseek41::validate_vision`, host test
`deepseek41-vision`, gate tool `strix-deepseek41-vision-gate` with the reference dumps of
`tools/vision/ds41_ref.py`.

Read from the file: 32 blocks, width 1024, 16 heads, RMSNorm (weight only, epsilon 1e-6), fused `wqkv` and `wo`
with bias, gated SwiGLU 2816 without bias, 2D rotary positions with base 10000, patch 14, a final norm, a 3x3
merge into `aligner.w1` (9216 to 5120) and `aligner.w2` (5120 to 5120), three learned rows (`image_start`,
`image_newline`, `image_end`), at most 1024 image rows, at least 295936 pixels, image token id 129264.

Measured on the tensors: the patch weight is laid out (channel, y, x) (neighbouring weights correlate along
those axes and not along the alternative); the aligner input is channel-major, index `c * 9 + slot` (the column
norms follow a 3 x 3 pattern in the slot and are flat otherwise).

**Not stated by the file** and not yet compared with the upstream processor and model code. Each is a field of
`deepseek41::VisionRecipe`, part of the adapter identity, and an option of the reference tool:

| Parameter | Value used | Basis |
|---|---|---|
| Resize rule | both sides to multiples of 42, area within [295936, 1024 x 42 x 42], aspect kept | the file's limits; the rounding rule is assumed |
| Resampler | bicubic with antialias | assumed |
| Mean, standard deviation | 0.5, 0.5 per channel | assumed |
| SwiGLU halves | first half of `mlp.w1` is the gate | assumed |
| Rotary layout | row angles then column angles, pairs `(i, i + 32)` | assumed |
| Aligner activation | GELU | assumed |
| Slot order in a 3 x 3 cell | `dy * 3 + dx` | assumed (channel-major is measured) |
| Span | start, each grid row followed by the row end, end | assumed |

The gate that compares these rows with the upstream implementation is still owed; until it has run, image
input for this family is not advertised by any service.

### Gate

```
python3 tools/vision/ds41_ref.py images --out REF
python3 tools/vision/ds41_ref.py preprocess --out REF
python3 tools/vision/ds41_ref.py encode --out REF --model VISION.gguf
strix-deepseek41-vision-gate --model VISION.gguf --ref REF --backend host --only rings_graph
strix-deepseek41-vision-gate-hip --model VISION.gguf --ref REF --backend hip
```

`encode` is an independent FP32 implementation in numpy, not the upstream code: it proves the wiring of the
shared tower and of the adapter, and it is the FP32 reference the device arithmetic (`vision-f16-v1`, class
FAST) is judged against. Thresholds: preprocessing max difference 1e-6 on the patch tensor; host backend
cosine at least 0.99999 per row; device backend mean cosine at least 0.9995, minimum 0.998, relative L2 at most
2 %; the special rows bit-equal to the file.

### Gate status on the pinned file

| Check | Result |
|---|---|
| Admission and strict validation | pass (266 tensors) |
| Preprocessing against Pillow, 6 images | patch tensor bit-equal |
| Host FP32 tower + adapter against the numpy implementation | cosine 1.000000, relative L2 0.00000 (rows differ only by their BF16 storage) |
| Device arithmetic `vision-f16-v1`, 7 images | single FP16 operands: 4 pass, 3 fail the row minimum; split activations: 1 of the 5 run fails it |
| Upstream processor and model | not run |

The device failures are on images with large flat areas (a smooth gradient, an upscaled 16 x 16 image, smooth
rings): the mean cosine stays at or above 0.998 while single rows fall far below it (row minimum 0.925, 0.996 and
0.978). `--diagnose` of the gate tool compares the residual stream after selected blocks with the host reference:

- Through block 11 the error is small and uniform, but already 30 times larger without split activations
  (relative L2 0.013 against 0.0003): on a flat patch the linears subtract large, nearly equal terms, so the
  rounding of a single FP16 operand is a large fraction of the result.
- Between blocks 12 and 15 the tower forms a few rows of very large norm (about 1000 against a median of 7).
  Which rows those become is decided by small differences between nearly identical patches, so the rounding
  error flips the choice: 69 of 1521 rows leave the reference at block 15 without split activations, 1 with them.

The error is therefore clustered in rows, input dependent and a precision effect, not a wiring fault. The adapter
requests split activations (`device_options()`); with them the gradient and the upscaled image pass and the rings
image misses the minimum on one row (0.9969; mean 0.99999, relative L2 0.3 %). The device arithmetic of this
tower is **not accepted yet**: the remaining single-row flips need either more precision in the first blocks
and in the attention operands, or a judgement on the language model's output instead of a per-row minimum.

Measured tower time on one node (warm, no per-layer synchronisation): 111 ms for 1,521 patches (the minimum
image), 388 ms for 3,888, 1,907 ms for 9,216 (the token cap); about 2.1 x, 1.8 x and 1.4 x that with split
activations. The aligner adds 22 to 83 ms.
