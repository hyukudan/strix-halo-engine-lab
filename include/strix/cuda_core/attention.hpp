#pragma once
#include "strix/attn/fp8_plan.hpp"
#include <cuda_bf16.h>

#include <cstddef>
#include <cstdint>
#include <span>

#include "strix/attn/prepared_lanes.hpp"
#include "strix/cuda_core/types.hpp"
#include "strix/cuda_core/attention_policy.hpp"  // FlashPipe, FlashTail

namespace strix::cuda_core {

struct AttentionDims {
    std::uint32_t heads = 0, kv_heads = 0, head_dim = 256;  // GQA: q head h reads kv head h / (heads / kv_heads)
    float scale = 1.0f;
    std::uint32_t max_positions = 0;  // KV cache capacity per kv head
};

// KV cache BF16, layout [kv_head][max_positions][head_dim] (one buffer each for K and V).
// pos0 is a DEVICE scalar (graph capture: the step's start position changes without re-capture).
// Stores rows k/v ([rows][kv_heads][head_dim] FP32, contiguous) at positions pos0 + r, rounding RNE to BF16.
cudaError_t kv_store(const AttentionDims& d, const float* k, const float* v, __nv_bfloat16* k_cache,
                     __nv_bfloat16* v_cache, const std::uint32_t* pos0_dev, std::uint32_t rows, cudaStream_t stream, const RowMap& rm = {});

// Token-tree commit (strix/spec/tree_window.hpp): copies the K and V row at index `from` to index `to` (every KV head) in each of
// the `layers` single-session caches (bits copied, no conversion). One launch for all layers.
constexpr std::uint32_t kMaxKvMoveLayers = 64;
struct KvLayerCaches {
    __nv_bfloat16* k[kMaxKvMoveLayers];
    __nv_bfloat16* v[kMaxKvMoveLayers];
};
cudaError_t kv_move_row(const AttentionDims& d, const KvLayerCaches& caches, std::uint32_t layers, std::uint32_t from, std::uint32_t to,
                        cudaStream_t stream);

constexpr std::uint32_t kAttnChunk = 256;  // positions per reduction chunk
[[nodiscard]] __host__ __device__ inline std::uint32_t attn_max_chunks(const AttentionDims& d) { return (d.max_positions + kAttnChunk - 1) / kAttnChunk; }

// Floats of scratch attention_decode needs for `rows`: scores + chunk max + chunk denominators + chunk value partials.
inline std::size_t attention_scratch_floats(const AttentionDims& d, std::uint32_t rows) {
    const std::size_t rh = static_cast<std::size_t>(rows) * d.heads, mc = attn_max_chunks(d);
    return rh * d.max_positions + 2 * rh * mc + rh * mc * d.head_dim;
}

// Row r (position pos0 + r) attends to cache positions 0 .. pos0 + r (causal inside the window; the window's own K/V
// must have been stored first). q: FP32 view (rows x heads x head_dim). gate (optional, same shape, its own strides):
// out *= sigmoid(gate). ORDER per (row, head), positions split into 256-position chunks by absolute position:
// score_p = fl(dot) * scale, dot = lane owns 8 consecutive dims sequentially then xor butterfly 16..1; m = max over all positions
// (exact); e_p = expf(s_p - m); chunk denominator = block tree (thread t holds position chunk*256 + t, zero past the length) of norms.hpp;
// chunk value partial_d = sum over the chunk positions ascending of fl(e_p * v[p][d]); denom = chunk denominators added ascending,
// acc = chunk partials added ascending, out = acc / denom. Independent of rows and launch shape. head_dim 256, group <= 8.
cudaError_t attention_decode(const AttentionDims& d, const float* q, const GroupedView& qv, const float* gate,
                             const GroupedView& gv, const __nv_bfloat16* k_cache, const __nv_bfloat16* v_cache,
                             const std::uint32_t* pos0_dev, std::uint32_t rows, float* out /*[rows][heads][head_dim]*/,
                             float* scratch, cudaStream_t stream, const RowMap& rm = {});

// Single-row, linear decode. Each phase enqueues one physical launch of attention_decode.
// Enqueue scores, values and combine in order with the same scratch buffer.
enum class AttentionDecodePhase : std::uint8_t { scores, values, combine };
cudaError_t attention_decode_row_phase(const AttentionDims& d, const float* q, const GroupedView& qv, const float* gate,
                                       const GroupedView& gv, const __nv_bfloat16* k_cache, const __nv_bfloat16* v_cache,
                                       const std::uint32_t* pos0_dev, float* out, float* scratch, AttentionDecodePhase phase,
                                       cudaStream_t stream);

// PREFILL attention (FAST class, NOT bitwise equal to attention_decode): causal flash attention over the BF16 KV cache for a chunk of
// `rows` query rows at positions pos0 .. pos0 + rows - 1 (the chunk's own K/V must already be stored by kv_store). Q is rounded to bf16,
// scores and the online softmax are FP32, P is rounded to bf16 for the PV mma (FP32 accumulate); out = o * sigmoid(gate) like
// attention_decode. head_dim 256, GQA group <= 8. q/gate/out use the same grouped views as attention_decode.
// FlashPipe = shared-memory pipeline of the K/V tiles only (scheduling): every pipe gives the SAME bits (same tiles, same order per row).
//   pad2: padded rows, 2 stages, two barriers per tile (the original form, kept as the off switch);
//   swz3: XOR-swizzled dense rows, 3 stages (prefetch distance 2), one barrier per tile.
// FlashTail = wave-tail split (scheduling only, SAME bits): with T = ceil(rows/128) * heads CTAs on S SMs (1 CTA/SM), the last
// T mod S tiles form a partial wave that bills a whole CTA time. split: those tiles run as jobs of w active warps (w = 1/2/4, the
// smallest with (T mod S) * 8 / w <= S, or tail_warps if non-zero), so the partial wave spreads over the idle SMs. Every warp keeps its
// 16 rows, its head and the tile count of its 128-row block, so each output row runs the same instructions on the same tiles.
cudaError_t attention_prefill_flash(const AttentionDims& d, const float* q, const GroupedView& qv, const float* gate, const GroupedView& gv,
                                    const __nv_bfloat16* k_cache, const __nv_bfloat16* v_cache, std::uint32_t pos0, std::uint32_t rows,
                                    float* out, cudaStream_t stream, FlashPipe pipe = FlashPipe::pad2, FlashTail tail = FlashTail::off,
                                    std::uint32_t tail_warps = 0);

struct PreparedBf16Flash {
    const float* relative_bias = nullptr;  // [query row][query head][backward distance], natural units.
    const float* sinks = nullptr;          // [query row][query head], natural units, zero values.
    std::uint32_t relative_extent = 0;
    std::uint32_t window = 0;  // Includes the current row. Zero means a full causal prefix.
    bool wrapped = false;     // Cache slot = absolute position % max_positions.
    bool bidirectional = false;  // Entire block plus window preceding pos0; no bias or sinks.
};

// Cache append for the prepared slot-major layout. K/V inputs are contiguous
// [row][kv_head][dim] FP32 and cache values are rounded to BF16 nearest-even.
// In wrapped mode, the caller reserves the live window tail as well as these rows.
cudaError_t kv_store_prepared_bf16(const AttentionDims& d, const float* k, const float* v,
                                   __nv_bfloat16* k_cache, __nv_bfloat16* v_cache, std::uint32_t pos0,
                                   std::uint32_t rows, bool wrapped, cudaStream_t stream,
                                   const std::uint32_t* device_position = nullptr);

// Validate the prepared binding before enqueuing any cache writes.
[[nodiscard]] bool can_launch_prepared_bf16_flash(const AttentionDims& d, const PreparedBf16Flash& prepared,
                                                  const float* q, const __nv_bfloat16* k_cache,
                                                  const __nv_bfloat16* v_cache, std::uint32_t pos0,
                                                  std::uint32_t rows, float* out, FlashPipe pipe);

// D128 prepared GQA binding of the shared BF16 flash core, for contiguous Q/output
// [row][head][dim] and BF16 K/V [slot][kv_head][dim]. Store the chunk's own K/V
// before launch. A wrapped cache must fit the whole live prefix of the chunk.
// FP32 online softmax and BF16 PV operands. Relative bias is added after QK scaling.
cudaError_t attention_prefill_prepared_bf16(const AttentionDims& d, const PreparedBf16Flash& prepared,
                                            const float* q, const __nv_bfloat16* k_cache, const __nv_bfloat16* v_cache,
                                            std::uint32_t pos0, std::uint32_t rows, float* out, cudaStream_t stream,
                                            FlashPipe pipe = FlashPipe::pad2,
                                            const std::uint32_t* device_position = nullptr);

// DECODE schedule of the prepared D128 GQA binding (FAST class, NOT bitwise equal to attention_prefill_prepared_bf16):
// one row at the DEVICE position *device_position, same operands, mask, relative bias and sink as the prepared flash.
// Keys are cut by strix/attention_split_plan.hpp (chunks counted from the first visible key; the plan depends only on
// (position, window)). Grid (kv_heads, decode_split_capacity): one CTA = one KV head, its heads / kv_heads <= 4 query
// heads (K/V read once per group) and one chunk; Q, scores, softmax and P V in FP32 (BF16 K/V widened), scores one key per
// thread, P V per (dimension pair, key quarter q: pass keys q, q + 4, ... ascending), quarters added in order, block
// reductions in fixed order. The last CTA of a KV head (device ticket, which it resets)
// merges the chunk partials in ascending chunk order after the sink term: deterministic, independent of the device and
// of CTA arrival. workspace: decode_split_workspace_bytes(d, window) bytes, 16-byte aligned; its ticket words must be
// zero before the first launch (decode_split_workspace_init) and are zero again after every completed launch.
[[nodiscard]] std::size_t decode_split_workspace_bytes(const AttentionDims& d, std::uint32_t window);
cudaError_t decode_split_workspace_init(void* workspace, const AttentionDims& d, std::uint32_t window, cudaStream_t stream);
[[nodiscard]] bool can_launch_prepared_decode_split(const AttentionDims& d, const PreparedBf16Flash& prepared,
                                                    const float* q, const __nv_bfloat16* k_cache,
                                                    const __nv_bfloat16* v_cache, const std::uint32_t* device_position,
                                                    float* out, const void* workspace, std::size_t workspace_bytes);
cudaError_t attention_decode_prepared_split_bf16(const AttentionDims& d, const PreparedBf16Flash& prepared,
                                                 const float* q, const __nv_bfloat16* k_cache,
                                                 const __nv_bfloat16* v_cache, const std::uint32_t* device_position,
                                                 float* out, void* workspace, std::size_t workspace_bytes,
                                                 cudaStream_t stream);

// LANES form of the decode schedule above (strix/attn/prepared_lanes.hpp): lane i is one query row at its own HOST
// position lanes[i].first_position over its own K / V planes (cache_slots, wrapped); q, out and the relative bias and
// sink images hold lane i at packed row i. Lane i runs the one-row body with its own plan
// (attn_split::decode_split_plan(position, window)), its own ticket words and record area: its bits are those of the
// one-row launch of that row (cohort / verify-width invariant). Attention only: lanes may share one cache (verify rows
// of one sequence). Grid capacity = decode_split_capacity(d.max_positions, window); a lane whose plan needs more chunks
// is refused (attn_split::lane_fits); a non-wrapped lane needs position < cache_slots. One launch per 16 lanes, in stream
// order. workspace: decode_split_lanes_workspace_bytes(d, window), 16-byte aligned, tickets zeroed once by
// decode_split_lanes_workspace_init and zero again after every completed launch.
[[nodiscard]] std::size_t decode_split_lanes_workspace_bytes(const AttentionDims& d, std::uint32_t window);
cudaError_t decode_split_lanes_workspace_init(void* workspace, const AttentionDims& d, std::uint32_t window,
                                              cudaStream_t stream);
[[nodiscard]] bool can_launch_prepared_decode_split_lanes(const AttentionDims& d, const PreparedBf16Flash& prepared,
                                                          const float* q,
                                                          std::span<const attn::lanes::PreparedBf16Lane> lanes,
                                                          const float* out, const void* workspace,
                                                          std::size_t workspace_bytes);
cudaError_t attention_decode_prepared_split_lanes_bf16(const AttentionDims& d, const PreparedBf16Flash& prepared,
                                                       const float* q,
                                                       std::span<const attn::lanes::PreparedBf16Lane> lanes, float* out,
                                                       void* workspace, std::size_t workspace_bytes,
                                                       cudaStream_t stream);

// Append then attend on the same stream. If supplied, failure is a device flag
// OR-ed with 1 for invalid operands or results before the stream completes.
cudaError_t append_attention_prepared_bf16(const AttentionDims& d, const PreparedBf16Flash& prepared,
                                           const float* q, const float* k, const float* v,
                                           __nv_bfloat16* k_cache, __nv_bfloat16* v_cache, std::uint32_t pos0,
                                           std::uint32_t rows, float* out, cudaStream_t stream,
                                           FlashPipe pipe = FlashPipe::pad2, unsigned* failure = nullptr,
                                           const std::uint32_t* device_position = nullptr);

// Decode lanes (strix/attn/prepared_lanes.hpp, the backend-neutral lane contract): lane l is packed
// row l of these images, with its own cache and position. Row widths: queries/output heads x dim,
// keys/values kv_heads x dim, relative logits heads x extent, sinks heads.
struct PreparedBf16LaneImages {
    const float* q = nullptr;
    const float* k = nullptr;
    const float* v = nullptr;
    const float* relative_bias = nullptr;
    const float* sinks = nullptr;
    float* out = nullptr;
};
// Lane-batched prepared append + attend: for every lane, the result of append_attention_prepared_bf16
// on that lane's row alone (rows = 1 at its first_position, its cache_slots and wrapped), issued as one
// cache store and one flash grid (second dimension = lane) per launch block, then one finite check over
// the packed rows. Each flash block runs the unchanged prepared flash body with exactly its lane's
// arguments, so a lane's bits do not depend on the other lanes (EXACT, batch invariant). `shape`
// supplies window and relative extent (its bias, sinks and wrapped are per lane); d.max_positions is
// ignored (per lane). Refused: bidirectional shapes, shared caches, any lane the one-lane call refuses;
// every lane is admitted before the first store.
[[nodiscard]] bool can_launch_prepared_bf16_lanes(const AttentionDims& d, const PreparedBf16Flash& shape,
                                                  const PreparedBf16LaneImages& images,
                                                  std::span<const attn::lanes::PreparedBf16Lane> lanes,
                                                  FlashPipe pipe);
cudaError_t append_attention_prepared_bf16_lanes(const AttentionDims& d, const PreparedBf16Flash& shape,
                                                 const PreparedBf16LaneImages& images,
                                                 std::span<const attn::lanes::PreparedBf16Lane> lanes,
                                                 cudaStream_t stream, FlashPipe pipe = FlashPipe::pad2,
                                                 unsigned* failure = nullptr);
// Launch plan of attention_prefill_flash (host arithmetic; exposed for tests and the effective-config print).
struct FlashPlan {
    std::uint32_t tiles = 0, full = 0, tail_tiles = 0, tail_warps = 8, jobs = 0;
};
FlashPlan flash_plan(std::uint32_t rows, std::uint32_t heads, std::uint32_t sms, FlashTail tail, std::uint32_t tail_warps = 0);
// Launch geometry of attention_prefill_flash for the wave planner (strix/wave_fit.hpp): q rows per CTA, CTAs per q-row tile
// (one per q head), SMs, resident CTAs per SM from the occupancy of the real kernel. Keep it in step with the kernel.
struct FlashGrid {
    std::uint32_t tile_rows = 0, ctas_per_tile = 0, sms = 0, ctas_per_sm = 0;
};
cudaError_t attention_prefill_flash_grid(const AttentionDims& d, FlashPipe pipe, int device, FlashGrid* out);
// FAST8 prefill attention (bit-changing; under the FAST8 prefill policy and its quality gate only). An e4m3 SHADOW of the BF16 KV cache
// (the BF16 cache stays the source of truth for decode and for every other path): per kv head K8 [P8][head_dim] e4m3 + ks [P8] FP32
// (k = K8 * ks, ks = amax / 448 per position), V8T [head_dim][P8] e4m3 transposed + vs [P8] (v = V8T * vs); P8 = max_positions rounded up
// to kAttn8Align. Allocate kv_shadow_e4m3_bytes and zero it once (positions never written must hold finite values).
constexpr std::uint32_t kAttn8Align = attn::fp8::capacity_alignment;
struct KvShadowE4m3 {
    std::uint8_t* k8 = nullptr;   // [kv_heads][P8][head_dim]
    float* ks = nullptr;          // [kv_heads][P8]
    std::uint8_t* v8t = nullptr;  // [kv_heads][head_dim][P8], mma key order inside 32-key groups
    float* vs = nullptr;          // [kv_heads][P8]
};
[[nodiscard]] std::uint32_t attn8_capacity(const AttentionDims& d);
[[nodiscard]] std::size_t kv_shadow_e4m3_bytes(const AttentionDims& d);  // k8 + ks + v8t + vs
// Re-quantises cache positions pos_begin .. pos_begin + count - 1 (all kv heads) from the BF16 cache into the shadow.
cudaError_t kv_shadow_e4m3(const AttentionDims& d, const __nv_bfloat16* k_cache, const __nv_bfloat16* v_cache, const KvShadowE4m3& sh,
                           std::uint32_t pos_begin, std::uint32_t count, cudaStream_t stream);
// Same contract as attention_prefill_flash (rows at pos0 .., causal, gated output) with Q quantised per (row, head) to e4m3, S = Q8 K8^T
// and O += P'' V8 on e4m3 tensor cores (FP32 accumulate), online softmax FP32. Keys 0 .. pos0 + rows - 1 must be in the shadow.
cudaError_t attention_prefill_flash8(const AttentionDims& d, const float* q, const GroupedView& qv, const float* gate, const GroupedView& gv,
                                     const KvShadowE4m3& sh, std::uint32_t pos0, std::uint32_t rows, float* out, cudaStream_t stream);
// FAST8 DECODE / VERIFY attention (bit-changing vs attention_decode; class "d8", own identity and quality gate; never under an exact
// profile). Same contract as attention_decode (rows at device positions *pos0_dev + r or rm.pos[r], ascending; row r attends keys
// 0 .. its position; gated output [rows][heads][head_dim]) over the e4m3 shadow, with the flash8 arithmetic (Q8 per (row, head),
// S and P''V on e4m3 mma, FP32 online softmax). Work item = (kAttn8Chunk-key chunk, kv head, 128 (row, q head) pairs): the q heads
// of a kv head and all rows share each K/V tile read. Per (row, head) the bits depend only on that row's q and the shadow keys
// 0 .. its position (absolute chunks, per-row tile skip and per-row P'' scale), so verify of R rows == R one-row calls and the
// launch shape never matters. Keys 0 .. last row's position must be in the shadow (kv_shadow_e4m3 / kv_shadow_e4m3_rows).
// rm.slot is not supported (the shadow has no slot dimension): returns cudaErrorNotSupported. Scratch: attention_decode8_scratch_floats.
constexpr std::uint32_t kAttn8Chunk = attn::fp8::split_keys;
[[nodiscard]] inline std::size_t attn8_max_chunks(const AttentionDims& d) { return (static_cast<std::size_t>(d.max_positions) + kAttn8Chunk - 1) / kAttn8Chunk; }
[[nodiscard]] std::size_t attention_decode8_scratch_floats(const AttentionDims& d, std::uint32_t rows);
// Quantises the shadow at the positions of `rows` decode rows (same bytes as kv_shadow_e4m3 for those positions); call after kv_store.
cudaError_t kv_shadow_e4m3_rows(const AttentionDims& d, const __nv_bfloat16* k_cache, const __nv_bfloat16* v_cache, const KvShadowE4m3& sh,
                                const std::uint32_t* pos0_dev, std::uint32_t rows, cudaStream_t stream, const RowMap& rm = {});
cudaError_t attention_decode8(const AttentionDims& d, const float* q, const GroupedView& qv, const float* gate, const GroupedView& gv,
                              const KvShadowE4m3& sh, const std::uint32_t* pos0_dev, std::uint32_t rows, float* out, float* scratch,
                              cudaStream_t stream, const RowMap& rm = {});
// Decode / verify attention of one layer with its K/V READ FORM as a trait: kv8 == nullptr -> the exact BF16 walk
// (attention_decode, row-shared plan); kv8 != nullptr -> the FAST e4m3 form (the step rows are quantised into the shadow,
// then attention_decode8). The BF16 cache stays the source of truth in both forms (kv_store before this call). Scratch:
// max(attention_scratch_floats, attention_decode8_scratch_floats). Engines bind this one entry, never the two bodies.
inline cudaError_t attention_decode_kv(const AttentionDims& d, const float* q, const GroupedView& qv, const float* gate,
                                       const GroupedView& gv, const __nv_bfloat16* k_cache, const __nv_bfloat16* v_cache,
                                       const KvShadowE4m3* kv8, const std::uint32_t* pos0_dev, std::uint32_t rows, float* out,
                                       float* scratch, cudaStream_t stream, const RowMap& rm = {}) {
    if (kv8 == nullptr) return attention_decode(d, q, qv, gate, gv, k_cache, v_cache, pos0_dev, rows, out, scratch, stream, rm);
    if (rm.self != nullptr) return cudaErrorNotSupported;  // the e4m3 shadow is indexed by position: no token-tree leaf K/V index
    if (const cudaError_t e = kv_shadow_e4m3_rows(d, k_cache, v_cache, *kv8, pos0_dev, rows, stream, rm); e != cudaSuccess) return e;
    return attention_decode8(d, q, qv, gate, gv, *kv8, pos0_dev, rows, out, scratch, stream, rm);
}
// p[i] = base + step * i for i < n (small device helper used to address attention windows).
cudaError_t iota_u32(std::uint32_t* p, std::uint32_t n, std::uint32_t base, std::uint32_t step, cudaStream_t stream);

}  // namespace strix::cuda_core
