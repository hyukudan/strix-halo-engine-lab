// Prefill attention (FAST class): causal flash attention, head_dim 256, GQA, BF16 KV cache, tensor cores (mma.sync m16n8k16 bf16, FP32
// accumulate). See attention.hpp attention_prefill_flash for the contract (NOT bitwise equal to attention_decode).
//
// CTA = 128 query rows x 1 query head, 8 warps of 16 query rows. Key tiles of 32 keys through a 2-stage cp.async pipeline. Q lives in
// registers as bf16 A fragments (the head-dimension k permutation: k step ks, quad lane q holds d = 16 ks + 4 q .. +3, the same
// permutation on the K side, so a thread reads its 4 K values with one LDS.64). S = Q K^T; online softmax in FP32 (exp2); P (bf16) feeds
// O += P V with V B-fragments from ldmatrix.trans. K rows are padded to 544 B (conflict-free LDS.64), V rows to 528 B (conflict-free ldmatrix).
#include "attn_decode.cuh"
#include "attn_lanes.cuh"

#include "common.cuh"
#include "attn_core/masks.hpp"
#include "attn_core/prepared_step.hpp"
#include "attn_core/relative_bias.hpp"
#include "core/row_transforms.hpp"

#include <cmath>
#include <limits>

namespace strix::cuda_core {
namespace {

constexpr int QB = 128, KT = 32;
constexpr auto step_position_limit = std::numeric_limits<unsigned>::max() - QB - KT;

// Shared-memory layouts of one K/V tile stage (byte offsets inside a stage). Both give conflict-free K LDS.64 and V ldmatrix;
// the arithmetic per row is identical (same tiles, same order), so every layout produces the same bits.
template <int Dim>
struct PadLayoutShape {  // FlashPipe::pad2: two stages, prefetch distance 1, two barriers per tile.
    static constexpr int D = Dim, KB = 2 * D + 32, VB = 2 * D + 16;
    static constexpr int STAGES = 2, K_TILE = KT * KB, STAGE = K_TILE + KT * VB;
    __device__ static std::uint32_t k_chunk(std::uint32_t r, std::uint32_t ch) { return r * KB + ch * 16; }
    __device__ static std::uint32_t k_read(std::uint32_t r, int ks, int q4) { return r * KB + 32 * ks + 8 * q4; }
    __device__ static std::uint32_t v_chunk(std::uint32_t key, std::uint32_t ch) { return K_TILE + key * VB + ch * 16; }
};
template <int Dim>
struct SwzLayoutShape {  // FlashPipe::swz3: XOR-swizzled, three stages, prefetch distance 2, one barrier per tile.
    static constexpr int D = Dim, BYTES = 2 * D;
    static constexpr int STAGES = 3, K_TILE = KT * BYTES, STAGE = K_TILE + KT * BYTES;
    __device__ static std::uint32_t k_chunk(std::uint32_t r, std::uint32_t ch) { return r * BYTES + ((((ch >> 1) ^ (r & 3)) << 5) | ((ch & 1) << 4)); }
    __device__ static std::uint32_t k_read(std::uint32_t r, int ks, int q4) { return r * BYTES + ((ks ^ (r & 3)) << 5) + 8 * q4; }
    __device__ static std::uint32_t v_chunk(std::uint32_t key, std::uint32_t ch) { return K_TILE + key * BYTES + ((ch ^ (key & 7)) << 4); }
};
struct PadLayout : PadLayoutShape<256> {};
struct SwzLayout : SwzLayoutShape<256> {};

template <class LY>
constexpr int smem_bytes() { return LY::STAGES * LY::STAGE; }

__device__ __forceinline__ void cp16(std::uint32_t smem_addr, const void* g) {
    asm volatile("cp.async.cg.shared.global [%0], [%1], 16;" ::"r"(smem_addr), "l"(g));
}
__device__ __forceinline__ void mma_bf16(float (&c)[4], const std::uint32_t (&a)[4], std::uint32_t b0, std::uint32_t b1) {
    asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 {%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};"
                 : "+f"(c[0]), "+f"(c[1]), "+f"(c[2]), "+f"(c[3])
                 : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b0), "r"(b1));
}
__device__ __forceinline__ std::uint32_t pack2(float lo, float hi) {
    std::uint32_t r;
    asm("cvt.rn.bf16x2.f32 %0, %1, %2;" : "=r"(r) : "f"(hi), "f"(lo));
    return r;
}
__device__ __forceinline__ void ldsm_x4_trans(std::uint32_t (&r)[4], std::uint32_t addr) {
    asm volatile("ldmatrix.sync.aligned.m8n8.x4.trans.shared.b16 {%0,%1,%2,%3}, [%4];" : "=r"(r[0]), "=r"(r[1]), "=r"(r[2]), "=r"(r[3]) : "r"(addr));
}

struct ContiguousFlashKeys { static constexpr bool step = false; };

struct StepFlashKeys {
    static constexpr bool step = true;
    const attn_core::PreparedStepCache& cache;
    const exec::StepBlock& descriptor;
    const state::SlotEntry& keys;
    const state::SlotEntry& values;
    unsigned row;

    __device__ __forceinline__ unsigned pending(unsigned position) const {
        return core::decode_state::pending_row(descriptor, row, position);
    }
    __device__ __forceinline__ bool visible(unsigned position) const {
        return position < keys.frontier || pending(position) != exec::kStepNoParent;
    }
    __device__ __forceinline__ void load(unsigned smem_k, unsigned smem_v, unsigned position,
                                         unsigned column, unsigned stride) const {
        if (position < keys.frontier) {
            cp16(smem_k, reinterpret_cast<const __nv_bfloat16*>(cache.keys.address(keys, position)) + column);
            cp16(smem_v, reinterpret_cast<const __nv_bfloat16*>(cache.values.address(values, position)) + column);
            return;
        }
        const unsigned source = pending(position);
        const bool present = source != exec::kStepNoParent;
        const auto offset = std::size_t{present ? source : row} * stride + column;
        const unsigned bytes = present ? 16U : 0U;
        asm volatile("cp.async.cg.shared.global [%0], [%1], 16, %2;" ::
                     "r"(smem_k), "l"(cache.pending_keys + offset), "r"(bytes));
        asm volatile("cp.async.cg.shared.global [%0], [%1], 16, %2;" ::
                     "r"(smem_v), "l"(cache.pending_values + offset), "r"(bytes));
    }
};

template <class LY, bool Prepared, class Keys>
__device__ __forceinline__ void load_kv_tile(std::uint32_t smem_stage, const __nv_bfloat16* kbase, const __nv_bfloat16* vbase,
                                             std::uint32_t key0, std::uint32_t nkeys_total, std::uint32_t first_key,
                                             std::uint32_t kvh, const AttentionDims& d, bool wrapped, const Keys& keys) {
    constexpr std::uint32_t D = LY::D, chunks = D / 8;
#pragma unroll
    for (int i = 0; i < KT * chunks / 256; ++i) {
        const std::uint32_t idx = threadIdx.x + 256 * i, r = idx / chunks, ch = idx % chunks;
        if constexpr (Keys::step) {
            const auto key = min(max(key0 + r, first_key), nkeys_total - 1U);
            keys.load(smem_stage + LY::k_chunk(r, ch), smem_stage + LY::v_chunk(r, ch), key,
                      kvh * D + ch * 8U, d.kv_heads * D);
        } else if constexpr (Prepared) {
            // Tile edges may load any finite row for absent keys. The score mask drops them.
            const std::uint32_t key = min(max(key0 + r, first_key), nkeys_total - 1);
            const std::uint32_t slot = wrapped ? key % d.max_positions : key;
            const std::size_t offset = (static_cast<std::size_t>(slot) * d.kv_heads + kvh) * D + ch * 8;
            cp16(smem_stage + LY::k_chunk(r, ch), kbase + offset);
            cp16(smem_stage + LY::v_chunk(r, ch), vbase + offset);
        } else {
            const std::uint32_t key = min(key0 + r, nkeys_total - 1);
            cp16(smem_stage + LY::k_chunk(r, ch), kbase + static_cast<std::size_t>(key) * D + ch * 8);
            cp16(smem_stage + LY::v_chunk(r, ch), vbase + static_cast<std::size_t>(key) * D + ch * 8);
        }
    }
}

template <class LY, bool Prepared, bool Bf16Gate = false, bool Bidirectional = false, class Keys = ContiguousFlashKeys>
__device__ __forceinline__ void flash_body(AttentionDims d, const float* q, GroupedView qv, const float* gate, GroupedView gv,
                                           const __nv_bfloat16* kc, const __nv_bfloat16* vc, std::uint32_t pos0,
                                           std::uint32_t rows, float* out, std::uint32_t nqb, std::uint32_t n_full,
                                           std::uint32_t tail_w, const PreparedBf16Flash& prepared, const Keys& keys = {}) {
    constexpr int D = LY::D;
    extern __shared__ __align__(16) unsigned char smem[];
    // job -> tile (linear order = the former 2D grid: head-major, x fastest). Jobs below n_full are whole tiles (8 warps); the rest
    // split the remaining tiles into 8 / tail_w jobs of tail_w active warps (inactive warps still copy K/V and meet every barrier).
    std::uint32_t tile = blockIdx.x, wlo = 0, wn = 8;
    if (blockIdx.x >= n_full) {
        const std::uint32_t k = blockIdx.x - n_full, per = 8 / tail_w;
        tile = n_full + k / per;
        wlo = (k % per) * tail_w;
        wn = tail_w;
    }
    const std::uint32_t qb = nqb - 1 - tile % nqb;  // heavy (late) blocks first
    const std::uint32_t head = tile / nqb, G = d.heads / d.kv_heads, kvh = head / G;
    const int warp = threadIdx.x >> 5, lane = threadIdx.x & 31, g = lane >> 2, q4 = lane & 3;
    const bool act = static_cast<std::uint32_t>(warp) - wlo < wn;  // warp-uniform
    const std::uint32_t row_base = qb * QB, wrow0 = row_base + warp * 16;
    const auto fixed = attn_core::mask::block_range(pos0, rows, prepared.window);
    const std::uint32_t nkeys = Bidirectional ? fixed.end : pos0 + min(rows, row_base + QB);
    const std::uint32_t first_key = Bidirectional ? fixed.begin :
        (Prepared ? attn_core::mask::window_lo(pos0 + row_base, prepared.window) : 0U);
    const std::uint32_t tile_start = first_key / KT * KT;
    const std::uint32_t ntiles = (nkeys - tile_start + KT - 1) / KT;
    const __nv_bfloat16* kbase = Prepared ? kc : kc + static_cast<std::size_t>(kvh) * d.max_positions * D;
    const __nv_bfloat16* vbase = Prepared ? vc : vc + static_cast<std::size_t>(kvh) * d.max_positions * D;
    const std::uint32_t smem_base = static_cast<std::uint32_t>(__cvta_generic_to_shared(smem));

    // prefetch tiles 0 .. STAGES - 2 (one commit group per tile slot; empty groups past the end keep the count uniform)
#pragma unroll
    for (int st = 0; st < LY::STAGES - 1; ++st) {
        if (st < static_cast<int>(ntiles)) {
            load_kv_tile<LY, Prepared>(smem_base + st * LY::STAGE, kbase, vbase, tile_start + st * KT,
                                       nkeys, first_key, kvh, d, prepared.wrapped, keys);
        }
        asm volatile("cp.async.commit_group;");
    }

    // Q fragments (rows g and g + 8 of the warp's 16)
    std::uint32_t qa[D / 16][4];
    {
        const std::uint32_t r0 = wrow0 + g, r1 = wrow0 + g + 8;
        const float* p0 = q + static_cast<std::size_t>(min(r0, rows - 1)) * qv.rs() + static_cast<std::size_t>(head) * qv.gs() + 4 * q4;
        const float* p1 = q + static_cast<std::size_t>(min(r1, rows - 1)) * qv.rs() + static_cast<std::size_t>(head) * qv.gs() + 4 * q4;
#pragma unroll
        for (int ks = 0; ks < D / 16; ++ks) {
            const float4 a = *reinterpret_cast<const float4*>(p0 + 16 * ks), b = *reinterpret_cast<const float4*>(p1 + 16 * ks);
            qa[ks][0] = pack2(a.x, a.y);
            qa[ks][1] = pack2(b.x, b.y);
            qa[ks][2] = pack2(a.z, a.w);
            qa[ks][3] = pack2(b.z, b.w);
        }
    }
    float o[D / 8][4];
#pragma unroll
    for (int i = 0; i < D / 8; ++i) o[i][0] = o[i][1] = o[i][2] = o[i][3] = 0.0f;
    float m0 = attn_core::mask::excluded_score, m1 = attn_core::mask::excluded_score, l0 = 0.0f, l1 = 0.0f;
    constexpr float log2e = 1.4426950408889634f;
    const float sl2 = d.scale * log2e;
    const std::uint32_t qpos0 = pos0 + wrow0 + g, qpos1 = qpos0 + 8;
    const std::uint32_t qrow0 = wrow0 + g, qrow1 = qrow0 + 8;
    const attn_core::RelativeBiasRows bias{prepared.relative_bias, d.heads, prepared.relative_extent};
    if constexpr (Prepared) {
        if (!Bidirectional && prepared.sinks != nullptr) {
            const attn_core::AttentionSinkRows sinks{prepared.sinks, d.heads};
            if (qrow0 < rows) { m0 = sinks.sink(qrow0, head) * log2e; l0 = q4 == 0 ? 1.0f : 0.0f; }
            if (qrow1 < rows) { m1 = sinks.sink(qrow1, head) * log2e; l1 = q4 == 0 ? 1.0f : 0.0f; }
        }
    }

    std::uint32_t stage_idx = 0;  // t % STAGES
    for (std::uint32_t t = 0; t < ntiles; ++t) {
        const std::uint32_t stage = stage_idx * LY::STAGE;
        if constexpr (LY::STAGES == 2) {
            if (t + 1 < ntiles) {
                load_kv_tile<LY, Prepared>(smem_base + (stage_idx ^ 1) * LY::STAGE, kbase, vbase,
                                           tile_start + (t + 1) * KT, nkeys, first_key, kvh, d, prepared.wrapped, keys);
                asm volatile("cp.async.commit_group;\ncp.async.wait_group 1;" ::: "memory");
            } else {
                asm volatile("cp.async.wait_group 0;" ::: "memory");
            }
            __syncthreads();
        } else {
            // committed groups: tiles 0 .. t + STAGES - 2; tile t has landed once at most STAGES - 2 groups are pending
            asm volatile("cp.async.wait_group %0;" ::"n"(LY::STAGES - 2) : "memory");
            __syncthreads();  // tile t visible to every warp, and every warp is done with tile t - 1, whose stage is refilled now
            const std::uint32_t t_load = t + LY::STAGES - 1;
            if (t_load < ntiles) {
                const std::uint32_t ls = stage_idx == 0 ? LY::STAGES - 1 : stage_idx - 1;  // (t + STAGES - 1) % STAGES
                load_kv_tile<LY, Prepared>(smem_base + ls * LY::STAGE, kbase, vbase,
                                           tile_start + t_load * KT, nkeys, first_key, kvh, d, prepared.wrapped, keys);
            }
            asm volatile("cp.async.commit_group;");
        }
        stage_idx = stage_idx + 1 == static_cast<std::uint32_t>(LY::STAGES) ? 0 : stage_idx + 1;
        if (!act) {  // copy/barrier participant only
            if constexpr (LY::STAGES == 2) __syncthreads();
            continue;
        }
        const unsigned char* sK = smem + stage;
        const std::uint32_t sV = smem_base + stage;
        float s[4][4];
#pragma unroll
        for (int nt = 0; nt < 4; ++nt) s[nt][0] = s[nt][1] = s[nt][2] = s[nt][3] = 0.0f;
#pragma unroll
        for (int ks = 0; ks < D / 16; ++ks) {
#pragma unroll
            for (int nt = 0; nt < 4; ++nt) {
                const uint2 b = *reinterpret_cast<const uint2*>(sK + LY::k_read(nt * 8 + g, ks, q4));
                mma_bf16(s[nt], qa[ks], b.x, b.y);
            }
        }
        const std::uint32_t key0 = tile_start + t * KT;
        // scale (log2 domain) and causal mask
        const bool need_mask = key0 + KT - 1 > pos0 + wrow0;
#pragma unroll
        for (int nt = 0; nt < 4; ++nt)
#pragma unroll
            for (int j = 0; j < 4; ++j) {
                float v = s[nt][j] * sl2;
                if constexpr (Prepared) {
                    const std::uint32_t kp = key0 + nt * 8 + 2 * q4 + (j & 1);
                    const std::uint32_t qp = j < 2 ? qpos0 : qpos1, qr = j < 2 ? qrow0 : qrow1;
                    const auto lo = attn_core::mask::window_lo(qp, prepared.window);
                    if constexpr (Bidirectional) {
                        if (qr >= rows || !fixed.contains(kp)) v = attn_core::mask::excluded_score;
                    } else {
                        bool visible = qr < rows && attn_core::mask::causal_ok(kp, qp, lo);
                        if constexpr (Keys::step) visible = visible && keys.visible(kp);
                        if (!visible) {
                            v = attn_core::mask::excluded_score;
                        } else {
                            v += bias.score_bias(qr, head, qp, kp) * log2e;
                        }
                    }
                } else if (need_mask) {
                    const std::uint32_t kp = key0 + nt * 8 + 2 * q4 + (j & 1), qp = j < 2 ? qpos0 : qpos1;
                    if (kp > qp) v = attn_core::mask::excluded_score;
                }
                s[nt][j] = v;
            }
        float mx0 = attn_core::mask::excluded_score, mx1 = attn_core::mask::excluded_score;
#pragma unroll
        for (int nt = 0; nt < 4; ++nt) {
            mx0 = fmaxf(mx0, fmaxf(s[nt][0], s[nt][1]));
            mx1 = fmaxf(mx1, fmaxf(s[nt][2], s[nt][3]));
        }
        mx0 = fmaxf(mx0, __shfl_xor_sync(0xffffffffu, mx0, 1));
        mx0 = fmaxf(mx0, __shfl_xor_sync(0xffffffffu, mx0, 2));
        mx1 = fmaxf(mx1, __shfl_xor_sync(0xffffffffu, mx1, 1));
        mx1 = fmaxf(mx1, __shfl_xor_sync(0xffffffffu, mx1, 2));
        const float n0 = fmaxf(m0, mx0), n1 = fmaxf(m1, mx1);
        const float n0s = n0 == attn_core::mask::excluded_score ? 0.0f : n0, n1s = n1 == attn_core::mask::excluded_score ? 0.0f : n1;
        const float al0 = exp2f(m0 - n0s), al1 = exp2f(m1 - n1s);  // m = -inf -> 0
        m0 = n0;
        m1 = n1;
        l0 *= al0;
        l1 *= al1;
#pragma unroll
        for (int i = 0; i < D / 8; ++i) {
            o[i][0] *= al0; o[i][1] *= al0; o[i][2] *= al1; o[i][3] *= al1;
        }
        std::uint32_t pa[2][4];
#pragma unroll
        for (int nt = 0; nt < 4; ++nt) {
            const float p00 = exp2f(s[nt][0] - n0s), p01 = exp2f(s[nt][1] - n0s), p10 = exp2f(s[nt][2] - n1s), p11 = exp2f(s[nt][3] - n1s);
            l0 += p00 + p01;
            l1 += p10 + p11;
            const std::uint32_t lo = pack2(p00, p01), hi = pack2(p10, p11);
            pa[nt >> 1][(nt & 1) * 2 + 0] = lo;
            pa[nt >> 1][(nt & 1) * 2 + 1] = hi;
        }
        // O += P V : a0 = (row g, keys 2q..) of n-tile 2j, a1 = (row g+8, n-tile 2j), a2 = (row g, n-tile 2j+1), a3 = (g+8, 2j+1)
#pragma unroll
        for (int j = 0; j < 2; ++j) {
            const std::uint32_t a[4] = {pa[j][0], pa[j][1], pa[j][2], pa[j][3]};
#pragma unroll
            for (int np = 0; np < D / 16; ++np) {
                std::uint32_t b[4];
                const std::uint32_t key = 16 * j + ((lane >> 3) & 1) * 8 + (lane & 7), dcol = 8 * (2 * np + (lane >> 4));
                ldsm_x4_trans(b, sV + LY::v_chunk(key, dcol / 8));
                mma_bf16(o[2 * np], a, b[0], b[1]);
                mma_bf16(o[2 * np + 1], a, b[2], b[3]);
            }
        }
        if constexpr (LY::STAGES == 2) __syncthreads();
    }
    if (!act) return;  // no barrier below
    // finalize
    l0 += __shfl_xor_sync(0xffffffffu, l0, 1);
    l0 += __shfl_xor_sync(0xffffffffu, l0, 2);
    l1 += __shfl_xor_sync(0xffffffffu, l1, 1);
    l1 += __shfl_xor_sync(0xffffffffu, l1, 2);
    const float i0 = 1.0f / l0, i1 = 1.0f / l1;
    const std::uint32_t r0 = wrow0 + g, r1 = r0 + 8;
#pragma unroll
    for (int nt = 0; nt < D / 8; ++nt) {
        const std::uint32_t dd = 8 * nt + 2 * q4;
        for (int h = 0; h < 2; ++h) {
            const std::uint32_t r = h ? r1 : r0;
            if (r >= rows) continue;
            float v0 = o[nt][2 * h] * (h ? i1 : i0), v1 = o[nt][2 * h + 1] * (h ? i1 : i0);
            if constexpr (Bf16Gate) {
                const auto index = (static_cast<std::size_t>(r) * d.heads + head) * D + dd;
                using ::strix::core::row_transforms::Bf16SigmoidProduct;
                Bf16SigmoidProduct{&v0, gate + index, &v0}(0);
                Bf16SigmoidProduct{&v1, gate + index + 1U, &v1}(0);
            } else if (gate != nullptr) {
                const float2 gt = *reinterpret_cast<const float2*>(gate + static_cast<std::size_t>(r) * gv.rs() + static_cast<std::size_t>(head) * gv.gs() + dd);
                v0 *= cc_sigmoid(gt.x);
                v1 *= cc_sigmoid(gt.y);
            }
            *reinterpret_cast<float2*>(out + (static_cast<std::size_t>(r) * d.heads + head) * D + dd) = make_float2(v0, v1);
        }
    }
}

template <class LY>
__global__ void __launch_bounds__(256, 1)
    flash_kernel(AttentionDims d, const float* q, GroupedView qv, const float* gate, GroupedView gv,
                 const __nv_bfloat16* kc, const __nv_bfloat16* vc, std::uint32_t pos0,
                 std::uint32_t rows, float* out, std::uint32_t nqb, std::uint32_t n_full, std::uint32_t tail_w) {
    flash_body<LY, false>(d, q, qv, gate, gv, kc, vc, pos0, rows, out, nqb, n_full, tail_w, {});
}

template <class LY, bool Bidirectional>
__global__ void __launch_bounds__(256, 1)
    flash_prepared_kernel(AttentionDims d, PreparedBf16Flash prepared, const float* q,
                          const __nv_bfloat16* kc, const __nv_bfloat16* vc, std::uint32_t pos0,
                          std::uint32_t rows, float* out, std::uint32_t tiles) {
    flash_body<LY, true, false, Bidirectional>(d, q, {rows, d.heads, d.head_dim}, nullptr, {}, kc, vc,
                         pos0, rows, out, tiles, tiles * d.heads, 8, prepared);
}

template <class LY>
__global__ __launch_bounds__(256, 1)
void flash_prepared_step_kernel(AttentionDims d, PreparedBf16Flash prepared, const float* q,
                                const __nv_bfloat16* kc, const __nv_bfloat16* vc,
                                const std::uint32_t* position, float* out) {
    flash_body<LY, true>(d, q, {1U, d.heads, d.head_dim}, nullptr, {}, kc, vc,
                         *position, 1U, out, 1U, d.heads, 8, prepared);
}

template <unsigned Rows, class LY>
__global__ __launch_bounds__(256, 1) void flash_prepared_step_rows(attn_core::PreparedStepFlash args) {
    const unsigned r = blockIdx.y;
    if (r >= Rows || !exec::step_row_valid(*args.step, r)) return;
    const auto& row = args.step->row[r];
    const auto* keys = args.cache.keys.slots.entry(row.slot);
    const auto* values = args.cache.values.slots.entry(row.slot);
    using core::decode_state::accepts;
    if (!accepts(keys, row.position) || !accepts(values, row.position) || keys->frontier != values->frontier ||
        row.position > step_position_limit) return;

    const AttentionDims dims{args.heads, args.kv_heads, LY::D, args.score_scale, keys->capacity};
    const auto heads = std::size_t{r} * args.heads;
    const auto first = heads * LY::D;
    const PreparedBf16Flash prepared{
        args.relative_logits ? args.relative_logits + heads * args.relative_extent : nullptr,
        args.sinks ? args.sinks + heads : nullptr, args.relative_extent, args.window, false};
    flash_body<LY, true>(dims, args.queries + first, {1U, args.heads, LY::D}, nullptr, {}, nullptr, nullptr,
        row.position, 1U, args.output + first, 1U, args.heads, 8U,
        prepared, StepFlashKeys{args.cache, *args.step, *keys, *values, r});
}

template <class LY, bool Bidirectional = false>
cudaError_t launch_flash_prepared(const AttentionDims& d, const PreparedBf16Flash& prepared,
                                 const float* q, const __nv_bfloat16* k_cache, const __nv_bfloat16* v_cache,
                                 std::uint32_t pos0, std::uint32_t rows, float* out, cudaStream_t stream,
                                 const std::uint32_t* device_position) {
    if (device_position) {
        if (auto e = cudaFuncSetAttribute(flash_prepared_step_kernel<LY>, cudaFuncAttributeMaxDynamicSharedMemorySize,
                                          smem_bytes<LY>()); e != cudaSuccess) return e;
        flash_prepared_step_kernel<LY><<<d.heads, 256, smem_bytes<LY>(), stream>>>(
            d, prepared, q, k_cache, v_cache, device_position, out);
        return cudaGetLastError();
    }
    const auto tiles = (rows + QB - 1U) / QB;
    if (auto e = cudaFuncSetAttribute(flash_prepared_kernel<LY, Bidirectional>, cudaFuncAttributeMaxDynamicSharedMemorySize,
                                      smem_bytes<LY>()); e != cudaSuccess) return e;
    flash_prepared_kernel<LY, Bidirectional><<<tiles * d.heads, 256, smem_bytes<LY>(), stream>>>(
        d, prepared, q, k_cache, v_cache, pos0, rows, out, tiles);
    return cudaGetLastError();
}

template <bool Bf16Gate>
__global__ void __launch_bounds__(256, 1)
flash_planned_kernel(AttentionDims d, PreparedBf16Flash prepared, const float* q,
                     const __nv_bfloat16* k, const __nv_bfloat16* v,
                     cuda::core::attn::detail::PlanPosition position,
                     std::uint32_t rows, const float* gate, float* out) {
    const auto first = position.device ? *position.device : position.host;
    if (first > position.last_start) return;
    // K4c plans admit at most 32 query rows, within a single Q tile.
    flash_body<PadLayoutShape<128>, true, Bf16Gate>(d, q, {rows, d.heads, d.head_dim}, gate, {},
        k, v, first, rows, out, 1U, d.heads, 8, prepared);
}

template <bool Bf16Gate>
cudaError_t launch_flash_planned(const AttentionDims& dims, const PreparedBf16Flash& prepared,
                                 const float* q, const __nv_bfloat16* k, const __nv_bfloat16* v,
                                 cuda::core::attn::detail::PlanPosition position, std::uint32_t rows,
                                 const float* gate, float* out, cudaStream_t stream) {
    constexpr auto smem = smem_bytes<PadLayoutShape<128>>();
    if (auto status = cudaFuncSetAttribute(flash_planned_kernel<Bf16Gate>,
            cudaFuncAttributeMaxDynamicSharedMemorySize, smem); status != cudaSuccess) return status;
    flash_planned_kernel<Bf16Gate><<<dims.heads, 256, smem, stream>>>(dims, prepared, q, k, v,
                                                                     position, rows, gate, out);
    return cudaGetLastError();
}

__global__ void iota_kernel(std::uint32_t* p, std::uint32_t n, std::uint32_t base, std::uint32_t step) {
    const std::uint32_t i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) p[i] = base + step * i;
}

}  // namespace

cudaError_t attention_prefill_flash(const AttentionDims& d, const float* q, const GroupedView& qv, const float* gate, const GroupedView& gv,
                                    const __nv_bfloat16* k_cache, const __nv_bfloat16* v_cache, std::uint32_t pos0, std::uint32_t rows,
                                    float* out, cudaStream_t stream, FlashPipe pipe, FlashTail tail, std::uint32_t tail_warps) {
    if (d.head_dim != PadLayout::D || rows == 0 || d.kv_heads == 0 || d.heads % d.kv_heads != 0 || d.heads / d.kv_heads > 8 || pos0 + rows > d.max_positions)
        return cudaErrorInvalidValue;
    static int sms = 0;
    if (sms == 0 && tail != FlashTail::off) {
        int dev = 0;
        if (auto e = cudaGetDevice(&dev); e != cudaSuccess) return e;
        if (auto e = cudaDeviceGetAttribute(&sms, cudaDevAttrMultiProcessorCount, dev); e != cudaSuccess) return e;
    }
    const FlashPlan pl = flash_plan(rows, d.heads, static_cast<std::uint32_t>(sms), tail, tail_warps);
    const std::uint32_t nqb = (rows + QB - 1) / QB;
    const dim3 grid(pl.jobs);
    // Attributes belong to the current device; CUDA serializes the update.
    if (pipe == FlashPipe::swz3) {
        if (auto e = cudaFuncSetAttribute(flash_kernel<SwzLayout>, cudaFuncAttributeMaxDynamicSharedMemorySize, smem_bytes<SwzLayout>());
            e != cudaSuccess)
            return e;
        flash_kernel<SwzLayout><<<grid, 256, smem_bytes<SwzLayout>(), stream>>>(d, q, qv, gate, gv, k_cache, v_cache, pos0, rows, out, nqb,
                                                                                 pl.full, pl.tail_warps);
    } else {
        if (auto e = cudaFuncSetAttribute(flash_kernel<PadLayout>, cudaFuncAttributeMaxDynamicSharedMemorySize, smem_bytes<PadLayout>());
            e != cudaSuccess)
            return e;
        flash_kernel<PadLayout><<<grid, 256, smem_bytes<PadLayout>(), stream>>>(d, q, qv, gate, gv, k_cache, v_cache, pos0, rows, out, nqb,
                                                                                 pl.full, pl.tail_warps);
    }
    return cudaGetLastError();
}

bool can_launch_prepared_bf16_flash(const AttentionDims& d, const PreparedBf16Flash& prepared,
                                    const float* q, const __nv_bfloat16* k_cache, const __nv_bfloat16* v_cache,
                                    std::uint32_t pos0, std::uint32_t rows, float* out, FlashPipe pipe) {
    constexpr auto limit = std::numeric_limits<std::uint32_t>::max() - QB - KT;
    constexpr auto grid_limit = static_cast<std::uint32_t>(std::numeric_limits<int>::max());
    constexpr auto max_floats = std::numeric_limits<std::size_t>::max() / sizeof(float);
    constexpr auto max_bf16 = std::numeric_limits<std::size_t>::max() / sizeof(__nv_bfloat16);
    const auto aligned = [](const void* ptr) { return ptr && reinterpret_cast<std::uintptr_t>(ptr) % 16U == 0; };
    if (d.head_dim != 128 || rows == 0 || d.heads == 0 || d.kv_heads == 0 || d.heads % d.kv_heads != 0 ||
        d.heads / d.kv_heads > 8 || !std::isfinite(d.scale) || d.heads > limit / d.head_dim ||
        pos0 > limit || rows > limit - pos0 || d.heads > grid_limit / ((rows + QB - 1U) / QB) ||
        !aligned(q) || !aligned(k_cache) || !aligned(v_cache) || !aligned(out) || q == out || k_cache == v_cache ||
        (pipe != FlashPipe::pad2 && pipe != FlashPipe::swz3) || d.max_positions == 0 ||
        d.heads > max_floats / rows / d.head_dim || d.kv_heads > max_bf16 / d.max_positions / d.head_dim ||
        (prepared.relative_extent != 0 &&
            (!prepared.relative_bias || d.heads > max_floats / rows / prepared.relative_extent))) return false;

    const auto end = pos0 + rows;
    if (prepared.bidirectional && (prepared.relative_bias || prepared.relative_extent || prepared.sinks)) return false;
    const auto begin = prepared.bidirectional ? attn_core::mask::block_range(pos0, rows, prepared.window).begin :
        attn_core::mask::window_lo(pos0, prepared.window);
    return prepared.wrapped ? prepared.window != 0 && d.max_positions >= end - begin : d.max_positions >= end;
}

cudaError_t attention_prefill_prepared_bf16(const AttentionDims& d, const PreparedBf16Flash& prepared,
                                            const float* q, const __nv_bfloat16* k_cache, const __nv_bfloat16* v_cache,
                                            std::uint32_t pos0, std::uint32_t rows, float* out, cudaStream_t stream,
                                            FlashPipe pipe, const std::uint32_t* device_position) {
    if ((device_position && (rows != 1U || prepared.bidirectional)) || !can_launch_prepared_bf16_flash(d, prepared, q, k_cache, v_cache, pos0, rows, out, pipe)) {
        return cudaErrorInvalidValue;
    }
    if (pipe == FlashPipe::swz3) {
        if (prepared.bidirectional)
            return launch_flash_prepared<SwzLayoutShape<128>, true>(d, prepared, q, k_cache, v_cache, pos0, rows, out, stream,
                                                                   device_position);
        return launch_flash_prepared<SwzLayoutShape<128>>(d, prepared, q, k_cache, v_cache, pos0, rows, out, stream,
                                                        device_position);
    }
    if (prepared.bidirectional)
        return launch_flash_prepared<PadLayoutShape<128>, true>(d, prepared, q, k_cache, v_cache, pos0, rows, out, stream,
                                                               device_position);
    return launch_flash_prepared<PadLayoutShape<128>>(d, prepared, q, k_cache, v_cache, pos0, rows, out, stream,
                                                     device_position);
}

template <unsigned Rows>
cudaError_t prepared_step_flash(const attn_core::PreparedStepFlash& a, cudaStream_t stream) {
    using Layout = PadLayoutShape<128>;
    const auto aligned = [](const void* p) { return p && reinterpret_cast<std::uintptr_t>(p) % 16U == 0U; };
    const auto covers = [&](unsigned ring) { return ring == 0U || (a.window != 0U && ring >= a.window); };
    if (!a.step || a.kv_heads == 0U || a.heads == 0U || a.heads % a.kv_heads != 0U || a.heads / a.kv_heads > 8U ||
        a.heads > static_cast<unsigned>(std::numeric_limits<int>::max()) / Layout::D || !std::isfinite(a.score_scale) ||
        !aligned(a.queries) || !aligned(a.output) || a.queries == a.output ||
        !aligned(a.cache.pending_keys) || !aligned(a.cache.pending_values) ||
        !attn_core::valid_prepared_cache(a.cache, a.kv_heads, Layout::D) ||
        !covers(a.cache.keys.ring) || !covers(a.cache.values.ring) ||
        (a.relative_extent && !a.relative_logits)) return cudaErrorInvalidValue;
    if (auto e = cudaFuncSetAttribute(flash_prepared_step_rows<Rows, Layout>,
            cudaFuncAttributeMaxDynamicSharedMemorySize, smem_bytes<Layout>()); e != cudaSuccess) return e;
    flash_prepared_step_rows<Rows, Layout><<<dim3(a.heads, Rows), 256, smem_bytes<Layout>(), stream>>>(a);
    return cudaGetLastError();
}

template cudaError_t prepared_step_flash<1>(const attn_core::PreparedStepFlash&, cudaStream_t);
template cudaError_t prepared_step_flash<2>(const attn_core::PreparedStepFlash&, cudaStream_t);
template cudaError_t prepared_step_flash<4>(const attn_core::PreparedStepFlash&, cudaStream_t);
template cudaError_t prepared_step_flash<8>(const attn_core::PreparedStepFlash&, cudaStream_t);
template cudaError_t prepared_step_flash<16>(const attn_core::PreparedStepFlash&, cudaStream_t);

namespace {

// blockIdx.y = lane of the block, blockIdx.x = query head: each block runs the prepared flash body with
// exactly the arguments flash_prepared_kernel receives for that lane's row alone (one query tile).
template <class LY>
__global__ void __launch_bounds__(256, 1)
    flash_prepared_lanes_kernel(AttentionDims d, PreparedBf16Flash shape, PreparedBf16LaneImages images,
                                attn::lanes::LaneBlock block) {
    const auto b = lanes_detail::bind_lane(d, shape, images, block.lane[blockIdx.y],
                                           std::size_t{block.first_row} + blockIdx.y);
    flash_body<LY, true, false, false>(b.dims, b.q, {1U, d.heads, d.head_dim}, nullptr, {}, b.k_cache, b.v_cache,
                                       b.pos0, 1U, b.out, 1U, d.heads, 8, b.prepared);
}

template <class LY>
cudaError_t launch_flash_prepared_lanes(const AttentionDims& d, const PreparedBf16Flash& shape,
                                        const PreparedBf16LaneImages& images, const attn::lanes::LaneBlock& block,
                                        cudaStream_t stream) {
    if (block.count == 0U) return cudaErrorInvalidValue;
    if (auto e = cudaFuncSetAttribute(flash_prepared_lanes_kernel<LY>, cudaFuncAttributeMaxDynamicSharedMemorySize,
                                      smem_bytes<LY>()); e != cudaSuccess) return e;
    flash_prepared_lanes_kernel<LY><<<dim3(d.heads, block.count), 256, smem_bytes<LY>(), stream>>>(
        d, shape, images, block);
    return cudaGetLastError();
}

// blockIdx.y = lane of a replayed step, resolved on the device; the body and its arguments are those of
// flash_prepared_lanes_kernel for the same lane (bind_lane), so a lane's bits equal its one-row call.
template <class LY>
__global__ void __launch_bounds__(256, 1)
    flash_prepared_step_lanes_kernel(AttentionDims d, PreparedBf16Flash shape, PreparedBf16LaneImages images,
                                     PreparedBf16StepLanes lanes) {
    const auto b = lanes_detail::bind_lane(d, shape, images, lanes_detail::resolve_step_lane(lanes, blockIdx.y),
                                           blockIdx.y);
    flash_body<LY, true, false, false>(b.dims, b.q, {1U, d.heads, d.head_dim}, nullptr, {}, b.k_cache, b.v_cache,
                                       b.pos0, 1U, b.out, 1U, d.heads, 8, b.prepared);
}

template <class LY>
cudaError_t launch_flash_prepared_step_lanes(const AttentionDims& d, const PreparedBf16Flash& shape,
                                             const PreparedBf16LaneImages& images, const PreparedBf16StepLanes& lanes,
                                             cudaStream_t stream) {
    if (lanes.lanes.lanes == 0U) return cudaErrorInvalidValue;
    if (auto e = cudaFuncSetAttribute(flash_prepared_step_lanes_kernel<LY>, cudaFuncAttributeMaxDynamicSharedMemorySize,
                                      smem_bytes<LY>()); e != cudaSuccess) return e;
    flash_prepared_step_lanes_kernel<LY><<<dim3(d.heads, lanes.lanes.lanes), 256, smem_bytes<LY>(), stream>>>(
        d, shape, images, lanes);
    return cudaGetLastError();
}

}  // namespace

cudaError_t lanes_detail::launch_prepared_bf16_step_lanes_flash(const AttentionDims& d, const PreparedBf16Flash& shape,
                                                                const PreparedBf16LaneImages& images,
                                                                const PreparedBf16StepLanes& lanes,
                                                                cudaStream_t stream, FlashPipe pipe) {
    return pipe == FlashPipe::swz3 ? launch_flash_prepared_step_lanes<SwzLayoutShape<128>>(d, shape, images, lanes, stream)
                                   : launch_flash_prepared_step_lanes<PadLayoutShape<128>>(d, shape, images, lanes, stream);
}

cudaError_t lanes_detail::launch_prepared_bf16_lanes_flash(const AttentionDims& d, const PreparedBf16Flash& shape,
                                                           const PreparedBf16LaneImages& images,
                                                           const attn::lanes::LaneBlock& block, cudaStream_t stream,
                                                           FlashPipe pipe) {
    return pipe == FlashPipe::swz3 ? launch_flash_prepared_lanes<SwzLayoutShape<128>>(d, shape, images, block, stream)
                                   : launch_flash_prepared_lanes<PadLayoutShape<128>>(d, shape, images, block, stream);
}

bool can_launch_prepared_bf16_lanes(const AttentionDims& d, const PreparedBf16Flash& shape,
                                    const PreparedBf16LaneImages& images,
                                    std::span<const attn::lanes::PreparedBf16Lane> lanes, FlashPipe pipe) {
    if (lanes.empty() || shape.bidirectional || !attn::lanes::distinct_lane_caches(lanes)) return false;
    for (std::size_t row = 0; row < lanes.size(); ++row) {
        const auto b = lanes_detail::bind_lane(d, shape, images, lanes[row], row);
        if (!can_launch_prepared_bf16_flash(b.dims, b.prepared, b.q, b.k_cache, b.v_cache, b.pos0, 1U, b.out, pipe))
            return false;
    }
    return true;
}

FlashPlan flash_plan(std::uint32_t rows, std::uint32_t heads, std::uint32_t sms, FlashTail tail, std::uint32_t tail_warps) {
    FlashPlan p;
    p.tiles = (rows + QB - 1) / QB * heads;
    p.full = p.jobs = p.tiles;
    if (tail == FlashTail::off || sms == 0 || p.tiles <= sms) return p;  // no partial wave behind a full one
    const std::uint32_t r = p.tiles % sms;
    if (r == 0) return p;
    std::uint32_t w = 0;
    if (tail_warps == 1 || tail_warps == 2 || tail_warps == 4) {
        w = tail_warps;
    } else {
        for (std::uint32_t c : {1u, 2u, 4u})
            if (r * (8 / c) <= sms) {
                w = c;
                break;
            }
    }
    if (w == 0) return p;
    p.full = p.tiles - r;
    p.tail_tiles = r;
    p.tail_warps = w;
    p.jobs = p.full + r * (8 / w);
    return p;
}

cudaError_t attention_prefill_flash_grid(const AttentionDims& d, FlashPipe pipe, int device, FlashGrid* out) {
    if (out == nullptr || d.heads == 0) return cudaErrorInvalidValue;
    int sms = 0, per_sm = 0;
    if (auto e = cudaDeviceGetAttribute(&sms, cudaDevAttrMultiProcessorCount, device); e != cudaSuccess) return e;
    cudaError_t e = cudaSuccess;
    if (pipe == FlashPipe::swz3) {
        e = cudaFuncSetAttribute(flash_kernel<SwzLayout>, cudaFuncAttributeMaxDynamicSharedMemorySize, smem_bytes<SwzLayout>());
        if (e == cudaSuccess) e = cudaOccupancyMaxActiveBlocksPerMultiprocessor(&per_sm, flash_kernel<SwzLayout>, 256, smem_bytes<SwzLayout>());
    } else {
        e = cudaFuncSetAttribute(flash_kernel<PadLayout>, cudaFuncAttributeMaxDynamicSharedMemorySize, smem_bytes<PadLayout>());
        if (e == cudaSuccess) e = cudaOccupancyMaxActiveBlocksPerMultiprocessor(&per_sm, flash_kernel<PadLayout>, 256, smem_bytes<PadLayout>());
    }
    if (e != cudaSuccess) return e;
    *out = FlashGrid{static_cast<std::uint32_t>(QB), d.heads, static_cast<std::uint32_t>(sms), static_cast<std::uint32_t>(per_sm)};
    return cudaSuccess;
}

cudaError_t iota_u32(std::uint32_t* p, std::uint32_t n, std::uint32_t base, std::uint32_t step, cudaStream_t stream) {
    iota_kernel<<<(n + 127) / 128, 128, 0, stream>>>(p, n, base, step);
    return cudaGetLastError();
}

}  // namespace strix::cuda_core

namespace strix::cuda::core::attn::detail {

cudaError_t decode_prepared(const cuda_core::AttentionDims& dims,
                            const cuda_core::PreparedBf16Flash& prepared, const float* q,
                            const __nv_bfloat16* k, const __nv_bfloat16* v, PlanPosition position,
                            std::uint32_t rows, const float* gate, float* out, cudaStream_t stream) {
    if (!gate && !position.device) {
        return cuda_core::attention_prefill_prepared_bf16(dims, prepared, q, k, v, position.host,
                                                         rows, out, stream);
    }
    if (gate)
        return cuda_core::launch_flash_planned<true>(dims, prepared, q, k, v, position, rows, gate, out, stream);
    return cuda_core::launch_flash_planned<false>(dims, prepared, q, k, v, position, rows, gate, out, stream);
}

}  // namespace strix::cuda::core::attn::detail
