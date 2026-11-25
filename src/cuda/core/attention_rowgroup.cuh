// Shared CUDA core: ROW-GROUP sparse flash attention (FAST forms; mma.sync m16n8k16 BF16 -> FP32, sm_80+).
//
// Why: a per-row sparse attention gathers every selected K and V row once per query row. At Qwen QSA 8K that is 356 GB of
// L2 -> SM traffic per prefill and it binds the time (score and output kernels both run at ~2.7 TB/s). Neighbouring rows
// select mostly the same keys (Qwen 8K exact: 4-row union / per-row count = 1.37, HIP union counter), so a group of R rows
// that walks the union of its rows' lists reads each K/V row once for all R rows: bytes / (R / inflation).
//
// Two kernels per call, both templated on a binding Policy (one per model; the core has no model constants):
//   rowgroup_union_kernel  grid (tiles), 256 threads: the tile's union list + per-entry row masks (rowgroup_union.hpp).
//   rowgroup_flash_kernel  grid (kv heads, tiles), 32 * R threads: warp w = query row w of the tile, its Heads heads
//                          as the M16 rows (padded with zero queries); Q stays in registers as A fragments for the walk.
//                          16-key union chunks are staged once per CTA by cp.async (K and V, two chunks in the ring) and
//                          every warp whose row holds a key of the chunk runs QK, the masked online softmax, PV.
// Policy (device-copyable struct):
//   static constexpr unsigned D (head dim, multiple of 16), Heads (<= 16 query heads per KV head), Rows (R <= 8),
//   Cap (max selected keys per row); member rows (query rows of the call);
//   __device__ unsigned count(row) const;  __device__ std::uint32_t key(row, j) const;   (ascending lists expected)
//   __device__ const std::uint16_t* key_row(key, kvh) const; value_row(key, kvh) const;  (BF16, D contiguous, 16-B aligned)
//   __device__ float query(row, kvh, head, d) const;            (rounded to BF16 here)
//   __device__ void store2(row, kvh, head, d, a0, a1) const;      (attention values of features d, d + 1: epilogue/gate)
//   static constexpr float kScaleLog2 (softmax scale times log2 e).
// Numerics (identity of the binding's FAST form): BF16 Q, K, V; FP32 MMA accumulation in hardware order; base-2 online
// softmax with ex2.approx over union chunks ascending; P rounded to BF16; out = O / l. Not bit-comparable with exact paths.
#pragma once

#include "attn_core/rowgroup_plan.hpp"
#include "attention_tile.cuh"
#include "ptx.cuh"
#include "rowgroup_union.hpp"

#include <cstddef>
#include <cstdint>

namespace strix::cuda::core::rowgroup {

namespace detail {
// Exclusive scan of one value per thread over a 256-thread block; returns the prefix, total in `total`.
__device__ __forceinline__ unsigned block_scan256(unsigned v, unsigned* warp_sums, unsigned& total) {
    const unsigned lane = threadIdx.x & 31U, warp = threadIdx.x >> 5U;
    unsigned x = v;
#pragma unroll
    for (unsigned d = 1; d < 32U; d <<= 1U) {
        const unsigned y = __shfl_up_sync(0xffffffffU, x, d);
        if (lane >= d) x += y;
    }
    if (lane == 31U) warp_sums[warp] = x;
    __syncthreads();
    unsigned before = 0;
    total = 0;
    for (unsigned w = 0; w < 8U; ++w) {
        if (w < warp) before += warp_sums[w];
        total += warp_sums[w];
    }
    __syncthreads();
    return before + x - v;
}
}  // namespace detail

template <class P> __device__ __forceinline__ void union_body(const P& p, Union u) {
    using S = Shape<P>;
    constexpr unsigned R = S::R, Cap = S::Cap;
    extern __shared__ __align__(16) std::uint8_t rg_union_smem[];
    auto* lists = reinterpret_cast<std::uint32_t*>(rg_union_smem);
    auto* pre = reinterpret_cast<std::uint16_t*>(rg_union_smem + R * Cap * sizeof(std::uint32_t));
    __shared__ unsigned warp_sums[8];
    __shared__ unsigned irregular;
    const unsigned tid = threadIdx.x, tile = blockIdx.x;
    unsigned n[R];
#pragma unroll
    for (unsigned r = 0; r < R; ++r) {
        const unsigned row = tile * R + r;
        n[r] = row < p.rows ? p.count(row) : 0U;
    }
    if (tid == 0) irregular = 0U;
    __syncthreads();
#pragma unroll
    for (unsigned r = 0; r < R; ++r)
        for (unsigned j = tid; j < n[r]; j += blockDim.x) lists[r * Cap + j] = p.key(tile * R + r, j);
    __syncthreads();
#pragma unroll
    for (unsigned r = 0; r < R; ++r)
        for (unsigned j = tid + 1U; j < n[r]; j += blockDim.x)
            if (lists[r * Cap + j] <= lists[r * Cap + j - 1U]) irregular = 1U;
    __syncthreads();
    std::uint32_t* const tok = u.tok + std::size_t{tile} * S::UnionCap;
    std::uint8_t* const mask = u.mask + std::size_t{tile} * S::UnionCap;
    if (irregular) {  // concatenation: every (row, key) entry once, one mask bit
        unsigned base = 0;
#pragma unroll
        for (unsigned r = 0; r < R; ++r) {
            for (unsigned j = tid; j < n[r]; j += blockDim.x) { tok[base + j] = lists[r * Cap + j]; mask[base + j] = static_cast<std::uint8_t>(1U << r); }
            base += n[r];
        }
        if (tid == 0) u.count[tile] = base;
        return;
    }
    // owned prefix per list: thread t scans a contiguous segment of `per` entries
#pragma unroll
    for (unsigned r = 0; r < R; ++r) {
        const unsigned per = (n[r] + blockDim.x - 1U) / blockDim.x, j0 = tid * per;
        unsigned local = 0;
        for (unsigned j = j0; j < j0 + per && j < n[r]; ++j) local += owned(lists, Cap, n, r, lists[r * Cap + j]) ? 1U : 0U;
        unsigned total = 0;
        unsigned run = detail::block_scan256(local, warp_sums, total);
        for (unsigned j = j0; j < j0 + per && j < n[r]; ++j) {
            pre[r * (Cap + 1U) + j] = static_cast<std::uint16_t>(run);
            run += owned(lists, Cap, n, r, lists[r * Cap + j]) ? 1U : 0U;
        }
        if (tid == 0) pre[r * (Cap + 1U) + n[r]] = static_cast<std::uint16_t>(total);
    }
    __syncthreads();
    unsigned count = 0;
#pragma unroll
    for (unsigned r = 0; r < R; ++r) count += pre[r * (Cap + 1U) + n[r]];
#pragma unroll
    for (unsigned r = 0; r < R; ++r)
        for (unsigned j = tid; j < n[r]; j += blockDim.x) {
            if (!owned(lists, Cap, n, r, lists[r * Cap + j])) continue;
            unsigned m = 0;
            const unsigned at = rank_and_mask<R>(lists, Cap, n, pre, Cap + 1U, r, j, m);
            tok[at] = lists[r * Cap + j];
            mask[at] = static_cast<std::uint8_t>(m);
        }
    if (tid == 0) u.count[tile] = count;
}

template <class P> __device__ __forceinline__ void flash_body(const P& p, Union u) {
    using S = Shape<P>;
    namespace cp = strix::cuda::core::ptx;
    constexpr unsigned D = S::D, R = S::R, Pitch = S::Pitch, Slot = S::Slot, NT = D / 8U, Pieces = D / 8U;  // 16-B pieces per row
    extern __shared__ __align__(16) std::uint16_t rg_flash_smem[];
    const unsigned tid = threadIdx.x, warp = tid >> 5U, lane = tid & 31U;
    const unsigned kvh = blockIdx.x, tile = blockIdx.y, row = tile * R + warp;
    const bool row_ok = row < p.rows;
    // Q of this warp's row: staged once (BF16, heads >= Heads zero), then held as A fragments for the whole walk.
    std::uint16_t* const qs = rg_flash_smem + warp * 16U * Pitch;
    for (unsigned i = lane; i < 16U * D; i += 32U) {
        const unsigned h = i / D, d = i % D;
        float x = 0.0F;
        if (row_ok && h < P::Heads) x = p.query(row, kvh, h, d);
        unsigned packed;
        asm("cvt.rn.bf16x2.f32 %0, %1, %2;\n" : "=r"(packed) : "f"(0.0F), "f"(x));
        qs[h * Pitch + d] = static_cast<std::uint16_t>(packed & 0xffffU);
    }
    __syncwarp();
    unsigned qa[D / 16][4];
    attn::load_q_frags<D, Pitch>(qa, qs, lane);
    __syncthreads();  // the ring reuses the Q staging memory
    const unsigned count = u.count[tile];
    const std::uint32_t* const tok = u.tok + std::size_t{tile} * S::UnionCap;
    const std::uint8_t* const mask = u.mask + std::size_t{tile} * S::UnionCap;
    const unsigned chunks = (count + S::Chunk - 1U) / S::Chunk;
    const auto issue = [&](unsigned c) {
        if (c < chunks) {
            std::uint16_t* const ring = rg_flash_smem + (c & 1U) * 2U * Slot;
            for (unsigned piece = tid; piece < 2U * S::Chunk * Pieces; piece += S::Threads) {
                const unsigned which = piece / (S::Chunk * Pieces), rest = piece % (S::Chunk * Pieces);
                const unsigned k = rest / Pieces, col = rest % Pieces, e = c * S::Chunk + k;
                const std::uint32_t key = tok[e < count ? e : 0U];
                const std::uint16_t* src = (which ? p.value_row(key, kvh) : p.key_row(key, kvh)) + col * 8U;
                cp::cp_async16(ring + which * Slot + k * Pitch + col * 8U, src);
            }
        }
        cp::cp_async_commit();
    };
    float o[NT][4];
#pragma unroll
    for (unsigned i = 0; i < NT; ++i) { o[i][0] = 0.0F; o[i][1] = 0.0F; o[i][2] = 0.0F; o[i][3] = 0.0F; }
    float m[2] = {__uint_as_float(0xff800000U), __uint_as_float(0xff800000U)}, l[2] = {0.0F, 0.0F};
    issue(0U);
#pragma unroll 1
    for (unsigned c = 0; c < chunks; ++c) {
        issue(c + 1U);
        cp::cp_async_wait<1>();
        __syncthreads();
        const unsigned e = c * S::Chunk + (lane & 15U);
        const unsigned bit = lane < 16U && e < count ? (mask[e] >> warp) & 1U : 0U;
        const unsigned keys = __ballot_sync(0xffffffffU, bit != 0U) & 0xffffU;
        if (keys != 0U && row_ok) {
            const std::uint16_t* const ring = rg_flash_smem + (c & 1U) * 2U * Slot;
            float s[2][4];
            attn::qk_tile16_qreg<D, Pitch>(s, qa, ring, lane);
            unsigned pa[4];
            attn::online_softmax16_mask<NT>(s, keys, P::kScaleLog2, m, l, o, pa, lane);
            attn::pv_tile16<D, Pitch>(o, pa, ring + Slot, lane);
        }
        __syncthreads();  // chunk c's slots are refilled by issue(c + 2)
    }
    cp::cp_async_wait<0>();
    if (!row_ok) return;
    l[0] += __shfl_xor_sync(0xffffffffU, l[0], 1); l[0] += __shfl_xor_sync(0xffffffffU, l[0], 2);
    l[1] += __shfl_xor_sync(0xffffffffU, l[1], 1); l[1] += __shfl_xor_sync(0xffffffffU, l[1], 2);
    const unsigned g = lane >> 2U, t = lane & 3U;
#pragma unroll
    for (unsigned i = 0; i < NT; ++i) {
        const unsigned d = i * 8U + 2U * t;
        if (g < P::Heads) p.store2(row, kvh, g, d, o[i][0] / l[0], o[i][1] / l[0]);
        if (g + 8U < P::Heads) p.store2(row, kvh, g + 8U, d, o[i][2] / l[1], o[i][3] / l[1]);
    }
}

}  // namespace strix::cuda::core::rowgroup
