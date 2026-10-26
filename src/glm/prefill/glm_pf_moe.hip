// GLM-5.3-Flash prefill MoE: grouped WMMA launches over all routed experts of a chunk (identity: glm_pf_moe.hpp).
// Port of the MiniMax v4 grouped core (3e1897a1, src/minimax/moe/mmx_prefill_core.hpp item_core_v3 Tiled,
// mmx_prefill.hip gather_transform_tiled / gate_up_finish_tiled / down_combine) to the GLM shapes with ONE K split.
//
// Library contract mirrored (strix-trellis e7f801d, strix_gemm_pinned(splits = 1)):
//   input   t = RN16(H128(RN16(materialize(x * suh))))    transforms.hpp:8-26 input_transform, hadamard.hpp:9-37
//   core    per output: acc = +0; acc = WMMA_f32_16x16x16_f16(A_k, B_k, acc) for k-tiles 0 .. K/16 - 1 ascending,
//           A_k = the transformed f16 row, B_k = codebook_value(state, mul1) of the trellis tile (gemm.hip:15-77
//           wmma_core; wmma_tiled.hip tiled_core and wmma_packed decoder_core claim the same bits per output)
//   finish  v = (+0 + p); y = H128(v) * svh (* alpha = 1, no bias, beta = 0)   transforms.hpp:28-51 output_transform
//           (wmma_tiled.hip fused epilogue: the same per-group arithmetic)
// Every helper keeps that operation order; only addressing, scheduling and launch structure differ.
// Compile with -ffp-contract=off -fno-fast-math -fno-gpu-flush-denormals-to-zero.
#include "glm/prefill/glm_pf_moe.hpp"

#include "glm/common/glm_math.hpp"

#include <hip/hip_fp16.h>

#include <algorithm>
#include <cstdlib>
#include <stdexcept>
#include <string>
#include <vector>

namespace strix::glm::pf {

namespace {

constexpr unsigned kE = kMoeExperts, kTop = kMoeTop, kH = kMoeHidden, kI = kMoeInter;
constexpr unsigned kHB = kH / 128, kIB = kI / 128;  // 32, 8

using Half16 = _Float16 __attribute__((ext_vector_type(16)));
using Float8 = float __attribute__((ext_vector_type(8)));

// ---------------------------------------------------------------- library pieces (verbatim arithmetic)
// hadamard.hpp:9-14 materialize_fp32: RN32 boundary before a later RN16 conversion.
__device__ __forceinline__ float materialize(float v) {
    asm volatile("" : "+v"(v));
    return v;
}
// hadamard.hpp:18-37 hadamard128 (values held as lane + 32 j).
__device__ __forceinline__ void hadamard128(float (&v)[4], unsigned lane) {
#pragma unroll
    for (int stride = 1; stride < 32; stride *= 2) {
#pragma unroll
        for (int j = 0; j < 4; ++j) {
            const float peer = __shfl_xor(v[j], stride, 32);
            v[j] = (lane & static_cast<unsigned>(stride)) ? peer - v[j] : v[j] + peer;
        }
    }
#pragma unroll
    for (int stride = 1; stride < 4; stride *= 2) {
#pragma unroll
        for (int j = 0; j < 4; ++j)
            if (!(j & stride)) {
                const float l = v[j], r = v[j + stride];
                v[j] = l + r;
                v[j + stride] = l - r;
            }
    }
#pragma unroll
    for (int j = 0; j < 4; ++j) v[j] = materialize(v[j] * 0x1.6a09e6p-4F);
}
__device__ __forceinline__ float half_bits(std::uint16_t h) { return __half2float(__ushort_as_half(h)); }

// trellis_device.hpp:8-15 codebook_value(state, 2) (mul1).
__device__ __forceinline__ float codebook_mul1(std::uint32_t state) {
    const std::uint32_t product = state * 0x83dcd12dU;
    const unsigned sum = __builtin_amdgcn_udot4(product, 0x01010101U, 0, false);
    return __half2float(__float2half_rn(fmaf(static_cast<float>(1024 + sum), __half2float(__ushort_as_half(0x1eee)),
                                             __half2float(__ushort_as_half(0xc931)))));
}

// Core v2 (STRIX_GLM_PF_MOE_CORE=2) form of codebook_mul1: the same integers and the same fma / RN16; float(1024 + sum)
// for sum <= 1020 is the bit pattern 0x44800000 | sum << 13 (no cvt; checked exhaustively for all sums 0..1020).
__device__ __forceinline__ float codebook_mul1_v2(std::uint32_t state) {
    const std::uint32_t product = state * 0x83dcd12dU;
    const unsigned sum = __builtin_amdgcn_udot4(product, 0x01010101U, 0, false);
    return __half2float(__float2half_rn(fmaf(__uint_as_float(0x44800000U | (sum << 13)), __half2float(__ushort_as_half(0x1eee)),
                                             __half2float(__ushort_as_half(0xc931)))));
}
template <typename T>
using Glob = const __attribute__((address_space(1))) T*;  // global address space: global_load, not flat

// ---------------------------------------------------------------- grouped WMMA core (mmx_prefill_core.hpp v3/v4)
// Normalized window of the v3 core: every lane funnel-shifts its kNL trellis words once per k-tile so that state r of
// its column is alignbit(N[g(r) - 1], N[g(r)], shift(r)) with compile-time g / shift. Equal to WaveTile::state +
// the wmma_core q layout for Bits 1..8 (design notes, 4.1 M checks).
template <int Bits>
struct Norm {
    static constexpr int kWords = Bits * 8;
    static constexpr int kNL = ((47 + 27 * Bits) >> 5) + 2;
    static constexpr int delta(int r) { return 8 * ((r & 7) >> 1) + (r & 1) + 2 * (r >> 3); }
    static constexpr int e(int r) { return 48 + delta(r) * Bits; }
    static constexpr int g(int r) { return (e(r) - 1) >> 5; }
    static constexpr unsigned shift(int r) { return static_cast<unsigned>((32 - (e(r) & 31)) & 31); }
};

__device__ __forceinline__ Half16 as_half16(const std::uint32_t (&f)[8]) {
    Half16 out;
#pragma unroll
    for (int i = 0; i < 8; ++i) {
        out[2 * i] = __builtin_bit_cast(_Float16, static_cast<std::uint16_t>(f[i] & 0xffffU));
        out[2 * i + 1] = __builtin_bit_cast(_Float16, static_cast<std::uint16_t>(f[i] >> 16));
    }
    return out;
}

// One grouped work item: one expert matrix over its group's rows.
struct Item {
    const std::uint32_t* trellis;
    const _Float16* x;  // tile-major transformed rows of the group: [rows/16 padded][K/16][16][16]
    float* part;        // [rows][N] fp32 WMMA results (one split)
    unsigned rows, cta_begin, gy, pad;
};

// CTA = 4 waves = 8 tile columns = 128 outputs x MT row tiles of 16. Wave w owns tiles 2p, 2p + 1 (p = 4 bx + w):
// lanes 0-15 decode column lane of tile 2p, lanes 16-31 column lane - 16 of tile 2p + 1; v_permlanex16 hands each
// half the other's column so both B fragments (lane = column, k = 0..15) are complete without LDS.
// V2 (core v2, same bits): codebook_mul1_v2, global-address-space loads and the k loop unrolled by 2 (no ln -> lc
// register rotation). PF (core v3 = V2 with PF 4): the trellis words of k-tile t + PF are loaded while tile t is
// computed (a register ring of PF tiles, compile-time slots), so more weight bytes are in flight (k12: the weight
// stream was serialized with the WMMAs). Only load timing changes.
template <int Bits, int MT, int K, int N, bool V2, int PF = 1>
__device__ __forceinline__ void item_core(const Item& it, unsigned bx, unsigned by) {
    using NW = Norm<Bits>;
    constexpr int tiles_n = N / 16, tiles_k = K / 16;
    const int lane = static_cast<int>(threadIdx.x & 31U), wave = static_cast<int>(threadIdx.x >> 5U);
    const int c = lane & 15;
    const bool upper = lane >= 16;
    const int rows = static_cast<int>(it.rows);
    const int tile_pair = static_cast<int>(bx) * 4 + wave;
    const int my_tile = 2 * tile_pair + (upper ? 1 : 0);
    const int first_row = static_cast<int>(by) * MT * 16;
    const int base = 32 * (c & 7) + 4 * (c >> 3);
    const int S = (base + 1) * Bits - 16 + 256 * Bits - 32;
    const int d = S & 31, wb = (S >> 5) - (d == 0 ? 1 : 0);
    const unsigned sh = static_cast<unsigned>((32 - d) & 31);
    int loff[NW::kNL];
#pragma unroll
    for (int i = 0; i < NW::kNL; ++i) loff[i] = my_tile * NW::kWords + (wb + i) % NW::kWords;
    constexpr std::size_t k_stride = static_cast<std::size_t>(tiles_n) * NW::kWords;
    // A: the lane's 32 bytes (row first_row + 16 m + c) in k tile 0 of its row tile; null past the group's rows.
    const _Float16* xr[MT];
#pragma unroll
    for (int m = 0; m < MT; ++m) {
        const int row = first_row + 16 * m + c;
        const _Float16* p = it.x + (static_cast<std::size_t>(first_row / 16 + m) * static_cast<std::size_t>(tiles_k)) * 256 +
                            static_cast<std::size_t>(c) * 16;
        // V2: rows past the group read row tile 0 of this CTA instead of zeros (no branch). A WMMA output row depends
        // only on its own A row and those rows are never stored, so every stored value keeps its bits.
        xr[m] = row < rows ? p : V2 ? it.x + static_cast<std::size_t>(first_row / 16) * static_cast<std::size_t>(tiles_k) * 256 : nullptr;
    }
    Float8 acc0[MT] = {}, acc1[MT] = {};
    static_assert(PF == 1 || V2, "prefetch ring only on the v2 path");
    static_assert(tiles_k % (2 * PF) == 0, "k loop unrolled by 2 (PF 1) or by PF");
    auto tword = [&](int tile, int i) -> std::uint32_t {
        const std::uint32_t* kb = it.trellis + static_cast<std::size_t>(tile) * k_stride;  // uniform
        return V2 ? ((Glob<std::uint32_t>)kb)[loff[i]] : kb[loff[i]];
    };
    std::uint32_t ring[PF][NW::kNL];
#pragma unroll
    for (int q = 0; q < PF; ++q)
#pragma unroll
        for (int i = 0; i < NW::kNL; ++i) ring[q][i] = tword(q, i);
    constexpr int kUnroll = V2 && PF == 1 ? 2 : 1;
#pragma unroll kUnroll
    for (int tk0 = 0; tk0 < tiles_k; tk0 += PF) {
#pragma unroll
     for (int q = 0; q < PF; ++q) {
        const int tile_k = tk0 + q;
        std::uint32_t lc[NW::kNL];
#pragma unroll
        for (int i = 0; i < NW::kNL; ++i) lc[i] = ring[q][i];
        if (tile_k + PF < tiles_k) {
#pragma unroll
            for (int i = 0; i < NW::kNL; ++i) ring[q][i] = tword(tile_k + PF, i);
        }
        std::uint32_t nrm[NW::kNL - 1];
#pragma unroll
        for (int i = 0; i < NW::kNL - 1; ++i) nrm[i] = __builtin_amdgcn_alignbit(lc[i], lc[i + 1], sh);
        std::uint32_t mine[8];
#pragma unroll
        for (int r = 0; r < 16; r += 2) {
            std::uint16_t v[2];
#pragma unroll
            for (int i = 0; i < 2; ++i) {
                const int rr = r + i;
                const std::uint32_t state = __builtin_amdgcn_alignbit(nrm[NW::g(rr) - 1], nrm[NW::g(rr)], NW::shift(rr)) & 65535U;
                v[i] = __builtin_bit_cast(std::uint16_t, static_cast<_Float16>(V2 ? codebook_mul1_v2(state) : codebook_mul1(state)));
            }
            mine[r / 2] = static_cast<std::uint32_t>(v[0]) | (static_cast<std::uint32_t>(v[1]) << 16);
        }
        std::uint32_t other[8], f0[8], f1[8];
#pragma unroll
        for (int i = 0; i < 8; ++i) {
            other[i] = __builtin_amdgcn_permlanex16(mine[i], mine[i], 0x76543210U, 0xfedcba98U, false, false);
            f0[i] = upper ? other[i] : mine[i];
            f1[i] = upper ? mine[i] : other[i];
        }
        const Half16 b0 = as_half16(f0), b1 = as_half16(f1);
        constexpr int G = MT < 4 ? MT : 4;  // A fragments in groups of <= 4 row tiles (VGPR budget at MT = 8)
#pragma unroll
        for (int g = 0; g < MT; g += G) {
            Half16 a[G];
#pragma unroll
            for (int i = 0; i < G; ++i) {
                const int m = g + i;
                if constexpr (V2) {
                    a[i] = *((Glob<Half16>)__builtin_assume_aligned(xr[m] + static_cast<std::size_t>(tile_k) * 256, 32));
                } else if (xr[m] != nullptr) {  // 32 bytes, 32-byte aligned (512-byte tiles, 256-byte aligned regions)
                    const auto* src = static_cast<const _Float16*>(__builtin_assume_aligned(xr[m] + static_cast<std::size_t>(tile_k) * 256, 32));
                    __builtin_memcpy(&a[i], src, sizeof(Half16));
                } else {
                    a[i] = Half16{};
                }
            }
#pragma unroll
            for (int i = 0; i < G; ++i) {
                acc0[g + i] = __builtin_amdgcn_wmma_f32_16x16x16_f16_w32(a[i], b0, acc0[g + i]);
                acc1[g + i] = __builtin_amdgcn_wmma_f32_16x16x16_f16_w32(a[i], b1, acc1[g + i]);
            }
        }
     }
    }
    // acc[m][j] -> row first_row + 16 m + 2 j + lane / 16, column tile * 16 + lane % 16 (wmma_core store layout).
    const int col0 = (2 * tile_pair) * 16 + c, col1 = col0 + 16;
#pragma unroll
    for (int m = 0; m < MT; ++m) {
#pragma unroll
        for (int j = 0; j < 8; ++j) {
            const int row = first_row + 16 * m + 2 * j + (lane >> 4);
            if (row < rows) {
                float* dst = it.part + static_cast<std::size_t>(row) * N;
                dst[col0] = acc0[m][j];
                dst[col1] = acc1[m][j];
            }
        }
    }
}

template <int Bits, int MT, int K, int N, bool V2, int PF>
__global__ __launch_bounds__(128) void grouped_kernel(const Item* items, unsigned count) {
    unsigned lo = 0, hi = count;  // last item with cta_begin <= blockIdx.x
    while (hi - lo > 1) {
        const unsigned mid = (lo + hi) / 2;
        if (items[mid].cta_begin <= blockIdx.x) lo = mid;
        else hi = mid;
    }
    const Item it = items[lo];
    constexpr unsigned gx = N / 128;
    const unsigned local = blockIdx.x - it.cta_begin;
    // MT 8 at PF 4 spills (256 VGPR + scratch): ring of 2 there for Bits <= 4, 1 above (Bits 5/6 spill even at 2)
    constexpr int kPF = MT == 8 && PF > 1 ? (Bits <= 4 ? 2 : 1) : PF;
    item_core<Bits, MT, K, N, V2, kPF>(it, local % gx, local / gx);
}

// Row-tile class of a group with `rows` rows: MT in {1, 2, 4, 8} (index 0..3).
unsigned mt_class(unsigned rows) { return rows <= 16 ? 0U : rows <= 32 ? 1U : rows <= 64 ? 2U : 3U; }

// ---------------------------------------------------------------- tables and element-wise stages
struct GroupDev {  // one routed expert of the chunk
    MoeExpert w;
    unsigned n, pair_begin, tbegin, pad;  // tbegin: first row in the tile-major layout (multiple of 16)
};
struct Tables {
    const GroupDev* groups;
    const std::uint32_t* tile_group;  // [tpairs / 16] group of each 16-row tile (gather v3)
    const std::uint32_t* pair_row;    // [P] token of the pair
    const std::uint32_t* pair_group;  // [P]
    const std::uint32_t* tok_pair;    // [T][8] pair of route (t, k)
    const float* tok_w;               // [T][8]
    _Float16* xt;                     // [2][tpairs][4096] tile-major (gate, up)
    _Float16* at;                     // [tpairs][1024] tile-major
    float* part_g;                    // [P][1024]
    float* part_u;                    // [P][1024]
    float* part_d;                    // [P][4096]
    unsigned pairs, rows, tpairs, pad;
};

// Tile-major position of padded row R, column ch of a K-wide matrix: [R/16][ch/16][R%16][ch%16].
__device__ __forceinline__ std::size_t tiled_at(unsigned R, unsigned ch, unsigned K) {
    return ((static_cast<std::size_t>(R >> 4U) * (K >> 4U) + (ch >> 4U)) << 8U) + ((R & 15U) << 4U) + (ch & 15U);
}

// One wave per (pair, matrix, 128-block): input_transform of the pair's token row with that matrix's suh.
__global__ __launch_bounds__(256) void gather_transform_kernel(const float* x, Tables t) {
    const unsigned wave = threadIdx.x >> 5U, lane = threadIdx.x & 31U;
    const unsigned q = blockIdx.x * 8U + wave;
    if (q >= t.pairs * 2 * kHB) return;
    const unsigned p = q / (2 * kHB), rem = q % (2 * kHB), m = rem / kHB, b = rem % kHB;
    const GroupDev& g = t.groups[t.pair_group[p]];
    const std::uint16_t* suh = m == 0 ? g.w.gate.suh : g.w.up.suh;
    const float* xr = x + static_cast<std::size_t>(t.pair_row[p]) * kH;
    float v[4];
#pragma unroll
    for (int j = 0; j < 4; ++j) {
        const unsigned c = b * 128U + lane + 32U * static_cast<unsigned>(j);
        const float scaled = materialize(xr[c] * half_bits(suh[c]));
        v[j] = static_cast<float>(static_cast<_Float16>(scaled));
    }
    hadamard128(v, lane);
    _Float16* out = t.xt + static_cast<std::size_t>(m) * t.tpairs * kH;
    const unsigned R = g.tbegin + (p - g.pair_begin);
#pragma unroll
    for (int j = 0; j < 4; ++j) out[tiled_at(R, b * 128U + lane + 32U * static_cast<unsigned>(j), kH)] = static_cast<_Float16>(v[j]);
}

// Gather v3 (STRIX_GLM_PF_MOE_GATHER=3, same bits): CTA = one 16-row tile of the tile-major layout x one matrix x 8
// consecutive 128-blocks. The tile's group comes from a host-built table (one load; gather v2's per-CTA binary search
// over GroupDev made it 2.1x slower, k13). Per block, each wave transforms 2 of the tile's rows exactly as
// gather_transform_kernel does into LDS; the CTA then writes the block's 8 consecutive 16 x 16 tiles (4 KB) with one
// 16-byte store per thread. Rows of the tile past the group are written as +0 (never read for a stored output).
constexpr unsigned kG3Pitch = 136, kG3Blocks = 8;  // halves per LDS row (rows 4 banks apart); 128-blocks per CTA
__global__ __launch_bounds__(256) void gather_transform_v3_kernel(const float* x, Tables t) {
    __shared__ alignas(16) _Float16 sv[16][kG3Pitch];
    const unsigned wave = threadIdx.x >> 5U, lane = threadIdx.x & 31U, tile = blockIdx.x, m = blockIdx.y, R0 = 16U * tile;
    const GroupDev& g = t.groups[t.tile_group[tile]];
    const std::uint16_t* suh = m == 0 ? g.w.gate.suh : g.w.up.suh;
    const float* xr[2];
    bool live[2];
#pragma unroll
    for (unsigned h = 0; h < 2; ++h) {
        const unsigned local = R0 + wave + 8U * h - g.tbegin;
        live[h] = local < g.n;
        xr[h] = live[h] ? x + static_cast<std::size_t>(t.pair_row[g.pair_begin + local]) * kH : x;
    }
    const unsigned ct = threadIdx.x / 32U, w = threadIdx.x % 32U, r = w / 2U, c0 = 16U * ct + 8U * (w % 2U);
    for (unsigned bi = 0; bi < kG3Blocks; ++bi) {
        const unsigned b = blockIdx.z * kG3Blocks + bi;
#pragma unroll
        for (unsigned h = 0; h < 2; ++h) {
            float v[4] = {0.0F, 0.0F, 0.0F, 0.0F};
            if (live[h]) {  // uniform per wave
#pragma unroll
                for (int j = 0; j < 4; ++j) {
                    const unsigned c = b * 128U + lane + 32U * static_cast<unsigned>(j);
                    const float scaled = materialize(xr[h][c] * half_bits(suh[c]));
                    v[j] = static_cast<float>(static_cast<_Float16>(scaled));
                }
                hadamard128(v, lane);
            }
#pragma unroll
            for (int j = 0; j < 4; ++j) sv[wave + 8U * h][lane + 32U * static_cast<unsigned>(j)] = static_cast<_Float16>(v[j]);
        }
        __syncthreads();
        _Float16* out = t.xt + static_cast<std::size_t>(m) * t.tpairs * kH + tiled_at(R0, b * 128U, kH) + 256U * ct + 8U * w;
        *reinterpret_cast<uint4*>(out) = *reinterpret_cast<const uint4*>(&sv[r][c0]);
        __syncthreads();  // sv reused by the next block
    }
}

// output_transform of one 128-block of one row with one split: v = +0 + p; H128; * svh.
__device__ __forceinline__ void finish_block(const float* row, unsigned b, const std::uint16_t* svh, unsigned lane, float (&v)[4]) {
#pragma unroll
    for (int j = 0; j < 4; ++j) v[j] = 0.0F;
#pragma unroll
    for (int j = 0; j < 4; ++j) v[j] += row[b * 128 + lane + 32U * static_cast<unsigned>(j)];
    hadamard128(v, lane);
#pragma unroll
    for (int j = 0; j < 4; ++j) v[j] = v[j] * half_bits(svh[b * 128 + lane + 32U * static_cast<unsigned>(j)]);
}

// R15 (glm_tier_a.hip swiglu_kernel): gg = min(g, 10); uu = clamp(u, -10, 10); a = silu(gg) * uu.
__device__ __forceinline__ float swiglu_r15(float g, float u) {
    const float gg = g > kClamp ? kClamp : g;
    const float uu = u > kClamp ? kClamp : (u < -kClamp ? -kClamp : u);
    return silu(gg) * uu;
}

// One wave per (pair, 128-block of I): gate / up output transforms, R15, down input transform, tile-major store.
__global__ __launch_bounds__(256) void finish_kernel(Tables t) {
    const unsigned wave = threadIdx.x >> 5U, lane = threadIdx.x & 31U;
    const unsigned q = blockIdx.x * 8U + wave;
    if (q >= t.pairs * kIB) return;
    const unsigned p = q / kIB, b = q % kIB;
    const GroupDev& g = t.groups[t.pair_group[p]];
    float gv[4], uv[4];
    finish_block(t.part_g + static_cast<std::size_t>(p) * kI, b, g.w.gate.svh, lane, gv);
    finish_block(t.part_u + static_cast<std::size_t>(p) * kI, b, g.w.up.svh, lane, uv);
    float v[4];
#pragma unroll
    for (int j = 0; j < 4; ++j) {
        const unsigned c = b * 128 + lane + 32U * static_cast<unsigned>(j);
        const float act = swiglu_r15(gv[j], uv[j]);
        const float scaled = materialize(act * half_bits(g.w.down.suh[c]));
        v[j] = static_cast<float>(static_cast<_Float16>(scaled));
    }
    hadamard128(v, lane);
    const unsigned R = g.tbegin + (p - g.pair_begin);
#pragma unroll
    for (int j = 0; j < 4; ++j) t.at[tiled_at(R, b * 128 + lane + 32U * static_cast<unsigned>(j), kI)] = static_cast<_Float16>(v[j]);
}

// One wave per (token, 128-block of H): down output transform of the token's 8 routes in route order, R16, + shared.
__global__ __launch_bounds__(256) void combine_kernel(Tables t, const float* shared, float* out, float* eo) {
    const unsigned wave = threadIdx.x >> 5U, lane = threadIdx.x & 31U;
    const unsigned q = blockIdx.x * 8U + wave;
    if (q >= t.rows * kHB) return;
    const unsigned tok = q / kHB, b = q % kHB;
    float acc[4] = {0.0F, 0.0F, 0.0F, 0.0F};
    for (unsigned k = 0; k < kTop; ++k) {
        const unsigned p = t.tok_pair[tok * kTop + k];
        const GroupDev& g = t.groups[t.pair_group[p]];
        float v[4];
        finish_block(t.part_d + static_cast<std::size_t>(p) * kH, b, g.w.down.svh, lane, v);
        if (eo != nullptr) {
#pragma unroll
            for (int j = 0; j < 4; ++j)
                eo[(static_cast<std::size_t>(tok) * kTop + k) * kH + b * 128 + lane + 32U * static_cast<unsigned>(j)] = v[j];
        }
        const float w = t.tok_w[tok * kTop + k];
#pragma unroll
        for (int j = 0; j < 4; ++j) acc[j] = __builtin_fmaf(w, v[j], acc[j]);
    }
#pragma unroll
    for (int j = 0; j < 4; ++j) {
        const std::size_t i = static_cast<std::size_t>(tok) * kH + b * 128 + lane + 32U * static_cast<unsigned>(j);
        out[i] = acc[j] + shared[i];
    }
}

// ---------------------------------------------------------------- host side
void launched(const char* what) {
    const hipError_t e = hipGetLastError();
    if (e != hipSuccess) throw std::runtime_error(std::string("glm prefill moe ") + what + ": " + hipGetErrorString(e));
}
void ok(hipError_t e, const char* what) {
    if (e != hipSuccess) throw std::runtime_error(std::string("glm prefill moe ") + what + ": " + hipGetErrorString(e));
}

constexpr std::size_t align256(std::size_t v) { return (v + 255) & ~std::size_t{255}; }
bool bits_ok(unsigned b) { return b >= 3 && b <= 6; }

struct Plan {
    std::vector<GroupDev> groups;
    std::vector<std::uint32_t> pair_row, pair_group, tok_pair;
    std::vector<float> tok_w;
    std::vector<Item> items;  // gate/up classes 0..3, then down classes 0..3
    unsigned gu_count[4] = {}, dn_count[4] = {}, gu_ctas[4] = {}, dn_ctas[4] = {};
    unsigned tpairs = 0;
    std::vector<std::uint32_t> tile_group;  // [tpairs / 16]
    std::size_t off_tiles = 0;
    std::size_t off_groups = 0, off_pair_row = 0, off_pair_group = 0, off_tok_pair = 0, off_tok_w = 0, off_items = 0,
                off_xt = 0, off_at = 0, off_pg = 0, off_pu = 0, off_pd = 0, bytes = 0;
};

bool exl3_ok(const MoeExl3& m) {
    return m.trellis != nullptr && m.suh != nullptr && m.svh != nullptr && (reinterpret_cast<std::uintptr_t>(m.trellis) & 3U) == 0 &&
           (reinterpret_cast<std::uintptr_t>(m.suh) & 1U) == 0 && (reinterpret_cast<std::uintptr_t>(m.svh) & 1U) == 0;
}

Plan make_plan(const PrefillMoeView& v) {
    if (v.rows == 0) throw std::invalid_argument("glm prefill moe: rows must be > 0");
    if (v.x == nullptr || v.ids == nullptr || v.weights == nullptr || v.experts == nullptr || v.shared == nullptr || v.out == nullptr)
        throw std::invalid_argument("glm prefill moe: null pointer in view");
    if (!bits_ok(v.bits_gu) || !bits_ok(v.bits_dn)) throw std::invalid_argument("glm prefill moe: bits must be 3..6");
    const unsigned T = v.rows;
    if (T > (1U << 24)) throw std::invalid_argument("glm prefill moe: too many rows");
    Plan p;
    std::vector<std::vector<std::uint32_t>> erows(kE);
    std::vector<std::uint32_t> slot(std::size_t{T} * kTop);  // index of route (t, k) inside its expert's row list
    for (unsigned t = 0; t < T; ++t)
        for (unsigned k = 0; k < kTop; ++k) {
            const std::uint32_t e = v.ids[std::size_t{t} * kTop + k];
            if (e >= kE) throw std::invalid_argument("glm prefill moe: expert id out of range");
            slot[std::size_t{t} * kTop + k] = static_cast<std::uint32_t>(erows[e].size());
            erows[e].push_back(t);  // ascending token order
        }
    std::vector<std::uint32_t> group_of(kE, ~0U);
    for (unsigned e = 0; e < kE; ++e) {
        if (erows[e].empty()) continue;
        const MoeExpert& w = v.experts[e];
        if (!exl3_ok(w.gate) || !exl3_ok(w.up) || !exl3_ok(w.down))
            throw std::invalid_argument("glm prefill moe: routed expert " + std::to_string(e) + " has null or misaligned weights");
        GroupDev g{};
        g.w = w;
        g.n = static_cast<unsigned>(erows[e].size());
        g.pair_begin = static_cast<unsigned>(p.pair_row.size());
        g.tbegin = p.tpairs;
        p.tpairs += (g.n + 15U) / 16U * 16U;
        p.tile_group.insert(p.tile_group.end(), (g.n + 15U) / 16U, static_cast<std::uint32_t>(p.groups.size()));
        group_of[e] = static_cast<std::uint32_t>(p.groups.size());
        for (const std::uint32_t r : erows[e]) {
            p.pair_row.push_back(r);
            p.pair_group.push_back(group_of[e]);
        }
        p.groups.push_back(g);
    }
    p.tok_pair.resize(std::size_t{T} * kTop);
    p.tok_w.resize(std::size_t{T} * kTop);
    for (std::size_t i = 0; i < p.tok_pair.size(); ++i) {
        p.tok_pair[i] = p.groups[group_of[v.ids[i]]].pair_begin + slot[i];
        p.tok_w[i] = v.weights[i];
    }
    const std::size_t P = p.pair_row.size();
    // Workspace layout (every region 256-byte aligned).
    std::size_t off = 0;
    auto take = [&](std::size_t bytes) {
        const std::size_t at = off;
        off += align256(bytes);
        return at;
    };
    p.off_groups = take(p.groups.size() * sizeof(GroupDev));
    p.off_pair_row = take(P * 4);
    p.off_pair_group = take(P * 4);
    p.off_tok_pair = take(p.tok_pair.size() * 4);
    p.off_tok_w = take(p.tok_w.size() * 4);
    p.off_items = take(3 * p.groups.size() * sizeof(Item));
    p.off_tiles = take(p.tile_group.size() * 4);
    p.off_xt = take(2 * std::size_t{p.tpairs} * kH * 2);
    p.off_at = take(std::size_t{p.tpairs} * kI * 2);
    p.off_pg = take(P * kI * 4);
    p.off_pu = take(P * kI * 4);
    p.off_pd = take(P * kH * 4);
    p.bytes = off;
    return p;
}

void fill_items(Plan& p, unsigned char* ws) {
    auto* xt = reinterpret_cast<_Float16*>(ws + p.off_xt);
    auto* at = reinterpret_cast<_Float16*>(ws + p.off_at);
    auto* pg = reinterpret_cast<float*>(ws + p.off_pg);
    auto* pu = reinterpret_cast<float*>(ws + p.off_pu);
    auto* pd = reinterpret_cast<float*>(ws + p.off_pd);
    std::vector<Item> gu[4], dn[4];
    for (const GroupDev& g : p.groups) {
        const unsigned cls = mt_class(g.n), tm = 16U << cls;
        const unsigned gy = (g.n + tm - 1) / tm;
        for (unsigned m = 0; m < 2; ++m) {
            Item it{};
            it.trellis = m == 0 ? g.w.gate.trellis : g.w.up.trellis;
            it.x = xt + (static_cast<std::size_t>(m) * p.tpairs + g.tbegin) * kH;
            it.part = (m == 0 ? pg : pu) + static_cast<std::size_t>(g.pair_begin) * kI;
            it.rows = g.n;
            it.gy = gy;
            it.cta_begin = p.gu_ctas[cls];
            p.gu_ctas[cls] += (kI / 128) * gy;
            gu[cls].push_back(it);
        }
        Item it{};
        it.trellis = g.w.down.trellis;
        it.x = at + static_cast<std::size_t>(g.tbegin) * kI;
        it.part = pd + static_cast<std::size_t>(g.pair_begin) * kH;
        it.rows = g.n;
        it.gy = gy;
        it.cta_begin = p.dn_ctas[cls];
        p.dn_ctas[cls] += (kH / 128) * gy;
        dn[cls].push_back(it);
    }
    for (unsigned c = 0; c < 4; ++c) {
        p.gu_count[c] = static_cast<unsigned>(gu[c].size());
        p.items.insert(p.items.end(), gu[c].begin(), gu[c].end());
    }
    for (unsigned c = 0; c < 4; ++c) {
        p.dn_count[c] = static_cast<unsigned>(dn[c].size());
        p.items.insert(p.items.end(), dn[c].begin(), dn[c].end());
    }
}

bool gather_v3() {  // default gather v3 (same bits; measurement: -12/-13 % at T 2048/4096); STRIX_GLM_PF_MOE_GATHER=1: v1
    static const bool v3 = [] {                     // (v2 removed: 2.1x slower, k13)
        const char* e = std::getenv("STRIX_GLM_PF_MOE_GATHER");
        return !(e != nullptr && std::string(e) == "1");
    }();
    return v3;
}
unsigned core_version() {  // default core v3 (same bits; measurement: gate/up and down -6 % at T 2048, -3 % at 1024, ~-1 %
    // at 4096); STRIX_GLM_PF_MOE_CORE=1 / 2: core v1 / v2
    static const unsigned v = [] {
        const char* e = std::getenv("STRIX_GLM_PF_MOE_CORE");
        if (e != nullptr && std::string(e) == "1") return 1U;
        if (e != nullptr && std::string(e) == "2") return 2U;
        return 3U;
    }();
    return v;
}
template <int Bits, int K, int N, bool V2, int PF>
void launch_class_v(const Item* items_d, unsigned n, unsigned ctas, unsigned cls, hipStream_t stream) {
    switch (cls) {
        case 0: hipLaunchKernelGGL((grouped_kernel<Bits, 1, K, N, V2, PF>), dim3(ctas), dim3(128), 0, stream, items_d, n); break;
        case 1: hipLaunchKernelGGL((grouped_kernel<Bits, 2, K, N, V2, PF>), dim3(ctas), dim3(128), 0, stream, items_d, n); break;
        case 2: hipLaunchKernelGGL((grouped_kernel<Bits, 4, K, N, V2, PF>), dim3(ctas), dim3(128), 0, stream, items_d, n); break;
        default: hipLaunchKernelGGL((grouped_kernel<Bits, 8, K, N, V2, PF>), dim3(ctas), dim3(128), 0, stream, items_d, n); break;
    }
}
template <int Bits, int K, int N>
void launch_class(const Item* items_d, unsigned n, unsigned ctas, unsigned cls, hipStream_t stream) {
    if (n == 0) return;
    switch (core_version()) {
        case 3: launch_class_v<Bits, K, N, true, 4>(items_d, n, ctas, cls, stream); break;
        case 2: launch_class_v<Bits, K, N, true, 1>(items_d, n, ctas, cls, stream); break;
        default: launch_class_v<Bits, K, N, false, 1>(items_d, n, ctas, cls, stream); break;
    }
}
template <int K, int N>
void launch_stage(unsigned bits, const Item* items_d, const unsigned (&count)[4], const unsigned (&ctas)[4], hipStream_t stream) {
    std::size_t at = 0;
    for (unsigned c = 0; c < 4; ++c) {
        switch (bits) {
            case 3: launch_class<3, K, N>(items_d + at, count[c], ctas[c], c, stream); break;
            case 4: launch_class<4, K, N>(items_d + at, count[c], ctas[c], c, stream); break;
            case 5: launch_class<5, K, N>(items_d + at, count[c], ctas[c], c, stream); break;
            case 6: launch_class<6, K, N>(items_d + at, count[c], ctas[c], c, stream); break;
            default: throw std::invalid_argument("glm prefill moe: bits must be 3..6");
        }
        at += count[c];
    }
}

PrefillMoeMarker g_marker = nullptr;
void* g_marker_user = nullptr;
void mark(const char* label, hipStream_t stream) {
    if (g_marker != nullptr) g_marker(label, stream, g_marker_user);
}

}  // namespace

void set_prefill_moe_marker(PrefillMoeMarker marker, void* user) {
    g_marker = marker;
    g_marker_user = user;
}

std::size_t prefill_moe_workspace_bytes(const PrefillMoeView& v) { return make_plan(v).bytes; }

std::size_t prefill_moe_workspace_bound(std::uint32_t rows) {
    const std::size_t P = std::size_t{kTop} * rows, G = std::min<std::size_t>(kE, P);
    const std::size_t PX = P + 15 * G;  // every group padded to 16 rows
    std::size_t b = align256(G * sizeof(GroupDev)) + 2 * align256(P * 4) + 2 * align256(P * 4) + align256(3 * G * sizeof(Item));
    b += align256(PX / 16 * 4);  // tile -> group table (gather v3)
    b += align256(2 * PX * kH * 2) + align256(PX * kI * 2);
    b += 2 * align256(P * kI * 4) + align256(P * kH * 4);
    return b;
}

void enqueue_prefill_moe_glm(const PrefillMoeView& v, void* workspace, std::size_t workspace_bytes, hipStream_t stream) {
    Plan p = make_plan(v);
    if (workspace == nullptr || (reinterpret_cast<std::uintptr_t>(workspace) & 255U) != 0 || workspace_bytes < p.bytes)
        throw std::invalid_argument("glm prefill moe: workspace null, misaligned or too small");
    auto* ws = static_cast<unsigned char*>(workspace);
    fill_items(p, ws);
    ok(hipMemcpyAsync(ws + p.off_groups, p.groups.data(), p.groups.size() * sizeof(GroupDev), hipMemcpyHostToDevice, stream), "groups");
    ok(hipMemcpyAsync(ws + p.off_pair_row, p.pair_row.data(), p.pair_row.size() * 4, hipMemcpyHostToDevice, stream), "pairs");
    ok(hipMemcpyAsync(ws + p.off_pair_group, p.pair_group.data(), p.pair_group.size() * 4, hipMemcpyHostToDevice, stream), "pairs");
    ok(hipMemcpyAsync(ws + p.off_tok_pair, p.tok_pair.data(), p.tok_pair.size() * 4, hipMemcpyHostToDevice, stream), "tokens");
    ok(hipMemcpyAsync(ws + p.off_tok_w, p.tok_w.data(), p.tok_w.size() * 4, hipMemcpyHostToDevice, stream), "tokens");
    ok(hipMemcpyAsync(ws + p.off_items, p.items.data(), p.items.size() * sizeof(Item), hipMemcpyHostToDevice, stream), "items");
    ok(hipMemcpyAsync(ws + p.off_tiles, p.tile_group.data(), p.tile_group.size() * 4, hipMemcpyHostToDevice, stream), "tiles");
    ok(hipStreamSynchronize(stream), "tables");  // the host vectors die with this call
    Tables t{};
    t.groups = reinterpret_cast<const GroupDev*>(ws + p.off_groups);
    t.tile_group = reinterpret_cast<const std::uint32_t*>(ws + p.off_tiles);
    t.pair_row = reinterpret_cast<const std::uint32_t*>(ws + p.off_pair_row);
    t.pair_group = reinterpret_cast<const std::uint32_t*>(ws + p.off_pair_group);
    t.tok_pair = reinterpret_cast<const std::uint32_t*>(ws + p.off_tok_pair);
    t.tok_w = reinterpret_cast<const float*>(ws + p.off_tok_w);
    t.xt = reinterpret_cast<_Float16*>(ws + p.off_xt);
    t.at = reinterpret_cast<_Float16*>(ws + p.off_at);
    t.part_g = reinterpret_cast<float*>(ws + p.off_pg);
    t.part_u = reinterpret_cast<float*>(ws + p.off_pu);
    t.part_d = reinterpret_cast<float*>(ws + p.off_pd);
    t.pairs = static_cast<unsigned>(p.pair_row.size());
    t.rows = v.rows;
    t.tpairs = p.tpairs;
    const auto* items_d = reinterpret_cast<const Item*>(ws + p.off_items);
    std::size_t n_gu = 0;
    for (unsigned c = 0; c < 4; ++c) n_gu += p.gu_count[c];
    mark("start", stream);
    if (gather_v3())
        hipLaunchKernelGGL(gather_transform_v3_kernel, dim3(t.tpairs / 16, 2, kHB / kG3Blocks), dim3(256), 0, stream, v.x, t);
    else
        hipLaunchKernelGGL(gather_transform_kernel, dim3((t.pairs * 2 * kHB + 7) / 8), dim3(256), 0, stream, v.x, t);
    launched("gather transform");
    mark("gather", stream);
    launch_stage<kH, kI>(v.bits_gu, items_d, p.gu_count, p.gu_ctas, stream);
    launched("gate/up");
    mark("gate_up", stream);
    hipLaunchKernelGGL(finish_kernel, dim3((t.pairs * kIB + 7) / 8), dim3(256), 0, stream, t);
    launched("finish");
    mark("finish", stream);
    launch_stage<kI, kH>(v.bits_dn, items_d + n_gu, p.dn_count, p.dn_ctas, stream);
    launched("down");
    mark("down", stream);
    hipLaunchKernelGGL(combine_kernel, dim3((v.rows * kHB + 7) / 8), dim3(256), 0, stream, t, v.shared, v.out, v.eo_debug);
    launched("combine");
    mark("combine", stream);
}

MoeExl3 moe_exl3_from_weights(strix_weights w, unsigned N, unsigned K, unsigned bits) {
    const void* data = nullptr;
    std::uint64_t bytes = 0;
    if (strix_weights_device_span(w, &data, &bytes) != STRIX_SUCCESS || data == nullptr)
        throw std::invalid_argument("glm prefill moe: strix_weights_device_span failed");
    if (N % 128 != 0 || K % 128 != 0 || bits < 1 || bits > 8) throw std::invalid_argument("glm prefill moe: bad matrix shape");
    constexpr std::size_t A = STRIX_TRELLIS_WORKSPACE_ALIGNMENT;  // api.hip prepare_regions alignment
    auto al = [](std::size_t v) { return (v + A - 1) / A * A; };
    const std::size_t tb = std::size_t{N} * K / 8 * bits;
    const std::size_t suh_off = al(tb), svh_off = al(suh_off + std::size_t{K} * 2), total = al(svh_off + std::size_t{N} * 2);
    if (bytes != total)
        throw std::invalid_argument("glm prefill moe: weight span " + std::to_string(bytes) + " bytes, expected " + std::to_string(total) +
                                    " (no bias, layout trellis | suh | svh)");
    const auto* base = static_cast<const unsigned char*>(data);
    MoeExl3 m;
    m.trellis = reinterpret_cast<const std::uint32_t*>(base);
    m.suh = reinterpret_cast<const std::uint16_t*>(base + suh_off);
    m.svh = reinterpret_cast<const std::uint16_t*>(base + svh_off);
    return m;
}

}  // namespace strix::glm::pf
