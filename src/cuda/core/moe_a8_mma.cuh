#pragma once
// FAST arithmetic gsq-iq-bf16mma-cuda-v1 (contract and oracle geometry: native_mma.hpp). GSQ experts on BF16 tensor cores
// with exact operands: integer codes and the BF16 rows the exact route reads; no activation quantization. Selected only by
// the explicit expert arithmetic (qwen_moe.cu enqueue_routed); the exact native kernels of moe_a8.cuh are not touched.
// One block = one CSR tile (<= 64 rows = 4 m16 blocks), 8 warps. A warp owns one n8 column block per matrix and decodes
// the codes of exactly the weight rows it multiplies straight into B registers: every code is decoded once per tile, with
// no shared-memory weight stage and no weight barrier. Only activations are staged (double buffered, one barrier per
// stage) and all 8 warps reuse them. Included at the end of moe_a8.cuh (tile constants, at(), cp.async helpers).
#include "native_mma.hpp"

namespace strix::cuda::core::moe_a8 {

constexpr unsigned kMmaThreads = native_mma::kThreads;
constexpr unsigned kMmaProjectOutputs = native_mma::kProjectOutputs;  // per block and matrix: grid (tiles, ffn / 64); warp w: outputs 8 w .. 8 w + 7
constexpr unsigned kMmaDownOutputs = native_mma::kDownOutputs;    // grid (tiles, hidden / 128); warp w: outputs 16 w .. 16 w + 15
constexpr unsigned kMmaProjectK = native_mma::kProjectK;       // BF16 values per project stage (4 groups)
constexpr unsigned kMmaDownK = native_mma::kDownK;           // h values per down stage (2 groups)
constexpr unsigned kMmaProjectPitch = kMmaProjectK + 16U;  // 72 words per row: the 8-byte A loads of rows g are 8 banks apart
constexpr unsigned kMmaDownPitch = kMmaDownK + 16U;        // 40 words per row: same property
static_assert(kNativeTileRows == 64U, "the tensor-core mapping covers 4 m16 blocks");

// D += A(16x16 bf16, row) * B(16x8 bf16, col), FP32 accumulate.
__device__ __forceinline__ void mma_bf16(float (&d)[4], const unsigned (&a)[4], unsigned b0, unsigned b1) {
    asm("mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 {%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};\n"
        : "+f"(d[0]), "+f"(d[1]), "+f"(d[2]), "+f"(d[3])
        : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b0), "r"(b1));
}
// A fragment for rows (row, row + 8), logical k .. k + 3 of the half (native_mma::logical_k): one 8-byte load per row.
__device__ __forceinline__ void mma_rows(const std::uint16_t* base, unsigned pitch, unsigned row, unsigned k, unsigned (&a)[4]) {
    const uint2 r0 = *reinterpret_cast<const uint2*>(base + row * pitch + k);
    const uint2 r1 = *reinterpret_cast<const uint2*>(base + (row + 8U) * pitch + k);
    native_mma::a_regs(r0.x, r0.y, r1.x, r1.y, a);
}
// Field bytes of one weight row (column g of this lane) for one group: the lane's two words (word_index h = 0, 1) and the
// scales. Loads only, so the loops issue them one group ahead (weights do not depend on the shared-memory stage).
template <class Decode>
__device__ __forceinline__ typename Decode::Raw mma_raw(const std::uint8_t* row, unsigned group, unsigned lane) {
    return Decode::load(row, group, native_mma::word_index(lane, 0U), native_mma::word_index(lane, 1U));
}
// B registers and C-column scales of that row and group from its fields (branch-free decode, codebook in shared memory).
// C element i of a lane is column 2 q + (i & 1): its scale comes from the lanes that own those columns (lanes 8 q and
// 8 q + 4).
template <class Decode>
__device__ __forceinline__ void mma_weights(const typename Decode::Raw& raw, unsigned group, const weight::GsqSmemGrid& grid,
                                            unsigned lane, unsigned (&b)[2][2], float (&scale)[2][2]) {
    namespace nm = native_mma;
#pragma unroll
    for (unsigned h = 0; h < 2U; ++h) nm::code_pairs(Decode::word(raw, h, nm::word_index(lane, h), grid), b[h][0], b[h][1]);
    float first = 0.0F, second = 0.0F;
    Decode::scales(raw, group, first, second);
    scale[0][0] = __shfl_sync(0xFFFFFFFFU, first, nm::scale_lane(lane, 0U));
    scale[0][1] = __shfl_sync(0xFFFFFFFFU, first, nm::scale_lane(lane, 1U));
    if constexpr (Decode::halves) {
        scale[1][0] = __shfl_sync(0xFFFFFFFFU, second, nm::scale_lane(lane, 0U));
        scale[1][1] = __shfl_sync(0xFFFFFFFFU, second, nm::scale_lane(lane, 1U));
    } else {
        scale[1][0] = scale[0][0]; scale[1][1] = scale[0][1];
    }
}
__device__ __forceinline__ void mma_scale_acc(float (&acc)[4], const float (&p)[4], const float (&scale)[2]) {
#pragma unroll
    for (unsigned i = 0; i < 4U; ++i) acc[i] = __fmaf_rn(p[i], scale[i & 1U], acc[i]);
}

// Field loader of the BF16 tensor-core kernels (lane words native_mma::word_index). The TURBO int8 kernels pass their own
// loader (moe_turbo_i8.cuh TurboRaw: the int8 fragment's words) to the same ring.
template <class Decode> struct MmaRaw {
    static __device__ __forceinline__ typename Decode::Raw get(const std::uint8_t* row, unsigned group, unsigned lane) {
        return mma_raw<Decode>(row, group, lane);
    }
};
// Weight-field ring of depth D (native_mma::ring_group): slot 0 is the group consumed next. Loads only, issued D groups
// ahead; D = 1 is the one-group lookahead of the tile form. One ring for every tensor-core expert kernel of the backend
// (BF16 and TURBO int8): Load supplies the lane's field bytes of one row and group.
template <class Decode, unsigned D, class Load = MmaRaw<Decode>>
__device__ __forceinline__ void ring_fill(typename Decode::Raw (&ring)[D][2], const std::uint8_t* row0, const std::uint8_t* row1,
                                          unsigned groups, unsigned lane) {
#pragma unroll
    for (unsigned i = 0; i < D; ++i) {
        const unsigned group = native_mma::ring_group(0U, i, groups);
        ring[i][0] = Load::get(row0, group, lane);
        ring[i][1] = Load::get(row1, group, lane);
    }
}
// After the caller copied slot 0 (group `group`): shift the ring and load group ring_group(group, D) into the last slot.
template <class Decode, unsigned D, class Load = MmaRaw<Decode>>
__device__ __forceinline__ void ring_advance(typename Decode::Raw (&ring)[D][2], const std::uint8_t* row0, const std::uint8_t* row1,
                                             unsigned group, unsigned groups, unsigned lane) {
    const unsigned next = native_mma::ring_group(group, D, groups);
#pragma unroll
    for (unsigned i = 0; i + 1U < D; ++i) { ring[i][0] = ring[i + 1U][0]; ring[i][1] = ring[i + 1U][1]; }
    ring[D - 1U][0] = Load::get(row0, next, lane);
    ring[D - 1U][1] = Load::get(row1, next, lane);
}

// grid (tiles, ffn / 64), tile-major raster (native_mma::tile_major), 256 threads. Writes h = native_activation(gate, up) into intermediates[row][slot * 2 * ffn + o]
// with the exact route's layout and fail-closed flags. Gate and up of one output are in the same lane (same C element).
// MB m16 blocks per tile and a D-deep weight ring: (kTileMBlocks, kTileAhead) for any tile, (kDecodeMBlocks,
// kDecodeAhead) when native_mma::decode_form(rows) holds for the launch; bitwise the same outputs (see native_mma.hpp).
template <unsigned T, class Binding, unsigned MB = native_mma::kTileMBlocks, unsigned D = native_mma::kTileAhead>
__global__ __launch_bounds__(kMmaThreads, 2) void native_mma_project_kernel(const std::uint8_t* gate, const std::uint8_t* up,
                                                                            typename Binding::Rows v) {
    static_assert(MB >= 1U && MB <= native_mma::kTileMBlocks && D >= 1U, "tile capacity and ring depth");
    using Decode = weight::GsqDecode<T>;
    constexpr unsigned hidden = Binding::hidden, ffn = Binding::ffn, projection_nonfinite = Binding::projection_nonfinite;
    constexpr unsigned stages = hidden / kMmaProjectK, groups_per_stage = kMmaProjectK / 32U;
    constexpr std::uint64_t row_bytes = std::uint64_t(hidden / Decode::be) * Decode::bb;
    static_assert(hidden % kMmaProjectK == 0U && hidden % Decode::be == 0U && ffn % kMmaProjectOutputs == 0U,
                  "tensor-core project covers whole stages, storage blocks and output blocks");
    const auto& csr = v.csr;
    const native_mma::TileSlice ts = native_mma::tile_major(blockIdx.x, blockIdx.y, gridDim.x, ffn / kMmaProjectOutputs);
    if (csr.summary[0] || ts.tile >= *csr.tile_count) return;
    const unsigned tile_begin = csr.tile_begins[ts.tile], count = csr.tile_begins[ts.tile + 1U] - tile_begin;
    const unsigned expert = csr.tile_experts[ts.tile];
    if constexpr (MB < native_mma::kTileMBlocks) if (count > 16U * MB) {  // decode form over a wider tile: fail closed
        if (threadIdx.x == 0U) atomicOr(csr.summary, projection_nonfinite);
        return;
    }
    __shared__ alignas(16) std::uint16_t X[2][16U * MB][kMmaProjectPitch];
    __shared__ unsigned row_of[kNativeTileRows], slot_of[kNativeTileRows];
    __shared__ std::uint32_t grid_words[Decode::grid_words ? Decode::grid_words : 1U];  // the type's codebook (<= 8 KB)
    const unsigned t = threadIdx.x, lane = t & 31U, warp = t >> 5, g = lane >> 2, q = lane & 3U;
    const unsigned out_base = native_mma::project_output(ts.slice, warp, 0U);
    if (t < kNativeTileRows) {
        const unsigned a = t < count ? t : 0U;
        row_of[t] = csr.rows[tile_begin + a]; slot_of[t] = csr.slots[tile_begin + a];
    }
    weight::gsq_grid_fill<T>(grid_words, t, kMmaThreads);
    __syncthreads();
    const weight::GsqSmemGrid grid{grid_words};
    // Thread t copies word t % 64 of rows t / 64 + 4 i. Rows >= count are not written: an MMA row reaches only its own C
    // row, and those C rows are never stored.
    const auto load = [&](unsigned buf, unsigned s) {
        const unsigned w = t & 63U;
        for (unsigned r = t >> 6; r < count; r += kMmaThreads / 64U) {
            const std::uint16_t* src = at(v.input, row_of[r]) + s * kMmaProjectK + 2U * w;
            std::uint16_t* dst = &X[buf][r][2U * w];
            if ((reinterpret_cast<std::uintptr_t>(src) & 3U) == 0U) ptx::cp_async4(dst, src);
            else { dst[0] = src[0]; dst[1] = src[1]; }
        }
        cp_commit();
    };
    const std::uint64_t wr = std::uint64_t(expert) * ffn + out_base + g;
    const std::uint8_t* wrow[2] = {gate + wr * row_bytes, up + wr * row_bytes};
    const unsigned mblocks = native_mma::m_blocks(count);  // block-uniform, <= MB
    float acc[MB][2][4];
#pragma unroll
    for (unsigned m = 0; m < MB; ++m)
#pragma unroll
        for (unsigned mat = 0; mat < 2U; ++mat)
#pragma unroll
            for (unsigned i = 0; i < 4U; ++i) acc[m][mat][i] = 0.0F;
    load(0U, 0U);
    // Weight fields of the next D groups are loaded before this group's decode and MMAs (and across the stage barrier):
    // the global round trips overlap D groups of work. The last groups re-load the last one (no read past the row).
    constexpr unsigned groups = stages * groups_per_stage;
    typename Decode::Raw ahead[D][2];
    ring_fill<Decode>(ahead, wrow[0], wrow[1], groups, lane);
#pragma unroll 1
    for (unsigned s = 0; s < stages; ++s) {
        cp_wait_all();
        __syncthreads();  // stage s landed; every warp finished stage s - 1, so its buffer is free
        if (s + 1U < stages) load((s + 1U) & 1U, s + 1U);
        const std::uint16_t* xs = &X[s & 1U][0][0];
#pragma unroll
        for (unsigned j = 0; j < groups_per_stage; ++j) {
            const unsigned group = s * groups_per_stage + j;
            const typename Decode::Raw raw[2] = {ahead[0][0], ahead[0][1]};
            ring_advance<Decode>(ahead, wrow[0], wrow[1], group, groups, lane);
            unsigned b[2][2][2];
            float scale[2][2][2];
            mma_weights<Decode>(raw[0], group, grid, lane, b[0], scale[0]);
            mma_weights<Decode>(raw[1], group, grid, lane, b[1], scale[1]);
#pragma unroll
            for (unsigned m = 0; m < MB; ++m) {
                if (m >= mblocks) break;
                unsigned a0[4], a1[4];
                mma_rows(xs, kMmaProjectPitch, 16U * m + g, 32U * j + 4U * q, a0);
                mma_rows(xs, kMmaProjectPitch, 16U * m + g, 32U * j + 16U + 4U * q, a1);
#pragma unroll
                for (unsigned mat = 0; mat < 2U; ++mat) {
                    float p[4] = {0.0F, 0.0F, 0.0F, 0.0F};
                    mma_bf16(p, a0, b[mat][0][0], b[mat][0][1]);
                    if constexpr (Decode::halves) {
                        mma_scale_acc(acc[m][mat], p, scale[mat][0]);
                        p[0] = p[1] = p[2] = p[3] = 0.0F;
                    }
                    mma_bf16(p, a1, b[mat][1][0], b[mat][1][1]);
                    mma_scale_acc(acc[m][mat], p, scale[mat][1]);
                }
            }
        }
    }
#pragma unroll
    for (unsigned m = 0; m < MB; ++m)
#pragma unroll
        for (unsigned i = 0; i < 4U; ++i) {
            const unsigned row = 16U * m + native_mma::c_row(lane, i);
            if (row < count) {
                const float h = Binding::native_activation(acc[m][0][i], acc[m][1][i]);
                at(v.intermediates, row_of[row])[slot_of[row] * 2U * ffn + out_base + native_mma::c_col(lane, i)] = h;
                if (!isfinite(h)) { atomicOr(at(v.status, row_of[row]), projection_nonfinite); atomicOr(csr.summary, projection_nonfinite); }
            }
        }
}

// grid (tiles, hidden / 128), tile-major raster (native_mma::tile_major), 256 threads. Reads h from intermediates[row][slot * 2 * ffn + k] and writes FP32
// rawdots[row][slot * hidden + o] (the exact route's storage; merge / commit unchanged). h enters as hi + lo BF16
// (native_mma::split); both MMAs of a half share the code fragment. MB and D as in the project kernel.
template <unsigned T, class Binding, unsigned MB = native_mma::kTileMBlocks, unsigned D = native_mma::kTileAhead>
__global__ __launch_bounds__(kMmaThreads, 2) void native_mma_down_kernel(const std::uint8_t* down, typename Binding::Rows v) {
    static_assert(MB >= 1U && MB <= native_mma::kTileMBlocks && D >= 1U, "tile capacity and ring depth");
    namespace nm = native_mma;
    using Decode = weight::GsqDecode<T>;
    constexpr unsigned hidden = Binding::hidden, ffn = Binding::ffn;
    constexpr unsigned stages = ffn / kMmaDownK, groups_per_stage = kMmaDownK / 32U;
    constexpr std::uint64_t row_bytes = std::uint64_t(ffn / Decode::be) * Decode::bb;
    static_assert(ffn % kMmaDownK == 0U && ffn % Decode::be == 0U && hidden % kMmaDownOutputs == 0U,
                  "tensor-core down covers whole stages, storage blocks and output blocks");
    const auto& csr = v.csr;
    const native_mma::TileSlice ts = native_mma::tile_major(blockIdx.x, blockIdx.y, gridDim.x, hidden / kMmaDownOutputs);
    if (csr.summary[0] || ts.tile >= *csr.tile_count) return;
    const unsigned tile_begin = csr.tile_begins[ts.tile], count = csr.tile_begins[ts.tile + 1U] - tile_begin;
    const unsigned expert = csr.tile_experts[ts.tile];
    if constexpr (MB < native_mma::kTileMBlocks) if (count > 16U * MB) {  // as in the project kernel: fail closed
        if (threadIdx.x == 0U) atomicOr(csr.summary, Binding::projection_nonfinite);
        return;
    }
    __shared__ alignas(16) std::uint16_t Xh[2][16U * MB][kMmaDownPitch];
    __shared__ alignas(16) std::uint16_t Xl[2][16U * MB][kMmaDownPitch];
    __shared__ unsigned row_of[kNativeTileRows], slot_of[kNativeTileRows];
    __shared__ std::uint32_t grid_words[Decode::grid_words ? Decode::grid_words : 1U];  // the type's codebook (none: 4 B)
    const unsigned t = threadIdx.x, lane = t & 31U, warp = t >> 5, g = lane >> 2, q = lane & 3U;
    const unsigned out_base = native_mma::down_output(ts.slice, warp, 0U, 0U);
    if (t < kNativeTileRows) {
        const unsigned a = t < count ? t : 0U;
        row_of[t] = csr.rows[tile_begin + a]; slot_of[t] = csr.slots[tile_begin + a];
    }
    weight::gsq_grid_fill<T>(grid_words, t, kMmaThreads);
    __syncthreads();
    const weight::GsqSmemGrid grid{grid_words};
    // Thread t holds 16 h values of the next stage (row t / 4, values 16 (t % 4) ..) in registers across the MMAs and then
    // writes their hi / lo halves: the split needs the values, so this is a register prefetch, not a shared-memory copy.
    const unsigned sr = t >> 2, sk = 16U * (t & 3U);
    const bool sload = sr < count;
    const float* hrow = sload ? at(v.intermediates, row_of[sr]) + slot_of[sr] * 2U * ffn + sk : nullptr;
    const bool vec = (reinterpret_cast<std::uintptr_t>(hrow) & 15U) == 0U;
    float hv[16];
    const auto fetch = [&](unsigned s) {
        if (!sload) return;
        if (vec) {
#pragma unroll
            for (unsigned k = 0; k < 4U; ++k) {
                const float4 f = reinterpret_cast<const float4*>(hrow + s * kMmaDownK)[k];
                hv[4U * k] = f.x; hv[4U * k + 1U] = f.y; hv[4U * k + 2U] = f.z; hv[4U * k + 3U] = f.w;
            }
        } else {
#pragma unroll
            for (unsigned k = 0; k < 16U; ++k) hv[k] = hrow[s * kMmaDownK + k];
        }
    };
    const auto put = [&](unsigned buf) {
        if (!sload) return;
        unsigned hi[8], lo[8];
#pragma unroll
        for (unsigned k = 0; k < 8U; ++k) {
            std::uint32_t h0, l0, h1, l1;
            nm::split(hv[2U * k], h0, l0); nm::split(hv[2U * k + 1U], h1, l1);
            hi[k] = h0 | (h1 << 16U); lo[k] = l0 | (l1 << 16U);
        }
        uint4* dh = reinterpret_cast<uint4*>(&Xh[buf][sr][sk]);
        uint4* dl = reinterpret_cast<uint4*>(&Xl[buf][sr][sk]);
        dh[0] = make_uint4(hi[0], hi[1], hi[2], hi[3]); dh[1] = make_uint4(hi[4], hi[5], hi[6], hi[7]);
        dl[0] = make_uint4(lo[0], lo[1], lo[2], lo[3]); dl[1] = make_uint4(lo[4], lo[5], lo[6], lo[7]);
    };
    const std::uint8_t* wrow[2];
#pragma unroll
    for (unsigned n = 0; n < 2U; ++n) wrow[n] = down + (std::uint64_t(expert) * hidden + out_base + 8U * n + g) * row_bytes;
    const unsigned mblocks = native_mma::m_blocks(count);  // block-uniform, <= MB
    float acc[MB][2][4];
#pragma unroll
    for (unsigned m = 0; m < MB; ++m)
#pragma unroll
        for (unsigned n = 0; n < 2U; ++n)
#pragma unroll
            for (unsigned i = 0; i < 4U; ++i) acc[m][n][i] = 0.0F;
    fetch(0U); put(0U);
    // Weight fields D groups ahead, as in the project kernel (the last groups re-load the last one).
    constexpr unsigned groups = stages * groups_per_stage;
    typename Decode::Raw ahead[D][2];
    ring_fill<Decode>(ahead, wrow[0], wrow[1], groups, lane);
    __syncthreads();
#pragma unroll 1
    for (unsigned s = 0; s < stages; ++s) {
        const bool more = s + 1U < stages;
        if (more) fetch(s + 1U);  // in flight during this stage's MMAs
        const std::uint16_t* xh = &Xh[s & 1U][0][0];
        const std::uint16_t* xl = &Xl[s & 1U][0][0];
#pragma unroll
        for (unsigned j = 0; j < groups_per_stage; ++j) {
            const unsigned group = s * groups_per_stage + j;
            const typename Decode::Raw raw[2] = {ahead[0][0], ahead[0][1]};
            ring_advance<Decode>(ahead, wrow[0], wrow[1], group, groups, lane);
            unsigned b[2][2][2];
            float scale[2][2][2];
            mma_weights<Decode>(raw[0], group, grid, lane, b[0], scale[0]);
            mma_weights<Decode>(raw[1], group, grid, lane, b[1], scale[1]);
#pragma unroll
            for (unsigned m = 0; m < MB; ++m) {
                if (m >= mblocks) break;
                unsigned h0[4], l0[4], h1[4], l1[4];
                mma_rows(xh, kMmaDownPitch, 16U * m + g, 32U * j + 4U * q, h0);
                mma_rows(xl, kMmaDownPitch, 16U * m + g, 32U * j + 4U * q, l0);
                mma_rows(xh, kMmaDownPitch, 16U * m + g, 32U * j + 16U + 4U * q, h1);
                mma_rows(xl, kMmaDownPitch, 16U * m + g, 32U * j + 16U + 4U * q, l1);
#pragma unroll
                for (unsigned n = 0; n < 2U; ++n) {
                    float p[4] = {0.0F, 0.0F, 0.0F, 0.0F};
                    mma_bf16(p, h0, b[n][0][0], b[n][0][1]);
                    mma_bf16(p, l0, b[n][0][0], b[n][0][1]);
                    if constexpr (Decode::halves) {
                        mma_scale_acc(acc[m][n], p, scale[n][0]);
                        p[0] = p[1] = p[2] = p[3] = 0.0F;
                    }
                    mma_bf16(p, h1, b[n][1][0], b[n][1][1]);
                    mma_bf16(p, l1, b[n][1][0], b[n][1][1]);
                    mma_scale_acc(acc[m][n], p, scale[n][1]);
                }
            }
        }
        if (more) put((s + 1U) & 1U);  // the other buffer: last read in stage s - 1, before the previous barrier
        __syncthreads();
    }
#pragma unroll
    for (unsigned m = 0; m < MB; ++m)
#pragma unroll
        for (unsigned n = 0; n < 2U; ++n)
#pragma unroll
            for (unsigned i = 0; i < 4U; i += 2U) {
                const unsigned row = 16U * m + nm::c_row(lane, i);
                if (row < count) {
                    float* dst = at(v.rawdots, row_of[row]) + slot_of[row] * hidden + out_base + 8U * n + nm::c_col(lane, i);
                    *reinterpret_cast<float2*>(dst) = make_float2(acc[m][n][i], acc[m][n][i + 1U]);
                }
            }
}

} // namespace strix::cuda::core::moe_a8
