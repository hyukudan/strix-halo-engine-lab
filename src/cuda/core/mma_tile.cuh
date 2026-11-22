// Shared CUDA core: FP16 tensor-core tile mechanics for FAST forms (sm_80+, built for sm_120). One mechanism, two drivers:
//   (1) register-B warp GEMM (warp_gemm_k32 + Int8Codes / Int4Codes traits, perm8 K order, below), and
//   (2) the CTA-level double-buffered smem tile (Tile<NT8, WeightFmt>, at the end): weights staged to smem as exact f16 codes,
//       B through ldmatrix; WeightFmt (Q40Blocks = 18-byte Q4_0 blocks) is the weight-format trait of its load / store half.
// Both share the exact unpack (i8/u4 magic numbers) and the per-32-K-block FP32 scale epilogue (block_fma).
// The staged CTA mainloop also serves grouped BF16: its inputs are already BF16, the packed weights
// decode to BF16 before MMA, and the running FP32 accumulator has no deferred scale epilogue.
// A = activations (FP32 in global, FP16 in smem), B = quantised weights (int8 or int4 codes + one FP32 scale per 32 K),
// accumulation FP32 (mma.sync m16n8k16 f16). Codes are exact in FP16; the scale is applied to the FP32 block partial:
//   acc += scale * (sum over the 32-K block of act * code)     -> the weights are never rounded.
//
// K permutation "perm8" (A and B agree, no runtime cost). Inside every 32-wide K block (the scale block) and for MMA step
// j in {0,1}, the thread t (lane & 3) holds the actual K indices 8t + 4j + {0,1} in MMA slots 2t, 2t+1 and 8t + 4j + {2,3} in
// slots 8 + 2t, 9 + 2t. So one thread's 8 consecutive codes of a weight row (uint2 of int8 / one uint32 of nibbles)
// feed both k16 steps of the block, and A is stored to smem in permuted order by stage_k4().
//
// Fragment map (PTX m16n8k16, g = lane >> 2, t = lane & 3): a0 (row g, slots 2t..2t+1), a1 (row g+8), a2 (row g, slots 8+2t..),
// a3 (row g+8); b0 (slots 2t..2t+1, col g), b1 (slots 8+2t.., col g); c0,c1 (row g, cols 2t, 2t+1), c2,c3 (row g+8).
#pragma once

#include "../../moe_core/grouped_microscaled_rows.hpp"
#include "../../moe_core/grouped_tall_plan.hpp"
#include "native_mma.hpp"

#if defined(__CUDACC__) && !defined(__HIPCC__)
#define STRIX_MMA_TILE_HD __host__ __device__ __forceinline__
#else
#define STRIX_MMA_TILE_HD inline
#endif

namespace strix::cuda::core::mma::detail {

struct LdsmAddress {
    unsigned row;
    unsigned segment;
};

STRIX_MMA_TILE_HD constexpr LdsmAddress ldsm_a_address(unsigned lane) {
    return {lane & 15U, lane >> 4U};
}

template<unsigned Words>
STRIX_MMA_TILE_HD constexpr LdsmAddress ldsm_b_address(unsigned lane) {
    return {(Words == 4U ? (lane >> 4U) * 8U : 0U) + (lane & 7U), (lane >> 3U) & 1U};
}

}  // namespace strix::cuda::core::mma::detail

#if defined(__CUDACC__) && !defined(__HIPCC__)
#include <cuda_fp16.h>
#include <cuda_runtime.h>

#include <cstdint>

#include "ptx.cuh"
#include "../weight/arch/sm120_sources.cuh"

namespace strix::cuda::core::mma {

constexpr unsigned kPadHalves = 8;  // smem row pad (16 B): ldmatrix rows hit distinct banks when pitch/8 is odd

// FP32 x4 (actual K = 4m .. 4m+3 of a 32-block, m in 0..7) -> FP16, stored in perm8 order into the 32-half block at `blk`.
__device__ __forceinline__ void stage_k4(__half* blk, unsigned m, float4 v) {
    const unsigned t = m >> 1U, j = m & 1U;
    *reinterpret_cast<__half2*>(blk + j * 16U + 2U * t) = __floats2half2_rn(v.x, v.y);
    *reinterpret_cast<__half2*>(blk + j * 16U + 8U + 2U * t) = __floats2half2_rn(v.z, v.w);
}

// A 16x16 fragment at tile row 0. Pitch is in units of the shared storage element.
template<class Storage>
__device__ __forceinline__ void ldsm_a(unsigned (&a)[4], const Storage* tile, unsigned pitch, unsigned lane) {
    constexpr unsigned chunk = 16U / sizeof(Storage);
    const auto address = detail::ldsm_a_address(lane);
    ptx::ldsm_x4(a, tile + address.row * pitch + address.segment * chunk);
}

// One or two adjacent B fragments. Lanes 16..31 provide addresses only for the two-fragment load.
template<unsigned Words, class Storage>
__device__ __forceinline__ void ldsm_b(unsigned (&b)[Words], const Storage* tile, unsigned pitch, unsigned lane) {
    static_assert(Words == 2U || Words == 4U);
    constexpr unsigned chunk = 16U / sizeof(Storage);
    const auto index = detail::ldsm_b_address<Words>(lane);
    const auto* address = tile + index.row * pitch + index.segment * chunk;
    if constexpr (Words == 4U) ptx::ldsm_x4(b, address);
    else ptx::ldsm_x2(b, address);
}

template<bool Bf16 = false>
__device__ __forceinline__ void mma_k16(float (&acc)[4], const unsigned (&a)[4], unsigned b0, unsigned b1) {
    if constexpr (Bf16) ptx::mma_bf16(acc, a[0], a[1], a[2], a[3], b0, b1);
    else ptx::mma_f16(acc, a[0], a[1], a[2], a[3], b0, b1);
}

// Block-uniform K mainloop for staged MMA operands. Load the next step into registers before
// computing the current one, then publish it to the other shared buffer. Callers own the
// arithmetic and fragment layouts. The final barrier leaves both buffers safe to reuse.
template<class Count, class Load, class Store, class Compute>
__device__ __forceinline__ void run_staged(Count steps, Load load, Store store, Compute compute) {
    load(0);
    store(0);
    __syncthreads();
    for (Count step = 0; step < steps; ++step) {
        const int current = static_cast<int>(step & 1);
        if (step + 1 < steps) load(step + 1);
        compute(current);
        if (step + 1 < steps) store(current ^ 1);
        __syncthreads();
    }
}

// Exact integer -> half2 unpack (magic-number trick: 0x6400 | u8 = 1024 + u8).
__device__ __forceinline__ unsigned i8_pair_to_half2(unsigned w, unsigned sel) {  // sel: 0x4140 (bytes 0,1) or 0x4342 (bytes 2,3)
    const unsigned v = __byte_perm(w ^ 0x80808080U, 0x64646464U, sel);  // {1024 + (c0^0x80), 1024 + (c1^0x80)}
    const __half2 r = __hsub2(*reinterpret_cast<const __half2*>(&v), __float2half2_rn(1152.0F));
    return *reinterpret_cast<const unsigned*>(&r);
}
__device__ __forceinline__ unsigned u4_pair_to_half2(unsigned w, unsigned shift) {  // nibbles at bit shift, shift + 4; offset 8
    const unsigned v = (((w >> shift) & 0xFU) | (((w >> (shift + 4U)) & 0xFU) << 16U)) | 0x64006400U;
    const __half2 r = __hsub2(*reinterpret_cast<const __half2*>(&v), __float2half2_rn(1032.0F));
    return *reinterpret_cast<const unsigned*>(&r);
}
// Same, for u already shifted so the two nibbles sit at bits 0 and 8 (Q4_0 byte pairs: low / high halves of the block).
__device__ __forceinline__ unsigned u4_bytes_to_half2(unsigned u) {
    return ::strix::cuda::weight::Q40F16Source::unpack_pair(u);
}

// Per-32-K-block FP32 scale epilogue: acc += scale * (block partial) for one c fragment (s0 / s1: scales of columns 2t, 2t+1).
__device__ __forceinline__ void block_fma(float (&acc)[4], const float (&blk)[4], float s0, float s1) {
    acc[0] = __fmaf_rn(s0, blk[0], acc[0]);
    acc[1] = __fmaf_rn(s1, blk[1], acc[1]);
    acc[2] = __fmaf_rn(s0, blk[2], acc[2]);
    acc[3] = __fmaf_rn(s1, blk[3], acc[3]);
}

// Code traits: W words hold the thread's 8 consecutive codes (k 8t .. 8t+7 of the 32-block) of one weight column n = g.
struct Int8Codes {  // signed int8 (Q8_0 style), 2 words, byte i of word 0 = k 8t + i, word 1 = k 8t + 4 + i
    static constexpr unsigned W = 2;
    __device__ static __forceinline__ void b_frag(const unsigned (&w)[W], unsigned step, unsigned& b0, unsigned& b1) {
        b0 = i8_pair_to_half2(w[step], 0x4140U);
        b1 = i8_pair_to_half2(w[step], 0x4342U);
    }
};
struct Int4Codes {  // unsigned nibbles with offset 8 (Q4_0 style value = nibble - 8), 1 word, nibble i = k 8t + i
    static constexpr unsigned W = 1;
    __device__ static __forceinline__ void b_frag(const unsigned (&w)[W], unsigned step, unsigned& b0, unsigned& b1) {
        b0 = u4_pair_to_half2(w[0], 16U * step);
        b1 = u4_pair_to_half2(w[0], 16U * step + 8U);
    }
};

// One 32-K block for MT m16 tiles x one n8 tile. a[mt][step] are the A fragments (ldsm_a of the 2 steps of the block),
// w the thread's codes of column g, s0/s1 the FP32 scales of the columns 2t and 2t+1 (c-fragment columns).
template <unsigned MT, class Codes>
__device__ __forceinline__ void warp_gemm_k32(float (&acc)[MT][4], const unsigned (&a)[MT][2][4], const unsigned (&w)[Codes::W],
    float s0, float s1) {
    unsigned b[2][2];
    Codes::b_frag(w, 0U, b[0][0], b[0][1]);
    Codes::b_frag(w, 1U, b[1][0], b[1][1]);
#pragma unroll
    for (unsigned mt = 0; mt < MT; ++mt) {
        float blk[4] = {0.0F, 0.0F, 0.0F, 0.0F};
        ptx::mma_f16(blk, a[mt][0][0], a[mt][0][1], a[mt][0][2], a[mt][0][3], b[0][0], b[0][1]);
        ptx::mma_f16(blk, a[mt][1][0], a[mt][1][1], a[mt][1][2], a[mt][1][3], b[1][0], b[1][1]);
        block_fma(acc[mt], blk, s0, s1);
    }
}


// Weight-format trait of the smem tile: Q4_0-style blocks, 18 B (2-byte aligned): fp16 d, 16 bytes; byte j low nibble = element j,
// high nibble = element j + 16. Regs = one block as 9 u16 (loaded for all threads, two threads per output row); store() turns the
// nibbles into exact f16 codes (code = nibble - 8) of elements kh * 16 .. +15 and the scale into FP32 (kh == 0 only).
using Q40Blocks = ::strix::cuda::weight::Q40TileSource;

// CTA tile: BM = 128 rows x BN = 16 * NT8 outputs per CTA of 256 threads (8 warps: 4 along M x 2 along N, warp tile 32 x 8 * NT8).
template <int NT8, class WeightFmt = Q40Blocks>
struct Tile {
    static constexpr int kBM = 128, kBK = 32, kLd = 40;  // smem row stride in halves (80 B: conflict-free ldmatrix)
    static constexpr int kBN = 16 * NT8;
    static_assert(kBN == 128 || kBN == 64, "weight loader: two threads per output row");
    struct Smem {
        __half a[2][kBM * kLd];
        __half w[2][kBN * kLd];
        float s[2][kBN];
    };
    struct Regs { float4 a[4]; typename WeightFmt::Regs w; };

    // Global -> registers for K step kb. x: A row base of the stream (row stride ld floats, K offset applied by caller);
    // wq: weight block 0 of the stream for output 0 (row stride row_bytes); n0: first output of this CTA.
    __device__ static __forceinline__ void load(Regs& r, const float* x, std::uint64_t ld, unsigned m0, unsigned rows, int kb,
                                                const std::uint8_t* wq, std::uint64_t row_bytes, unsigned n0) {
        const unsigned t = threadIdx.x, row = t >> 1U, kh = t & 1U;
        const float* p = x + static_cast<std::uint64_t>(min(m0 + row, rows - 1U)) * ld + kb * kBK + kh * 16U;
#pragma unroll
        for (int i = 0; i < 4; ++i) r.a[i] = *reinterpret_cast<const float4*>(p + 4 * i);
        const unsigned nl = (t % (kBN * 2U)) >> 1U;
        WeightFmt::load(r.w, wq + static_cast<std::uint64_t>(n0 + nl) * row_bytes + static_cast<std::uint64_t>(kb) * WeightFmt::kBlockBytes);
    }
    // Registers -> shared (FP32 -> f16 for A; nibbles -> exact f16 codes for W; scale -> FP32 for the epilogue).
    __device__ static __forceinline__ void store(const Regs& r, Smem& sm, int buf) {
        const unsigned t = threadIdx.x, row = t >> 1U, kh = t & 1U;
        uint4 pa[2];
        unsigned* pw = reinterpret_cast<unsigned*>(pa);
#pragma unroll
        for (int i = 0; i < 4; ++i) {
            __half2 lo = __floats2half2_rn(r.a[i].x, r.a[i].y), hi = __floats2half2_rn(r.a[i].z, r.a[i].w);
            pw[2 * i] = *reinterpret_cast<unsigned*>(&lo);
            pw[2 * i + 1] = *reinterpret_cast<unsigned*>(&hi);
        }
        uint4* da = reinterpret_cast<uint4*>(&sm.a[buf][row * kLd + kh * 16U]);
        da[0] = pa[0];
        da[1] = pa[1];
        // W: thread t -> output t / 2, half kh: elements kh * 16 .. +15 are nibble kh of bytes 0..15.
        if (t < kBN * 2U) {
            const unsigned nl = t >> 1U;
            WeightFmt::store(r.w, &sm.w[buf][nl * kLd + kh * 16U], &sm.s[buf][nl], kh);
        }
    }
    // One K step (2 x k16) on shared buffer `buf`: partial = A * codes in fresh accumulators, acc += scale * partial.
    __device__ static __forceinline__ void compute(float (&acc)[2][NT8][4], const Smem& sm, int buf) {
        const unsigned lane = threadIdx.x & 31U, wid = threadIdx.x >> 5U, wm = wid & 3U, wn = wid >> 2U;
        unsigned af[2][2][4];
#pragma unroll
        for (int ks = 0; ks < 2; ++ks)
#pragma unroll
            for (int mi = 0; mi < 2; ++mi)
                ldsm_a(af[ks][mi], &sm.a[buf][(wm * 32U + mi * 16U) * kLd + ks * 16U], kLd, lane);
#pragma unroll
        for (int jp = 0; jp < NT8 / 2; ++jp) {
            unsigned bf[2][4];
            const unsigned nrow = wn * (NT8 * 8U) + jp * 16U;
#pragma unroll
            for (int ks = 0; ks < 2; ++ks)
                ldsm_b(bf[ks], &sm.w[buf][nrow * kLd + ks * 16U], kLd, lane);
#pragma unroll
            for (int jj = 0; jj < 2; ++jj) {
                const int j = jp * 2 + jj;
                const float2 sc = *reinterpret_cast<const float2*>(&sm.s[buf][wn * (NT8 * 8U) + j * 8U + 2U * (lane & 3U)]);
#pragma unroll
                for (int mi = 0; mi < 2; ++mi) {
                    float t[4] = {0.0F, 0.0F, 0.0F, 0.0F};
#pragma unroll
                    for (int ks = 0; ks < 2; ++ks)
                        mma_k16(t, af[ks][mi], bf[ks][2 * jj], bf[ks][2 * jj + 1]);
                    block_fma(acc[mi][j], t, sc.x, sc.y);
                }
            }
        }
    }
    // acc = sum over `kblocks` K steps of one stream. Ends with a barrier (shared buffers reusable by the next stream).
    __device__ static void run(float (&acc)[2][NT8][4], Smem& sm, const float* x, std::uint64_t ld, unsigned m0, unsigned rows,
                               const std::uint8_t* wq, std::uint64_t row_bytes, unsigned n0, int kblocks) {
#pragma unroll
        for (int mi = 0; mi < 2; ++mi)
#pragma unroll
            for (int j = 0; j < NT8; ++j) acc[mi][j][0] = acc[mi][j][1] = acc[mi][j][2] = acc[mi][j][3] = 0.0F;
        Regs r;
        run_staged(kblocks, [&](int kb) { load(r, x, ld, m0, rows, kb, wq, row_bytes, n0); },
                   [&](int buf) { store(r, sm, buf); }, [&](int buf) { compute(acc, sm, buf); });
    }
    // Tile-local output column / row of accumulator element e (0..3) of fragment (mi, j) for this thread.
    __device__ static __forceinline__ unsigned col(int j, int e) {
        return ((threadIdx.x >> 5U) >> 2U) * (NT8 * 8U) + j * 8U + 2U * (threadIdx.x & 3U) + (e & 1);
    }
    __device__ static __forceinline__ unsigned row(int mi, int e) {
        return ((threadIdx.x >> 5U) & 3U) * 32U + mi * 16U + ((threadIdx.x & 31U) >> 2U) + (e >> 1) * 8U;
    }
};

}  // namespace strix::cuda::core::mma
#endif

namespace strix::cuda::core::mma::microscaled {

namespace view = ::strix::moe_core::grouped_microscaled;
namespace packed = ::strix::weight::microscaled;
namespace tall = ::strix::moe_core::grouped_tall;

inline constexpr unsigned kKStep = 16U;
// Plan rows per tile: the tall CTA extent. The staged tile form (64-row CTAs) takes a larger tile as row slices of 64.
inline constexpr unsigned kRowsPerTile = tall::kMaxRows;
inline constexpr unsigned kDecodeRows = 16U * native_mma::kDecodeMBlocks;
inline constexpr unsigned kTileOutputs = 128U;
inline constexpr unsigned kDecodeOutputs = 32U;
inline constexpr unsigned kInputAlignment = 32U;

template<unsigned MBlocks, unsigned OutputTile>
struct TileShape {
    static constexpr bool kDecode = MBlocks == native_mma::kDecodeMBlocks;
    static_assert(kDecode ? OutputTile == kDecodeOutputs
                          : MBlocks == native_mma::kTileMBlocks && OutputTile == kTileOutputs);
    static constexpr unsigned kRows = 16U * MBlocks;
    static constexpr unsigned kThreads = kDecode ? 128U : native_mma::kThreads;
    static constexpr unsigned kWarpMBlocks = kDecode ? 1U : 2U;
    static constexpr unsigned kWarpNBlocks = kDecode ? 1U : 4U;
    static constexpr unsigned kNBlocksPerLoad = kDecode ? 1U : 2U;
    static constexpr unsigned kColumnWarps = OutputTile / (8U * kWarpNBlocks);
    static constexpr unsigned kPairs = kKStep / 2U;
    static constexpr unsigned kActivationLoads = kRows * kPairs / kThreads;
    static constexpr unsigned kWeightPairs = kDecode ? kPairs : kPairs / 2U;

    STRIX_MMA_TILE_HD static constexpr unsigned first_row(unsigned slice) { return slice * kRows; }
    STRIX_MMA_TILE_HD static constexpr unsigned first_output(unsigned block) { return block * OutputTile; }
    STRIX_MMA_TILE_HD static constexpr unsigned k_start(unsigned step) { return step * kKStep; }
    STRIX_MMA_TILE_HD static constexpr unsigned activation_row(unsigned thread, unsigned load) {
        return thread / kPairs + load * (kThreads / kPairs);
    }
    STRIX_MMA_TILE_HD static constexpr unsigned activation_pair(unsigned thread) { return thread % kPairs; }
    STRIX_MMA_TILE_HD static constexpr unsigned weight_row(unsigned thread) { return kDecode ? thread : thread / 2U; }
    STRIX_MMA_TILE_HD static constexpr unsigned weight_first_pair(unsigned thread) {
        return kDecode ? 0U : (thread % 2U) * kWeightPairs;
    }
    STRIX_MMA_TILE_HD static constexpr unsigned fragment_row(unsigned warp, unsigned m) {
        return 16U * (warp / kColumnWarps * kWarpMBlocks + m);
    }
    STRIX_MMA_TILE_HD static constexpr unsigned fragment_column(unsigned warp, unsigned n) {
        return 8U * (warp % kColumnWarps * kWarpNBlocks + n);
    }
    STRIX_MMA_TILE_HD static unsigned output_row(unsigned warp, unsigned m, unsigned lane, unsigned i) {
        return fragment_row(warp, m) + native_mma::c_row(lane, i);
    }
    STRIX_MMA_TILE_HD static unsigned output_column(unsigned warp, unsigned n, unsigned lane, unsigned i) {
        return fragment_column(warp, n) + native_mma::c_col(lane, i);
    }
};

inline constexpr unsigned output_tile(unsigned outputs, unsigned max_tile_rows) {
    const bool decode = max_tile_rows != 0U && max_tile_rows <= kDecodeRows;
    return !decode && outputs % kTileOutputs == 0U ? kTileOutputs : kDecodeOutputs;
}

inline constexpr unsigned row_slices(unsigned max_tile_rows, unsigned rows_per_cta) {
    const unsigned rows = max_tile_rows == 0U || max_tile_rows > kRowsPerTile ? kRowsPerTile : max_tile_rows;
    return (rows + rows_per_cta - 1U) / rows_per_cta;
}

inline bool fits(const view::Args& args) {
    return view::fits_args(args, kInputAlignment, output_tile(args.outputs, args.max_tile_rows));
}

template<packed::MatrixFormat Format>
inline bool fits_tile(const view::TileRows& tile, unsigned inputs, unsigned outputs) {
    return outputs % output_tile(outputs, tile.rows) == 0U &&
           view::fits_rows<Format>(tile, inputs, outputs, kInputAlignment, kRowsPerTile);
}

#if defined(__CUDACC__) && !defined(__HIPCC__)
namespace detail {

inline constexpr unsigned kWordPitch = (kKStep + kPadHalves) / 2U;

// Decode Pairs BF16 pairs from their code words (little-endian code bytes) and one scale code. Shift: NVFP4 operands decode
// with PackedOperand::decode_bf16_shift (bitwise decode_bf16, view::Decode::shift).
template<bool Shift = false, class Operand, unsigned Pairs, unsigned Words>
__device__ __forceinline__ void decode_words(const Operand& source, const std::uint32_t (&words)[Words],
                                            std::uint8_t scale_code, std::uint32_t (&decoded)[Pairs]) {
    constexpr unsigned code_bytes = Pairs * (Operand::kNvfp4 ? 1U : 2U);
    static_assert(code_bytes == 4U * Words);
    if constexpr (Shift && Operand::kNvfp4) {
        source.decode_bf16_shift(words, scale_code, decoded);
    } else {
        std::uint8_t unpacked[code_bytes];
#pragma unroll
        for (unsigned i = 0; i < code_bytes; ++i)
            unpacked[i] = static_cast<std::uint8_t>(words[i / 4U] >> (8U * (i % 4U)));
        source.decode_bf16(unpacked, scale_code, decoded);
    }
}

template<bool Shift = false, class Operand, unsigned Pairs>
__device__ __forceinline__ void decode_aligned(const Operand& source, unsigned row, unsigned k, unsigned first_pair,
                                              std::uint32_t (&decoded)[Pairs]) {
    const auto address = source.tile16_address(row, k);
    constexpr unsigned bytes_per_pair = Operand::kNvfp4 ? 1U : 2U;
    const auto* codes = address.codes + first_pair * bytes_per_pair;
    constexpr unsigned code_bytes = Pairs * bytes_per_pair;
    std::uint32_t words[code_bytes / 4U];
    if constexpr (code_bytes == 4U) {
        asm("ld.global.u32 %0, [%1];" : "=r"(words[0]) : "l"(codes));
    } else if constexpr (code_bytes == 8U) {
        asm("ld.global.v2.u32 {%0, %1}, [%2];" : "=r"(words[0]), "=r"(words[1]) : "l"(codes));
    } else {
        static_assert(code_bytes == 16U);
        asm("ld.global.v4.u32 {%0, %1, %2, %3}, [%4];"
            : "=r"(words[0]), "=r"(words[1]), "=r"(words[2]), "=r"(words[3]) : "l"(codes));
    }
    decode_words<Shift>(source, words, *address.scale_code, decoded);
}

// One lane owns each live (row, column), accumulating from +0 in ascending K16
// steps with BF16 m16n8k16 and FP32 accumulators. Each step sees the same BF16 A
// row and B column (a pure decode of codes and scale), so the forms agree bitwise.
// Padded A rows are zero, and only live rows are stored. Shift selects the
// exponent-shift weight decode (bitwise the same B operands).
template<packed::MatrixFormat Format, unsigned MBlocks, unsigned OutputTile, bool Shift = false>
__global__ __launch_bounds__(TileShape<MBlocks, OutputTile>::kThreads, Shift ? 4 : 0) void projection(view::Args args) {
    using Shape = TileShape<MBlocks, OutputTile>;
    if ((args.failure && *args.failure != 0U) || (args.tile_count && blockIdx.z >= *args.tile_count)) return;
    view::TileRows rows = args.tiles[blockIdx.z];
    {  // row slice y (both forms: a tile above the CTA rows runs as slices that keep each row's m16 position)
        const unsigned first_row = Shape::first_row(blockIdx.y);
        if (first_row >= rows.rows) return;
        rows.input += std::size_t{first_row} * rows.input_row;
        rows.output += std::size_t{first_row} * rows.output_row;
        rows.rows -= first_row;
    }
    const unsigned thread = threadIdx.x, lane = thread & 31U, warp = thread >> 5U;
    const unsigned pair = Shape::activation_pair(thread);
    const unsigned first_output = Shape::first_output(blockIdx.x);
    const unsigned weight_row = Shape::weight_row(thread), weight_pair = Shape::weight_first_pair(thread);
    __shared__ __align__(16) std::uint32_t activations[2][Shape::kRows][kWordPitch];
    __shared__ __align__(16) std::uint32_t weights[2][OutputTile][kWordPitch];
    const packed::PackedOperand<Format> source{rows.weights};
    float acc[Shape::kWarpMBlocks][Shape::kWarpNBlocks][4]{};
    std::uint32_t activation[Shape::kActivationLoads]{}, decoded[Shape::kWeightPairs];

    run_staged(args.inputs / kKStep, [&](unsigned step) {
        const unsigned k = Shape::k_start(step);
#pragma unroll
        for (unsigned load = 0; load < Shape::kActivationLoads; ++load) {
            const unsigned input_row = Shape::activation_row(thread, load);
            if (input_row < rows.rows) {
                const auto* values = rows.input + std::size_t{input_row} * rows.input_row + k + 2U * pair;
                asm("ld.global.u32 %0, [%1];" : "=r"(activation[load]) : "l"(values));
            }
        }
        if (weight_row < OutputTile)
            decode_aligned<Shift>(source, first_output + weight_row, k, weight_pair, decoded);
    }, [&](int buffer) {
#pragma unroll
        for (unsigned load = 0; load < Shape::kActivationLoads; ++load)
            activations[buffer][Shape::activation_row(thread, load)][pair] = activation[load];
        if (weight_row < OutputTile) {
#pragma unroll
            for (unsigned i = 0; i < Shape::kWeightPairs; ++i)
                weights[buffer][weight_row][weight_pair + i] = decoded[i];
        }
    }, [&](int buffer) {
        unsigned a[Shape::kWarpMBlocks][4];
#pragma unroll
        for (unsigned m = 0; m < Shape::kWarpMBlocks; ++m)
            ldsm_a(a[m], &activations[buffer][Shape::fragment_row(warp, m)][0], kWordPitch, lane);
#pragma unroll
        for (unsigned n = 0; n < Shape::kWarpNBlocks; n += Shape::kNBlocksPerLoad) {
            unsigned b[2U * Shape::kNBlocksPerLoad];
            ldsm_b(b, &weights[buffer][Shape::fragment_column(warp, n)][0], kWordPitch, lane);
#pragma unroll
            for (unsigned m = 0; m < Shape::kWarpMBlocks; ++m) {
#pragma unroll
                for (unsigned part = 0; part < Shape::kNBlocksPerLoad; ++part)
                    mma_k16<true>(acc[m][n + part], a[m], b[2U * part], b[2U * part + 1U]);
            }
        }
    });

#pragma unroll
    for (unsigned m = 0; m < Shape::kWarpMBlocks; ++m) {
#pragma unroll
        for (unsigned n = 0; n < Shape::kWarpNBlocks; ++n) {
#pragma unroll
            for (unsigned i = 0; i < 4U; ++i) {
                const unsigned row = Shape::output_row(warp, m, lane, i);
                const unsigned column = first_output + Shape::output_column(warp, n, lane, i);
                if (row < rows.rows) rows.output[std::size_t{row} * rows.output_row + column] = acc[m][n][i];
            }
        }
    }
}

template<packed::MatrixFormat Format, unsigned MBlocks, unsigned OutputTile>
inline cudaError_t launch_form(const view::Args& args, cudaStream_t stream) {
    using Shape = TileShape<MBlocks, OutputTile>;
    const dim3 grid(args.outputs / OutputTile, row_slices(args.max_tile_rows, Shape::kRows), args.count);
    if constexpr (Format == packed::MatrixFormat::nvfp4) {
        if (args.decode == view::Decode::shift) {
            projection<Format, MBlocks, OutputTile, true><<<grid, Shape::kThreads, 0, stream>>>(args);
            return cudaGetLastError();
        }
    }
    projection<Format, MBlocks, OutputTile><<<grid, Shape::kThreads, 0, stream>>>(args);
    return cudaGetLastError();
}

// Tall slab schedule of the same arithmetic (plan, maps and the bitwise argument: moe_core/grouped_tall_plan.hpp). Grid
// (outputs / 128, row slices, tiles). Activations and raw codes arrive by cp.async in a ring of F.stages slots; the weight
// slab of a stage is decoded once into one of two BF16 slabs that every M warp reads; one barrier per stage.
template<packed::MatrixFormat Format, tall::FormId Id, bool Shift = false>
__global__ __launch_bounds__(tall::form(Id).threads(), 1) void projection_tall(view::Args args) {
    constexpr tall::Form F = tall::form(Id);
    using Operand = packed::PackedOperand<Format>;
    constexpr tall::Layout L = tall::layout(F, Operand::kNvfp4);
    static_assert(tall::valid(F, Operand::kNvfp4));
    constexpr unsigned kWM = F.warp_m_blocks(), kWN = F.warp_n_blocks();
    constexpr unsigned kActLoads = tall::act_chunks_per_thread(F), kCodeLoads = tall::code_chunks_per_thread(F);
    constexpr unsigned kColumns = tall::decode_columns(F), kPairs = kColumns / 2U;
    constexpr unsigned kCodeBytes = kColumns * L.code_tile / 16U, kWords = kCodeBytes / 4U;
    constexpr std::size_t kCodeStride = std::size_t{L.code_row};  // global code bytes per stage of one weight row
    constexpr std::size_t kScaleStride = F.k_slab / Operand::kBlockColumns;
    constexpr unsigned kActWords = L.act_pitch / 4U, kSlabWords = L.slab_pitch / 4U;

    if ((args.failure && *args.failure != 0U) || (args.tile_count && blockIdx.z >= *args.tile_count)) return;
    const view::TileRows rows = args.tiles[blockIdx.z];
    const unsigned first_row = blockIdx.y * F.rows;
    if (first_row >= rows.rows) return;
    const unsigned live = min(rows.rows - first_row, F.rows);
    const std::uint16_t* input = rows.input + std::size_t{first_row} * rows.input_row;
    float* output = rows.output + std::size_t{first_row} * rows.output_row;

    extern __shared__ __align__(16) std::uint8_t smem[];
    const auto base = static_cast<std::uint32_t>(__cvta_generic_to_shared(smem));
    const unsigned thread = threadIdx.x, lane = thread & 31U, warp = thread >> 5U;
    const unsigned first_output = blockIdx.x * tall::kOutputs;
    const Operand source{rows.weights};

    // Copy sources and slot-0 destinations, resolved once; a stage advances them by k_slab inputs.
    const std::uint16_t* act_src[kActLoads];
    std::uint32_t act_dst[kActLoads];
    bool act_ok[kActLoads];
#pragma unroll
    for (unsigned j = 0; j < kActLoads; ++j) {
        const tall::ActChunk c = tall::act_chunk(F, thread, j);
        act_ok[j] = c.row < live;
        act_src[j] = input + std::size_t{act_ok[j] ? c.row : 0U} * rows.input_row + 8U * c.part;
        act_dst[j] = base + tall::act_offset(L, c.row, c.part);
    }
    const std::uint8_t* code_src[kCodeLoads];
    std::uint32_t code_dst[kCodeLoads];
#pragma unroll
    for (unsigned j = 0; j < kCodeLoads; ++j) {
        const tall::CodeChunk c = tall::code_chunk(F, thread, j);
        const bool chunk = tall::code_chunk_live(F, thread, j);
        code_src[j] = source.tile16_address(first_output + (chunk ? c.output : 0U), 16U * c.tile).codes;
        code_dst[j] = base + tall::code_offset(L, c.output, c.tile);
    }
    const auto issue = [&](unsigned stage) {
        const std::uint32_t slot = tall::slot(F, stage) * L.stage_bytes;
#pragma unroll
        for (unsigned j = 0; j < kActLoads; ++j)
            ptx::cp_async16_zfill(act_dst[j] + slot, act_src[j] + std::size_t{stage} * F.k_slab, act_ok[j]);
#pragma unroll
        for (unsigned j = 0; j < kCodeLoads; ++j) {
            if (!tall::code_chunk_live(F, thread, j)) continue;
            if constexpr (Operand::kNvfp4)
                ptx::cp_async8_zfill(code_dst[j] + slot, code_src[j] + std::size_t{stage} * kCodeStride, true);
            else
                ptx::cp_async16_zfill(code_dst[j] + slot, code_src[j] + std::size_t{stage} * kCodeStride, true);
        }
    };

    // Exponent-shift decode: the tables of the 128 scale codes that can have one, built once per CTA (outer scale fixed per
    // tile); decode_bf16_shift = shift_table, then decode_bf16_shifted or the decode_bf16 fallback, so the bits are equal.
    constexpr bool kTable = Shift && Operand::kNvfp4;
    if constexpr (kTable) {
        if (thread < tall::kShiftCodes) {
            std::uint32_t low[2]{}, high[2]{};
            const bool in_domain = source.shift_table(static_cast<std::uint8_t>(thread), low, high);
            reinterpret_cast<uint4*>(smem + L.table)[thread] = make_uint4(low[0], low[1], high[0], high[1]);
            smem[L.table + 16U * tall::kShiftCodes + thread] = in_domain ? 1U : 0U;
        }
    }

    // Decode job: kColumns inputs of one output per stage; its scale byte is prefetched one iteration ahead.
    const tall::DecodeJob job = tall::decode_job(F, thread);
    const std::uint8_t* scale_src = source.tile16_address(first_output + job.output, job.column).scale_code;
    const auto decode = [&](unsigned stage, std::uint8_t scale_code) {
        const std::uint8_t* raw = smem + tall::slot(F, stage) * L.stage_bytes + tall::decode_code_offset(L, job);
        std::uint32_t words[kWords];
        if constexpr (kWords == 1U) {
            words[0] = *reinterpret_cast<const std::uint32_t*>(raw);
        } else if constexpr (kWords == 2U) {
            const uint2 v = *reinterpret_cast<const uint2*>(raw);
            words[0] = v.x, words[1] = v.y;
        } else {
            static_assert(kWords == 4U);
            const uint4 v = *reinterpret_cast<const uint4*>(raw);
            words[0] = v.x, words[1] = v.y, words[2] = v.z, words[3] = v.w;
        }
        std::uint32_t decoded[kPairs];
        if constexpr (kTable) {
            if (scale_code < tall::kShiftCodes && smem[L.table + 16U * tall::kShiftCodes + scale_code] != 0U) {
                const uint4 entry = reinterpret_cast<const uint4*>(smem + L.table)[scale_code];
                const std::uint32_t low[2] = {entry.x, entry.y}, high[2] = {entry.z, entry.w};
                Operand::decode_bf16_shifted(words, low, high, decoded);
            } else {
                decode_words<false>(source, words, scale_code, decoded);
            }
        } else {
            decode_words<Shift>(source, words, scale_code, decoded);
        }
        auto* slab = reinterpret_cast<uint4*>(smem + L.ring_bytes + tall::slab(stage) * L.slab_bytes +
                                              tall::slab_offset(L, job.output, job.column));
#pragma unroll
        for (unsigned i = 0; i < kPairs / 4U; ++i)
            slab[i] = make_uint4(decoded[4U * i], decoded[4U * i + 1U], decoded[4U * i + 2U], decoded[4U * i + 3U]);
    };

    // A warp whose rows all lie past the tile issues no ldmatrix or MMA (warp-uniform; its outputs are never stored). A
    // partly live warp multiplies its zero-filled rows too: rows of an MMA never mix, and those rows are not stored.
    const bool warp_live = tall::fragment_row(F, warp, 0U) < live;
    const bool decode_first = tall::decode_first(F, warp);
    float acc[kWM][kWN][4]{};
    const auto compute = [&](unsigned stage) {
        const auto* act = reinterpret_cast<const std::uint32_t*>(smem + tall::slot(F, stage) * L.stage_bytes);
        const auto* slab = reinterpret_cast<const std::uint32_t*>(smem + L.ring_bytes + tall::slab(stage) * L.slab_bytes);
#pragma unroll
        for (unsigned kk = 0; kk < F.k16_steps(); ++kk) {
            unsigned a[kWM][4];
#pragma unroll
            for (unsigned m = 0; m < kWM; ++m)
                ldsm_a(a[m], act + tall::fragment_row(F, warp, m) * kActWords + 8U * kk, kActWords, lane);
#pragma unroll
            for (unsigned n = 0; n < kWN; n += 2U) {
                unsigned b[4];
                ldsm_b(b, slab + tall::fragment_column(F, warp, n) * kSlabWords + 8U * kk, kSlabWords, lane);
#pragma unroll
                for (unsigned m = 0; m < kWM; ++m) {
                    mma_k16<true>(acc[m][n], a[m], b[0], b[1]);
                    mma_k16<true>(acc[m][n + 1U], a[m], b[2], b[3]);
                }
            }
        }
    };

    const unsigned steps = args.inputs / F.k_slab;
    std::uint8_t scale_code = *scale_src;
#pragma unroll
    for (unsigned s = 0; s + 1U < F.stages; ++s) {
        if (s < steps) issue(s);
        ptx::cp_async_commit();
    }
    ptx::cp_async_wait<static_cast<int>(tall::prologue_pending(F))>();
    __syncthreads();
    decode(0U, scale_code);
    if (steps > 1U) scale_code = scale_src[kScaleStride];
    for (unsigned s = 0; s < steps; ++s) {
        ptx::cp_async_wait<static_cast<int>(tall::loop_pending(F))>();
        __syncthreads();  // stage s + 1 landed for all; every warp is past stage s - 1 (its slot and slab are free)
        if (tall::copy_stage(F, s) < steps) issue(tall::copy_stage(F, s));
        ptx::cp_async_commit();
#pragma unroll 1
        for (unsigned phase = 0; phase < 2U; ++phase) {  // decode stage s + 1 and multiply stage s, in the warp row's order
            if ((phase == 0U) == decode_first) {
                if (s + 1U < steps) {
                    decode(s + 1U, scale_code);
                    if (s + 2U < steps) scale_code = scale_src[std::size_t{s + 2U} * kScaleStride];
                }
            } else if (warp_live) {
                compute(s);
            }
        }
    }
    ptx::cp_async_wait<0>();

#pragma unroll
    for (unsigned m = 0; m < kWM; ++m) {
#pragma unroll
        for (unsigned n = 0; n < kWN; ++n) {
#pragma unroll
            for (unsigned i = 0; i < 4U; ++i) {
                const unsigned row = tall::fragment_row(F, warp, m) + native_mma::c_row(lane, i);
                const unsigned column = first_output + tall::fragment_column(F, warp, n) + native_mma::c_col(lane, i);
                if (row < live) output[std::size_t{row} * rows.output_row + column] = acc[m][n][i];
            }
        }
    }
}

// Per-block shared bytes the current device grants on opt-in (0 when it cannot be read).
inline std::size_t shared_budget() {
    int device = 0, bytes = 0;
    if (cudaGetDevice(&device) != cudaSuccess ||
        cudaDeviceGetAttribute(&bytes, cudaDevAttrMaxSharedMemoryPerBlockOptin, device) != cudaSuccess) {
        (void)cudaGetLastError();
        return 0U;
    }
    return static_cast<std::size_t>(bytes);
}

template<packed::MatrixFormat Format, tall::FormId Id>
inline cudaError_t launch_tall_form(const view::Args& args, cudaStream_t stream) {
    constexpr tall::Form F = tall::form(Id);
    constexpr tall::Layout L = tall::layout(F, Format == packed::MatrixFormat::nvfp4);
    const dim3 grid(args.outputs / tall::kOutputs, tall::row_slices(F, args.max_tile_rows), args.count);
    auto* kernel = projection_tall<Format, Id, false>;
    if constexpr (Format == packed::MatrixFormat::nvfp4)
        if (args.decode == view::Decode::shift) kernel = projection_tall<Format, Id, true>;
    if (const auto status = cudaFuncSetAttribute(kernel, cudaFuncAttributeMaxDynamicSharedMemorySize,
                                                 static_cast<int>(L.smem)); status != cudaSuccess) return status;
    kernel<<<grid, F.threads(), L.smem, stream>>>(args);
    return cudaGetLastError();
}

// The forms select() returns; the bench instantiates the others through launch_tall_form.
template<packed::MatrixFormat Format>
inline cudaError_t launch_tall(tall::FormId id, const view::Args& args, cudaStream_t stream) {
    switch (id) {
    case tall::FormId::r128_k32_s4: return launch_tall_form<Format, tall::FormId::r128_k32_s4>(args, stream);
    case tall::FormId::r64_k32_s4: return launch_tall_form<Format, tall::FormId::r64_k32_s4>(args, stream);
    default: return cudaErrorInvalidValue;
    }
}

}  // namespace detail

// Schedule trait (view::Schedule, never changes bits): automatic takes the tall slab schedule wherever the staged 128-output
// tile form would run and the tall form fits (shape and the device's opt-in shared memory), else the staged forms; staged
// keeps the incumbent forms; tall runs the slab schedule for any admitted 128-output shape or refuses.
template<packed::MatrixFormat Format>
inline cudaError_t launch(const view::Args& args, cudaStream_t stream) {
    if (!fits(args)) return cudaErrorInvalidValue;
    const bool tile = output_tile(args.outputs, args.max_tile_rows) == kTileOutputs;
    if (args.schedule == view::Schedule::tall || (args.schedule == view::Schedule::automatic && tile)) {
        const auto id = tall::select(args.max_tile_rows);
        if (tall::fits(args, id, Format == packed::MatrixFormat::nvfp4, detail::shared_budget()))
            return detail::launch_tall<Format>(id, args, stream);
        if (args.schedule == view::Schedule::tall) return cudaErrorInvalidValue;
    }
    if (tile) return detail::launch_form<Format, native_mma::kTileMBlocks, kTileOutputs>(args, stream);
    return detail::launch_form<Format, native_mma::kDecodeMBlocks, kDecodeOutputs>(args, stream);
}
#endif

}  // namespace strix::cuda::core::mma::microscaled

#undef STRIX_MMA_TILE_HD
