// Large-M dense GEMM on tensor cores with the decode kernels' reduction order (see dense_big.hpp for the contract).
//
// Tile: 128 outputs (A = weights, mma m dimension) x BN activation rows (mma n dimension), 8 warps (2 along out x 4 along rows),
// BK = 32 (one decode "block": two k16 steps), STAGES-deep cp.async pipeline. The decode kernels sum each output as SEGMENTS of
// K (warp partials / split-K ranges), each segment a chain of mma accumulations from zero, then the segments in ascending order.
// Here the CTA walks the segments one after the other (a per-shape table lists the 32-blocks of every segment in chain order),
// accumulates the chain in `acc`, and folds it into the running sum S at the end of the segment: same bits, any tile size.
// Shared memory rows are 32 bf16/f16 (64 B); the 8-byte unit u of a row is stored at u ^ ((row >> 1) & 1) so the LDS.64 fragment
// reads (4 rows x 4 quad lanes per half warp) are conflict free.
#include "strix/cuda_core/dense_big.hpp"

#include <algorithm>
#include <map>
#include <mutex>
#include <memory>
#include <tuple>
#include <vector>

#include "strix/cuda_core/nvfp4.hpp"
#include "strix/cuda/core/nvfp4_tile_cuda.hpp"
namespace ftile = ::strix::cuda::core::nvfp4::tile;  // FAST4 operand layouts and kernels (shared CUDA core)
#include "../common.cuh"

namespace strix::cuda_core {
struct Table {
    int* kbs = nullptr;
    int* seg_end = nullptr;
    std::uint32_t nseg = 0, nkb = 0;
    ~Table() {
        if (kbs) (void)cudaFree(kbs);
        if (seg_end) (void)cudaFree(seg_end);
    }
};
// Scratch of one stream context: activation planes, expanded weight, q8 scales (grow-only).
struct BigCtx {
    int device = -1;
    std::map<std::tuple<int, std::uint32_t, std::uint32_t, std::uint32_t>, std::unique_ptr<Table>> tables;
    ~BigCtx();
    void* planes[2] = {nullptr, nullptr};  // slot 0: bf16 planes, slot 1: f16 plane (both may be live for one x)
    std::uint64_t planes_bytes[2] = {0, 0};
    void* wexp = nullptr;
    std::uint64_t wexp_bytes = 0;
    float* scales = nullptr;
    std::uint64_t scales_bytes = 0;
    float* rscale = nullptr;  // f8 activation row scales [rows_pad]
    std::uint64_t rscale_bytes = 0;
    BigOperandCache* cache = nullptr;  // resident expanded operands shared with other contexts (not owned)
    ftile::Feed feed = ftile::kDefaultFeed;  // stage feed of the FAST4 tile GEMMs on this context (scheduling only, bitwise)
};
// Resident expanded operands (dense_big.hpp): one arena, one entry per (weight data, operand class), filled on first use.
struct BigOperandCache {
    struct Entry {
        std::uint64_t offset = 0;
        std::uint32_t out = 0, in = 0;
        operand_cache::EntryState state;
        cudaEvent_t filled = nullptr;
    };
    std::mutex mu;
    int device = -1;
    bool reserved = false;
    void* arena = nullptr;
    BigCacheReport report;
    std::map<std::pair<const void*, int>, Entry> entries;
    ~BigOperandCache() {
        int previous = -1;
        (void)cudaGetDevice(&previous);
        if (device != -1) (void)cudaSetDevice(device);
        for (auto& kv : entries)
            if (kv.second.filled) (void)cudaEventDestroy(kv.second.filled);
        if (arena) (void)cudaFree(arena);
        if (previous != -1) (void)cudaSetDevice(previous);
    }
};

namespace {

constexpr int BM = 128, BK = 32;

enum Mode { kChain = 0, kFold = 1, kSplit = 2 };  // bf16 | q8_0 fold per block | nvfp4 (split-K ranges, gscale)

struct BigArgs {
    const std::uint16_t* A;     // [out][in] 16-bit weights (bf16 / f16 / int8-as-bf16)
    const std::uint16_t* P;     // planes [T][rows_pad][in]
    const float* scales;        // fold: [out][in/32]
    const int* kbs;             // block sequence in chain order
    const int* seg_end;         // cumulative block count per segment
    float* y;
    const float* aux;
    std::uint32_t out, in, rows, rows_pad, nseg, nkb;
    std::uint32_t tiles_x, tiles_y, group;  // CTA raster: groups of `group` out tiles walk all row tiles (weight panel stays in L2)
    float gscale;
    int epi;
    const float* rscale;        // f8 kernel: per activation row dequant scale
};

__device__ __forceinline__ void cp8(std::uint32_t smem_addr, const void* g) {
    asm volatile("cp.async.ca.shared.global [%0], [%1], 8;" ::"r"(smem_addr), "l"(g));
}
__device__ __forceinline__ void cp_commit() { asm volatile("cp.async.commit_group;"); }
template <int N>
__device__ __forceinline__ void cp_wait() { asm volatile("cp.async.wait_group %0;" ::"n"(N)); }

template <bool F16>
__device__ __forceinline__ void mma(float (&c)[4], const std::uint32_t (&a)[4], std::uint32_t b0, std::uint32_t b1) {
    if constexpr (F16)
        asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32 {%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};"
                     : "+f"(c[0]), "+f"(c[1]), "+f"(c[2]), "+f"(c[3])
                     : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b0), "r"(b1));
    else
        asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 {%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};"
                     : "+f"(c[0]), "+f"(c[1]), "+f"(c[2]), "+f"(c[3])
                     : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b0), "r"(b1));
}

// BKB = 32-blocks per pipeline unit (1 or 2): a unit is BKB consecutive blocks of one segment.
template <int T, int NT, int BKB>
struct Cfg {
    static constexpr int BN = 4 * NT * 8;
    static constexpr int UK = BK * BKB, RU = 8 * BKB;  // elements per row of a unit, 8-byte units per row
    static constexpr int A_ELEMS = BM * UK, B_ELEMS = T * BN * UK;
    static constexpr int STAGE_ELEMS = A_ELEMS + B_ELEMS;
    static constexpr int STAGES = (STAGE_ELEMS * 2 * 4 <= 99 * 1024) ? 4 : (STAGE_ELEMS * 2 * 3 <= 99 * 1024) ? 3 : 2;
    static constexpr int SMEM = STAGES * STAGE_ELEMS * 2;
};
// XOR mask on the 8-byte unit index of a shared row (conflict-free LDS.64 fragment reads)
template <int BKB>
__device__ __forceinline__ int swz_mask(std::uint32_t row) {
    if constexpr (BKB == 1) return (row >> 1) & 1;
    else return (row & 1) | (((row >> 1) & 1) << 3);
}

template <int T, int NT, int BKB>
__device__ __forceinline__ void load_unit(const BigArgs& a, std::uint16_t* sbase, std::uint32_t o0, std::uint32_t r0, std::uint32_t kb) {
    using C = Cfg<T, NT, BKB>;
    const std::uint32_t tid = threadIdx.x;
    const std::uint32_t sb = static_cast<std::uint32_t>(__cvta_generic_to_shared(sbase));
#pragma unroll
    for (int i = 0; i < BM * C::RU / 256; ++i) {
        const std::uint32_t idx = tid + 256 * i, r = idx / C::RU, u = idx % C::RU;
        const std::uint32_t orow = min(o0 + r, a.out - 1);
        const std::uint16_t* src = a.A + static_cast<std::size_t>(orow) * a.in + kb * 32 + u * 4;
        cp8(sb + (r * C::UK + ((u ^ swz_mask<BKB>(r)) * 4)) * 2, src);
    }
#pragma unroll
    for (int t = 0; t < T; ++t)
#pragma unroll
        for (int i = 0; i < C::BN * C::RU / 256; ++i) {
            const std::uint32_t idx = tid + 256 * i, r = idx / C::RU, u = idx % C::RU;
            const std::uint16_t* src = a.P + (static_cast<std::size_t>(t) * a.rows_pad + r0 + r) * a.in + kb * 32 + u * 4;
            cp8(sb + (C::A_ELEMS + (t * C::BN + r) * C::UK + ((u ^ swz_mask<BKB>(r)) * 4)) * 2, src);
        }
}

// Epilogue of big_kernel and big_kernel_f8: fragment slot [mt][nt][h * 2 + j] = element (out o0 + wm 64 + mt 16 + h 8 + g,
// row r0 + wn NT 8 + nt 8 + 2 q + j); y = epi(value(mt, nt, h * 2 + j, j), aux). y may alias aux (residual / silu_mul run in
// place) and the f8 row scales, so written per element the compiler had to issue every load after the previous store: 16 NT
// serialised load latencies per thread (32 NT for residual f8) while the 1-CTA/SM tile held its SM (same pattern measured in the
// FAST4 tile GEMM: +0.19-0.20 ms per residual launch, rtx_dense3 11-18). Each element is read and written only by the thread
// that owns its accumulator, so a mt group's aux loads all go before its stores (f8 row scales are hoisted by the caller): the
// same operations on the same values, bitwise equal.
template <int NT, class Value>
__device__ __forceinline__ void big_epilogue(const BigArgs& a, std::uint32_t o0, std::uint32_t r0, int wm, int wn, int g, int q,
                                             const Value& value) {
    const bool use_aux = a.epi == 1 || a.epi == 2;
#pragma unroll
    for (int mt = 0; mt < 4; ++mt) {
        float av[2][NT][2];
#pragma unroll
        for (int h = 0; h < 2; ++h)
#pragma unroll
            for (int nt = 0; nt < NT; ++nt)
#pragma unroll
                for (int j = 0; j < 2; ++j) {
                    const std::uint32_t o = o0 + wm * 64 + mt * 16 + h * 8 + g;
                    const std::uint32_t r = r0 + wn * NT * 8 + nt * 8 + 2 * q + j;
                    av[h][nt][j] = use_aux && o < a.out && r < a.rows ? a.aux[static_cast<std::size_t>(r) * a.out + o] : 0.0f;
                }
#pragma unroll
        for (int h = 0; h < 2; ++h) {
            const std::uint32_t o = o0 + wm * 64 + mt * 16 + h * 8 + g;
            if (o >= a.out) continue;
#pragma unroll
            for (int nt = 0; nt < NT; ++nt)
#pragma unroll
                for (int j = 0; j < 2; ++j) {
                    const std::uint32_t r = r0 + wn * NT * 8 + nt * 8 + 2 * q + j;
                    if (r >= a.rows) continue;
                    float v = value(mt, nt, h * 2 + j, j);
                    if (a.epi == 1) v = __fadd_rn(av[h][nt][j], v);
                    else if (a.epi == 2) v = __fmul_rn(cc_silu(av[h][nt][j]), v);
                    a.y[static_cast<std::size_t>(r) * a.out + o] = v;
                }
        }
    }
}

template <bool F16, int T, int NT, int MODE, int BKB>
__global__ void __launch_bounds__(256, 1) big_kernel(const BigArgs a) {
    using C = Cfg<T, NT, BKB>;
    extern __shared__ __align__(16) std::uint16_t smem[];
    std::uint32_t tx, ty;
    {
        const std::uint32_t per = a.group * a.tiles_y, grp = blockIdx.x / per, rem = blockIdx.x % per;
        const std::uint32_t gsz = min(a.group, a.tiles_x - grp * a.group);
        ty = rem / gsz;
        tx = grp * a.group + rem % gsz;
    }
    const std::uint32_t o0 = tx * BM, r0 = ty * C::BN;
    const int warp = threadIdx.x >> 5, lane = threadIdx.x & 31, g = lane >> 2, q = lane & 3;
    const int swz = swz_mask<BKB>(g);  // (row & 3) == (g & 3) for every row this thread reads
    const int wm = warp & 1, wn = warp >> 1;
    float S[4][NT][4], acc[4][NT][4];
#pragma unroll
    for (int i = 0; i < 4; ++i)
#pragma unroll
        for (int j = 0; j < NT; ++j)
#pragma unroll
            for (int k = 0; k < 4; ++k) S[i][j][k] = acc[i][j][k] = 0.0f;

    const std::uint32_t nkb = a.nkb;
    std::uint32_t nload = 0;
#pragma unroll
    for (int s = 0; s < C::STAGES - 1; ++s) {
        if (nload < nkb) load_unit<T, NT, BKB>(a, smem + s * C::STAGE_ELEMS, o0, r0, a.kbs[nload]);
        ++nload;
        cp_commit();
    }
    std::uint32_t n = 0;
    std::uint32_t seg_begin = 0;
    for (std::uint32_t seg = 0; seg < a.nseg; ++seg) {
        const std::uint32_t seg_stop = a.seg_end[seg];
        for (; n < seg_stop; ++n) {
            cp_wait<C::STAGES - 2>();
            __syncthreads();
            if (nload < nkb) load_unit<T, NT, BKB>(a, smem + (nload % C::STAGES) * C::STAGE_ELEMS, o0, r0, a.kbs[nload]);
            ++nload;
            cp_commit();
            const std::uint16_t* sA = smem + (n % C::STAGES) * C::STAGE_ELEMS;
            const std::uint16_t* sB = sA + C::A_ELEMS;
#pragma unroll
            for (int blk = 0; blk < BKB; ++blk) {
            float scl[4][2];
            [[maybe_unused]] float part[4][NT][4];
            if constexpr (MODE == kFold) {
                const std::uint32_t kb = a.kbs[n] + blk;
                const std::uint32_t nb = a.in >> 5;
#pragma unroll
                for (int mt = 0; mt < 4; ++mt)
#pragma unroll
                    for (int h = 0; h < 2; ++h) {
                        const std::uint32_t orow = min(o0 + wm * 64 + mt * 16 + h * 8 + g, a.out - 1);
                        scl[mt][h] = __ldg(a.scales + static_cast<std::size_t>(orow) * nb + kb);
                    }
#pragma unroll
                for (int i = 0; i < 4; ++i)
#pragma unroll
                    for (int j = 0; j < NT; ++j)
#pragma unroll
                        for (int k = 0; k < 4; ++k) part[i][j][k] = 0.0f;
            }
#pragma unroll
            for (int s = 0; s < 2; ++s) {
                const int pu = (blk * 8 + 2 * q + s) ^ swz;
                uint2 af[4][2];
#pragma unroll
                for (int mt = 0; mt < 4; ++mt)
#pragma unroll
                    for (int h = 0; h < 2; ++h)
                        af[mt][h] = *reinterpret_cast<const uint2*>(sA + (wm * 64 + mt * 16 + h * 8 + g) * C::UK + pu * 4);
#pragma unroll
                for (int nt = 0; nt < NT; ++nt) {
                    uint2 bf[T];
#pragma unroll
                    for (int t = 0; t < T; ++t)
                        bf[t] = *reinterpret_cast<const uint2*>(sB + (t * C::BN + wn * NT * 8 + nt * 8 + g) * C::UK + pu * 4);
#pragma unroll
                    for (int mt = 0; mt < 4; ++mt) {
                        const std::uint32_t av[4] = {af[mt][0].x, af[mt][1].x, af[mt][0].y, af[mt][1].y};
#pragma unroll
                        for (int t = T - 1; t >= 0; --t) {
                            if constexpr (MODE == kFold) mma<F16>(part[mt][nt], av, bf[t].x, bf[t].y);
                            else mma<F16>(acc[mt][nt], av, bf[t].x, bf[t].y);
                        }
                    }
                }
            }
            if constexpr (MODE == kFold) {
#pragma unroll
                for (int mt = 0; mt < 4; ++mt)
#pragma unroll
                    for (int nt = 0; nt < NT; ++nt) {
                        acc[mt][nt][0] = __fadd_rn(acc[mt][nt][0], __fmul_rn(scl[mt][0], part[mt][nt][0]));
                        acc[mt][nt][1] = __fadd_rn(acc[mt][nt][1], __fmul_rn(scl[mt][0], part[mt][nt][1]));
                        acc[mt][nt][2] = __fadd_rn(acc[mt][nt][2], __fmul_rn(scl[mt][1], part[mt][nt][2]));
                        acc[mt][nt][3] = __fadd_rn(acc[mt][nt][3], __fmul_rn(scl[mt][1], part[mt][nt][3]));
                    }
            }
            }  // blk
        }
        // segment end: S = S + acc (the first segment initialises S), acc = 0
#pragma unroll
        for (int i = 0; i < 4; ++i)
#pragma unroll
            for (int j = 0; j < NT; ++j)
#pragma unroll
                for (int k = 0; k < 4; ++k) {
                    S[i][j][k] = seg == 0 ? acc[i][j][k] : __fadd_rn(S[i][j][k], acc[i][j][k]);
                    acc[i][j][k] = 0.0f;
                }
        seg_begin = seg_stop;
    }
    (void)seg_begin;
    cp_wait<0>();
    // epilogue: c0,c1 = (out g, rows 2q, 2q+1), c2,c3 = (out g + 8, ...)
    big_epilogue<NT>(a, o0, r0, wm, wn, g, q, [&](int mt, int nt, int k, int) {
        float v = S[mt][nt][k];
        if constexpr (MODE == kSplit) v = __fmul_rn(v, a.gscale);
        return v;
    });
}


// ------------------------------------------------------------------------------------ FAST/TURBO e4m3 x e4m3 GEMM (sm_120)
// y[r][o] = gscale * 8 * rscale[r] * sum_k W8[o][k] * X8[r][k], W8 = nvfp4_expand_e4m3 (permuted inside every 128 inputs), X8 =
// quant_e4m3_kernel (same permutation). mma.sync m16n8k32 e4m3 (f32 accumulate); no segment order (bit-changing by design).
// A stage unit is BKB * 64 inputs = BKB * 64 bytes per row; the shared geometry is that of the 16-bit kernel with BKB blocks
// (rows of 32 * BKB 16-bit words): thread q reads 8-byte units 2q + s, s = 0, 1 (one k32 step each) = a0/a2 resp. b0/b1.
__device__ __forceinline__ void mma_e4m3(float (&c)[4], const std::uint32_t (&a)[4], std::uint32_t b0, std::uint32_t b1) {
#if defined(__CUDA_ARCH_FEAT_SM120_ALL)
    asm volatile("mma.sync.aligned.kind::f8f6f4.m16n8k32.row.col.f32.e4m3.e4m3.f32 {%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};"
                 : "+f"(c[0]), "+f"(c[1]), "+f"(c[2]), "+f"(c[3])
                 : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b0), "r"(b1));
#else
    __trap();  // built without sm_120a: the host never selects the e4m3 path
#endif
}

template <int NT, int BKB, int STG>
struct Cfg8 {
    static constexpr int BN = 4 * NT * 8;
    static constexpr int UK = BK * BKB, RU = 8 * BKB;  // 16-bit words per row of a unit, 8-byte units per row
    static constexpr int A_ELEMS = BM * UK, B_ELEMS = BN * UK;
    static constexpr int STAGE_ELEMS = A_ELEMS + B_ELEMS;
    static constexpr int SMEM = STG * STAGE_ELEMS * 2;
};

template <int NT, int BKB, int STG>
__device__ __forceinline__ void load_unit8(const BigArgs& a, std::uint16_t* sbase, std::uint32_t o0, std::uint32_t r0, std::uint32_t un) {
    using C = Cfg8<NT, BKB, STG>;
    const std::uint32_t tid = threadIdx.x;
    const std::uint32_t sb = static_cast<std::uint32_t>(__cvta_generic_to_shared(sbase));
    const unsigned char* A8 = reinterpret_cast<const unsigned char*>(a.A);
    const unsigned char* P8 = reinterpret_cast<const unsigned char*>(a.P);
#pragma unroll
    for (int i = 0; i < BM * C::RU / 256; ++i) {
        const std::uint32_t idx = tid + 256 * i, r = idx / C::RU, u = idx % C::RU;
        const std::uint32_t orow = min(o0 + r, a.out - 1);
        cp8(sb + (r * C::UK + ((u ^ swz_mask<BKB>(r)) * 4)) * 2, A8 + static_cast<std::size_t>(orow) * a.in + un * (64 * BKB) + u * 8);
    }
#pragma unroll
    for (int i = 0; i < C::BN * C::RU / 256; ++i) {
        const std::uint32_t idx = tid + 256 * i, r = idx / C::RU, u = idx % C::RU;
        cp8(sb + (C::A_ELEMS + r * C::UK + ((u ^ swz_mask<BKB>(r)) * 4)) * 2, P8 + static_cast<std::size_t>(r0 + r) * a.in + un * (64 * BKB) + u * 8);
    }
}

template <int NT, int BKB, int STG>
__global__ void __launch_bounds__(256, 1) big_kernel_f8(const BigArgs a) {
    using C = Cfg8<NT, BKB, STG>;
    extern __shared__ __align__(16) std::uint16_t smem[];
    std::uint32_t tx, ty;
    {
        const std::uint32_t per = a.group * a.tiles_y, grp = blockIdx.x / per, rem = blockIdx.x % per;
        const std::uint32_t gsz = min(a.group, a.tiles_x - grp * a.group);
        ty = rem / gsz;
        tx = grp * a.group + rem % gsz;
    }
    const std::uint32_t o0 = tx * BM, r0 = ty * C::BN;
    const int warp = threadIdx.x >> 5, lane = threadIdx.x & 31, g = lane >> 2, q = lane & 3;
    const int swz = swz_mask<BKB>(g);
    const int wm = warp & 1, wn = warp >> 1;
    float acc[4][NT][4];
#pragma unroll
    for (int i = 0; i < 4; ++i)
#pragma unroll
        for (int j = 0; j < NT; ++j)
#pragma unroll
            for (int k = 0; k < 4; ++k) acc[i][j][k] = 0.0f;
    const std::uint32_t nu = a.in / (64 * BKB);
    std::uint32_t nload = 0;
#pragma unroll
    for (int s = 0; s < STG - 1; ++s) {
        if (nload < nu) load_unit8<NT, BKB, STG>(a, smem + s * C::STAGE_ELEMS, o0, r0, nload);
        ++nload;
        cp_commit();
    }
    for (std::uint32_t n = 0; n < nu; ++n) {
        cp_wait<STG - 2>();
        __syncthreads();
        if (nload < nu) load_unit8<NT, BKB, STG>(a, smem + (nload % STG) * C::STAGE_ELEMS, o0, r0, nload);
        ++nload;
        cp_commit();
        const std::uint16_t* sA = smem + (n % STG) * C::STAGE_ELEMS;
        const std::uint16_t* sB = sA + C::A_ELEMS;
#pragma unroll
        for (int blk = 0; blk < BKB; ++blk)
#pragma unroll
            for (int s = 0; s < 2; ++s) {
                const int pu = (blk * 8 + 2 * q + s) ^ swz;
                uint2 af[4][2];
#pragma unroll
                for (int mt = 0; mt < 4; ++mt)
#pragma unroll
                    for (int h = 0; h < 2; ++h)
                        af[mt][h] = *reinterpret_cast<const uint2*>(sA + (wm * 64 + mt * 16 + h * 8 + g) * C::UK + pu * 4);
#pragma unroll
                for (int nt = 0; nt < NT; ++nt) {
                    const uint2 bf = *reinterpret_cast<const uint2*>(sB + (wn * NT * 8 + nt * 8 + g) * C::UK + pu * 4);
#pragma unroll
                    for (int mt = 0; mt < 4; ++mt) {
                        const std::uint32_t av[4] = {af[mt][0].x, af[mt][1].x, af[mt][0].y, af[mt][1].y};
                        mma_e4m3(acc[mt][nt], av, bf.x, bf.y);
                    }
                }
            }
    }
    cp_wait<0>();
    const float gs = a.gscale * 8.0f;
    float rsv[NT][2];  // this thread's row scales, loaded once before any store (rscale must not overlap y)
#pragma unroll
    for (int nt = 0; nt < NT; ++nt)
#pragma unroll
        for (int j = 0; j < 2; ++j) {
            const std::uint32_t r = r0 + wn * NT * 8 + nt * 8 + 2 * q + j;
            rsv[nt][j] = r < a.rows ? a.rscale[r] : 0.0f;
        }
    big_epilogue<NT>(a, o0, r0, wm, wn, g, q,
                     [&](int mt, int nt, int k, int j) { return __fmul_rn(acc[mt][nt][k], gs * rsv[nt][j]); });
}

// one block per row: scale = amax / 448, e4m3 bytes in the nvfp4_expand_e4m3 permutation (position 32t + 4j + 2p + e of a 128 block =
// input 16j + 8p + 2t + e); rows >= `rows` (padding) are zero with scale 1. thread = 4 consecutive output bytes.
__global__ void __launch_bounds__(256) quant_e4m3_kernel(const float* x, unsigned char* plane, float* rscale, std::uint32_t rows, std::uint32_t in) {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 890
    const std::uint32_t r = blockIdx.x;
    unsigned char* dst = plane + static_cast<std::size_t>(r) * in;
    const std::uint32_t nw = in >> 2;
    if (r >= rows) {
        for (std::uint32_t i = threadIdx.x; i < nw; i += 256) reinterpret_cast<std::uint32_t*>(dst)[i] = 0u;
        if (threadIdx.x == 0) rscale[r] = 1.0f;
        return;
    }
    const float* xr = x + static_cast<std::size_t>(r) * in;
    float mx = 0.f;
    for (std::uint32_t i = threadIdx.x; i < nw; i += 256) {
        const float4 v = *reinterpret_cast<const float4*>(xr + 4 * i);
        mx = fmaxf(mx, fmaxf(fmaxf(fabsf(v.x), fabsf(v.y)), fmaxf(fabsf(v.z), fabsf(v.w))));
    }
#pragma unroll
    for (int o = 16; o > 0; o >>= 1) mx = fmaxf(mx, __shfl_xor_sync(0xffffffffu, mx, o));
    __shared__ float red[8];
    if ((threadIdx.x & 31) == 0) red[threadIdx.x >> 5] = mx;
    __syncthreads();
    mx = red[0];
#pragma unroll
    for (int i = 1; i < 8; ++i) mx = fmaxf(mx, red[i]);
    const float sc = mx > 0.f ? mx * (1.0f / 448.0f) : 1.0f, inv = 1.0f / sc;
    if (threadIdx.x == 0) rscale[r] = sc;
    for (std::uint32_t m = threadIdx.x; m < nw; m += 256) {
        const std::uint32_t pos = 4 * m, blk = pos >> 7, rem = pos & 127, t = rem >> 5, j = (rem & 31) >> 2;
        const float* src = xr + blk * 128 + 16 * j + 2 * t;
        const float2 lo = *reinterpret_cast<const float2*>(src), hi = *reinterpret_cast<const float2*>(src + 8);
        std::uint32_t w0, w1;
        asm("{\n .reg .b16 h;\n cvt.rn.satfinite.e4m3x2.f32 h, %2, %1;\n cvt.u32.u16 %0, h;\n}" : "=r"(w0) : "f"(lo.x * inv), "f"(lo.y * inv));
        asm("{\n .reg .b16 h;\n cvt.rn.satfinite.e4m3x2.f32 h, %2, %1;\n cvt.u32.u16 %0, h;\n}" : "=r"(w1) : "f"(hi.x * inv), "f"(hi.y * inv));
        reinterpret_cast<std::uint32_t*>(dst)[m] = (w0 & 0xFFFFu) | (w1 << 16);
    }
#else
    // E4M3 conversion requires sm_89+; host FAST8 admission rejects this image.
    (void)x; (void)plane; (void)rscale; (void)rows; (void)in;
    __trap();
#endif
}

// ---------------------------------------------------------------------------------------------------- split / expand kernels
// planes: [T][rows_pad][in] bf16; thread = 8 consecutive k of one row.
template <int T>
__global__ void split_bf16_kernel(const float* x, std::uint16_t* planes, std::uint32_t rows, std::uint32_t rows_pad, std::uint32_t in) {
    const std::size_t i8 = static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    const std::size_t per_row = in >> 3;
    if (i8 >= per_row * rows_pad) return;
    const std::uint32_t r = static_cast<std::uint32_t>(i8 / per_row), c = static_cast<std::uint32_t>(i8 % per_row);
    float v[8];
    if (r < rows) {
        const float4 p0 = reinterpret_cast<const float4*>(x + static_cast<std::size_t>(r) * in)[2 * c];
        const float4 p1 = reinterpret_cast<const float4*>(x + static_cast<std::size_t>(r) * in)[2 * c + 1];
        v[0] = p0.x; v[1] = p0.y; v[2] = p0.z; v[3] = p0.w; v[4] = p1.x; v[5] = p1.y; v[6] = p1.z; v[7] = p1.w;
    } else {
#pragma unroll
        for (int k = 0; k < 8; ++k) v[k] = 0.0f;
    }
    std::uint32_t w[T][4];
#pragma unroll
    for (int k = 0; k < 4; ++k) {
        const __nv_bfloat16 h0 = __float2bfloat16_rn(v[2 * k]), h1 = __float2bfloat16_rn(v[2 * k + 1]);
        w[0][k] = static_cast<std::uint32_t>(__bfloat16_as_ushort(h0)) | (static_cast<std::uint32_t>(__bfloat16_as_ushort(h1)) << 16);
        if constexpr (T > 1) {
            const float r0 = __fsub_rn(v[2 * k], __bfloat162float(h0)), r1 = __fsub_rn(v[2 * k + 1], __bfloat162float(h1));
            const __nv_bfloat16 m0 = __float2bfloat16_rn(r0), m1 = __float2bfloat16_rn(r1);
            w[1][k] = static_cast<std::uint32_t>(__bfloat16_as_ushort(m0)) | (static_cast<std::uint32_t>(__bfloat16_as_ushort(m1)) << 16);
            if constexpr (T > 2) {
                const __nv_bfloat16 l0 = __float2bfloat16_rn(__fsub_rn(r0, __bfloat162float(m0)));
                const __nv_bfloat16 l1 = __float2bfloat16_rn(__fsub_rn(r1, __bfloat162float(m1)));
                w[2][k] = static_cast<std::uint32_t>(__bfloat16_as_ushort(l0)) | (static_cast<std::uint32_t>(__bfloat16_as_ushort(l1)) << 16);
            }
        }
    }
#pragma unroll
    for (int t = 0; t < T; ++t)
        reinterpret_cast<uint4*>(planes + (static_cast<std::size_t>(t) * rows_pad + r) * in)[c] = make_uint4(w[t][0], w[t][1], w[t][2], w[t][3]);
}

// f16 plane in the k-fragment order of nvfp4_expand_f16: thread = one 8-byte unit (4 elements) of a 32-block.
__global__ void split_f16_kernel(const float* x, std::uint16_t* plane, std::uint32_t rows, std::uint32_t rows_pad, std::uint32_t in) {
    const std::size_t i = static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    const std::size_t per_row = in >> 2;  // units
    if (i >= per_row * rows_pad) return;
    const std::uint32_t r = static_cast<std::uint32_t>(i / per_row), u = static_cast<std::uint32_t>(i % per_row);
    const std::uint32_t kb = u >> 3, pu = u & 7, q = pu >> 1, s = pu & 1;
    float v[4] = {0.f, 0.f, 0.f, 0.f};
    if (r < rows) {
        const float* xr = x + static_cast<std::size_t>(r) * in + kb * 32 + 16 * s + 2 * q;
        const float2 lo = *reinterpret_cast<const float2*>(xr), hi = *reinterpret_cast<const float2*>(xr + 8);
        v[0] = lo.x; v[1] = lo.y; v[2] = hi.x; v[3] = hi.y;
    }
    const __half2 a = __floats2half2_rn(v[0], v[1]), b = __floats2half2_rn(v[2], v[3]);
    reinterpret_cast<uint2*>(plane + static_cast<std::size_t>(r) * in)[u] =
        make_uint2(*reinterpret_cast<const std::uint32_t*>(&a), *reinterpret_cast<const std::uint32_t*>(&b));
}

// q8_0 [out][in/32] x {f16 d, int8 q[32]} -> bf16 ints [out][in] + float scales [out][in/32]; thread = one block.
__global__ void expand_q8_kernel(const unsigned char* w, std::size_t nblocks, std::uint16_t* dst, float* scales) {
    const std::size_t b = static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (b >= nblocks) return;
    const std::uint16_t* p = reinterpret_cast<const std::uint16_t*>(w + b * 34);
    scales[b] = __half2float(__ushort_as_half(p[0]));
    const std::int8_t* qv = reinterpret_cast<const std::int8_t*>(p + 1);
    std::uint32_t o[16];
#pragma unroll
    for (int i = 0; i < 16; ++i) {
        const __nv_bfloat16 h0 = __float2bfloat16_rn(static_cast<float>(qv[2 * i])), h1 = __float2bfloat16_rn(static_cast<float>(qv[2 * i + 1]));
        o[i] = static_cast<std::uint32_t>(__bfloat16_as_ushort(h0)) | (static_cast<std::uint32_t>(__bfloat16_as_ushort(h1)) << 16);
    }
    uint4* d = reinterpret_cast<uint4*>(dst + b * 32);
#pragma unroll
    for (int i = 0; i < 4; ++i) d[i] = make_uint4(o[4 * i], o[4 * i + 1], o[4 * i + 2], o[4 * i + 3]);
}

// ---------------------------------------------------------------------------------------------------- host state
using Scratch = BigCtx;
bool g_bkb2 = true;
std::uint32_t g_group = 16;
Scratch& scratch() {
    // The implicit single-stream context is isolated by host thread and device.
    thread_local std::map<int, std::unique_ptr<Scratch>> contexts;
    int device = -1;
    (void)cudaGetDevice(&device);
    auto& ctx = contexts[device];
    if (!ctx) ctx = std::make_unique<Scratch>();
    return *ctx;
}

cudaError_t bind_device(BigCtx& ctx) {
    int device = -1;
    if (auto e = cudaGetDevice(&device); e != cudaSuccess) return e;
    if (ctx.device != -1 && ctx.device != device) return cudaErrorInvalidValue;
    ctx.device = device;
    return cudaSuccess;
}

cudaError_t grow(void** p, std::uint64_t* have, std::uint64_t need) {
    if (*have >= need) return cudaSuccess;
    if (*p) cudaFree(*p);
    *p = nullptr;
    *have = 0;
    if (auto e = cudaMalloc(p, need); e != cudaSuccess) return e;
    *have = need;
    return cudaSuccess;
}

// The resident operand of w (class k) when ctx has a cache that admitted it: the first use enqueues `expand(dst)` on s and records
// the entry event; later uses (any stream) wait on it. *out = nullptr: not cached here, the caller expands into its scratch.
template <class Expand>
cudaError_t cached_operand(BigCtx& S, const DenseWeight& w, operand_cache::Kind k, cudaStream_t s, Expand&& expand, unsigned char** out) {
    *out = nullptr;
    BigOperandCache* C = S.cache;
    if (C == nullptr || !C->reserved || C->arena == nullptr || C->device != S.device) return cudaSuccess;
    cudaStreamCaptureStatus cap = cudaStreamCaptureStatusNone;
    if (auto e = cudaStreamIsCapturing(s, &cap); e != cudaSuccess) return e;
    if (cap != cudaStreamCaptureStatusNone) return cudaSuccess;
    std::lock_guard<std::mutex> lock(C->mu);
    const auto it = C->entries.find({w.data, static_cast<int>(k)});
    if (it == C->entries.end()) return cudaSuccess;
    auto& en = it->second;
    if (en.out != w.out || en.in != w.in) return cudaErrorInvalidValue;  // key reused by another weight: never serve stale bytes
    unsigned char* dst = static_cast<unsigned char*>(C->arena) + en.offset;
    if (operand_cache::on_use(en.state) == operand_cache::Use::fill) {
        cudaError_t e = expand(dst);
        if (e == cudaSuccess) e = cudaEventRecord(en.filled, s);
        if (e != cudaSuccess) {
            operand_cache::on_fill_failed(en.state);
            return e;
        }
    } else if (auto e = cudaStreamWaitEvent(s, en.filled, 0); e != cudaSuccess) {
        return e;
    }
    *out = dst;
    return cudaSuccess;
}

const Table* get_table(BigCtx& ctx, Mode mode, std::uint32_t out, std::uint32_t in, int bkb) {
    const std::uint32_t nkb = in >> 5;
    std::uint32_t key2 = 0;
    if (mode == kSplit) key2 = nvfp4_split_k(out, in);
    const auto key = std::make_tuple(static_cast<int>(mode), in, mode == kSplit ? key2 : 0u, static_cast<std::uint32_t>(bkb));
    auto it = ctx.tables.find(key);
    if (it != ctx.tables.end()) return it->second.get();
    std::vector<int> kbs, ends;
    if (mode == kSplit) {
        const std::uint32_t nch = in >> 7, ks = key2;
        for (std::uint32_t k = 0; k < ks; ++k) {
            const std::uint32_t c0 = static_cast<std::uint32_t>(static_cast<std::uint64_t>(k) * nch / ks);
            const std::uint32_t c1 = static_cast<std::uint32_t>(static_cast<std::uint64_t>(k + 1) * nch / ks);
            for (std::uint32_t c = c0; c < c1; ++c)
                for (std::uint32_t u = 0; u < 4; ++u) kbs.push_back(static_cast<int>(4 * c + u));
            ends.push_back(static_cast<int>(kbs.size()));
        }
    } else {
        const std::uint32_t U = mode == kChain ? 4 : 8, ngroups = (nkb + U - 1) / U;
        for (std::uint32_t w = 0; w < 8; ++w) {
            for (std::uint32_t gi = w; gi < ngroups; gi += 8)
                for (std::uint32_t u = 0; u < U; ++u)
                    if (gi * U + u < nkb) kbs.push_back(static_cast<int>(gi * U + u));
            ends.push_back(static_cast<int>(kbs.size()));
        }
    }
    if (bkb > 1) {  // units of bkb consecutive blocks (every segment is a multiple of bkb long and bkb-aligned: in % 128 == 0)
        std::vector<int> ub, ue;
        for (std::size_t i = 0; i < kbs.size(); i += bkb) ub.push_back(kbs[i]);
        for (int e : ends) ue.push_back(e / bkb);
        kbs = ub;
        ends = ue;
    }
    auto t = std::make_unique<Table>();
    t->nkb = static_cast<std::uint32_t>(kbs.size());
    t->nseg = static_cast<std::uint32_t>(ends.size());
    if (cudaMalloc(reinterpret_cast<void**>(&t->kbs), kbs.size() * sizeof(int)) != cudaSuccess) return nullptr;
    if (cudaMalloc(reinterpret_cast<void**>(&t->seg_end), ends.size() * sizeof(int)) != cudaSuccess) return nullptr;
    if (cudaMemcpy(t->kbs, kbs.data(), kbs.size() * sizeof(int), cudaMemcpyHostToDevice) != cudaSuccess) return nullptr;
    if (cudaMemcpy(t->seg_end, ends.data(), ends.size() * sizeof(int), cudaMemcpyHostToDevice) != cudaSuccess) return nullptr;
    return ctx.tables.emplace(key, std::move(t)).first->second.get();
}

// big_epilogue issues a thread's aux / row-scale loads before its y stores: aux must be y itself or disjoint from it, and the
// f8 row scales must not overlap y.
bool epilogue_alias_ok(const BigArgs& a, std::uint32_t out, std::uint32_t rows, bool row_scales) {
    const auto disjoint = [](const void* p, std::size_t pb, const void* q, std::size_t qb) {
        const std::uintptr_t x = reinterpret_cast<std::uintptr_t>(p), y = reinterpret_cast<std::uintptr_t>(q);
        return x + pb <= y || y + qb <= x;
    };
    const std::size_t ybytes = static_cast<std::size_t>(rows) * out * sizeof(float);
    if (a.aux && a.aux != a.y && !disjoint(a.aux, ybytes, a.y, ybytes)) return false;
    return !row_scales || !a.rscale || disjoint(a.rscale, static_cast<std::size_t>(rows) * sizeof(float), a.y, ybytes);
}

template <bool F16, int T, int NT, int MODE, int BKB>
cudaError_t launch(const BigArgs& a, std::uint32_t out, std::uint32_t rows, cudaStream_t s) {
    using C = Cfg<T, NT, BKB>;
    if (!epilogue_alias_ok(a, out, rows, false)) return cudaErrorInvalidValue;
    // CUDA applies attributes to the current device; no process-global initialization flag.
    if (auto e = cudaFuncSetAttribute(big_kernel<F16, T, NT, MODE, BKB>, cudaFuncAttributeMaxDynamicSharedMemorySize, C::SMEM); e != cudaSuccess) return e;
    BigArgs b = a;
    b.tiles_x = (out + BM - 1) / BM;
    b.tiles_y = (rows + C::BN - 1) / C::BN;
    b.group = std::min<std::uint32_t>(b.tiles_x, g_group);
    big_kernel<F16, T, NT, MODE, BKB><<<b.tiles_x * b.tiles_y, 256, C::SMEM, s>>>(b);
    return cudaGetLastError();
}

template <int NT, int BKB, int STG>
cudaError_t launch8(const BigArgs& a, std::uint32_t out, std::uint32_t rows, cudaStream_t s) {
    using C = Cfg8<NT, BKB, STG>;
    if (!epilogue_alias_ok(a, out, rows, true)) return cudaErrorInvalidValue;
    // CUDA applies attributes to the current device; no process-global initialization flag.
    if (auto e = cudaFuncSetAttribute(big_kernel_f8<NT, BKB, STG>, cudaFuncAttributeMaxDynamicSharedMemorySize, C::SMEM); e != cudaSuccess) return e;
    BigArgs b = a;
    b.tiles_x = (out + BM - 1) / BM;
    b.tiles_y = (rows + C::BN - 1) / C::BN;
    b.group = std::min<std::uint32_t>(b.tiles_x, g_group);
    big_kernel_f8<NT, BKB, STG><<<b.tiles_x * b.tiles_y, 256, C::SMEM, s>>>(b);
    return cudaGetLastError();
}

}  // namespace

bool dense_big_fast8_supported() {
#if defined(STRIX_CUDA_FAST8_SM120A)
    int device = -1;
    cudaDeviceProp prop{};
    return cudaGetDevice(&device) == cudaSuccess && cudaGetDeviceProperties(&prop, device) == cudaSuccess &&
           prop.major == 12 && prop.minor == 0;
#else
    return false;
#endif
}

bool dense_big_supported(const DenseWeight& w) {
    if (w.out % 16 != 0 || w.data == nullptr) return false;
    if (w.format == WeightFormat::bf16 || w.format == WeightFormat::q8_0) return w.in % 32 == 0;
    if (w.format == WeightFormat::nvfp4) return w.in % 128 == 0;
    return false;
}

cudaError_t dense_big_reserve(std::uint32_t max_rows, std::uint32_t max_in, std::uint64_t max_weight_elems, int max_terms, BigCtx* ctx) {
    Scratch& S = ctx ? *ctx : scratch();
    if (auto e = bind_device(S); e != cudaSuccess) return e;
    const std::uint32_t rp = (max_rows + 127) / 128 * 128;
    if (auto e = grow(&S.planes[0], &S.planes_bytes[0], static_cast<std::uint64_t>(std::max(1, max_terms)) * rp * max_in * 2); e != cudaSuccess) return e;
    if (auto e = grow(&S.planes[1], &S.planes_bytes[1], static_cast<std::uint64_t>(rp) * max_in * 2); e != cudaSuccess) return e;
    if (auto e = grow(&S.wexp, &S.wexp_bytes, max_weight_elems * 2); e != cudaSuccess) return e;
    if (auto e = grow(reinterpret_cast<void**>(&S.rscale), &S.rscale_bytes, static_cast<std::uint64_t>(rp) * 4); e != cudaSuccess) return e;
    return grow(reinterpret_cast<void**>(&S.scales), &S.scales_bytes, max_weight_elems / 32 * 4);
}

void dense_big_release(BigCtx* ctx) {
    Scratch& S = ctx ? *ctx : scratch();
    int previous = -1;
    (void)cudaGetDevice(&previous);
    if (S.device != -1) (void)cudaSetDevice(S.device);
    S.tables.clear();
    for (void* p : S.planes)
        if (p) cudaFree(p);
    if (S.wexp) cudaFree(S.wexp);
    if (S.scales) cudaFree(S.scales);
    if (S.rscale) cudaFree(S.rscale);
    S.planes[0] = S.planes[1] = S.wexp = nullptr;
    S.scales = S.rscale = nullptr;
    S.planes_bytes[0] = S.planes_bytes[1] = S.wexp_bytes = S.scales_bytes = S.rscale_bytes = 0;
    if (previous != -1) (void)cudaSetDevice(previous);
}
std::uint64_t dense_big_reserved_bytes(const BigCtx* ctx) {
    const Scratch& S = ctx ? *ctx : scratch();
    return S.planes_bytes[0] + S.planes_bytes[1] + S.wexp_bytes + S.scales_bytes + S.rscale_bytes;
}
BigCtx::~BigCtx() { dense_big_release(this); }
BigCtx* dense_big_ctx_create() { return new BigCtx(); }
void dense_big_ctx_destroy(BigCtx* ctx) {
    if (!ctx) return;
    delete ctx;
}

BigOperandCache* dense_big_cache_create() { return new BigOperandCache(); }
void dense_big_cache_destroy(BigOperandCache* cache) { delete cache; }
void dense_big_ctx_attach_cache(BigCtx* ctx, BigOperandCache* cache) {
    if (ctx) ctx->cache = cache;
}
void dense_big_ctx_set_tile_feed(BigCtx* ctx, ftile::Feed feed) {
    if (ctx) ctx->feed = feed;
}

cudaError_t dense_big_cache_reserve(BigOperandCache* cache, const DenseWeight* weights, std::size_t n, BigClass cls, operand_cache::Mode mode,
                                    std::uint64_t cap_bytes, BigCacheReport* report) {
    if (cache == nullptr || (n > 0 && weights == nullptr)) return cudaErrorInvalidValue;
    std::lock_guard<std::mutex> lock(cache->mu);
    if (cache->reserved) {
        if (report) *report = cache->report;
        return cudaSuccess;
    }
    int device = -1;
    if (auto e = cudaGetDevice(&device); e != cudaSuccess) return e;
    // the operand class dense_big_gemm uses for each weight (q8_0 always folds; nvfp4 is e4m3 under FAST8, the e2m1 W tile under
    // FAST4, f16 otherwise)
    const bool f8 = cls == BigClass::fast8, f4 = cls == BigClass::fast4;
    std::vector<operand_cache::Candidate> cand;
    std::vector<std::pair<const DenseWeight*, operand_cache::Kind>> who;
    for (std::size_t i = 0; i < n; ++i) {
        const DenseWeight& w = weights[i];
        if (!dense_big_supported(w)) continue;
        operand_cache::Kind k;
        if (w.format == WeightFormat::nvfp4) k = f4 ? operand_cache::Kind::e2m1_tile : f8 ? operand_cache::Kind::e4m3 : operand_cache::Kind::f16;
        else if (w.format == WeightFormat::q8_0) k = operand_cache::Kind::q8_bf16;
        else continue;  // bf16: the GEMM reads the weight directly
        if (!operand_cache::admits(mode, k)) continue;
        cand.push_back({w.data, operand_cache::entry_bytes(k, w.out, w.in)});
        who.emplace_back(&w, k);
    }
    std::size_t free_b = 0, total_b = 0;
    if (auto e = cudaMemGetInfo(&free_b, &total_b); e != cudaSuccess) return e;
    BigCacheReport rep;
    rep.f8 = f8;
    rep.f4 = f4;
    rep.budget_bytes = operand_cache::budget(mode, free_b, total_b, cap_bytes);
    const operand_cache::Plan p = operand_cache::plan(cand, rep.budget_bytes);
    rep.wanted_bytes = p.wanted_bytes;
    rep.arena_bytes = p.arena_bytes;
    rep.admitted = p.admitted;
    rep.rejected = p.rejected;
    if (p.arena_bytes > 0 && cudaMalloc(&cache->arena, p.arena_bytes) != cudaSuccess) {
        (void)cudaGetLastError();  // not sticky: degrade to per-call expansion, report nothing resident
        cache->arena = nullptr;
        rep.rejected += rep.admitted;
        rep.admitted = 0;
        rep.arena_bytes = 0;
    }
    if (cache->arena != nullptr) {
        for (std::size_t i = 0; i < who.size(); ++i) {
            if (p.offset[i] == operand_cache::kNone) continue;
            const auto key = std::make_pair(who[i].first->data, static_cast<int>(who[i].second));
            if (cache->entries.count(key)) continue;  // duplicate weight: one entry
            BigOperandCache::Entry en;
            en.offset = p.offset[i];
            en.out = who[i].first->out;
            en.in = who[i].first->in;
            if (auto e = cudaEventCreateWithFlags(&en.filled, cudaEventDisableTiming); e != cudaSuccess) {
                for (auto& kv : cache->entries) (void)cudaEventDestroy(kv.second.filled);
                cache->entries.clear();
                (void)cudaFree(cache->arena);
                cache->arena = nullptr;
                return e;
            }
            cache->entries.emplace(key, en);
        }
    }
    cache->device = device;
    cache->reserved = true;
    cache->report = rep;
    if (report) *report = rep;
    return cudaSuccess;
}

cudaError_t dense_big_split(const float* x, std::uint32_t rows, std::uint32_t in, WeightFormat fmt, int terms, BigAct* out, cudaStream_t s, BigCtx* ctx, bool f8) {
    if (f8 && !dense_big_fast8_supported()) return cudaErrorNotSupported;
    if (rows == 0 || in % 32 != 0 || terms < 1 || terms > 3) return cudaErrorInvalidValue;
    Scratch& S = ctx ? *ctx : scratch();
    if (auto e = bind_device(S); e != cudaSuccess) return e;
    const std::uint32_t rp = (rows + 127) / 128 * 128;
    const bool f16 = fmt == WeightFormat::nvfp4;
    const int T = f16 ? 1 : terms;
    const int slot = f16 ? 1 : 0;
    if (auto e = grow(&S.planes[slot], &S.planes_bytes[slot], static_cast<std::uint64_t>(T) * rp * in * 2); e != cudaSuccess) return e;
    std::uint16_t* pl = static_cast<std::uint16_t*>(S.planes[slot]);
    if (f8 && f16 && in % 128 == 0) {
        if (auto e = grow(reinterpret_cast<void**>(&S.rscale), &S.rscale_bytes, static_cast<std::uint64_t>(rp) * 4); e != cudaSuccess) return e;
        quant_e4m3_kernel<<<rp, 256, 0, s>>>(x, static_cast<unsigned char*>(S.planes[slot]), S.rscale, rows, in);
        out->f8 = true;
    } else if (f16) {
        out->f8 = false;
        const std::size_t n = static_cast<std::size_t>(rp) * (in >> 2);
        split_f16_kernel<<<static_cast<unsigned>((n + 255) / 256), 256, 0, s>>>(x, pl, rows, rp, in);
    } else {
        out->f8 = false;
        const std::size_t n = static_cast<std::size_t>(rp) * (in >> 3);
        const unsigned g = static_cast<unsigned>((n + 255) / 256);
        if (T == 1) split_bf16_kernel<1><<<g, 256, 0, s>>>(x, pl, rows, rp, in);
        else if (T == 2) split_bf16_kernel<2><<<g, 256, 0, s>>>(x, pl, rows, rp, in);
        else split_bf16_kernel<3><<<g, 256, 0, s>>>(x, pl, rows, rp, in);
    }
    if (auto e = cudaGetLastError(); e != cudaSuccess) return e;
    out->planes = S.planes[slot];
    out->rows = rows;
    out->rows_pad = rp;
    out->in = in;
    out->terms = T;
    out->f16 = f16;
    return cudaSuccess;
}

bool dense_big_fast4_supported() {
#if defined(STRIX_CUDA_FAST8_SM120A)
    return ftile::supported();
#else
    return false;
#endif
}

namespace {
// The FAST4 X tile + row scales of one activation into the context scratch; `quantise(x_tile, rs)` enqueues the producer.
template <class Quantise>
cudaError_t split4_into(std::uint32_t rows, std::uint32_t in, BigAct* out, BigCtx* ctx, Quantise quantise) {
    if (!dense_big_fast4_supported()) return cudaErrorNotSupported;
    if (out == nullptr || rows == 0 || in % ftile::kK != 0) return cudaErrorInvalidValue;
    Scratch& S = ctx ? *ctx : scratch();
    if (auto e = bind_device(S); e != cudaSuccess) return e;
    const std::uint32_t rp = ftile::rows_padded(rows);
    if (auto e = grow(&S.planes[1], &S.planes_bytes[1], ftile::x_tile_bytes(rp, in)); e != cudaSuccess) return e;
    if (auto e = grow(reinterpret_cast<void**>(&S.rscale), &S.rscale_bytes, static_cast<std::uint64_t>(rp) * 4); e != cudaSuccess) return e;
    if (auto e = quantise(static_cast<std::uint8_t*>(S.planes[1]), S.rscale); e != cudaSuccess) return e;
    *out = BigAct{};
    out->planes = S.planes[1];
    out->rows = rows;
    out->rows_pad = rp;
    out->in = in;
    out->terms = 1;
    out->f16 = true;  // the nvfp4 weight class
    out->f4 = true;
    return cudaSuccess;
}
}  // namespace

cudaError_t dense_big_split4(const float* x, std::uint32_t rows, std::uint32_t in, BigAct* out, cudaStream_t s, BigCtx* ctx) {
    if (!dense_big_fast4_supported()) return cudaErrorNotSupported;
    if (x == nullptr) return cudaErrorInvalidValue;
    return split4_into(rows, in, out, ctx, [&](std::uint8_t* xt, float* rs) { return ftile::quantize_enqueue(x, rows, in, xt, rs, s); });
}

bool dense_big_norm_split4_fits(std::uint32_t in) { return ftile::norm_quantize_fits(in); }

cudaError_t dense_big_norm_split4(const float* x, const float* gain, float eps, std::uint32_t rows, std::uint32_t in, BigAct* out,
                                  cudaStream_t s, BigCtx* ctx) {
    if (!dense_big_fast4_supported()) return cudaErrorNotSupported;
    if (x == nullptr || !ftile::norm_quantize_fits(in)) return cudaErrorInvalidValue;
    return split4_into(rows, in, out, ctx,
                       [&](std::uint8_t* xt, float* rs) { return ftile::norm_quantize_enqueue(x, gain, eps, rows, in, xt, rs, s); });
}

cudaError_t dense_big_gated_norm_split4(const float* x, const float* z, const float* gain, float eps, std::uint32_t rows, std::uint32_t in,
                                        BigAct* out, cudaStream_t s, BigCtx* ctx) {
    if (!dense_big_fast4_supported()) return cudaErrorNotSupported;
    if (x == nullptr || z == nullptr || gain == nullptr || !ftile::norm_quantize_fits(in)) return cudaErrorInvalidValue;
    return split4_into(rows, in, out, ctx, [&](std::uint8_t* xt, float* rs) {
        return ftile::gated_norm_quantize_enqueue(x, z, gain, eps, rows, in, xt, rs, s);
    });
}

namespace {
// W tile operand of w: the resident entry (repacked once) or, without a cache, a per-call repack into the context scratch at `slot`
// (0 or 1, so a fused gate/up call keeps both operands live).
cudaError_t tile_operand(Scratch& S, const DenseWeight& w, cudaStream_t s, int slot, const std::uint8_t** out) {
    const auto repack = [&](unsigned char* d) {
        return ftile::repack_enqueue(static_cast<const std::uint8_t*>(w.data), w.out, w.in, d, s);
    };
    unsigned char* res = nullptr;
    if (auto e = cached_operand(S, w, operand_cache::Kind::e2m1_tile, s, repack, &res); e != cudaSuccess) return e;
    if (res == nullptr) {
        const std::uint64_t bytes = ftile::w_tile_bytes(w.out, w.in);
        if (auto e = grow(&S.wexp, &S.wexp_bytes, 2 * bytes); e != cudaSuccess) return e;
        res = static_cast<unsigned char*>(S.wexp) + slot * bytes;
        if (auto e = repack(res); e != cudaSuccess) return e;
    }
    *out = res;
    return cudaSuccess;
}
cudaError_t gemm4(Scratch& S, const DenseWeight& w, const BigAct& a, float* y, const float* aux, BigEpi epi, cudaStream_t s) {
    ftile::GemmArgs g;
    if (auto e = tile_operand(S, w, s, 0, &g.w); e != cudaSuccess) return e;
    g.x = static_cast<const std::uint8_t*>(a.planes);
    g.rs = S.rscale;
    g.aux = aux;
    g.y = y;
    g.gs = w.gscale;
    g.out = w.out;
    g.in = w.in;
    g.rows = a.rows;
    g.rows_pad = a.rows_pad;
    g.epi = epi == BigEpi::residual ? ftile::Epi::residual : epi == BigEpi::silu_mul ? ftile::Epi::silu_mul : ftile::Epi::store;
    g.feed = S.feed;
    return ftile::gemm_enqueue(g, s);
}
}  // namespace

cudaError_t dense_big_glu4(const DenseWeight& gate, const DenseWeight& up, const BigAct& a, float* y, cudaStream_t s, BigCtx* ctx) {
    if (!a.f4 || !dense_big_fast4_supported()) return cudaErrorInvalidValue;
    if (gate.format != WeightFormat::nvfp4 || up.format != WeightFormat::nvfp4 || gate.out != up.out || gate.in != up.in ||
        a.in != gate.in || !dense_big_supported(gate) || !dense_big_supported(up))
        return cudaErrorInvalidValue;
    Scratch& S = ctx ? *ctx : scratch();
    if (auto e = bind_device(S); e != cudaSuccess) return e;
    ftile::GemmArgs g;
    if (auto e = tile_operand(S, gate, s, 0, &g.w); e != cudaSuccess) return e;
    if (auto e = tile_operand(S, up, s, 1, &g.w2); e != cudaSuccess) return e;
    g.x = static_cast<const std::uint8_t*>(a.planes);
    g.rs = S.rscale;
    g.y = y;
    g.gs = gate.gscale;
    g.gs2 = up.gscale;
    g.out = gate.out;
    g.in = gate.in;
    g.rows = a.rows;
    g.rows_pad = a.rows_pad;
    g.epi = ftile::Epi::glu;
    g.feed = S.feed;
    return ftile::gemm_enqueue(g, s);
}

cudaError_t dense_big_gemm(const DenseWeight& w, const BigAct& a, float* y, const float* aux, BigEpi epi, cudaStream_t s, BigCtx* ctx) {
    if (a.f8 && !dense_big_fast8_supported()) return cudaErrorNotSupported;
    if (!dense_big_supported(w) || a.in != w.in || a.f16 != (w.format == WeightFormat::nvfp4)) return cudaErrorInvalidValue;
    if (epi != BigEpi::none && aux == nullptr) return cudaErrorInvalidValue;
    Scratch& S = ctx ? *ctx : scratch();
    if (auto e = bind_device(S); e != cudaSuccess) return e;
    if (a.f4) {
        if (w.format != WeightFormat::nvfp4 || !dense_big_fast4_supported()) return cudaErrorInvalidValue;
        return gemm4(S, w, a, y, aux, epi, s);
    }
    BigArgs ba{};
    ba.P = static_cast<const std::uint16_t*>(a.planes);
    ba.y = y;
    ba.aux = aux;
    ba.out = w.out;
    ba.in = w.in;
    ba.rows = a.rows;
    ba.rows_pad = a.rows_pad;
    ba.gscale = w.gscale;
    ba.epi = static_cast<int>(epi);
    const std::uint64_t elems = static_cast<std::uint64_t>(w.out) * w.in;
    Mode mode = kChain;
    if (w.format == WeightFormat::bf16) {
        ba.A = static_cast<const std::uint16_t*>(w.data);
    } else if (w.format == WeightFormat::q8_0) {
        mode = kFold;
        const std::size_t nb = elems / 32;
        const std::uint64_t soff = operand_cache::scales_offset(operand_cache::Kind::q8_bf16, w.out, w.in);
        const auto expand_q8 = [&](std::uint16_t* vals, float* scales) {
            expand_q8_kernel<<<static_cast<unsigned>((nb + 255) / 256), 256, 0, s>>>(static_cast<const unsigned char*>(w.data), nb, vals, scales);
            return cudaGetLastError();
        };
        unsigned char* res = nullptr;
        if (auto e = cached_operand(S, w, operand_cache::Kind::q8_bf16, s,
                                    [&](unsigned char* d) { return expand_q8(reinterpret_cast<std::uint16_t*>(d), reinterpret_cast<float*>(d + soff)); }, &res);
            e != cudaSuccess)
            return e;
        if (res != nullptr) {
            ba.A = reinterpret_cast<const std::uint16_t*>(res);
            ba.scales = reinterpret_cast<float*>(res + soff);
        } else {
            if (auto e = grow(&S.wexp, &S.wexp_bytes, elems * 2); e != cudaSuccess) return e;
            if (auto e = grow(reinterpret_cast<void**>(&S.scales), &S.scales_bytes, elems / 32 * 4); e != cudaSuccess) return e;
            if (auto e = expand_q8(static_cast<std::uint16_t*>(S.wexp), S.scales); e != cudaSuccess) return e;
            ba.A = static_cast<const std::uint16_t*>(S.wexp);
            ba.scales = S.scales;
        }
    } else if (a.f8) {
        unsigned char* res = nullptr;
        if (auto e = cached_operand(S, w, operand_cache::Kind::e4m3, s, [&](unsigned char* d) { return nvfp4_expand_e4m3(w, d, s); }, &res);
            e != cudaSuccess)
            return e;
        if (res != nullptr) {
            ba.A = reinterpret_cast<const std::uint16_t*>(res);
        } else {
            if (auto e = grow(&S.wexp, &S.wexp_bytes, elems * 2); e != cudaSuccess) return e;
            if (auto e = nvfp4_expand_e4m3(w, S.wexp, s); e != cudaSuccess) return e;
            ba.A = static_cast<const std::uint16_t*>(S.wexp);
        }
        ba.rscale = S.rscale;
        return launch8<4, 2, 3>(ba, w.out, a.rows, s);
    } else {
        mode = kSplit;
        unsigned char* res = nullptr;
        if (auto e = cached_operand(S, w, operand_cache::Kind::f16, s, [&](unsigned char* d) { return nvfp4_expand_f16(w, d, s); }, &res);
            e != cudaSuccess)
            return e;
        if (res != nullptr) {
            ba.A = reinterpret_cast<const std::uint16_t*>(res);
        } else {
            if (auto e = grow(&S.wexp, &S.wexp_bytes, elems * 2); e != cudaSuccess) return e;
            if (auto e = nvfp4_expand_f16(w, S.wexp, s); e != cudaSuccess) return e;
            ba.A = static_cast<const std::uint16_t*>(S.wexp);
        }
    }
    const int bkb = (w.in % 128 == 0 && a.terms <= 2 && g_bkb2) ? 2 : 1;
    const Table* tb = get_table(S, mode, w.out, w.in, bkb);
    if (tb == nullptr) return cudaErrorMemoryAllocation;
    ba.kbs = tb->kbs;
    ba.seg_end = tb->seg_end;
    ba.nseg = tb->nseg;
    ba.nkb = tb->nkb;
    if (mode == kSplit) return bkb == 2 ? launch<true, 1, 4, kSplit, 2>(ba, w.out, a.rows, s) : launch<true, 1, 4, kSplit, 1>(ba, w.out, a.rows, s);
    if (mode == kFold) {
        if (a.terms == 1) return bkb == 2 ? launch<false, 1, 4, kFold, 2>(ba, w.out, a.rows, s) : launch<false, 1, 2, kFold, 1>(ba, w.out, a.rows, s);
        if (a.terms == 2) return bkb == 2 ? launch<false, 2, 2, kFold, 2>(ba, w.out, a.rows, s) : launch<false, 2, 2, kFold, 1>(ba, w.out, a.rows, s);
        return launch<false, 3, 2, kFold, 1>(ba, w.out, a.rows, s);
    }
    if (a.terms == 1) return bkb == 2 ? launch<false, 1, 4, kChain, 2>(ba, w.out, a.rows, s) : launch<false, 1, 4, kChain, 1>(ba, w.out, a.rows, s);
    if (a.terms == 2) return bkb == 2 ? launch<false, 2, 4, kChain, 2>(ba, w.out, a.rows, s) : launch<false, 2, 4, kChain, 1>(ba, w.out, a.rows, s);
    return launch<false, 3, 4, kChain, 1>(ba, w.out, a.rows, s);
}

}  // namespace strix::cuda_core
