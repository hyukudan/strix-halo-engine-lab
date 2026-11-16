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
#include <tuple>
#include <vector>

#include "strix/cuda_core/nvfp4.hpp"
#include "../common.cuh"

namespace strix::cuda_core {
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
    float gscale;
    int epi;
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

template <int T, int NT>
struct Cfg {
    static constexpr int BN = 4 * NT * 8;
    static constexpr int A_ELEMS = BM * BK, B_ELEMS = T * BN * BK;
    static constexpr int STAGE_ELEMS = A_ELEMS + B_ELEMS;
    static constexpr int STAGES = (STAGE_ELEMS * 2 * 4 <= 99 * 1024) ? 4 : 3;
    static constexpr int SMEM = STAGES * STAGE_ELEMS * 2;
};

template <int T, int NT>
__device__ __forceinline__ void load_unit(const BigArgs& a, std::uint16_t* sbase, std::uint32_t o0, std::uint32_t r0, std::uint32_t kb) {
    using C = Cfg<T, NT>;
    const std::uint32_t tid = threadIdx.x;
    const std::uint32_t sb = static_cast<std::uint32_t>(__cvta_generic_to_shared(sbase));
#pragma unroll
    for (int i = 0; i < BM * 8 / 256; ++i) {
        const std::uint32_t idx = tid + 256 * i, r = idx >> 3, u = idx & 7;
        const std::uint32_t orow = min(o0 + r, a.out - 1);
        const std::uint16_t* src = a.A + static_cast<std::size_t>(orow) * a.in + kb * 32 + u * 4;
        cp8(sb + (r * 32 + ((u ^ ((r >> 1) & 1)) * 4)) * 2, src);
    }
#pragma unroll
    for (int t = 0; t < T; ++t)
#pragma unroll
        for (int i = 0; i < C::BN * 8 / 256; ++i) {
            const std::uint32_t idx = tid + 256 * i, r = idx >> 3, u = idx & 7;
            const std::uint16_t* src = a.P + (static_cast<std::size_t>(t) * a.rows_pad + r0 + r) * a.in + kb * 32 + u * 4;
            cp8(sb + (C::A_ELEMS + (t * C::BN + r) * 32 + ((u ^ ((r >> 1) & 1)) * 4)) * 2, src);
        }
}

template <bool F16, int T, int NT, int MODE>
__global__ void __launch_bounds__(256, 1) big_kernel(const BigArgs a) {
    using C = Cfg<T, NT>;
    extern __shared__ __align__(16) std::uint16_t smem[];
    const std::uint32_t o0 = blockIdx.x * BM, r0 = blockIdx.y * C::BN;
    const int warp = threadIdx.x >> 5, lane = threadIdx.x & 31, g = lane >> 2, q = lane & 3, swz = (g >> 1) & 1;
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
        if (nload < nkb) load_unit<T, NT>(a, smem + s * C::STAGE_ELEMS, o0, r0, a.kbs[nload]);
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
            if (nload < nkb) load_unit<T, NT>(a, smem + (nload % C::STAGES) * C::STAGE_ELEMS, o0, r0, a.kbs[nload]);
            ++nload;
            cp_commit();
            const std::uint16_t* sA = smem + (n % C::STAGES) * C::STAGE_ELEMS;
            const std::uint16_t* sB = sA + C::A_ELEMS;
            float scl[4][2];
            [[maybe_unused]] float part[4][NT][4];
            if constexpr (MODE == kFold) {
                const std::uint32_t kb = a.kbs[n];
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
                const int pu = (2 * q + s) ^ swz;
                uint2 af[4][2];
#pragma unroll
                for (int mt = 0; mt < 4; ++mt)
#pragma unroll
                    for (int h = 0; h < 2; ++h)
                        af[mt][h] = *reinterpret_cast<const uint2*>(sA + (wm * 64 + mt * 16 + h * 8 + g) * 32 + pu * 4);
#pragma unroll
                for (int nt = 0; nt < NT; ++nt) {
                    uint2 bf[T];
#pragma unroll
                    for (int t = 0; t < T; ++t)
                        bf[t] = *reinterpret_cast<const uint2*>(sB + (t * C::BN + wn * NT * 8 + nt * 8 + g) * 32 + pu * 4);
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
#pragma unroll
    for (int mt = 0; mt < 4; ++mt)
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
                    const std::size_t idx = static_cast<std::size_t>(r) * a.out + o;
                    float v = S[mt][nt][h * 2 + j];
                    if constexpr (MODE == kSplit) v = __fmul_rn(v, a.gscale);
                    if (a.epi == 1) v = __fadd_rn(a.aux[idx], v);
                    else if (a.epi == 2) v = __fmul_rn(cc_silu(a.aux[idx]), v);
                    a.y[idx] = v;
                }
        }
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
struct Table {
    int* kbs = nullptr;
    int* seg_end = nullptr;
    std::uint32_t nseg = 0, nkb = 0;
};
std::map<std::tuple<int, std::uint32_t, std::uint32_t, std::uint32_t>, Table>& tables() {
    static std::map<std::tuple<int, std::uint32_t, std::uint32_t, std::uint32_t>, Table> t;
    return t;
}
struct Scratch {
    void* planes[2] = {nullptr, nullptr};  // slot 0: bf16 planes, slot 1: f16 plane (both may be live for one x)
    std::uint64_t planes_bytes[2] = {0, 0};
    void* wexp = nullptr;
    std::uint64_t wexp_bytes = 0;
    float* scales = nullptr;
    std::uint64_t scales_bytes = 0;
};
Scratch& scratch() {
    static Scratch s;
    return s;
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

const Table* get_table(Mode mode, std::uint32_t out, std::uint32_t in) {
    const std::uint32_t nkb = in >> 5;
    std::uint32_t key2 = 0;
    if (mode == kSplit) key2 = nvfp4_split_k(out, in);
    const auto key = std::make_tuple(static_cast<int>(mode), in, mode == kSplit ? key2 : 0u, 0u);
    auto it = tables().find(key);
    if (it != tables().end()) return &it->second;
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
    Table t;
    t.nkb = static_cast<std::uint32_t>(kbs.size());
    t.nseg = static_cast<std::uint32_t>(ends.size());
    if (cudaMalloc(reinterpret_cast<void**>(&t.kbs), kbs.size() * 4) != cudaSuccess) return nullptr;
    if (cudaMalloc(reinterpret_cast<void**>(&t.seg_end), ends.size() * 4) != cudaSuccess) return nullptr;
    cudaMemcpy(t.kbs, kbs.data(), kbs.size() * 4, cudaMemcpyHostToDevice);
    cudaMemcpy(t.seg_end, ends.data(), ends.size() * 4, cudaMemcpyHostToDevice);
    return &tables().emplace(key, t).first->second;
}

template <bool F16, int T, int NT, int MODE>
cudaError_t launch(const BigArgs& a, std::uint32_t out, std::uint32_t rows, cudaStream_t s) {
    using C = Cfg<T, NT>;
    static bool attr = false;
    if (!attr) {
        if (auto e = cudaFuncSetAttribute(big_kernel<F16, T, NT, MODE>, cudaFuncAttributeMaxDynamicSharedMemorySize, C::SMEM); e != cudaSuccess) return e;
        attr = true;
    }
    const dim3 grid((out + BM - 1) / BM, (rows + C::BN - 1) / C::BN);
    big_kernel<F16, T, NT, MODE><<<grid, 256, C::SMEM, s>>>(a);
    return cudaGetLastError();
}

}  // namespace

bool dense_big_supported(const DenseWeight& w) {
    if (w.out % 16 != 0 || w.data == nullptr) return false;
    if (w.format == WeightFormat::bf16 || w.format == WeightFormat::q8_0) return w.in % 32 == 0;
    if (w.format == WeightFormat::nvfp4) return w.in % 128 == 0;
    return false;
}

cudaError_t dense_big_reserve(std::uint32_t max_rows, std::uint32_t max_in, std::uint64_t max_weight_elems, int max_terms) {
    Scratch& S = scratch();
    const std::uint32_t rp = (max_rows + 127) / 128 * 128;
    if (auto e = grow(&S.planes[0], &S.planes_bytes[0], static_cast<std::uint64_t>(std::max(1, max_terms)) * rp * max_in * 2); e != cudaSuccess) return e;
    if (auto e = grow(&S.planes[1], &S.planes_bytes[1], static_cast<std::uint64_t>(rp) * max_in * 2); e != cudaSuccess) return e;
    if (auto e = grow(&S.wexp, &S.wexp_bytes, max_weight_elems * 2); e != cudaSuccess) return e;
    return grow(reinterpret_cast<void**>(&S.scales), &S.scales_bytes, max_weight_elems / 32 * 4);
}

void dense_big_release() {
    Scratch& S = scratch();
    for (void* p : S.planes)
        if (p) cudaFree(p);
    if (S.wexp) cudaFree(S.wexp);
    if (S.scales) cudaFree(S.scales);
    S = Scratch{};
}
std::uint64_t dense_big_reserved_bytes() { return scratch().planes_bytes[0] + scratch().planes_bytes[1] + scratch().wexp_bytes + scratch().scales_bytes; }

cudaError_t dense_big_split(const float* x, std::uint32_t rows, std::uint32_t in, WeightFormat fmt, int terms, BigAct* out, cudaStream_t s) {
    if (rows == 0 || in % 32 != 0 || terms < 1 || terms > 3) return cudaErrorInvalidValue;
    Scratch& S = scratch();
    const std::uint32_t rp = (rows + 127) / 128 * 128;
    const bool f16 = fmt == WeightFormat::nvfp4;
    const int T = f16 ? 1 : terms;
    const int slot = f16 ? 1 : 0;
    if (auto e = grow(&S.planes[slot], &S.planes_bytes[slot], static_cast<std::uint64_t>(T) * rp * in * 2); e != cudaSuccess) return e;
    std::uint16_t* pl = static_cast<std::uint16_t*>(S.planes[slot]);
    if (f16) {
        const std::size_t n = static_cast<std::size_t>(rp) * (in >> 2);
        split_f16_kernel<<<static_cast<unsigned>((n + 255) / 256), 256, 0, s>>>(x, pl, rows, rp, in);
    } else {
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

cudaError_t dense_big_gemm(const DenseWeight& w, const BigAct& a, float* y, const float* aux, BigEpi epi, cudaStream_t s) {
    if (!dense_big_supported(w) || a.in != w.in || a.f16 != (w.format == WeightFormat::nvfp4)) return cudaErrorInvalidValue;
    if (epi != BigEpi::none && aux == nullptr) return cudaErrorInvalidValue;
    Scratch& S = scratch();
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
        if (auto e = grow(&S.wexp, &S.wexp_bytes, elems * 2); e != cudaSuccess) return e;
        if (auto e = grow(reinterpret_cast<void**>(&S.scales), &S.scales_bytes, elems / 32 * 4); e != cudaSuccess) return e;
        const std::size_t nb = elems / 32;
        expand_q8_kernel<<<static_cast<unsigned>((nb + 255) / 256), 256, 0, s>>>(static_cast<const unsigned char*>(w.data), nb,
                                                                               static_cast<std::uint16_t*>(S.wexp), S.scales);
        if (auto e = cudaGetLastError(); e != cudaSuccess) return e;
        ba.A = static_cast<const std::uint16_t*>(S.wexp);
        ba.scales = S.scales;
    } else {
        mode = kSplit;
        if (auto e = grow(&S.wexp, &S.wexp_bytes, elems * 2); e != cudaSuccess) return e;
        if (auto e = nvfp4_expand_f16(w, S.wexp, s); e != cudaSuccess) return e;
        ba.A = static_cast<const std::uint16_t*>(S.wexp);
    }
    const Table* tb = get_table(mode, w.out, w.in);
    if (tb == nullptr) return cudaErrorMemoryAllocation;
    ba.kbs = tb->kbs;
    ba.seg_end = tb->seg_end;
    ba.nseg = tb->nseg;
    ba.nkb = tb->nkb;
    if (mode == kSplit) return launch<true, 1, 4, kSplit>(ba, w.out, a.rows, s);
    if (mode == kFold) {
        if (a.terms == 1) return launch<false, 1, 2, kFold>(ba, w.out, a.rows, s);
        if (a.terms == 2) return launch<false, 2, 2, kFold>(ba, w.out, a.rows, s);
        return launch<false, 3, 2, kFold>(ba, w.out, a.rows, s);
    }
    if (a.terms == 1) return launch<false, 1, 4, kChain>(ba, w.out, a.rows, s);
    if (a.terms == 2) return launch<false, 2, 4, kChain>(ba, w.out, a.rows, s);
    return launch<false, 3, 4, kChain>(ba, w.out, a.rows, s);
}

}  // namespace strix::cuda_core
