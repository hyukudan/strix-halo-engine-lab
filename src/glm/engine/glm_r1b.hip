// R1b generated identity (default; default prompts use R1) (approved arithmetic contract): device kernels of the blocked chains (glm_math.hpp
// r1b_dot). Scope and allowlist: glm_r1b.hpp. Bitwise equal to the host emulation by construction (same fmaf
// sequence per block, same tree, same R6 / collapse code as dev::hc_pre), gated by tests/hip_glm_r1b_tests.hip.
// Build flags as the Tier A kernels (-ffp-contract=off -fno-fast-math, denormals kept).
#include "glm/engine/glm_r1b.hpp"

#include "core/fma_rows.hpp"
#include "glm/common/glm_hc.hpp"
#include "glm/common/glm_math.hpp"

#include <atomic>
#include <cstdlib>
#include <stdexcept>
#include <string>

namespace strix::glm::r1b {
namespace {

std::atomic<bool> g_active{false};

void ok(hipError_t e, const char* what) {
    if (e != hipSuccess) throw std::runtime_error(std::string("glm r1b ") + what + ": " + hipGetErrorString(e));
}

constexpr unsigned kMaxRows = 8;

namespace rows_core = ::strix::core::fma_rows;
using DenseReduction = rows_core::Reduction<rows_core::Reduce::blocked16_tree>;
static_assert(DenseReduction::inputs == kHidden && DenseReduction::inputs_per_block == kR1bBlock);

void dense_jobs(const rows_core::Blocked16Group& group, hipStream_t s, int variant) {
    using Form = rows_core::Blocked16Form;
    if (variant < 0) variant = dense_variant();
    if (variant == 7) variant = group.rows == 1 ? 5 : 6;
    if (variant == 5 && group.rows != 1) throw std::invalid_argument("r1b dense v5: packed weights require rows == 1");
    Form form = Form::split;
    if (variant == 6) form = Form::packed;
    else if (variant == 5) form = Form::staged_packed;
    else if (variant == 4 && group.rows == 1) form = Form::staged;
    ok(rows_core::launch<rows_core::Reduce::blocked16_tree>(group, form, s), "dense_k4096");
}

// Lossless permutation: destination vectors [c8][output][block_j][8].
__global__ void pack_k4096_kernel(const std::uint16_t* W, unsigned n, std::uint16_t* packed) {
    const std::size_t v = static_cast<std::size_t>(blockIdx.x) * 256 + threadIdx.x;
    if (v >= static_cast<std::size_t>(n) * 512) return;
    const unsigned j = static_cast<unsigned>(v % 16), o = static_cast<unsigned>((v / 16) % n),
                   c8 = static_cast<unsigned>(v / (static_cast<std::size_t>(16) * n));
    *reinterpret_cast<uint4*>(packed + v * 8) =
        *reinterpret_cast<const uint4*>(W + static_cast<std::size_t>(o) * 4096 + j * 256 + c8 * 8);
}

// ---------------------------------------------------------------- mHC hc_pre (24 x 16,384 F32 chains)
constexpr unsigned kN = kHc * kHidden;          // 16384
constexpr unsigned kHcBlocks = kN / kR1bBlock;  // 64
constexpr unsigned kPad = kR1bBlock + 1;        // LDS row stride (bank spread across chains / rows)
static_assert(kHcBlocks == kR1bMaxBlocks && (kHcBlocks & (kHcBlocks - 1)) == 0, "64-block tree");

__device__ float g_rr[kMaxRows];
__device__ float g_part[kMaxRows * kHcMix * kHcBlocks];

// R2 tree in LDS over L lanes (identical to glm_tier_a.hip lds_tree).
template <unsigned L> __device__ __forceinline__ float lds_tree(float* s, float part) {
    const unsigned l = threadIdx.x;
    s[l] = part;
    __syncthreads();
    for (unsigned h = L / 2; h > 0; h >>= 1U) {
        if (l < h) s[l] = s[l] + s[l + h];
        __syncthreads();
    }
    const float r = s[0];
    __syncthreads();
    return r;
}

// Launch 1: per row, the R2/R3 sum of squares (256 lane-strided fmaf partials + tree) -> rr = 1 / sqrt(ss / n + eps).
__global__ __launch_bounds__(256) void hc_rr_kernel(const float* X) {
    __shared__ float s[256];
    const unsigned r = blockIdx.x, l = threadIdx.x;
    const float* Xr = X + static_cast<std::size_t>(r) * kN;
    float a = 0.0F;
    for (unsigned i = l; i < kN; i += 256) a = __builtin_fmaf(Xr[i], Xr[i], a);
    const float ss = lds_tree<256>(s, a);
    if (l == 0) g_rr[r] = inv_sqrt(ss / static_cast<float>(kN) + kEps);
}

// Launch 2: workgroup j = block j of 256 inputs. F = X * rr (one rounding, as dev::hc_pre), written out; thread
// (row r, chain o) runs the ascending fmaf chain of block j from +0.
constexpr unsigned kPartThreads = kHcMix * kMaxRows;  // 192
template <unsigned Rows>
__global__ __launch_bounds__(kPartThreads) void hc_part_kernel(const float* X, const float* fn, float* F, unsigned rows) {
    __shared__ float fs[kHcMix][kPad];
    __shared__ float xs[Rows][kPad];
    const unsigned j = blockIdx.x, t = threadIdx.x, i0 = j * kR1bBlock;
    for (unsigned idx = t; idx < kHcMix * kR1bBlock; idx += kPartThreads) {
        const unsigned o = idx / kR1bBlock, i = idx % kR1bBlock;
        fs[o][i] = fn[static_cast<std::size_t>(o) * kN + i0 + i];
    }
    for (unsigned idx = t; idx < rows * kR1bBlock; idx += kPartThreads) {
        const unsigned r = idx / kR1bBlock, i = idx % kR1bBlock;
        const std::size_t g = static_cast<std::size_t>(r) * kN + i0 + i;
        const float v = X[g] * g_rr[r];
        xs[r][i] = v;
        F[g] = v;
    }
    __syncthreads();
    const unsigned o = t % kHcMix, r = t / kHcMix;
    if (r >= rows) return;
    float acc = 0.0F;
    for (unsigned i = 0; i < kR1bBlock; ++i) acc = __builtin_fmaf(xs[r][i], fs[o][i], acc);
    g_part[(static_cast<std::size_t>(r) * kHcMix + o) * kHcBlocks + j] = acc;
}

// Launch 3: per row, the 64-partial tree (offsets 32 .. 1) per chain, then R6 + collapse exactly as dev::hc_pre.
// Coherent: the partials were stored by other workgroups of the same launch (hc_site_kernel), read at agent scope.
template <bool Distributed, bool ParallelR6 = false, bool Norm = false, bool Coherent = false>
__device__ __forceinline__ void hc_fin_row(unsigned r, const float* X, const float* base, const float* scale, float* mix, float* xc,
                                           const std::uint16_t* norm, float* xn) {
    __shared__ float z[kHcMix];
    __shared__ float m[kHcMix];
    const unsigned l = threadIdx.x;
    const float* Xr = X + static_cast<std::size_t>(r) * kN;
    if constexpr (Distributed) {
        // Lane a owns indices a + 8*k. These local stages are exactly the
        // serial tree's offsets 32, 16, 8, including the left operand.
        if (l < kHcMix * 8) {
            const unsigned chain = l / 8, lane = l % 8;
            const float* src = g_part + (static_cast<std::size_t>(r) * kHcMix + chain) * kHcBlocks;
            float p[8];
#pragma unroll
            for (unsigned k = 0; k < 8; ++k)
                p[k] = Coherent ? __hip_atomic_load(&src[lane + 8 * k], __ATOMIC_RELAXED, __HIP_MEMORY_SCOPE_AGENT) : src[lane + 8 * k];
#pragma unroll
            for (unsigned h = 4; h > 0; h >>= 1U) {
#pragma unroll
                for (unsigned k = 0; k < h; ++k) p[k] = p[k] + p[k + h];
            }
            float acc = p[0];
            for (unsigned h = 4; h > 0; h >>= 1U) {
                const float peer = __shfl_down(acc, h, 8);
                if (lane < h) acc = acc + peer;
            }
            if (lane == 0) z[chain] = acc;
        }
    } else if (l < kHcMix) {
        float p[kHcBlocks];
        const float* src = g_part + (static_cast<std::size_t>(r) * kHcMix + l) * kHcBlocks;
        for (unsigned j = 0; j < kHcBlocks; ++j) p[j] = src[j];
        for (unsigned hh = kHcBlocks / 2; hh >= 1; hh /= 2)
            for (unsigned i = 0; i < hh; ++i) p[i] = p[i] + p[i + hh];
        z[l] = p[0];
    }
    __syncthreads();
    if constexpr (ParallelR6) {
        core::hc::mix_final_wave<HcR6>(z, base, scale, kHcEps, m);
    } else if (l == 0) {
        core::hc::mix_final_packed<HcR6>(z, scale, base, kHcEps, m);
        for (unsigned k = 0; k < kHcMix; ++k) mix[static_cast<std::size_t>(r) * kHcMix + k] = m[k];
    }
    __syncthreads();
    if constexpr (ParallelR6)
        if (l < kHcMix) mix[static_cast<std::size_t>(r) * kHcMix + l] = m[l];
    float* xr = xc + static_cast<std::size_t>(r) * kHidden;
    for (unsigned d = l; d < kHidden; d += 256)
        xr[d] = core::hc::packed_dot(m, Xr, d, kHidden);
    if constexpr (Norm) {
        // Materialize xc before reading it, exactly as the separate rmsnorm launch.
        __syncthreads();
        __shared__ float ns[256];
        float a = 0.0F;
        for (unsigned i = l; i < kHidden; i += 256) a = __builtin_fmaf(xr[i], xr[i], a);
        const float ss = lds_tree<256>(ns, a);
        const float rr = inv_sqrt(ss / static_cast<float>(kHidden) + kEps);
        float* yr = xn + static_cast<std::size_t>(r) * kHidden;
        for (unsigned i = l; i < kHidden; i += 256)
            yr[i] = norm ? (xr[i] * rr) * bf16_to_float(norm[i]) : xr[i] * rr;
    }
}
template <bool Distributed, bool ParallelR6 = false, bool Norm = false>
__global__ __launch_bounds__(256) void hc_fin_kernel(const float* X, const float* base, const float* scale, float* mix, float* xc,
                                                     const std::uint16_t* norm = nullptr, float* xn = nullptr) {
    hc_fin_row<Distributed, ParallelR6, Norm>(blockIdx.x, X, base, scale, mix, xc, norm, xn);
}

// One launch per decode mHC site (hc_pre_norm with STRIX_GLM_HC_SITE): launches 1 + 2 (+ 3 when Fin) of hc_pre_norm in
// one grid of kHcBlocks workgroups. Every workgroup recomputes rr of each row exactly as hc_rr_kernel (the same 256
// lane-strided fmaf chains and the same lds_tree pairs, one tree per row), then runs hc_part's block j with that rr; with
// Fin the last workgroup to arrive (device-scope counter, as variant 2) runs hc_fin<true, true, true> row after row. Fin
// is for one row: the last workgroup would run rows serially, so multi-row calls launch hc_fin<..., Norm> separately.
__device__ unsigned g_site_cnt;
template <unsigned Rows, bool Fin>
__global__ __launch_bounds__(256) void hc_site_kernel(const float* X, const float* fn, float* F, const float* base, const float* scale,
                                                      float* mix, float* xc, const std::uint16_t* norm, float* xn, unsigned rows) {
    __shared__ float fs[kHcMix][kPad];
    __shared__ float xs[Rows][kPad];
    __shared__ float ss[Rows][256];
    __shared__ float rrs[Rows];
    __shared__ unsigned last;
    const unsigned j = blockIdx.x, t = threadIdx.x, i0 = j * kR1bBlock;
    for (unsigned r = 0; r < rows; ++r) {
        // All 64 loads in flight first, then hc_rr's chain in its order (i = t, t + 256, ...).
        const float* Xr = X + static_cast<std::size_t>(r) * kN + t;
        float v[kN / 256];
#pragma unroll
        for (unsigned k = 0; k < kN / 256; ++k) v[k] = Xr[k * 256];
        float a = 0.0F;
#pragma unroll
        for (unsigned k = 0; k < kN / 256; ++k) a = __builtin_fmaf(v[k], v[k], a);
        ss[r][t] = a;
    }
    for (unsigned idx = t; idx < kHcMix * kR1bBlock; idx += 256) {
        const unsigned o = idx / kR1bBlock, i = idx % kR1bBlock;
        fs[o][i] = fn[static_cast<std::size_t>(o) * kN + i0 + i];
    }
    __syncthreads();
    for (unsigned h = 128; h > 0; h >>= 1U) {
        for (unsigned r = 0; r < rows; ++r)
            if (t < h) ss[r][t] = ss[r][t] + ss[r][t + h];
        __syncthreads();
    }
    if (t < rows) rrs[t] = inv_sqrt(ss[t][0] / static_cast<float>(kN) + kEps);
    __syncthreads();
    for (unsigned idx = t; idx < rows * kR1bBlock; idx += 256) {
        const unsigned r = idx / kR1bBlock, i = idx % kR1bBlock;
        const std::size_t g = static_cast<std::size_t>(r) * kN + i0 + i;
        const float v = X[g] * rrs[r];
        xs[r][i] = v;
        F[g] = v;
    }
    __syncthreads();
    const unsigned o = t % kHcMix, r = t / kHcMix;
    if (t < kPartThreads && r < rows) {
        float acc = 0.0F;
        for (unsigned i = 0; i < kR1bBlock; ++i) acc = __builtin_fmaf(xs[r][i], fs[o][i], acc);
        float* dst = &g_part[(static_cast<std::size_t>(r) * kHcMix + o) * kHcBlocks + j];
        if constexpr (Fin) __hip_atomic_store(dst, acc, __ATOMIC_RELAXED, __HIP_MEMORY_SCOPE_AGENT);
        else *dst = acc;
    }
    if constexpr (Fin) {
        __threadfence();
        __syncthreads();
        if (t == 0) last = atomicAdd(&g_site_cnt, 1U) == gridDim.x - 1 ? 1U : 0U;
        __syncthreads();
        if (last == 0) return;
        __threadfence();
        for (unsigned q = 0; q < rows; ++q) {
            __syncthreads();
            hc_fin_row<true, true, true, true>(q, X, base, scale, mix, xc, norm, xn);
        }
        if (t == 0) atomicExch(&g_site_cnt, 0U);
    }
}


// Variant 2 (STRIX_GLM_R1B_HC=2): launches 2 + 3 fused. Every workgroup stores its block partials; the last workgroup
// to arrive (device-scope counter) does the trees, R6 and the collapse for all rows. Same arithmetic as variant 1.
__device__ unsigned g_hc_cnt;
__global__ __launch_bounds__(kPartThreads) void hc_part_fin_kernel(const float* X, const float* fn, float* F, const float* base,
                                                                   const float* scale, float* mix, float* xc, unsigned rows) {
    __shared__ float fs[kHcMix][kPad];
    __shared__ float xs[kMaxRows][kPad];
    __shared__ float zs[kMaxRows][kHcMix];
    __shared__ float ms[kMaxRows][kHcMix];
    __shared__ unsigned last;
    const unsigned j = blockIdx.x, t = threadIdx.x, i0 = j * kR1bBlock;
    for (unsigned idx = t; idx < kHcMix * kR1bBlock; idx += kPartThreads) {
        const unsigned o = idx / kR1bBlock, i = idx % kR1bBlock;
        fs[o][i] = fn[static_cast<std::size_t>(o) * kN + i0 + i];
    }
    for (unsigned idx = t; idx < rows * kR1bBlock; idx += kPartThreads) {
        const unsigned r = idx / kR1bBlock, i = idx % kR1bBlock;
        const std::size_t g = static_cast<std::size_t>(r) * kN + i0 + i;
        const float v = X[g] * g_rr[r];
        xs[r][i] = v;
        F[g] = v;
    }
    __syncthreads();
    const unsigned o = t % kHcMix, r = t / kHcMix;
    if (r < rows) {
        float acc = 0.0F;
        for (unsigned i = 0; i < kR1bBlock; ++i) acc = __builtin_fmaf(xs[r][i], fs[o][i], acc);
        __hip_atomic_store(&g_part[(static_cast<std::size_t>(r) * kHcMix + o) * kHcBlocks + j], acc, __ATOMIC_RELAXED,
                           __HIP_MEMORY_SCOPE_AGENT);
    }
    __threadfence();
    __syncthreads();
    if (t == 0) last = atomicAdd(&g_hc_cnt, 1U) == gridDim.x - 1 ? 1U : 0U;
    __syncthreads();
    if (last == 0) return;
    __threadfence();
    if (r < rows) {
        float p[kHcBlocks];
        const std::size_t base_i = (static_cast<std::size_t>(r) * kHcMix + o) * kHcBlocks;
        for (unsigned b = 0; b < kHcBlocks; ++b) p[b] = __hip_atomic_load(&g_part[base_i + b], __ATOMIC_RELAXED, __HIP_MEMORY_SCOPE_AGENT);
        for (unsigned hh = kHcBlocks / 2; hh >= 1; hh /= 2)
            for (unsigned i = 0; i < hh; ++i) p[i] = p[i] + p[i + hh];
        zs[r][o] = p[0];
    }
    __syncthreads();
    if (t < rows) {
        core::hc::mix_final_packed<HcR6>(zs[t], scale, base, kHcEps, ms[t]);
        for (unsigned k = 0; k < kHcMix; ++k) mix[static_cast<std::size_t>(t) * kHcMix + k] = ms[t][k];
    }
    __syncthreads();
    for (unsigned idx = t; idx < rows * kHidden; idx += kPartThreads) {
        const unsigned rr = idx / kHidden, d = idx % kHidden;
        const float* Xr = X + static_cast<std::size_t>(rr) * kN;
        const float* m = ms[rr];
        xc[static_cast<std::size_t>(rr) * kHidden + d] =
            core::hc::packed_dot(m, Xr, d, kHidden);
    }
    if (t == 0) atomicExch(&g_hc_cnt, 0U);
}

}  // namespace

void set_active(bool on) { g_active.store(on); }
bool active() { return g_active.load(); }

int dense_variant() {
    static const int v = std::getenv("STRIX_GLM_R1B_DENSE") != nullptr ? std::atoi(std::getenv("STRIX_GLM_R1B_DENSE")) : 7;
    return v;
}

void pack_k4096(const std::uint16_t* W, unsigned n, std::uint16_t* packed, hipStream_t s) {
    if (n == 0) throw std::invalid_argument("r1b pack_k4096: n >= 1");
    pack_k4096_kernel<<<static_cast<unsigned>((static_cast<std::size_t>(n) * 512 + 255) / 256), 256, 0, s>>>(W, n, packed);
    ok(hipGetLastError(), "pack_k4096");
}

void dense_f16_k4096(const float* x, unsigned ldx, const std::uint16_t* W, unsigned n, float* y, unsigned ldy, unsigned rows,
                     hipStream_t s, int variant) {
    rows_core::Blocked16Group group{x, {W}, {y}, {n}, ldx, {ldy}, 1, rows};
    dense_jobs(group, s, variant);
}

void dense_f16_k4096_x3(const float* x, unsigned ldx, const std::uint16_t* const W[3], const unsigned n[3], float* const y[3],
                        const unsigned ldy[3], unsigned rows, hipStream_t s, int variant) {
    rows_core::Blocked16Group group{x, {}, {}, {}, ldx, {}, 3, rows};
    for (unsigned q = 0; q < 3; ++q) {
        group.w[q] = W[q];
        group.y[q] = y[q];
        group.n[q] = n[q];
        group.ldy[q] = ldy[q];
    }
    dense_jobs(group, s, variant);
}

void router_dense(const float* x, const std::uint16_t* router, float* logits, unsigned rows, hipStream_t s, int variant) {
    dense_f16_k4096(x, kHidden, router, kExperts, logits, kExperts, rows, s, variant);
}

void hc_pre(const float* X, const float* fn, const float* base, const float* scale, float* F, float* mix, float* xc,
            unsigned rows, hipStream_t s, int variant) {
    if (rows == 0 || rows > kMaxRows) throw std::invalid_argument("r1b hc_pre: rows 1..8");
    hc_rr_kernel<<<rows, 256, 0, s>>>(X);
    ok(hipGetLastError(), "hc_rr");
    static const int env_variant = std::getenv("STRIX_GLM_R1B_HC") != nullptr ? std::atoi(std::getenv("STRIX_GLM_R1B_HC")) : 4;
    if (variant < 0) variant = env_variant;
    if (variant == 2) {
        hc_part_fin_kernel<<<kHcBlocks, kPartThreads, 0, s>>>(X, fn, F, base, scale, mix, xc, rows);
        ok(hipGetLastError(), "hc_part_fin");
        return;
    }
    if ((variant == 3 || variant == 4) && rows == 1) {
        hc_part_kernel<1><<<kHcBlocks, kPartThreads, 0, s>>>(X, fn, F, rows);
    } else {
        hc_part_kernel<kMaxRows><<<kHcBlocks, kPartThreads, 0, s>>>(X, fn, F, rows);
    }
    ok(hipGetLastError(), "hc_part");
    if (variant == 4) {
        hc_fin_kernel<true, true><<<rows, 256, 0, s>>>(X, base, scale, mix, xc);
    } else if (variant == 3) {
        hc_fin_kernel<true><<<rows, 256, 0, s>>>(X, base, scale, mix, xc);
    } else {
        hc_fin_kernel<false><<<rows, 256, 0, s>>>(X, base, scale, mix, xc);
    }
    ok(hipGetLastError(), "hc_fin");
}

bool hc_norm_enabled() {
    static const bool enabled = [] {
        const char* v = std::getenv("STRIX_GLM_HC_NORM");
        // The fused finish is hc_pre variant 4 (the default); any other STRIX_GLM_R1B_HC keeps its own unfused path.
        const char* hc = std::getenv("STRIX_GLM_R1B_HC");
        return v != nullptr && std::string(v) == "1" && (hc == nullptr || std::atoi(hc) == 4);
    }();
    return enabled;
}

int hc_site_mode() {
    // 1 (default): one launch per site at one row, three at 2..8 rows; 2: also two launches (rr + part, fin + norm) at
    // 2..8 rows (measured +0.7 % R8 ms/step: the per-workgroup rr recompute costs more than the saved launch); 0: three.
    static const int mode = [] {
        const char* v = std::getenv("STRIX_GLM_HC_SITE");
        return v != nullptr && *v != '\0' ? std::atoi(v) : 1;
    }();
    return mode;
}

void hc_pre_norm(const float* X, const float* fn, const float* base, const float* scale, float* F, float* mix, float* xc,
                 const std::uint16_t* norm, float* xn, unsigned rows, hipStream_t s, int site) {
    if (rows == 0 || rows > kMaxRows) throw std::invalid_argument("r1b hc_pre_norm: rows 1..8");
    if (site < 0) site = hc_site_mode();
    if (site == 2 || (site == 1 && rows == 1)) {
        if (rows == 1) {
            hc_site_kernel<1, true><<<kHcBlocks, 256, 0, s>>>(X, fn, F, base, scale, mix, xc, norm, xn, rows);
            ok(hipGetLastError(), "hc_site");
            return;
        }
        hc_site_kernel<kMaxRows, false><<<kHcBlocks, 256, 0, s>>>(X, fn, F, base, scale, mix, xc, norm, xn, rows);
        ok(hipGetLastError(), "hc_site_part");
        hc_fin_kernel<true, true, true><<<rows, 256, 0, s>>>(X, base, scale, mix, xc, norm, xn);
        ok(hipGetLastError(), "hc_site_fin");
        return;
    }
    hc_rr_kernel<<<rows, 256, 0, s>>>(X);
    ok(hipGetLastError(), "hc_norm_rr");
    if (rows == 1) hc_part_kernel<1><<<kHcBlocks, kPartThreads, 0, s>>>(X, fn, F, rows);
    else hc_part_kernel<kMaxRows><<<kHcBlocks, kPartThreads, 0, s>>>(X, fn, F, rows);
    ok(hipGetLastError(), "hc_norm_part");
    hc_fin_kernel<true, true, true><<<rows, 256, 0, s>>>(X, base, scale, mix, xc, norm, xn);
    ok(hipGetLastError(), "hc_norm_fin");
}

}  // namespace strix::glm::r1b
