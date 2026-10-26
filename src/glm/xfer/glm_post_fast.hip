// Exact faster forms of the GLM post-exchange consumers (see glm_post_fast.hpp). Build with -ffp-contract=off
// -fno-fast-math -fno-gpu-flush-denormals-to-zero, as the Tier A kernels.
#include "glm_post_fast.hpp"

#include "glm/common/glm_math.hpp"

#include <stdexcept>
#include <type_traits>
#include <utility>
#include <string>

namespace strix::glm::xfer {

namespace {

void ok(hipError_t e, const char* what) {
    if (e != hipSuccess) throw std::runtime_error(std::string(what) + ": " + hipGetErrorString(e));
}

constexpr unsigned kN = kHc * kHidden;  // 16384
constexpr unsigned kT = 256;            // tile width (values per row per tile)

// R2 tree in LDS over 256 lanes (as Tier A lds_tree<256>).
__device__ __forceinline__ float tree256(float* s, float part) {
    const unsigned l = threadIdx.x;
    s[l] = part;
    __syncthreads();
    for (unsigned h = 128; h > 0; h >>= 1U) {
        if (l < h) s[l] = s[l] + s[l + h];
        __syncthreads();
    }
    const float r = s[0];
    __syncthreads();
    return r;
}

// R6 on wave 0 from z[24] in LDS into m[24] (hc_pre_v2's form: same op sequence per value as Tier A's lane 0).
__device__ __forceinline__ void r6_wave(const float* z, float* m, const float* base, const float* scale) {
    const unsigned l = threadIdx.x;
    if (l >= 32) return;
    const unsigned t = l < 16 ? l : 0, j = t / kHc, k = t % kHc;
    if (l >= 16 && l < 20) m[l - 16] = sigmoid(z[l - 16] * scale[0] + base[l - 16]) + kHcEps;
    if (l >= 20 && l < 24) m[kHc + l - 20] = 2.0F * sigmoid(z[kHc + l - 20] * scale[1] + base[kHc + l - 20]);
    float c = z[2 * kHc + t] * scale[2] + base[2 * kHc + t];
    {
        float e[kHc], cr[kHc];
        #pragma unroll
        for (unsigned q = 0; q < kHc; ++q) cr[q] = __shfl(c, static_cast<int>(j * kHc + q));
        float mx = cr[0];
        for (unsigned q = 1; q < kHc; ++q) mx = cr[q] > mx ? cr[q] : mx;
        #pragma unroll
        for (unsigned q = 0; q < kHc; ++q) e[q] = declared_exp(cr[q] - mx);
        const float sum = ((e[0] + e[1]) + e[2]) + e[3];
        const float ek = k == 0 ? e[0] : k == 1 ? e[1] : k == 2 ? e[2] : e[3];  // constant indices only (no scratch)
        c = ek / sum + kHcEps;
    }
    for (unsigned it = 0; it < kHcIters; ++it) {
        if (it > 0) {
            const float sr = (((__shfl(c, static_cast<int>(j * kHc + 0)) + __shfl(c, static_cast<int>(j * kHc + 1))) +
                               __shfl(c, static_cast<int>(j * kHc + 2))) +
                              __shfl(c, static_cast<int>(j * kHc + 3))) +
                             kHcEps;
            c = c / sr;
        }
        const float sc = (((__shfl(c, static_cast<int>(0 * kHc + k)) + __shfl(c, static_cast<int>(1 * kHc + k))) +
                           __shfl(c, static_cast<int>(2 * kHc + k))) +
                          __shfl(c, static_cast<int>(3 * kHc + k))) +
                         kHcEps;
        c = c / sc;
    }
    if (l < 16) m[2 * kHc + l] = c;
}

// 16 chain steps from two blocks of 4 float4 (ascending element order).
__device__ __forceinline__ float fma16(const float4* a, const float4* w, float acc) {
    #pragma unroll
    for (unsigned q = 0; q < 4; ++q) {
        acc = __builtin_fmaf(a[q].x, w[q].x, acc);
        acc = __builtin_fmaf(a[q].y, w[q].y, acc);
        acc = __builtin_fmaf(a[q].z, w[q].z, acc);
        acc = __builtin_fmaf(a[q].w, w[q].w, acc);
    }
    return acc;
}

// NOUT R1 chains of length K over LDS tiles: acc_j = chain over i ascending of fmaf(x[i], W_j[i], acc). Tiles of TW
// values in NST stages; waves 1..7 load the tile NST - 1 ahead (float4, or 8 x F16 converted exactly); lanes 0..NOUT-1
// of wave 0 read each tile in register batches of 16 values with the next batch prefetched, so the dependent FMA chain
// never waits for one load per step. Every thread of the block calls it (barriers inside). Rows j >= rows_valid are
// not loaded (their lanes compute nothing that is stored).
template <unsigned NOUT, unsigned TW, unsigned NST, bool F16W>
__device__ __forceinline__ float lds_chains(const float* x, const void* W, std::size_t ldw, unsigned rows_valid, unsigned K,
                                            float (*wt)[NOUT][TW + 4], float (*xt)[TW]) {
    const unsigned l = threadIdx.x;
    const unsigned tiles = K / TW;
    constexpr unsigned kV = F16W ? 8 : 4;  // weight values per load item
    auto load = [&](unsigned t, unsigned buf) {
        constexpr unsigned kWItems = NOUT * (TW / kV), kItems = kWItems + TW / 4;
        for (unsigned idx = l - 32; idx < kItems; idx += 224) {
            if (idx < kWItems) {
                const unsigned j = idx / (TW / kV), q = idx % (TW / kV);
                if (j >= rows_valid) continue;
                float* dst = &wt[buf][j][q * kV];
                if constexpr (F16W) {
                    const auto* src = static_cast<const std::uint16_t*>(W) + static_cast<std::size_t>(j) * ldw + t * TW + q * 8;
                    const uint4 u = *reinterpret_cast<const uint4*>(src);
                    const std::uint32_t wv[4] = {u.x, u.y, u.z, u.w};
                    #pragma unroll
                    for (unsigned h = 0; h < 4; ++h) {
                        dst[2 * h] = f16_to_float(static_cast<std::uint16_t>(wv[h] & 0xFFFFU));
                        dst[2 * h + 1] = f16_to_float(static_cast<std::uint16_t>(wv[h] >> 16U));
                    }
                } else {
                    const auto* src = static_cast<const float*>(W) + static_cast<std::size_t>(j) * ldw + t * TW + q * 4;
                    *reinterpret_cast<float4*>(dst) = *reinterpret_cast<const float4*>(src);
                }
            } else {
                const unsigned q = idx - kWItems;
                *reinterpret_cast<float4*>(&xt[buf][q * 4]) = *reinterpret_cast<const float4*>(x + t * TW + q * 4);
            }
        }
    };
    if (l >= 32)
        for (unsigned t = 0; t + 1 < NST && t < tiles; ++t) load(t, t);
    __syncthreads();
    float acc = 0.0F;
    for (unsigned t = 0; t < tiles; ++t) {
        const unsigned b = t % NST;
        if (l >= 32 && t + NST - 1 < tiles) load(t + NST - 1, (t + NST - 1) % NST);
        if (l < NOUT) {
            const auto* w4 = reinterpret_cast<const float4*>(wt[b][l]);
            const auto* x4 = reinterpret_cast<const float4*>(xt[b]);
            float4 wa[4], xa[4], wb[4], xb[4];
            #pragma unroll
            for (unsigned q = 0; q < 4; ++q) {
                wa[q] = w4[q];
                xa[q] = x4[q];
            }
            for (unsigned k = 0; k < TW / 4; k += 8) {  // two batches of 16 values per step
                #pragma unroll
                for (unsigned q = 0; q < 4; ++q) {
                    wb[q] = w4[k + 4 + q];
                    xb[q] = x4[k + 4 + q];
                }
                acc = fma16(xa, wa, acc);
                if (k + 8 < TW / 4)
                    #pragma unroll
                    for (unsigned q = 0; q < 4; ++q) {
                        wa[q] = w4[k + 8 + q];
                        xa[q] = x4[k + 8 + q];
                    }
                acc = fma16(xb, wb, acc);
            }
        }
        __syncthreads();
    }
    return acc;
}

template <int V>
__global__ __launch_bounds__(256) void hc_pre_v3_kernel(const float* X, const float* fn, const float* base, const float* scale,
                                                        float* F, float* mix, float* xc) {
    __shared__ float s[256];
    __shared__ float z[kHcMix];
    __shared__ float m[kHcMix];
    __shared__ float wt[V == 1 ? 2 : 1][V == 1 ? kHcMix : 1][V == 1 ? kT + 1 : 1];
    __shared__ float xt[V == 1 ? 2 : 1][V == 1 ? kT : 1];
    __shared__ __attribute__((aligned(16))) float wt4[V == 2 ? 4 : 1][V == 2 ? kHcMix : 1][V == 2 ? 128 + 4 : 4];
    __shared__ __attribute__((aligned(16))) float xt4[V == 2 ? 4 : 1][V == 2 ? 128 : 4];
    const unsigned r = blockIdx.x, l = threadIdx.x;
    const float* Xr = X + static_cast<std::size_t>(r) * kN;
    float* Fr = F + static_cast<std::size_t>(r) * kN;
    float a = 0.0F;
    for (unsigned i = l; i < kN; i += 256) a = __builtin_fmaf(Xr[i], Xr[i], a);
    const float ss = tree256(s, a);
    const float rr = inv_sqrt(ss / static_cast<float>(kN) + kEps);
    for (unsigned i = l; i < kN; i += 256) Fr[i] = Xr[i] * rr;
    __syncthreads();
    if constexpr (V == 0) {
        if (l < kHcMix) {
            const auto* f4 = reinterpret_cast<const float4*>(Fr);
            const auto* w4 = reinterpret_cast<const float4*>(fn + static_cast<std::size_t>(l) * kN);
            float4 fa[4], wa[4], fb[4], wb[4];
            #pragma unroll
            for (unsigned q = 0; q < 4; ++q) {
                fa[q] = f4[q];
                wa[q] = w4[q];
            }
            float acc = 0.0F;
            constexpr unsigned kBlocks = kN / 16;
            for (unsigned b = 0; b < kBlocks; b += 2) {
                #pragma unroll
                for (unsigned q = 0; q < 4; ++q) {  // prefetch block b + 1 (always exists: kBlocks is even)
                    fb[q] = f4[(b + 1) * 4 + q];
                    wb[q] = w4[(b + 1) * 4 + q];
                }
                acc = fma16(fa, wa, acc);
                if (b + 2 < kBlocks)
                    #pragma unroll
                    for (unsigned q = 0; q < 4; ++q) {
                        fa[q] = f4[(b + 2) * 4 + q];
                        wa[q] = w4[(b + 2) * 4 + q];
                    }
                acc = fma16(fb, wb, acc);
            }
            z[l] = acc;
        }
    } else if constexpr (V == 2) {
        const float acc = lds_chains<kHcMix, 128, 4, false>(Fr, fn, kN, kHcMix, kN, wt4, xt4);
        if (l < kHcMix) z[l] = acc;
    } else {
        constexpr unsigned kTiles = kN / kT;
        auto load = [&](unsigned t, unsigned buf) {  // waves 1..7
            for (unsigned idx = l - 32; idx < (kHcMix + 1) * kT; idx += 224) {
                const unsigned j = idx / kT, k = idx % kT;
                if (j < kHcMix) wt[buf][j][k] = fn[static_cast<std::size_t>(j) * kN + t * kT + k];
                else xt[buf][k] = Fr[t * kT + k];
            }
        };
        if (l >= 32) load(0, 0);
        __syncthreads();
        float acc = 0.0F;
        for (unsigned t = 0; t < kTiles; ++t) {
            const unsigned b = t & 1U;
            if (l >= 32 && t + 1 < kTiles) load(t + 1, b ^ 1U);
            if (l < kHcMix)
                for (unsigned k = 0; k < kT; ++k) acc = __builtin_fmaf(xt[b][k], wt[b][l][k], acc);
            __syncthreads();
        }
        if (l < kHcMix) z[l] = acc;
    }
    __syncthreads();
    r6_wave(z, m, base, scale);
    __syncthreads();
    if (l < kHcMix) mix[static_cast<std::size_t>(r) * kHcMix + l] = m[l];
    float* xr = xc + static_cast<std::size_t>(r) * kHidden;
    for (unsigned d = l; d < kHidden; d += 256)
        xr[d] = ((m[0] * Xr[d] + m[1] * Xr[kHidden + d]) + m[2] * Xr[2 * kHidden + d]) + m[3] * Xr[3 * kHidden + d];
}

// R1 router: block (group of 32 outputs, row); LDS tiles of 32 rows x 256 inputs (F16 -> float, exact) + the x tile,
// loaded by waves 1..7; lanes 0..31 of wave 0 run the 32 chains.
constexpr unsigned kTR = 128;  // router tile width: 2 x 32 x 129 floats + x tiles = 34 KB of LDS
__global__ __launch_bounds__(256) void router_dense_kernel(const float* x, const std::uint16_t* W, float* y) {
    __shared__ float wt[2][32][kTR + 1];
    __shared__ float xt[2][kTR];
    const unsigned g = blockIdx.x, r = blockIdx.y, l = threadIdx.x;
    const unsigned o0 = g * 32, nout = kExperts - o0 < 32 ? kExperts - o0 : 32;
    const float* xr = x + static_cast<std::size_t>(r) * kHidden;
    constexpr unsigned kTiles = kHidden / kTR;
    auto load = [&](unsigned t, unsigned buf) {
        for (unsigned idx = l - 32; idx < (32 + 1) * kTR; idx += 224) {
            const unsigned j = idx / kTR, k = idx % kTR;
            if (j < 32) {
                if (j < nout) wt[buf][j][k] = f16_to_float(W[static_cast<std::size_t>(o0 + j) * kHidden + t * kTR + k]);
            } else {
                xt[buf][k] = xr[t * kTR + k];
            }
        }
    };
    if (l >= 32) load(0, 0);
    __syncthreads();
    float acc = 0.0F;
    for (unsigned t = 0; t < kTiles; ++t) {
        const unsigned b = t & 1U;
        if (l >= 32 && t + 1 < kTiles) load(t + 1, b ^ 1U);
        if (l < nout)
            for (unsigned k = 0; k < kTR; ++k) acc = __builtin_fmaf(xt[b][k], wt[b][l][k], acc);
        __syncthreads();
    }
    if (l < nout) y[static_cast<std::size_t>(r) * kExperts + o0 + l] = acc;
}

// R1 router, pipelined: block (group of 32 outputs, row), 3-stage LDS tiles of 32 x 128 (F16 -> float, exact).
__global__ __launch_bounds__(256) void router_dense_v2_kernel(const float* x, const std::uint16_t* W, float* y) {
    __shared__ __attribute__((aligned(16))) float wt[3][32][128 + 4];
    __shared__ __attribute__((aligned(16))) float xt[3][128];
    const unsigned g = blockIdx.x, r = blockIdx.y, l = threadIdx.x;
    const unsigned o0 = g * 32, nout = kExperts - o0 < 32 ? kExperts - o0 : 32;
    const float acc = lds_chains<32, 128, 3, true>(x + static_cast<std::size_t>(r) * kHidden, W + static_cast<std::size_t>(o0) * kHidden,
                                                   kHidden, nout, kHidden, wt, xt);
    if (l < nout) y[static_cast<std::size_t>(r) * kExperts + o0 + l] = acc;
}

// f(std::integral_constant<unsigned, i>{}) for i = 0..N-1, unrolled at compile time (register-array indices stay constant).
template <unsigned N, class Fn> __device__ __forceinline__ void static_for(Fn&& fn) {
    [&]<unsigned... I>(std::integer_sequence<unsigned, I...>) { (fn(std::integral_constant<unsigned, I>{}), ...); }(std::make_integer_sequence<unsigned, N>{});
}

// Workgroup barrier for LDS traffic only. A plain __syncthreads is a workgroup release fence, i.e. s_waitcnt vmcnt(0)
// in WGP mode: it waits for every global load in flight and so drains lds_chains_deep's register ring at every tile
// (measurement: ~0.9 us per hc_pre tile = one DRAM round trip). These fences cover only the LDS address space (lgkmcnt). The
// ring's loads must be global_* (vmcnt only), not flat_* (which also count in lgkmcnt); checked in the ISA.
__device__ __forceinline__ void lds_barrier() {
    __builtin_amdgcn_fence(__ATOMIC_RELEASE, "workgroup", "local");
    __builtin_amdgcn_s_barrier();
    __builtin_amdgcn_fence(__ATOMIC_ACQUIRE, "workgroup", "local");
}

// Deep form of lds_chains for 1024-thread blocks: waves 1..31 are loaders, each owning at most ONE load item (a float4,
// or 8 F16) of every tile, and keeping D tiles of its item in flight in registers (slot = tile % D, unrolled so the
// slot index is a compile-time constant); at iteration t a loader stores tile t+NST-1 (issued D iterations earlier) into
// its LDS stage and issues tile t+NST-1+D. Lanes 0..NOUT-1 of wave 0 consume exactly as lds_chains. tiles % D == 0.
// LB: tile barriers are lds_barrier() (the ring stays in flight) instead of __syncthreads (v5). Same arithmetic.
// PF > 0 (design notes): the consumer keeps a ring of RG register batches (16 values of its w row + 16 of x) and issues
// the LDS loads of batch g+PF before computing batch g; sched_barriers pin that order (otherwise the scheduler moves
// every ds_load next to its use and the chain waits one LDS round trip per 4 steps). XB: the prefetch runs on into
// tile t+1 (stage (t+1)%NST, stored in iteration t-1 and published by barrier t-1; needs NST >= 3; its next overwrite
// stores tile t+4 in iteration t+2, after the consumer passed barrier t+1). On the last tile the XB loads read the
// in-bounds, initialized (stale) next stage and their registers are never used. The LDS-only fence before each
// barrier drains pending LDS reads (their values are carried across it in registers). Same values, same ascending fmaf
// order per chain.
template <unsigned NOUT, unsigned TW, unsigned NST, unsigned D, bool F16W, unsigned NT, bool LB = false, unsigned PF = 0,
          unsigned RG = 2, bool XB = false>
__device__ __forceinline__ float lds_chains_deep(const float* x, const void* W, std::size_t ldw, unsigned rows_valid, unsigned K,
                                                 float (*wt)[NOUT][TW + 4], float (*xt)[TW]) {
    static_assert(D >= NST, "D must cover the LDS stages");
    static_assert(PF == 0 || (RG > PF && (TW / 16) % RG == 0 && (D * (TW / 16)) % RG == 0), "consumer ring");
    static_assert(!XB || (PF > 0 && NST >= 3 && PF <= TW / 16), "cross-barrier prefetch needs a published next stage");
    const unsigned l = threadIdx.x;
    const unsigned tiles = K / TW;
    constexpr unsigned kV = F16W ? 8 : 4;
    constexpr unsigned kWItems = NOUT * (TW / kV), kItems = kWItems + TW / 4;
    static_assert(kItems <= NT - 32, "one item per loader thread");
    static_assert((TW / 4) % 8 == 0, "two 16-value batches per step");
    const unsigned item = l - 32;
    const bool loader = l >= 32 && item < kItems;
    const bool witem = loader && item < kWItems;
    const unsigned j = witem ? item / (TW / kV) : 0, q = witem ? item % (TW / kV) : item - kWItems;
    const bool active = loader && (!witem || j < rows_valid);
    uint4 reg[D];
    // LB: every lane issues every tile, branch-free (tile index clamped, non-loaders and invalid rows read a valid
    // address that is never stored), so the compiler's in-order vmcnt bookkeeping sees straight-line loads and waits
    // only for the oldest one before each store (a conditional issue makes it drain with vmcnt(0)).
    constexpr std::size_t kEsz = F16W ? 2 : 4;
    const unsigned jl = loader && witem ? (j < rows_valid ? j : rows_valid - 1) : 0, ql = loader ? q : 0;
    const char* const src = loader && !witem ? reinterpret_cast<const char*>(x) + std::size_t{ql} * 16
                                             : static_cast<const char*>(W) + (std::size_t{jl} * ldw) * kEsz + std::size_t{ql} * 16;
    const std::size_t step = loader && !witem ? std::size_t{TW} * 4 : std::size_t{TW} * kEsz;
    auto issue = [&](unsigned t, uint4& r) {
        if constexpr (LB) {
            const unsigned tt = t < tiles ? t : tiles - 1;
            r = *reinterpret_cast<const uint4*>(src + tt * step);
        } else {
            if (!active || t >= tiles) return;
            if (witem) {
                if constexpr (F16W) r = *reinterpret_cast<const uint4*>(static_cast<const std::uint16_t*>(W) + static_cast<std::size_t>(j) * ldw + t * TW + q * 8);
                else r = *reinterpret_cast<const uint4*>(static_cast<const float*>(W) + static_cast<std::size_t>(j) * ldw + t * TW + q * 4);
            } else {
                r = *reinterpret_cast<const uint4*>(x + t * TW + q * 4);
            }
        }
    };
    auto store = [&](unsigned t, const uint4& r) {
        if (!active || t >= tiles) return;
        const unsigned buf = t % NST;
        if (witem) {
            float* dst = &wt[buf][j][q * kV];
            if constexpr (F16W) {
                const std::uint32_t wv[4] = {r.x, r.y, r.z, r.w};
                #pragma unroll
                for (unsigned h = 0; h < 4; ++h) {
                    dst[2 * h] = f16_to_float(static_cast<std::uint16_t>(wv[h] & 0xFFFFU));
                    dst[2 * h + 1] = f16_to_float(static_cast<std::uint16_t>(wv[h] >> 16U));
                }
            } else {
                const std::uint32_t wv[4] = {r.x, r.y, r.z, r.w};
                for (unsigned h = 0; h < 4; ++h) dst[h] = __builtin_bit_cast(float, wv[h]);
            }
        } else {
            const std::uint32_t xv[4] = {r.x, r.y, r.z, r.w};
            float* xd = &xt[buf][q * 4];
            for (unsigned h = 0; h < 4; ++h) xd[h] = __builtin_bit_cast(float, xv[h]);
        }
    };
    // prologue: tiles 0..D-1 in flight, tiles 0..NST-2 stored, their slots refilled with tiles D..D+NST-2
    static_for<D>([&](auto u) { issue(u.value, reg[u.value]); });
    static_for<NST - 1>([&](auto u) {
        store(u.value, reg[u.value]);
        issue(u.value + D, reg[u.value]);
    });
    if constexpr (LB) lds_barrier();
    else __syncthreads();
    float acc = 0.0F;
    constexpr unsigned kNB = TW / 16;  // 16-value batches per tile
    float4 rw[PF > 0 ? RG : 1][4], rx[PF > 0 ? RG : 1][4];
    auto ld = [&](unsigned stage, unsigned k, float4 (&w)[4], float4 (&xv)[4]) {
        const auto* w4 = reinterpret_cast<const float4*>(wt[stage][l]);
        const auto* x4 = reinterpret_cast<const float4*>(xt[stage]);
        #pragma unroll
        for (unsigned e = 0; e < 4; ++e) {
            w[e] = w4[4 * k + e];
            xv[e] = x4[4 * k + e];
        }
    };
    if constexpr (PF > 0 && XB) {  // the ring starts once, on tile 0 (stage 0, published by the prologue barrier)
        if (l < NOUT) static_for<PF>([&](auto pc) { ld(0, pc.value, rw[pc.value % RG], rx[pc.value % RG]); });
    }
    for (unsigned t0 = 0; t0 < tiles; t0 += D) {
        static_for<D>([&](auto jc) {  // compile-time jj: the ring slot (jj + NST - 1) % D is a constant
            constexpr unsigned jj = decltype(jc)::value, kOff = NST - 1, slot = (jj + kOff) % D;
            const unsigned t = t0 + jj;
            store(t + kOff, reg[slot]);
            issue(t + kOff + D, reg[slot]);
            const unsigned b = t % NST;
            if constexpr (PF > 0) {
                if (l < NOUT) {
                    if constexpr (!XB) static_for<PF>([&](auto pc) { ld(b, pc.value, rw[pc.value % RG], rx[pc.value % RG]); });
                    const unsigned bn = (t + 1) % NST;
                    static_for<kNB>([&](auto kc) {
                        constexpr unsigned k = decltype(kc)::value, g = (XB ? jj * kNB : 0) + k, cs = g % RG, ns = (g + PF) % RG;
                        if constexpr (k + PF < kNB) ld(b, k + PF, rw[ns], rx[ns]);
                        else if constexpr (XB) ld(bn, k + PF - kNB, rw[ns], rx[ns]);
                        __builtin_amdgcn_sched_barrier(0);
                        acc = fma16(rx[cs], rw[cs], acc);
                        __builtin_amdgcn_sched_barrier(0);
                    });
                }
            } else if (l < NOUT) {
                const auto* w4 = reinterpret_cast<const float4*>(wt[b][l]);
                const auto* x4 = reinterpret_cast<const float4*>(xt[b]);
                float4 wa[4], xa[4], wb[4], xb[4];
                #pragma unroll
                for (unsigned e = 0; e < 4; ++e) {
                    wa[e] = w4[e];
                    xa[e] = x4[e];
                }
                for (unsigned k = 0; k < TW / 4; k += 8) {
                    #pragma unroll
                    for (unsigned e = 0; e < 4; ++e) {
                        wb[e] = w4[k + 4 + e];
                        xb[e] = x4[k + 4 + e];
                    }
                    acc = fma16(xa, wa, acc);
                    if (k + 8 < TW / 4)
                        #pragma unroll
                        for (unsigned e = 0; e < 4; ++e) {
                            wa[e] = w4[k + 8 + e];
                            xa[e] = x4[k + 8 + e];
                        }
                    acc = fma16(xb, wb, acc);
                }
            }
            if constexpr (LB) lds_barrier();
            else __syncthreads();
        });
    }
    return acc;
}

// hc_pre with 1024 threads (the router's lds_chains_deep shape: 3 stages of 24 x 128, 8 tiles in flight): R2/R3 exactly as Tier A on lanes 0..255 (the contract's 256 partials and tree), F written
// by all threads, the 24 chains through lds_chains_deep, R6 on wave 0, the collapse. LB: variant 4 (LDS-only tile barriers).
template <bool LB, unsigned PF = 0, unsigned RG = 2, bool XB = false>
__global__ __launch_bounds__(1024) void hc_pre_v5_kernel(const float* X, const float* fn, const float* base, const float* scale, float* F,
                                                         float* mix, float* xc) {
    __shared__ float s[256];
    __shared__ float z[kHcMix];
    __shared__ float m[kHcMix];
    __shared__ __attribute__((aligned(16))) float wt[3][kHcMix][128 + 4];
    __shared__ __attribute__((aligned(16))) float xt[3][128];
    const unsigned r = blockIdx.x, l = threadIdx.x;
    const float* Xr = X + static_cast<std::size_t>(r) * kN;
    float* Fr = F + static_cast<std::size_t>(r) * kN;
    float a = 0.0F;
    if (l < 256)
        for (unsigned i = l; i < kN; i += 256) a = __builtin_fmaf(Xr[i], Xr[i], a);
    if (l < 256) s[l] = a;
    __syncthreads();
    for (unsigned h = 128; h > 0; h >>= 1U) {
        if (l < h) s[l] = s[l] + s[l + h];
        __syncthreads();
    }
    const float rr = inv_sqrt(s[0] / static_cast<float>(kN) + kEps);
    for (unsigned i = l; i < kN; i += 1024) Fr[i] = Xr[i] * rr;
    __syncthreads();
    const float acc = lds_chains_deep<kHcMix, 128, 3, 8, false, 1024, LB, PF, RG, XB>(Fr, fn, kN, kHcMix, kN, wt, xt);
    if (l < kHcMix) z[l] = acc;
    __syncthreads();
    r6_wave(z, m, base, scale);
    __syncthreads();
    if (l < kHcMix) mix[static_cast<std::size_t>(r) * kHcMix + l] = m[l];
    float* xr = xc + static_cast<std::size_t>(r) * kHidden;
    for (unsigned d = l; d < kHidden; d += 1024)
        xr[d] = ((m[0] * Xr[d] + m[1] * Xr[kHidden + d]) + m[2] * Xr[2 * kHidden + d]) + m[3] * Xr[3 * kHidden + d];
}

__global__ __launch_bounds__(1024) void router_dense_v3_kernel(const float* x, const std::uint16_t* W, float* y) {
    __shared__ __attribute__((aligned(16))) float wt[3][32][128 + 4];
    __shared__ __attribute__((aligned(16))) float xt[3][128];
    const unsigned g = blockIdx.x, r = blockIdx.y, l = threadIdx.x;
    const unsigned o0 = g * 32, nout = kExperts - o0 < 32 ? kExperts - o0 : 32;
    const float acc = lds_chains_deep<32, 128, 3, 8, true, 1024>(x + static_cast<std::size_t>(r) * kHidden, W + static_cast<std::size_t>(o0) * kHidden,
                                                           kHidden, nout, kHidden, wt, xt);
    if (l < nout) y[static_cast<std::size_t>(r) * kExperts + o0 + l] = acc;
}

// Variant 3: 8 outputs per 256-thread block (36 blocks per row instead of 9, so more of the GPU streams the 2.36 MB),
// LDS-only tile barriers. 160 loader lanes (8 x 16 F16 items + 32 x items), lanes 0..7 run the chains.
__global__ __launch_bounds__(256) void router_dense_v4_kernel(const float* x, const std::uint16_t* W, float* y) {
    __shared__ __attribute__((aligned(16))) float wt[3][8][128 + 4];
    __shared__ __attribute__((aligned(16))) float xt[3][128];
    const unsigned g = blockIdx.x, r = blockIdx.y, l = threadIdx.x;
    const unsigned o0 = g * 8;
    static_assert(kExperts % 8 == 0, "36 full groups");
    const float acc = lds_chains_deep<8, 128, 3, 8, true, 256, true>(x + static_cast<std::size_t>(r) * kHidden, W + static_cast<std::size_t>(o0) * kHidden,
                                                                     kHidden, 8, kHidden, wt, xt);
    if (l < 8) y[static_cast<std::size_t>(r) * kExperts + o0 + l] = acc;
}

// Diagnostic (timing only, design notes): the 24 consumer chains of hc_pre over 128 tiles that all read ONE LDS stage
// filled once (no loaders, no barriers), with the consumer code of lds_chains_deep: PF = 0 the plain batch loop, PF > 0
// the pinned ring. Its time is the LDS-fed chain rate of the consumer itself.
template <unsigned PF, unsigned RG>
__global__ __launch_bounds__(64) void hc_chains_lds_only_kernel(const float* F, const float* fn, float* z) {
    __shared__ __attribute__((aligned(16))) float wt[kHcMix][128 + 4];
    __shared__ __attribute__((aligned(16))) float xt[128];
    const unsigned l = threadIdx.x;
    for (unsigned i = l; i < kHcMix * 128; i += 64) wt[i / 128][i % 128] = fn[static_cast<std::size_t>(i / 128) * kN + i % 128];
    for (unsigned i = l; i < 128; i += 64) xt[i] = F[i];
    __syncthreads();
    if (l >= kHcMix) return;
    const auto* w4 = reinterpret_cast<const float4*>(wt[l]);
    const auto* x4 = reinterpret_cast<const float4*>(xt);
    float acc = 0.0F;
    if constexpr (PF == 0) {
        for (unsigned t = 0; t < kN / 128; ++t)
            for (unsigned k = 0; k < 8; ++k) {
                float4 wa[4], xa[4];
                #pragma unroll
                for (unsigned e = 0; e < 4; ++e) {
                    wa[e] = w4[4 * k + e];
                    xa[e] = x4[4 * k + e];
                }
                acc = fma16(xa, wa, acc);
            }
    } else {
        float4 rw[RG][4], rx[RG][4];
        auto ld = [&](unsigned k, float4 (&w)[4], float4 (&xv)[4]) {
            #pragma unroll
            for (unsigned e = 0; e < 4; ++e) {
                w[e] = w4[4 * k + e];
                xv[e] = x4[4 * k + e];
            }
        };
        static_for<PF>([&](auto pc) { ld(pc.value, rw[pc.value % RG], rx[pc.value % RG]); });
        for (unsigned t = 0; t < kN / 128; ++t)
            static_for<8>([&](auto kc) {
                constexpr unsigned k = decltype(kc)::value, cs = k % RG, ns = (k + PF) % RG;
                ld((k + PF) % 8, rw[ns], rx[ns]);
                __builtin_amdgcn_sched_barrier(0);
                acc = fma16(rx[cs], rw[cs], acc);
                __builtin_amdgcn_sched_barrier(0);
            });
    }
    z[l] = acc;
}

// Parallel exact routing: one block of 288 lanes (+ padding to 320) per row.
__global__ __launch_bounds__(320) void route_v2_kernel(const float* logits, const float* bias, std::uint32_t* ids, float* w) {
    __shared__ float sg[kExperts];
    __shared__ std::uint32_t key[kExperts];
    __shared__ std::uint32_t pick[kTopE];
    __shared__ std::uint32_t rk[kExperts];
    const unsigned r = blockIdx.x, e = threadIdx.x;
    const float* lg = logits + static_cast<std::size_t>(r) * kExperts;
    if (e < kExperts) {
        sg[e] = sigmoid(lg[e]);
        key[e] = ordered_key(sg[e] + bias[e]);
    }
    __syncthreads();
    if (e < kExperts) {
        // rank under the total order (key desc, index asc): the Tier A scan picks exactly the 8 smallest ranks
        unsigned rank = 0;
        const std::uint32_t ke = key[e];
        for (unsigned f = 0; f < kExperts; ++f) {
            const std::uint32_t kf = key[f];
            rank += (kf > ke || (kf == ke && f < e)) ? 1U : 0U;
        }
        rk[e] = rank;
    }
    __syncthreads();
    if (e < kExperts && rk[e] < kTopE) {
        unsigned before = 0;  // position among the selected, ascending id
        for (unsigned f = 0; f < e; ++f) before += rk[f] < kTopE ? 1U : 0U;
        pick[before] = e;
    }
    __syncthreads();
    if (e == 0) {  // R14 on lane 0, ascending id, the Tier A expression order
        float den = 0.0F;
        for (unsigned k = 0; k < kTopE; ++k) den = den + sg[pick[k]];
        den = den + kRouteDenEps;
        for (unsigned k = 0; k < kTopE; ++k) {
            ids[static_cast<std::size_t>(r) * kTopE + k] = pick[k];
            w[static_cast<std::size_t>(r) * kTopE + k] = (sg[pick[k]] / den) * kRouteScale;
        }
    }
}

}  // namespace

void hc_pre_v3(const float* X, const float* fn, const float* base, const float* scale, float* F, float* mix, float* xc,
               unsigned rows, hipStream_t s, int variant) {
    if (variant == 0) hc_pre_v3_kernel<0><<<rows, 256, 0, s>>>(X, fn, base, scale, F, mix, xc);
    else if (variant == 1) hc_pre_v3_kernel<1><<<rows, 256, 0, s>>>(X, fn, base, scale, F, mix, xc);
    else if (variant == 2) hc_pre_v3_kernel<2><<<rows, 256, 0, s>>>(X, fn, base, scale, F, mix, xc);
    else if (variant == 3) hc_pre_v5_kernel<false><<<rows, 1024, 0, s>>>(X, fn, base, scale, F, mix, xc);
    else if (variant == 4) hc_pre_v5_kernel<true><<<rows, 1024, 0, s>>>(X, fn, base, scale, F, mix, xc);
    // design notes: pinned consumer prefetch (PF batches, ring RG), XB = across the tile barrier
    else if (variant == 5) hc_pre_v5_kernel<true, 1, 2, false><<<rows, 1024, 0, s>>>(X, fn, base, scale, F, mix, xc);
    else if (variant == 6) hc_pre_v5_kernel<true, 1, 2, true><<<rows, 1024, 0, s>>>(X, fn, base, scale, F, mix, xc);
    else hc_pre_v5_kernel<true, 3, 4, true><<<rows, 1024, 0, s>>>(X, fn, base, scale, F, mix, xc);
    ok(hipGetLastError(), "glm hc_pre_v3");
}
void router_dense(const float* x, const std::uint16_t* router, float* logits, unsigned rows, hipStream_t s, int variant) {
    if (variant < 0) variant = rows <= 2 ? 3 : 2;  // measured crossover between 2 and 4 rows
    if (variant == 0) router_dense_kernel<<<dim3((kExperts + 31) / 32, rows), 256, 0, s>>>(x, router, logits);
    else if (variant == 1) router_dense_v2_kernel<<<dim3((kExperts + 31) / 32, rows), 256, 0, s>>>(x, router, logits);
    else if (variant == 2) router_dense_v3_kernel<<<dim3((kExperts + 31) / 32, rows), 1024, 0, s>>>(x, router, logits);
    else router_dense_v4_kernel<<<dim3(kExperts / 8, rows), 256, 0, s>>>(x, router, logits);
    ok(hipGetLastError(), "glm router_dense");
}
void hc_chains_lds_only(const float* F, const float* fn, float* z, hipStream_t s, int pf) {
    if (pf == 0) hc_chains_lds_only_kernel<0, 2><<<1, 64, 0, s>>>(F, fn, z);
    else if (pf == 1) hc_chains_lds_only_kernel<1, 2><<<1, 64, 0, s>>>(F, fn, z);
    else hc_chains_lds_only_kernel<3, 4><<<1, 64, 0, s>>>(F, fn, z);
    ok(hipGetLastError(), "glm hc_chains_lds_only");
}
void route_v2(const float* logits, const float* bias, std::uint32_t* ids, float* w, unsigned rows, hipStream_t s) {
    route_v2_kernel<<<rows, 320, 0, s>>>(logits, bias, ids, w);
    ok(hipGetLastError(), "glm route_v2");
}

}  // namespace strix::glm::xfer
