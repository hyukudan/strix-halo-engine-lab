// Fast prefill profile, router component (included by hip_moe_route_launcher.hip after RouterTileRowsArgs).
//
// Router logits of the prefill tiles with v_wmma_f32_16x16x16_bf16. Every FP32 router weight w is split
// into Terms BF16 terms, h = bf16(w), m = bf16(w - h) (and l = bf16(w - h - m)), round to nearest even:
// h + m keeps 16 significant bits (relative error <= 2^-17), h + m + l is w exactly in the normal range.
// The BF16 activations are exact operands, so each product term is exact; the 16-wide dot products
// accumulate in FP32 over the K steps of 32 (terms in order h, m, l per step).
// A workgroup forms 128 rows x 64 experts: 8 waves of 32 x 32, A and the split B staged in LDS, the next K
// step's loads in registers while the current one computes.
namespace router_fast {
typedef __bf16 bf16x16 __attribute__((ext_vector_type(16)));
typedef float f32x8 __attribute__((ext_vector_type(8)));
typedef unsigned u32x4 __attribute__((ext_vector_type(4)));
constexpr unsigned kM = 128U, kN = 64U, kK = 32U, kPitch = kK + 8U;

__device__ __forceinline__ u32x4 load16(const void* p) {
    return *reinterpret_cast<const __attribute__((address_space(1))) u32x4*>(reinterpret_cast<std::uintptr_t>(p));
}
__device__ __forceinline__ void lds_barrier() { asm volatile("s_waitcnt lgkmcnt(0)\n\ts_barrier" ::: "memory"); }
__device__ __forceinline__ unsigned bf16_bits(float x) { // round to nearest even (finite x)
    const unsigned b = __float_as_uint(x);
    return (b + 0x7fffU + ((b >> 16U) & 1U)) >> 16U;
}
// Terms of four FP32 values, packed as BF16 pairs: t[term] = {values 0,1; values 2,3}.
__device__ __forceinline__ void split3(u32x4 v, uint2 (&t)[3]) {
    const unsigned w[4] = {v.x, v.y, v.z, v.w};
    unsigned h[4], m[4], l[4];
#pragma unroll
    for (unsigned i = 0; i < 4U; ++i) {
        const float x = __uint_as_float(w[i]);
        h[i] = bf16_bits(x);
        const float r1 = x - __uint_as_float(h[i] << 16U);
        m[i] = bf16_bits(r1);
        l[i] = bf16_bits(r1 - __uint_as_float(m[i] << 16U));
    }
    t[0] = make_uint2(h[0] | h[1] << 16U, h[2] | h[3] << 16U);
    t[1] = make_uint2(m[0] | m[1] << 16U, m[2] | m[3] << 16U);
    t[2] = make_uint2(l[0] | l[1] << 16U, l[2] | l[3] << 16U);
}

// rows: 8-row tiles (view r of tile i = row 8i + r); total_rows = 8 x tiles. hidden % 32 == 0, experts % 64 == 0.
template <unsigned Terms>
__global__ __launch_bounds__(256) void logits_kernel(RouterTileRowsArgs rows, const float* weights, std::uint32_t hidden,
    std::uint32_t total_rows) {
    static_assert(Terms == 2U || Terms == 3U);
    __shared__ __align__(16) std::uint16_t a_lds[kM][kPitch];
    __shared__ __align__(16) std::uint16_t b_lds[Terms][kN][kPitch];
    const unsigned t = threadIdx.x, lane = t % 32U, w = t / 32U, col = lane % 16U;
    const unsigned m0 = blockIdx.x * kM, n0 = blockIdx.y * kN, wm = (w % 4U) * 32U, wn = (w / 4U) * 32U;
    // Staging: A task j = (row t / 4 + 64 j, 8 activations 8 (t % 4)); B task j = (expert t / 8 + 32 j, 4 weights 4 (t % 8)).
    const std::uint8_t* a_src[2];
    const float* b_src[2];
#pragma unroll
    for (unsigned j = 0; j < 2U; ++j) {
        const unsigned g = min(m0 + t / 4U + 64U * j, total_rows - 1U);
        a_src[j] = reinterpret_cast<const std::uint8_t*>(rows.hidden[g % 8U]) + static_cast<std::uint64_t>(g / 8U) * rows.tile_hidden +
            16U * (t % 4U);
        b_src[j] = weights + static_cast<std::uint64_t>(n0 + t / 8U + 32U * j) * hidden + 4U * (t % 8U);
    }
    u32x4 a_reg[2], b_reg[2];
    const auto fetch = [&](unsigned k) {
#pragma unroll
        for (unsigned j = 0; j < 2U; ++j) { a_reg[j] = load16(a_src[j] + 2U * k); b_reg[j] = load16(b_src[j] + k); }
    };
    f32x8 acc[2][2] = {};
    fetch(0U);
    for (unsigned k = 0; k < hidden; k += kK) {
#pragma unroll
        for (unsigned j = 0; j < 2U; ++j) {
            *reinterpret_cast<u32x4*>(&a_lds[t / 4U + 64U * j][8U * (t % 4U)]) = a_reg[j];
            uint2 terms[3];
            split3(b_reg[j], terms);
#pragma unroll
            for (unsigned s = 0; s < Terms; ++s) *reinterpret_cast<uint2*>(&b_lds[s][t / 8U + 32U * j][4U * (t % 8U)]) = terms[s];
        }
        lds_barrier();
        if (k + kK < hidden) fetch(k + kK);
#pragma unroll
        for (unsigned kk = 0; kk < kK / 16U; ++kk) {
            bf16x16 a[2];
#pragma unroll
            for (unsigned i = 0; i < 2U; ++i) a[i] = *reinterpret_cast<const bf16x16*>(&a_lds[wm + 16U * i + col][16U * kk]);
#pragma unroll
            for (unsigned j = 0; j < 2U; ++j)
#pragma unroll
                for (unsigned s = 0; s < Terms; ++s) {
                    const bf16x16 b = *reinterpret_cast<const bf16x16*>(&b_lds[s][wn + 16U * j + col][16U * kk]);
#pragma unroll
                    for (unsigned i = 0; i < 2U; ++i) acc[i][j] = __builtin_amdgcn_wmma_f32_16x16x16_bf16_w32(a[i], b, acc[i][j]);
                }
        }
        lds_barrier();
    }
    // D layout: lane holds expert column col of each 16 x 16 tile, rows 2e + lane / 16.
#pragma unroll
    for (unsigned i = 0; i < 2U; ++i)
#pragma unroll
        for (unsigned e = 0; e < 8U; ++e) {
            const unsigned g = m0 + wm + 16U * i + 2U * e + lane / 16U;
            if (g >= total_rows) continue;
            float* out = reinterpret_cast<float*>(reinterpret_cast<std::uintptr_t>(rows.workspace[g % 8U]) +
                static_cast<std::uint64_t>(g / 8U) * rows.tile_workspace);
#pragma unroll
            for (unsigned j = 0; j < 2U; ++j) out[n0 + wn + 16U * j + col] = acc[i][j][e];
        }
}

// STRIX_PREFILL_ROUTER_TUNE (default 2, see the launcher): the same two-term logits from pre-split weights. split_weights writes h = bf16(w) and
// m = bf16(w - h) of every router weight once per call (logits_kernel<2> splits each weight again in all of its
// total_rows / kM workgroups), and logits_presplit_kernel stages K steps of 64 (half the barriers), for N = 64 experts
// per workgroup (waves of 32 x 32, as logits_kernel) or N = 128 (STRIX_PREFILL_ROUTER_TUNE=2: waves of 32 x 64, each A
// fragment feeds twice the WMMAs). Per logit the WMMA sequence is unchanged: K16 planes ascending, term h then m in
// each, so the FP32 results are bit-identical.
__global__ __launch_bounds__(256) void split_weights(const float* weights, std::uint16_t* h, std::uint16_t* m, std::uint32_t count) {
    for (std::uint32_t i = blockIdx.x * 256U + threadIdx.x; i < count; i += gridDim.x * 256U) {
        const float x = weights[i];
        const unsigned hb = bf16_bits(x);
        h[i] = static_cast<std::uint16_t>(hb);
        m[i] = static_cast<std::uint16_t>(bf16_bits(x - __uint_as_float(hb << 16U)));
    }
}
constexpr unsigned kK2 = 64U, kPitch2 = kK2 + 8U;
template <unsigned N>
__global__ __launch_bounds__(256) void logits_presplit_kernel(RouterTileRowsArgs rows, const std::uint16_t* wh, const std::uint16_t* wm_,
    std::uint32_t hidden, std::uint32_t total_rows) {
    static_assert(N == 64U || N == 128U);
    constexpr unsigned NT = N / 32U, BJ = N / 16U; // B tiles of 16 per wave; B staging tasks per thread
    __shared__ __align__(16) std::uint16_t a_lds[kM][kPitch2];
    __shared__ __align__(16) std::uint16_t b_lds[2][N][kPitch2];
    const unsigned t = threadIdx.x, lane = t % 32U, w = t / 32U, col = lane % 16U;
    const unsigned m0 = blockIdx.x * kM, n0 = blockIdx.y * N, wm = (w % 4U) * 32U, wn = (w / 4U) * (N / 2U);
    // Staging: A task j = (row t / 8 + 32 j, 8 activations 8 (t % 8)), j < 4; B task j = (term j / (BJ / 2), expert t / 8 +
    // 32 (j % (BJ / 2)), 8 weights 8 (t % 8)).
    const std::uint8_t* a_src[4];
    const std::uint16_t* b_src[BJ];
#pragma unroll
    for (unsigned j = 0; j < 4U; ++j) {
        const unsigned g = min(m0 + t / 8U + 32U * j, total_rows - 1U);
        a_src[j] = reinterpret_cast<const std::uint8_t*>(rows.hidden[g % 8U]) + static_cast<std::uint64_t>(g / 8U) * rows.tile_hidden +
            16U * (t % 8U);
    }
#pragma unroll
    for (unsigned j = 0; j < BJ; ++j)
        b_src[j] = (j < BJ / 2U ? wh : wm_) + static_cast<std::uint64_t>(n0 + t / 8U + 32U * (j % (BJ / 2U))) * hidden + 8U * (t % 8U);
    u32x4 a_reg[4], b_reg[BJ];
    const auto fetch = [&](unsigned k) {
#pragma unroll
        for (unsigned j = 0; j < 4U; ++j) a_reg[j] = load16(a_src[j] + 2U * k);
#pragma unroll
        for (unsigned j = 0; j < BJ; ++j) b_reg[j] = load16(b_src[j] + k);
    };
    f32x8 acc[2][NT] = {};
    fetch(0U);
    for (unsigned k = 0; k < hidden; k += kK2) {
#pragma unroll
        for (unsigned j = 0; j < 4U; ++j) *reinterpret_cast<u32x4*>(&a_lds[t / 8U + 32U * j][8U * (t % 8U)]) = a_reg[j];
#pragma unroll
        for (unsigned j = 0; j < BJ; ++j)
            *reinterpret_cast<u32x4*>(&b_lds[j / (BJ / 2U)][t / 8U + 32U * (j % (BJ / 2U))][8U * (t % 8U)]) = b_reg[j];
        lds_barrier();
        if (k + kK2 < hidden) fetch(k + kK2);
#pragma unroll
        for (unsigned kk = 0; kk < kK2 / 16U; ++kk) {
            bf16x16 a[2];
#pragma unroll
            for (unsigned i = 0; i < 2U; ++i) a[i] = *reinterpret_cast<const bf16x16*>(&a_lds[wm + 16U * i + col][16U * kk]);
#pragma unroll
            for (unsigned j = 0; j < NT; ++j)
#pragma unroll
                for (unsigned s = 0; s < 2U; ++s) {
                    const bf16x16 b = *reinterpret_cast<const bf16x16*>(&b_lds[s][wn + 16U * j + col][16U * kk]);
#pragma unroll
                    for (unsigned i = 0; i < 2U; ++i) acc[i][j] = __builtin_amdgcn_wmma_f32_16x16x16_bf16_w32(a[i], b, acc[i][j]);
                }
        }
        lds_barrier();
    }
#pragma unroll
    for (unsigned i = 0; i < 2U; ++i)
#pragma unroll
        for (unsigned e = 0; e < 8U; ++e) {
            const unsigned g = m0 + wm + 16U * i + 2U * e + lane / 16U;
            if (g >= total_rows) continue;
            float* out = reinterpret_cast<float*>(reinterpret_cast<std::uintptr_t>(rows.workspace[g % 8U]) +
                static_cast<std::uint64_t>(g / 8U) * rows.tile_workspace);
#pragma unroll
            for (unsigned j = 0; j < NT; ++j) out[n0 + wn + 16U * j + col] = acc[i][j][e];
        }
}
// STRIX_PREFILL_ROUTER_CHECK=1 (diagnostic): copy[r * experts + e] = logit (r, e) (pass 0), or count the differing bit
// patterns against copy (pass 1).
__global__ __launch_bounds__(256) void check_logits(RouterTileRowsArgs rows, float* copy, unsigned* mismatches, std::uint32_t experts,
    std::uint32_t total_rows, unsigned pass) {
    const std::uint32_t g = blockIdx.x;
    if (g >= total_rows) return;
    const float* out = reinterpret_cast<const float*>(reinterpret_cast<std::uintptr_t>(rows.workspace[g % 8U]) +
        static_cast<std::uint64_t>(g / 8U) * rows.tile_workspace);
    for (std::uint32_t e = threadIdx.x; e < experts; e += 256U) {
        float* c = copy + static_cast<std::uint64_t>(g) * experts + e;
        if (pass == 0U) *c = out[e];
        else if (__float_as_uint(*c) != __float_as_uint(out[e])) atomicAdd(mismatches, 1U);
    }
}
} // namespace router_fast
