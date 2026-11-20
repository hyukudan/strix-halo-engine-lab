// Included inside glm_prefill.hip's anonymous namespace (strict FP flags of that TU: -ffp-contract=off).
// GLM binding of the shared HC row expansion (src/core/hc_expand_rows.hpp): the expressions of hc_expand_norm_rows_kernel,
// unchanged (mx = ((c0 x0 + c1 x1) + c2 x2) + c3 x3, o = post * y + mx, y = y0 + y1 in rank order, rr = inv_sqrt(ss / n +
// eps) with ss from the same per-lane fmaf chain and lds_tree256_lane0's pairing). Placement only.
struct GlmHcExpandPolicy {
    static constexpr unsigned kStreams = kHc, kHidden = ::strix::glm::kHidden, kMixStride = kHcMix;
    static constexpr bool kNorm = true;
    static constexpr unsigned kMinWavesPerEu = 8;  // <= 192 VGPRs: spill-free with the hoisted y row
    struct Mix {
        float post[kHc];
        float comb[kHc * kHc];
    };
    __device__ __forceinline__ static Mix load_mix(const float* __restrict__ m) {
        Mix M;
#pragma unroll
        for (unsigned i = 0; i < kHc; ++i) M.post[i] = m[kHc + i];
#pragma unroll
        for (unsigned j = 0; j < kHc * kHc; ++j) M.comb[j] = m[2 * kHc + j];
        return M;
    }
    __device__ __forceinline__ static float expand(const Mix& M, unsigned i, const float* x, float yd) {
        const float mx = ((M.comb[0 * kHc + i] * x[0] + M.comb[1 * kHc + i] * x[1]) + M.comb[2 * kHc + i] * x[2]) +
                         M.comb[3 * kHc + i] * x[3];
        return M.post[i] * yd + mx;
    }
    // y0 + y1 with y0 as VOP src0, as the exchange's sum2 kernel and hc_expand_norm_rows_kernel emit it: with two NaN
    // inputs the quieted src0 payload is returned, so the operand order is pinned.
    __device__ __forceinline__ static float pair(float y0, float y1) {
        float s;
        asm("v_add_f32_e32 %0, %1, %2" : "=v"(s) : "v"(y0), "v"(y1));
        return s;
    }
    __device__ __forceinline__ static bool finite(float o) { return is_finite(o); }
    // lds_tree256_lane0's levels for lane l < 32 (s = the row's 256 lane partials).
    __device__ __forceinline__ static float tree256_lane0(const float* s, unsigned l) {
        const float t0 = s[l] + s[l + 128], t1 = s[l + 64] + s[l + 192], t2 = s[l + 32] + s[l + 160], t3 = s[l + 96] + s[l + 224];
        float w = (t0 + t1) + (t2 + t3);
#pragma unroll
        for (unsigned h = 16; h > 0; h >>= 1U) w = w + __shfl_down(w, h, 32);
        return w;
    }
    __device__ __forceinline__ static float finish(float ss) { return inv_sqrt(ss / static_cast<float>(kHc * kHidden) + kEps); }
};
static_assert(kHc == 4, "GlmHcExpandPolicy::expand spells out four streams");
// The GLM instance: one virtual lane per thread, streaming X loads and Xo stores (each row is touched once while the
// other compute queue's MoE / attention working set stays cached).
constexpr unsigned kHcExpandBwLanes = 1, kHcExpandBwNT = 2;
