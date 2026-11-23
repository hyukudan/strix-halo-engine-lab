// GLM post-exchange kernels (see glm_post_exchange.hpp). Build with -ffp-contract=off -fno-fast-math
// -fno-gpu-flush-denormals-to-zero, as the Tier A kernels (src/glm/kernels/glm_tier_a.hip).
#include "glm_post_exchange.hpp"
#include "glm_xfer_device.hpp"

#include "glm/common/glm_hc.hpp"

#include <stdexcept>
#include <string>

namespace strix::glm::xfer {

namespace {

void ok(hipError_t e, const char* what) {
    if (e != hipSuccess) throw std::runtime_error(std::string(what) + ": " + hipGetErrorString(e));
}

// Split-producer chunk frame (xport/split_frame.hpp, SplitLayout{rows = gridDim.y, kHidden, kSplitHeadCols}): the payload
// element (r, d) sits at the chunk-major index. kSplit = false is the row-major kernel, its code unchanged.
constexpr unsigned kSplitHeadCols = kHidden / 2;
template <bool kSplit>
__device__ __forceinline__ std::size_t payload_index(unsigned r, unsigned d) {
    if constexpr (kSplit) {
        return d < kSplitHeadCols ? static_cast<std::size_t>(r) * kSplitHeadCols + d
                                  : static_cast<std::size_t>(gridDim.y) * kSplitHeadCols +
                                        static_cast<std::size_t>(r) * (kHidden - kSplitHeadCols) + (d - kSplitHeadCols);
    } else {
        return static_cast<std::size_t>(r) * kHidden + d;
    }
}

template <bool Split>
struct ExchangeConsume {
    using View = RxView;
    static constexpr unsigned hidden = kHidden;
    __device__ __forceinline__ static void wait(const View& v) { glmx_wait(v); }
    __device__ __forceinline__ static std::size_t index(unsigned r, unsigned d) { return payload_index<Split>(r, d); }
    __device__ __forceinline__ static void flag(unsigned* out, float v) {
        if (out && !is_finite(v)) atomicOr(out, 1U);
    }
};

__global__ __launch_bounds__(256) void hc_bracket_kernel(const float* X, const float* mix, float* B) {
    const unsigned r = blockIdx.y, d = blockIdx.x * 256 + threadIdx.x;
    if (d >= kHidden) return;
    const float* c = mix + static_cast<std::size_t>(r) * kHcMix + 2 * kHc;
    const float* Xr = X + static_cast<std::size_t>(r) * kHc * kHidden;
    for (unsigned i = 0; i < kHc; ++i)
        B[static_cast<std::size_t>(r) * kHc * kHidden + i * kHidden + d] =
            core::hc::packed_dot(c + i, Xr, d, kHidden, kHc);
}

// R2 tree in LDS over 256 lanes (as Tier A lds_tree<256>).
__device__ __forceinline__ float lds_tree256(float* s, float part) {
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

__global__ __launch_bounds__(256) void hc_pre_v2_kernel(const float* X, const float* fn, const float* base, const float* scale,
                                                        float* F, float* mix, float* xc) {
    constexpr unsigned n = kHc * kHidden;
    __shared__ float s[256];
    __shared__ float z[kHcMix];
    __shared__ float m[kHcMix];
    const unsigned r = blockIdx.x, l = threadIdx.x;
    const float* Xr = X + static_cast<std::size_t>(r) * n;
    float* Fr = F + static_cast<std::size_t>(r) * n;
    float a = 0.0F;
    for (unsigned i = l; i < n; i += 256) a = __builtin_fmaf(Xr[i], Xr[i], a);
    const float ss = lds_tree256(s, a);
    const float rr = inv_sqrt(ss / static_cast<float>(n) + kEps);
    for (unsigned i = l; i < n; i += 256) Fr[i] = Xr[i] * rr;
    __syncthreads();
    if (l < kHcMix) {  // the 24 R1 chains, unchanged
        float acc = 0.0F;
        const float* row = fn + static_cast<std::size_t>(l) * n;
        for (unsigned i = 0; i < n; ++i) acc = __builtin_fmaf(Fr[i], row[i], acc);
        z[l] = acc;
    }
    __syncthreads();
    core::hc::mix_final_wave<HcR6>(z, base, scale, kHcEps, m);
    __syncthreads();
    if (l < kHcMix) mix[static_cast<std::size_t>(r) * kHcMix + l] = m[l];
    float* xr = xc + static_cast<std::size_t>(r) * kHidden;
    for (unsigned d = l; d < kHidden; d += 256)
        xr[d] = core::hc::packed_dot(m, Xr, d, kHidden);
}

}  // namespace

void hc_pre_v2(const float* X, const float* fn, const float* base, const float* scale, float* F, float* mix, float* xc,
               unsigned rows, hipStream_t s) {
    hc_pre_v2_kernel<<<rows, 256, 0, s>>>(X, fn, base, scale, F, mix, xc);
    ok(hipGetLastError(), "glm hc_pre_v2");
}

void post_expand(const float* X, const float* mix, const float* local, const float* peer, const RxView& view, unsigned rank,
                 float* Xout, unsigned* flag, unsigned rows, hipStream_t s, unsigned flag_stride) {
    core::hc::expand_kernel<ExchangeConsume<false>, false><<<dim3(kHidden / 256, rows), 256, 0, s>>>(X, mix, local, peer, view, rank, Xout, flag, flag_stride);
    ok(hipGetLastError(), "glm post_expand");
}
void post_expand_split(const float* X, const float* mix, const float* local, const float* peer, const RxView& view, unsigned rank,
                       float* Xout, unsigned* flag, unsigned rows, hipStream_t s, unsigned flag_stride) {
    core::hc::expand_kernel<ExchangeConsume<true>, false><<<dim3(kHidden / 256, rows), 256, 0, s>>>(X, mix, local, peer, view, rank, Xout, flag, flag_stride);
    ok(hipGetLastError(), "glm post_expand split");
}
void hc_bracket(const float* X, const float* mix, float* B, unsigned rows, hipStream_t s) {
    hc_bracket_kernel<<<dim3(kHidden / 256, rows), 256, 0, s>>>(X, mix, B);
    ok(hipGetLastError(), "glm hc_bracket");
}
void post_expand_bracket(const float* B, const float* mix, const float* local, const float* peer, const RxView& view,
                         unsigned rank, float* Xout, unsigned* flag, unsigned rows, hipStream_t s, unsigned flag_stride) {
    core::hc::expand_kernel<ExchangeConsume<false>, true><<<dim3(kHidden / 256, rows), 256, 0, s>>>(B, mix, local, peer, view, rank, Xout, flag, flag_stride);
    ok(hipGetLastError(), "glm post_expand_bracket");
}

}  // namespace strix::glm::xfer
