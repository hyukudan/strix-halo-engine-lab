#pragma once
// Engram gated injection, HIP kernel (the one kernel family of this operation on HIP; contract and host reference:
// include/strix/engram/inject.hpp). Engines bind it through InjectView + enqueue_inject; a form is a compile-time
// policy of the same kernel. Compile with the engine's strict arithmetic flags (-ffp-contract=off -fno-fast-math):
// the reductions follow ::strix::core::chunked_dot_of and the host reference bitwise.
//
// Geometry: one block per row, one wave per (copy, 640-element chunk) of hidden; lanes stride the chunk, a butterfly
// sums the wave, chunk partials are summed in ascending order by one thread per copy.
#include "strix/engram/inject.hpp"

#include <hip/hip_runtime.h>

#include <cstdint>
#include <stdexcept>

namespace strix::engram::hip {

__device__ inline float inject_butterfly(float v) {
    for (int d = 16; d; d >>= 1) v = v + __shfl_xor(v, d, static_cast<int>(inject_wave));
    return v;
}

// w0 / w1: see InjectView. The folded body is the DeepSeek V4.1 kernel as qualified (moved here unchanged).
template <InjectForm Form, unsigned Copies>
__global__ __launch_bounds__(1024) void inject_kernel(float* __restrict__ hc_stream, const float* __restrict__ kv,
                                                      const float* __restrict__ qn, const float* __restrict__ kn, unsigned hidden,
                                                      float eps, float inv_sqrt_hidden, const std::uint8_t* __restrict__ gate_off) {
    constexpr unsigned wave = inject_wave, chunk = inject_chunk;
    __shared__ float part[3][Copies][32];
    __shared__ float gates[Copies];
    const unsigned chunks = (hidden + chunk - 1U) / chunk;
    const unsigned w = threadIdx.x / wave, l = threadIdx.x % wave, cc = w / chunks, c = w % chunks;
    float* h = hc_stream + static_cast<std::size_t>(blockIdx.x) * Copies * hidden;
    const float* row_kv = kv + static_cast<std::size_t>(blockIdx.x) * (Copies + 1U) * hidden;
    if constexpr (Form == InjectForm::folded) {
        if (cc < Copies) {
            const float* hcp = h + cc * hidden;
            const float* key = row_kv + cc * hidden;
            const float* q = qn + cc * hidden;
            const float* k = kn + cc * hidden;
            const unsigned c0 = c * chunk, end = c0 + chunk < hidden ? c0 + chunk : hidden;
            float hh = 0.0F, kk = 0.0F, dk = 0.0F;
            for (unsigned i = c0 + l; i < end; i += wave) {
                hh = hh + hcp[i] * hcp[i];
                kk = kk + key[i] * key[i];
                dk = dk + (hcp[i] * (q[i] * k[i])) * key[i];
            }
            hh = inject_butterfly(hh);
            kk = inject_butterfly(kk);
            dk = inject_butterfly(dk);
            if (l == 0) { part[0][cc][c] = hh; part[1][cc][c] = kk; part[2][cc][c] = dk; }
        }
        __syncthreads();
        if (threadIdx.x < Copies) {
            const unsigned cp = threadIdx.x;
            float hh = 0.0F, kk = 0.0F, dk = 0.0F;
            for (unsigned j = 0; j < chunks; ++j) { hh = hh + part[0][cp][j]; kk = kk + part[1][cp][j]; dk = dk + part[2][cp][j]; }
            const float rh = ::strix::numeric::declared_rsqrt(hh / static_cast<float>(hidden) + eps);
            const float rk = ::strix::numeric::declared_rsqrt(kk / static_cast<float>(hidden) + eps);
            const float gate = ::strix::numeric::engram_gate_of((dk * (rh * rk)) * inv_sqrt_hidden);
            gates[cp] = gate_off != nullptr && gate_off[blockIdx.x] != 0U ? 0.0F : gate;
        }
        __syncthreads();
        const float* value = row_kv + Copies * hidden;
        for (unsigned i = threadIdx.x; i < Copies * hidden; i += blockDim.x) h[i] = h[i] + gates[i / hidden] * value[i % hidden];
    } else {
        // normalized_bf16: qn = gate weight [Copies][hidden]; kn = copy scale [Copies][hidden] then gamma [hidden].
        if (gate_off != nullptr && gate_off[blockIdx.x] != 0U) return;  // masked row: Engram output zero (uniform per block)
        __shared__ float scales[2U * Copies + 1U];  // rq[c], rk[c], rv
        const float* value = row_kv + Copies * hidden;
        const unsigned c0 = c * chunk, end = c0 + chunk < hidden ? c0 + chunk : hidden;
        if (cc < Copies) {  // pass 1: sums of squares (stream copy, key; copy 0's waves also the value)
            const float* hcp = h + cc * hidden;
            const float* key = row_kv + cc * hidden;
            float hh = 0.0F, kk = 0.0F, vv = 0.0F;
            for (unsigned i = c0 + l; i < end; i += wave) {
                hh = hh + hcp[i] * hcp[i];
                kk = kk + key[i] * key[i];
                if (cc == 0U) vv = vv + value[i] * value[i];
            }
            hh = inject_butterfly(hh);
            kk = inject_butterfly(kk);
            vv = inject_butterfly(vv);
            if (l == 0) { part[0][cc][c] = hh; part[1][cc][c] = kk; part[2][cc][c] = vv; }
        }
        __syncthreads();
        if (threadIdx.x <= Copies) {
            const unsigned cp = threadIdx.x;
            float a = 0.0F, b = 0.0F;
            for (unsigned j = 0; j < chunks; ++j) {
                if (cp < Copies) { a = a + part[0][cp][j]; b = b + part[1][cp][j]; }
                else a = a + part[2][0][j];
            }
            if (cp < Copies) {
                scales[cp] = ::strix::numeric::declared_rsqrt(a / static_cast<float>(hidden) + eps);
                scales[Copies + cp] = ::strix::numeric::declared_rsqrt(b / static_cast<float>(hidden) + eps);
            } else {
                scales[2U * Copies] = ::strix::numeric::declared_rsqrt(a / static_cast<float>(hidden) + eps);
            }
        }
        __syncthreads();
        if (cc < Copies) {  // pass 2: the gate dot of BF16-rounded normalized terms
            const float* hcp = h + cc * hidden;
            const float* key = row_kv + cc * hidden;
            const float* gw = qn + cc * hidden;
            const float rq = scales[cc], rk = scales[Copies + cc];
            float s = 0.0F;
            for (unsigned i = c0 + l; i < end; i += wave)
                s = s + ::strix::numeric::round_to_bf16(::strix::numeric::round_to_bf16(::strix::numeric::round_to_bf16(key[i] * rk) * gw[i]) * ::strix::numeric::round_to_bf16(hcp[i] * rq));
            s = inject_butterfly(s);
            if (l == 0) part[0][cc][c] = s;
        }
        __syncthreads();
        if (threadIdx.x < Copies) {
            const unsigned cp = threadIdx.x;
            float s = 0.0F;
            for (unsigned j = 0; j < chunks; ++j) s = s + part[0][cp][j];
            gates[cp] = normalized_gate_of(::strix::numeric::round_to_bf16(s * inv_sqrt_hidden));
        }
        __syncthreads();
        const float rv = scales[2U * Copies];
        const float* scale = kn;
        const float* gamma = kn + Copies * hidden;
        for (unsigned i = threadIdx.x; i < Copies * hidden; i += blockDim.x) {
            const unsigned e = i % hidden;
            const float m = ::strix::numeric::round_to_bf16(gamma[e] * ::strix::numeric::round_to_bf16(value[e] * rv));
            const float out = ::strix::numeric::round_to_bf16(::strix::numeric::round_to_bf16(gates[i / hidden] * m) + ::strix::numeric::round_to_bf16(scale[i] * m));
            h[i] = ::strix::numeric::round_to_bf16(h[i] + out);
        }
    }
}

// The form is a template argument so an engine instantiates only the form it binds (view.form must match).
template <InjectForm Form>
void enqueue_inject(const InjectView& v, hipStream_t stream) {
    if (v.form != Form || !inject_supported(v)) throw std::invalid_argument("Engram injection view");
    hipLaunchKernelGGL((inject_kernel<Form, 4U>), dim3(v.rows), dim3(inject_threads(v)), 0, stream, v.stream, v.kv, v.w0, v.w1,
                       v.hidden, v.eps, inv_sqrt_hidden_of(v.hidden), v.gate_off);
}

}  // namespace strix::engram::hip
