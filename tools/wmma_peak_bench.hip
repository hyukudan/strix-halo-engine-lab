// WMMA f16 peak of gfx1151 (diagnostic, not part of the build): v_wmma_f32_16x16x16_f16 throughput with register
// operands (independent accumulator chains) and with operands read from LDS by ds_load_b128 in the prefill-tile
// pattern (per K16 plane: 2 B fragments reused by 4 A fragments, 8 WMMAs). Reports TFLOP/s and the shader clock
// implied by s_memtime against wall time.
#include <hip/hip_runtime.h>
#include <cstdio>
#include <cstdlib>
#include <vector>
#include <algorithm>
typedef _Float16 h16 __attribute__((ext_vector_type(16)));
typedef float f8 __attribute__((ext_vector_type(8)));
#define CHECK(x) do { hipError_t e = (x); if (e != hipSuccess) { std::fprintf(stderr, "%s: %s\n", #x, hipGetErrorString(e)); std::exit(1); } } while (0)

template <unsigned Chains>
__global__ __launch_bounds__(256) void wmma_regs(float* out, unsigned iters, unsigned long long* clocks) {
    h16 a, b;
    for (unsigned k = 0; k < 16U; ++k) { a[k] = static_cast<_Float16>((threadIdx.x + k) % 7U); b[k] = static_cast<_Float16>((threadIdx.x * 3U + k) % 5U); }
    f8 acc[Chains] = {};
    const unsigned long long t0 = __builtin_readcyclecounter();
    for (unsigned it = 0; it < iters; ++it) {
#pragma unroll
        for (unsigned c = 0; c < Chains; ++c) acc[c] = __builtin_amdgcn_wmma_f32_16x16x16_f16_w32(a, b, acc[c]);
    }
    const unsigned long long t1 = __builtin_readcyclecounter();
    float s = 0.0F;
#pragma unroll
    for (unsigned c = 0; c < Chains; ++c) for (unsigned e = 0; e < 8U; ++e) s += acc[c][e];
    out[blockIdx.x * blockDim.x + threadIdx.x] = s;
    if (threadIdx.x == 0 && blockIdx.x == 0) *clocks = t1 - t0;
}

// LDS-fed: 8 waves, each 64 rows x 32 columns of a 64 x 256 tile at a 72-half pitch (the base MoE layout).
__global__ __launch_bounds__(256) void wmma_lds(float* out, unsigned iters) {
    __shared__ __align__(16) _Float16 a_s[64 * 72], b_s[256 * 72];
    for (unsigned k = threadIdx.x; k < 64U * 72U; k += 256U) a_s[k] = static_cast<_Float16>(k % 13U);
    for (unsigned k = threadIdx.x; k < 256U * 72U; k += 256U) b_s[k] = static_cast<_Float16>(k % 11U);
    __syncthreads();
    const unsigned lane = threadIdx.x & 31U, wave = threadIdx.x / 32U, i = lane & 15U;
    f8 acc[4][2] = {};
    const auto pair = [](const _Float16* row, unsigned lo, unsigned hi) {
        const uint4 x = *reinterpret_cast<const uint4*>(row + lo), y = *reinterpret_cast<const uint4*>(row + hi);
        h16 v; __builtin_memcpy(&v, &x, 16); __builtin_memcpy(reinterpret_cast<char*>(&v) + 16, &y, 16); return v;
    };
    for (unsigned it = 0; it < iters; ++it) {
#pragma unroll
        for (unsigned kk = 0; kk < 4U; ++kk) {
            const unsigned lo = 16U * kk, hi = lo + 8U;
            const h16 b0 = pair(b_s + (32U * wave + i) * 72U, lo, hi), b1 = pair(b_s + (32U * wave + 16U + i) * 72U, lo, hi);
#pragma unroll
            for (unsigned mi = 0; mi < 4U; ++mi) {
                const h16 a = pair(a_s + (16U * mi + i) * 72U, lo, hi);
                acc[mi][0] = __builtin_amdgcn_wmma_f32_16x16x16_f16_w32(a, b0, acc[mi][0]);
                acc[mi][1] = __builtin_amdgcn_wmma_f32_16x16x16_f16_w32(a, b1, acc[mi][1]);
            }
        }
    }
    float s = 0.0F;
    for (unsigned m = 0; m < 4U; ++m) for (unsigned n = 0; n < 2U; ++n) for (unsigned e = 0; e < 8U; ++e) s += acc[m][n][e];
    out[blockIdx.x * blockDim.x + threadIdx.x] = s;
}

// Co-issue probe: each iteration issues 8 independent WMMAs and V independent FP32 FMAs (4 chains) in one block.
// Time against WMMA-only (V=0) and VALU-only (W=0) tells whether VALU work overlaps the WMMA pipe.
template <unsigned W, unsigned V>
__global__ __launch_bounds__(256) void wmma_valu(float* out, unsigned iters) {
    h16 a, b;
    for (unsigned k = 0; k < 16U; ++k) { a[k] = static_cast<_Float16>((threadIdx.x + k) % 7U); b[k] = static_cast<_Float16>((threadIdx.x * 3U + k) % 5U); }
    f8 acc[8] = {};
    float x0 = threadIdx.x * 1e-3F, x1 = x0 + 1.0F, x2 = x0 + 2.0F, x3 = x0 + 3.0F;
    const float m = 1.0001F, c = 1e-7F;
    for (unsigned it = 0; it < iters; ++it) {
#pragma unroll
        for (unsigned w = 0; w < W; ++w) acc[w] = __builtin_amdgcn_wmma_f32_16x16x16_f16_w32(a, b, acc[w]);
#pragma unroll
        for (unsigned v = 0; v < V / 4U; ++v) { x0 = fmaf(x0, m, c); x1 = fmaf(x1, m, c); x2 = fmaf(x2, m, c); x3 = fmaf(x3, m, c); }
    }
    float s = x0 + x1 + x2 + x3;
#pragma unroll
    for (unsigned w = 0; w < 8U; ++w) for (unsigned e = 0; e < 8U; ++e) s += acc[w][e];
    out[blockIdx.x * blockDim.x + threadIdx.x] = s;
}

int main() {
    hipDeviceProp_t p; CHECK(hipGetDeviceProperties(&p, 0));
    std::printf("device %s CUs %d clock %d kHz\n", p.gcnArchName, p.multiProcessorCount, p.clockRate);
    float* out; unsigned long long* clk;
    const unsigned blocks_max = 40U * 64U;
    CHECK(hipMalloc(&out, blocks_max * 256U * sizeof(float))); CHECK(hipMalloc(&clk, 8));
    hipEvent_t e0, e1; CHECK(hipEventCreate(&e0)); CHECK(hipEventCreate(&e1));
    const auto time = [&](auto launch, double flops, const char* name) {
        launch(); CHECK(hipDeviceSynchronize());
        std::vector<float> ms;
        for (int r = 0; r < 5; ++r) {
            CHECK(hipEventRecord(e0)); launch(); CHECK(hipEventRecord(e1)); CHECK(hipEventSynchronize(e1));
            float t; CHECK(hipEventElapsedTime(&t, e0, e1)); ms.push_back(t);
        }
        std::sort(ms.begin(), ms.end());
        std::printf("%-44s %8.3f ms  %6.2f TFLOP/s\n", name, ms[2], flops / (ms[2] * 1e9));
        return ms[2];
    };
    const unsigned iters = 4096U;
    for (unsigned waves_per_simd : {1U, 2U, 4U, 8U}) {
        const unsigned blocks = 40U * 2U * waves_per_simd * 32U / 256U;  // 256-thread blocks: 8 waves
        char name[96];
        std::snprintf(name, sizeof name, "regs 8 chains, %u blocks (%u waves/SIMD)", blocks, waves_per_simd);
        const double flops = 2.0 * 16 * 16 * 16 * 8.0 * iters * blocks * 8.0;
        const float ms = time([&] { hipLaunchKernelGGL(wmma_regs<8>, dim3(blocks), dim3(256), 0, 0, out, iters, clk); }, flops, name);
        unsigned long long c; CHECK(hipMemcpy(&c, clk, 8, hipMemcpyDeviceToHost));
        std::printf("    block0 wave0: %llu cycles for %u WMMA -> %.1f cycles/WMMA/wave; implied clock %.0f MHz (ms %.3f)\n",
            c, iters * 8U, static_cast<double>(c) / (iters * 8.0), static_cast<double>(c) / (ms * 1e3), ms);
    }
    for (unsigned chains : {1U, 2U, 4U}) {
        const unsigned blocks = 40U * 2U * 4U * 32U / 256U;
        char name[96]; std::snprintf(name, sizeof name, "regs %u chains, 4 waves/SIMD", chains);
        const double flops = 2.0 * 4096.0 * chains * iters * blocks * 8.0;
        if (chains == 1U) time([&] { hipLaunchKernelGGL(wmma_regs<1>, dim3(blocks), dim3(256), 0, 0, out, iters, clk); }, flops, name);
        if (chains == 2U) time([&] { hipLaunchKernelGGL(wmma_regs<2>, dim3(blocks), dim3(256), 0, 0, out, iters, clk); }, flops, name);
        if (chains == 4U) time([&] { hipLaunchKernelGGL(wmma_regs<4>, dim3(blocks), dim3(256), 0, 0, out, iters, clk); }, flops, name);
    }
    for (unsigned per_wgp : {1U, 2U, 3U, 4U}) {
        const unsigned blocks = 20U * per_wgp * 8U, it = 1024U;
        char name[96]; std::snprintf(name, sizeof name, "lds 64x256 tile, %u CTA/WGP x8", per_wgp);
        const double flops = 2.0 * 4096.0 * 32.0 * it * blocks * 8.0;
        time([&] { hipLaunchKernelGGL(wmma_lds, dim3(blocks), dim3(256), 0, 0, out, it); }, flops, name);
    }
    {
        const unsigned blocks = 40U * 2U * 4U * 32U / 256U, it = 4096U;  // 4 waves/SIMD
        const auto co = [&]<unsigned W, unsigned V>(const char* name) {
            time([&] { hipLaunchKernelGGL(HIP_KERNEL_NAME(wmma_valu<W, V>), dim3(blocks), dim3(256), 0, 0, out, it); },
                2.0 * 4096.0 * W * it * blocks * 8.0 + 1.0, name);
        };
        co.template operator()<8U, 0U>("co-issue: 8 WMMA + 0 FMA / iter");
        co.template operator()<0U, 64U>("co-issue: 0 WMMA + 64 FMA / iter");
        co.template operator()<0U, 256U>("co-issue: 0 WMMA + 256 FMA / iter");
        co.template operator()<8U, 64U>("co-issue: 8 WMMA + 64 FMA / iter");
        co.template operator()<8U, 128U>("co-issue: 8 WMMA + 128 FMA / iter");
        co.template operator()<8U, 256U>("co-issue: 8 WMMA + 256 FMA / iter");
    }
    return 0;
}
