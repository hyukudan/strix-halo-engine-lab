#include "cuda/core/chain/args.hpp"
#include "cuda/models/minimax/mm_rtx_kernels.cuh"

#include <array>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <limits>

namespace {
namespace mm = strix::minimax::rtxpf::cuda;
namespace chain = strix::cuda::core::chain;

// Wave07 input norm and the pre-binding post-exchange norm use this row order.
__device__ __forceinline__ void norm_incumbent_row(const float* row, const float* w, float* out, float* red) {
    float sum = 0.0F;
    for (unsigned i = threadIdx.x; i < mm::kHidden; i += 256) sum = __fmaf_rn(row[i], row[i], sum);
    red[threadIdx.x] = sum;
    __syncthreads();
    for (unsigned k = 128; k != 0; k >>= 1U) {
        if (threadIdx.x < k) red[threadIdx.x] = red[threadIdx.x] + red[threadIdx.x + k];
        __syncthreads();
    }
    const float r = 1.0F / sqrtf(red[0] / static_cast<float>(mm::kHidden) + 1e-6F);
    for (unsigned i = threadIdx.x; i < mm::kHidden; i += 256) out[i] = (row[i] * r) * (1.0F + w[i]);
}

__global__ void minimax_norm_incumbent(const float* x, const float* w, float* y) {
    __shared__ float red[256];
    const std::size_t offset = static_cast<std::size_t>(blockIdx.x) * mm::kHidden;
    norm_incumbent_row(x + offset, w, y + offset, red);
}

__global__ void minimax_exchange_incumbent(float* h, const float* p0, const float* p1, const float* w, float* y) {
    __shared__ float red[256];
    const std::size_t offset = static_cast<std::size_t>(blockIdx.x) * mm::kHidden;
    for (unsigned i = threadIdx.x; i < mm::kHidden; i += 256) h[offset + i] = h[offset + i] + (p0[offset + i] + p1[offset + i]);
    __syncthreads();
    norm_incumbent_row(h + offset, w, y + offset, red);
}

void check(cudaError_t status) {
    if (status != cudaSuccess) {
        std::fprintf(stderr, "MiniMax RMSNorm: %s\n", cudaGetErrorString(status));
        std::exit(2);
    }
}

template <class T, std::size_t N>
T* upload(const std::array<T, N>& input) {
    T* result = nullptr;
    check(cudaMalloc(&result, sizeof(input)));
    check(cudaMemcpy(result, input.data(), sizeof(input), cudaMemcpyHostToDevice));
    return result;
}
}  // namespace

int main() {
    constexpr unsigned rows = 3;
    using Tile = std::array<float, rows * mm::kHidden>;
    Tile input{}, p0{}, p1{}, expected{}, actual{};
    for (unsigned i = 0; i < input.size(); ++i)
        input[i] = static_cast<float>(static_cast<int>(i * 13U % 251U) - 125) / 37.0F;
    input[17] = -0.0F;
    for (unsigned i = mm::kHidden; i < 2 * mm::kHidden; ++i) input[i] *= std::numeric_limits<float>::min();
    for (unsigned i = 2 * mm::kHidden; i < input.size(); ++i) input[i] *= 0x1p55F;
    for (unsigned i = 0; i < input.size(); ++i) {
        p0[i] = input[i] * 0.25F;
        p1[i] = input[i] * -0.5F;
    }
    input[0] = 1.0F;
    p0[0] = 0x1p24F;
    p1[0] = -0x1p24F;  // (h + p0) + p1 loses h; h + (p0 + p1) preserves it.
    std::array<float, mm::kHidden> raw_gain{}, effective_gain{};
    for (unsigned i = 0; i < raw_gain.size(); ++i) {
        raw_gain[i] = static_cast<float>(static_cast<int>(i % 19U) - 11) / 8.0F;
        effective_gain[i] = 1.0F + raw_gain[i];
    }
    auto* input_d = upload(input);
    auto* raw_gain_d = upload(raw_gain);
    auto* gain_d = upload(effective_gain);
    auto* norm_d = upload(std::array{chain::Norm{1e-6F, gain_d, nullptr, nullptr, chain::RmsOrder::bank_halving}});
    auto* expected_d = upload(expected);
    auto* actual_d = upload(actual);
    minimax_norm_incumbent<<<rows, 256>>>(input_d, raw_gain_d, expected_d);
    check(cudaGetLastError());
    mm::rmsnorm(input_d, norm_d, actual_d, rows, nullptr);
    check(cudaMemcpy(expected.data(), expected_d, sizeof(expected), cudaMemcpyDeviceToHost));
    check(cudaMemcpy(actual.data(), actual_d, sizeof(actual), cudaMemcpyDeviceToHost));
    if (std::memcmp(expected.data(), actual.data(), sizeof(expected)) != 0) {
        std::fputs("FAIL MiniMax initial RMSNorm bytes\n", stderr);
        return 1;
    }
    std::puts("PASS MiniMax initial RMSNorm bytes");

    auto* expected_h_d = upload(input);
    auto* p0_d = upload(p0);
    auto* p1_d = upload(p1);
    auto* residual_d = upload(std::array{chain::ScalarStore{input_d}});
    minimax_exchange_incumbent<<<rows, 256>>>(expected_h_d, p0_d, p1_d, raw_gain_d, expected_d);
    check(cudaGetLastError());
    mm::post_exchange(input_d, p0_d, p1_d, residual_d, norm_d, actual_d, rows, nullptr);
    check(cudaMemcpy(expected.data(), expected_d, sizeof(expected), cudaMemcpyDeviceToHost));
    check(cudaMemcpy(actual.data(), actual_d, sizeof(actual), cudaMemcpyDeviceToHost));
    const bool norm_equal = std::memcmp(expected.data(), actual.data(), sizeof(expected)) == 0;
    check(cudaMemcpy(expected.data(), expected_h_d, sizeof(expected), cudaMemcpyDeviceToHost));
    check(cudaMemcpy(actual.data(), input_d, sizeof(actual), cudaMemcpyDeviceToHost));
    const bool residual_equal = std::memcmp(expected.data(), actual.data(), sizeof(expected)) == 0;
    for (void* buffer : {static_cast<void*>(input_d), static_cast<void*>(raw_gain_d), static_cast<void*>(gain_d),
                         static_cast<void*>(norm_d), static_cast<void*>(expected_d), static_cast<void*>(actual_d),
                         static_cast<void*>(expected_h_d), static_cast<void*>(p0_d), static_cast<void*>(p1_d),
                         static_cast<void*>(residual_d)})
        check(cudaFree(buffer));
    if (!residual_equal || !norm_equal) {
        std::fprintf(stderr, "FAIL MiniMax post-exchange bytes residual=%d norm=%d\n", residual_equal, norm_equal);
        return 1;
    }
    std::puts("PASS MiniMax post-exchange residual and RMSNorm bytes");
}
