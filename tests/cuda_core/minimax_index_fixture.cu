#include "cuda/core/dense/rows_f32_input_major.cuh"
#include "cuda/models/minimax/mm_rtx_index.hpp"
#include "cuda/models/minimax/mm_rtx_kernels.cuh"
#include "../rows/mmidx_frozen.cuh"

#include <algorithm>
#include <bit>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <random>
#include <stdexcept>

namespace {

namespace mm = strix::minimax::rtxpf::cuda;
namespace admission = strix::minimax::rtxpf::index_detail;
namespace dense = strix::cuda_core;
constexpr unsigned kRows = 33, kIqRank = 256, kIq = 512, kIk = 128;
constexpr float kGuard = -12345.0F;
static_assert(mm::kHidden == mmidx_frozen::kHidden);
static_assert(admission::AdmissionTile::k == dense::F32InputMajorRows::k &&
              admission::AdmissionTile::outputs == dense::F32InputMajorRows::outputs);
constexpr unsigned kLastOutput = admission::kMaxOutputTiles * admission::AdmissionTile::outputs;
static_assert(dense::plan_f32_input_major_rows(1, mm::kHidden, kLastOutput, kLastOutput).valid());
static_assert(!dense::plan_f32_input_major_rows(1, mm::kHidden, kLastOutput + 1, kLastOutput + 1).valid());

struct Buffers {
    float x[kRows * mm::kHidden];
    float wiq0[mm::kHidden * kIqRank], wiq1[mm::kHidden * kIqRank], wik[mm::kHidden * kIk];
    float expected_iq[kRows * kIq + 2], actual_iq[kRows * kIq + 2];
    float expected_ik[kRows * kIk + 2], actual_ik[kRows * kIk + 2];
};

void check(cudaError_t status) {
    if (status != cudaSuccess) throw std::runtime_error(cudaGetErrorString(status));
}

void fill_inputs(Buffers& b) {
    std::mt19937 rng(0x4d4d4944U);
    const auto fill = [&]<std::size_t N>(float (&values)[N]) {
        for (float& value : values) {
            const auto bits = static_cast<std::uint32_t>(rng());
            value = std::bit_cast<float>((bits & 0x807fffffU) | (123U << 23U));
        }
    };
    fill(b.x);
    fill(b.wiq0);
    fill(b.wiq1);
    fill(b.wik);
}

template <std::size_t N>
void compare_plane(const float (&actual)[N], const float (&expected)[N], unsigned live,
                   const char* plane, unsigned rows) {
    const auto guard = std::bit_cast<std::uint32_t>(kGuard);
    for (std::size_t i = 0; i < N; ++i) {
        const auto a = std::bit_cast<std::uint32_t>(actual[i]);
        const auto b = std::bit_cast<std::uint32_t>(expected[i]);
        const bool written = i > 0 && i <= live;
        if (a == b && (written ? a != guard : a == guard)) continue;
        std::fprintf(stderr, "%s rows=%u element=%zu got=%08x expected=%08x\n", plane, rows, i, a, b);
        throw std::runtime_error("MMIDX output or guard bytes differ");
    }
}

void compare(Buffers& b, unsigned rows) {
    std::ranges::fill(b.expected_iq, kGuard);
    std::ranges::fill(b.actual_iq, kGuard);
    std::ranges::fill(b.expected_ik, kGuard);
    std::ranges::fill(b.actual_ik, kGuard);
    if (rows != 0) {
        const unsigned tiles = (rows + 15) / 16;
        mmidx_frozen::dense_t_kernel<<<dim3(tiles, 2), 128>>>(b.x, b.wiq0, b.expected_iq + 1, rows, kIqRank, kIq);
        check(cudaGetLastError());
        mmidx_frozen::dense_t_kernel<<<dim3(tiles, 2), 128>>>(b.x, b.wiq1, b.expected_iq + 1 + kIqRank, rows, kIqRank, kIq);
        check(cudaGetLastError());
        mmidx_frozen::dense_t_kernel<<<tiles, 128>>>(b.x, b.wik, b.expected_ik + 1, rows, kIk, kIk);
        check(cudaGetLastError());
    }
    mm::index_project(b.x, b.wiq0, b.wiq1, b.wik, b.actual_iq + 1, b.actual_ik + 1, rows, nullptr);
    check(cudaDeviceSynchronize());
    compare_plane(b.actual_iq, b.expected_iq, rows * kIq, "IQ", rows);
    compare_plane(b.actual_ik, b.expected_ik, rows * kIk, "IK", rows);
}

}  // namespace

int main() {
    try {
        Buffers* buffers = nullptr;
        check(cudaMallocManaged(&buffers, sizeof(Buffers)));
        fill_inputs(*buffers);
        for (const unsigned rows : {0U, 1U, 2U, 4U, 8U, 15U, 16U, 17U, 32U, kRows}) compare(*buffers, rows);
        check(cudaFree(buffers));
        std::puts("PASS MiniMax RTX MMIDX index projection bytes");
    } catch (const std::exception& error) {
        std::fprintf(stderr, "FAIL MiniMax RTX MMIDX: %s\n", error.what());
        return 1;
    }
}
