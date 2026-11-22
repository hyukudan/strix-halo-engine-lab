#pragma once

#include <cstddef>
#include <cstdint>

// Neutral layout/host contract: HIP translation units supply their native runtime
// declarations before including this header (tp_join.hpp already does so).
#if defined(__HIPCC__)
#define STRIX_TP_JOIN_HD __host__ __device__
#else
#define STRIX_TP_JOIN_HD
#endif

namespace strix::core::tp_join {

struct Sum {
    std::uint32_t count;

    STRIX_TP_JOIN_HD constexpr std::uint32_t size() const { return count; }
    STRIX_TP_JOIN_HD constexpr std::size_t output_index(std::uint32_t i) const { return i; }
};

// Each rank provides packed [rows][cols] values for the same output columns.
struct SumColumns {
    std::uint32_t rows, cols, col0, width;

    STRIX_TP_JOIN_HD constexpr std::uint32_t size() const { return rows * cols; }
    STRIX_TP_JOIN_HD constexpr std::size_t output_index(std::uint32_t i) const {
        return std::size_t{i / cols} * width + col0 + i % cols;
    }
};

template <class Layout>
STRIX_TP_JOIN_HD inline void join_sum_at(const float* a, const float* b, float* out, std::uint32_t i, Layout layout) {
    if (i < layout.size()) out[layout.output_index(i)] = a[i] + b[i];
}

// Sum overload of tp_join_kernel. Kept in its own header so callers can bind it
// without emitting the existing gather and argmax entry points in additional HIP TUs.
#if defined(__HIPCC__)
template <class Layout>
__global__ void tp_join_kernel(const float* __restrict__ a, const float* __restrict__ b, float* __restrict__ out, Layout layout) {
    const std::uint32_t i = blockIdx.x * blockDim.x + threadIdx.x;
    join_sum_at(a, b, out, i, layout);
}
#endif

}  // namespace strix::core::tp_join

#undef STRIX_TP_JOIN_HD
