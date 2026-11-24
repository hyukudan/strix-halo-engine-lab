#pragma once
// Exact TP sum joins and the pack / place copies of an exchange over one layout trait (Sum, SumColumns = Columns).
// Home: shared (backend core; HIP bodies under __HIPCC__). Bound by: deepseek41 HIP decode / batch / prefill TP sums,
// deepseek41 Engram row-TP pack / place and staged-ready gather / scatter.

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

// Each rank provides packed [rows][cols] values for the same output columns: the column slice [col0, col0 + cols) of
// [rows][width] rows, packed element i = (row i / cols, column col0 + i % cols).
struct SumColumns {
    std::uint32_t rows, cols, col0, width;

    STRIX_TP_JOIN_HD constexpr std::uint32_t size() const { return rows * cols; }
    STRIX_TP_JOIN_HD constexpr std::size_t output_index(std::uint32_t i) const {
        return std::size_t{i / cols} * width + col0 + i % cols;
    }
};
// The same slice where nothing is summed: the pack / place copies of an exchange.
using Columns = SumColumns;

template <class Layout>
STRIX_TP_JOIN_HD inline void join_sum_at(const float* a, const float* b, float* out, std::uint32_t i, Layout layout) {
    if (i < layout.size()) out[layout.output_index(i)] = a[i] + b[i];
}
// The copies around an exchange over the same layouts: pack_at gathers this rank's slice of the full rows into the
// contiguous message (packed[i] = full[output_index(i)]); place_at writes a received message into its slice
// (full[output_index(i)] = packed[i]). Plain copies: the words move unchanged.
template <class Layout>
STRIX_TP_JOIN_HD inline void pack_at(const float* full, float* packed, std::uint32_t i, Layout layout) {
    if (i < layout.size()) packed[i] = full[layout.output_index(i)];
}
template <class Layout>
STRIX_TP_JOIN_HD inline void place_at(const float* packed, float* full, std::uint32_t i, Layout layout) {
    if (i < layout.size()) full[layout.output_index(i)] = packed[i];
}

// Sum overload of tp_join_kernel. Kept in its own header so callers can bind it
// without emitting the existing gather and argmax entry points in additional HIP TUs.
#if defined(__HIPCC__)
template <class Layout>
__global__ void tp_join_kernel(const float* __restrict__ a, const float* __restrict__ b, float* __restrict__ out, Layout layout) {
    const std::uint32_t i = blockIdx.x * blockDim.x + threadIdx.x;
    join_sum_at(a, b, out, i, layout);
}
// One thread per packed element (grid = ceil(layout.size() / blockDim.x)).
template <class Layout>
__global__ void tp_pack_kernel(const float* __restrict__ full, float* __restrict__ packed, Layout layout) {
    const std::uint32_t i = blockIdx.x * blockDim.x + threadIdx.x;
    pack_at(full, packed, i, layout);
}
template <class Layout>
__global__ void tp_place_kernel(const float* __restrict__ packed, float* __restrict__ full, Layout layout) {
    const std::uint32_t i = blockIdx.x * blockDim.x + threadIdx.x;
    place_at(packed, full, i, layout);
}
#endif

}  // namespace strix::core::tp_join

#undef STRIX_TP_JOIN_HD
