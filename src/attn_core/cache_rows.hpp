#pragma once
// Shared HIP cache-row writer (contract include/strix/attn/cache_rows.hpp). One work-group per row: FP32 source row
// -> bf16 RNE at the row of its position (full cache or ring). Bits == cache_rows::store_rows_ref.
// Engines with a fused cache kernel (GLM DSA store: LayerNorm + flag + pooling in one launch) call the same
// cache_rows::encode / finite in their kernels instead of this launch.
#include <hip/hip_runtime.h>

#include "strix/attn/cache_rows.hpp"

#include <cstddef>
#include <cstdint>

namespace strix::attn_core::cache_rows {
namespace cr = strix::attn::cache_rows;

struct StoreArgs {
    const float* src;                // [rows][src_stride] FP32 (row k = the k-th token of the launch)
    std::size_t src_stride;          // floats between source rows (>= layout.width: a row may be a prefix of a wider one)
    const std::uint32_t* positions;  // device [rows] or nullptr: position = first_position + k
    std::uint32_t first_position;
    unsigned rows;
    cr::Rows layout;
    std::uint16_t* cache;  // bf16 rows
    unsigned* flag;        // nullptr or OR-ed with 1 when an input / encoded value is not finite
};

template <unsigned kBlock>
__global__ __launch_bounds__(kBlock) void store_rows_kernel(StoreArgs a) {
    const unsigned k = blockIdx.x;
    const std::uint64_t p = a.positions != nullptr ? a.positions[k] : std::uint64_t{a.first_position} + k;
    std::uint16_t* dst = a.cache + cr::row_offset(a.layout, p);
    const float* src = a.src + static_cast<std::size_t>(k) * a.src_stride;
    bool ok = true;
    for (unsigned i = threadIdx.x; i < a.layout.width; i += kBlock) {
        const float x = src[i];
        const std::uint16_t b = cr::encode(x);
        ok = ok && cr::finite(x) && cr::finite(b);
        dst[i] = b;
    }
    if (!ok && a.flag != nullptr) atomicOr(a.flag, 1U);
}

// rows == 0 is a no-op. Errors: width 0, a null cache / source.
inline hipError_t store_rows(const StoreArgs& a, hipStream_t stream) {
    if (a.rows == 0U) return hipSuccess;
    if (a.layout.width == 0U || a.cache == nullptr || a.src == nullptr || a.src_stride < a.layout.width) return hipErrorInvalidValue;
    constexpr unsigned kBlock = 128U;
    store_rows_kernel<kBlock><<<dim3(a.rows), dim3(kBlock), 0, stream>>>(a);
    return hipGetLastError();
}

}  // namespace strix::attn_core::cache_rows
