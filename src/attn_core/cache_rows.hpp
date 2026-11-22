#pragma once
// Shared HIP cache-row writer (contract include/strix/attn/cache_rows.hpp). One work-group per row: FP32 source row
// -> bf16 RNE at the row of its position (full cache or ring). Bits == cache_rows::store_rows_ref.
// Engines with a fused cache kernel (GLM DSA store: LayerNorm + flag + pooling in one launch) call the same
// cache_rows::encode / finite in their kernels instead of this launch.
#include <hip/hip_runtime.h>

#include "strix/attn/cache_rows.hpp"
#include "wmma_types.hpp"  // ImageEpilogue: the operand rounding the attention cores read (tile::to_bf16)

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
__device__ __forceinline__ void store_row(const StoreArgs& a) {
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

// store_rows_kernel keeps its own body (the same steps as store_row) so existing callers compile to the same
// machine code as before the paired entry existed.
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

template <unsigned kBlock>
__global__ __launch_bounds__(kBlock) void store_pair_kernel(StoreArgs first, StoreArgs second) {
    store_row<kBlock>(blockIdx.y == 0U ? first : second);
}

// The writer as a producer's store epilogue (policy E of core/f16_dense_q8a8.hpp StoreImage): the producer's FP32 output
// x is stored as tile::to_bf16(x) == encode(x) for every non-NaN x (a NaN raises the flag on both paths), and the flag
// is raised under the same rule as store_rows (encodes_finite). The image rows are the producer's rows (ring 0, rows
// as wide as the producer's out_stride): the bits store_rows writes from the producer's FP32 output, without it.
struct ImageEpilogue {
    using Args = unsigned*;  // the writer's flag (nullptr: none)
    static constexpr tile::Operand kOp = tile::Operand::bf16;
    static __device__ __forceinline__ void pair(const Args& flag, unsigned, unsigned, float& x0, float& x1) {
        if (flag != nullptr && !(cr::encodes_finite(x0) && cr::encodes_finite(x1))) atomicOr(flag, 1U);
    }
};

// rows == 0 is a no-op. Errors: width 0, a null cache / source.
inline bool valid_store(const StoreArgs& a) {
    return a.layout.width != 0U && a.cache != nullptr && a.src != nullptr && a.src_stride >= a.layout.width;
}

inline hipError_t store_rows(const StoreArgs& a, hipStream_t stream) {
    if (a.rows == 0U) return hipSuccess;
    if (!valid_store(a)) return hipErrorInvalidValue;
    constexpr unsigned kBlock = 128U;
    STRIX_HIP_LAUNCH(Rdna3Layout, store_rows_kernel<kBlock><<<dim3(a.rows), dim3(kBlock), 0, stream>>>(a));
    return hipGetLastError();
}

// Two disjoint destinations with the same number of rows, in one launch.
// The caller guarantees distinct slots for device-supplied positions.
inline hipError_t store_pair(const StoreArgs& first, const StoreArgs& second, hipStream_t stream) {
    if (first.rows != second.rows) return hipErrorInvalidValue;
    if (first.rows == 0U) return hipSuccess;
    const auto distinct_slots = [](const StoreArgs& a) {
        return a.positions != nullptr || a.layout.ring == 0U || a.rows <= a.layout.ring;
    };
    if (first.cache == second.cache || !valid_store(first) || !valid_store(second) ||
        !distinct_slots(first) || !distinct_slots(second)) return hipErrorInvalidValue;
    constexpr unsigned kBlock = 128U;
    store_pair_kernel<kBlock><<<dim3(first.rows, 2U), dim3(kBlock), 0, stream>>>(first, second);
    return hipGetLastError();
}

// One K row and one V row per decode lane, each to the lane's own caches at the lane's own position: lane l reads row l
// of the packed key / value sources. Every element is encode(x) under the store_rows flag rule, so each lane's cache
// rows hold the bits store_pair writes for that lane alone. Destinations of different lanes must be distinct caches.
struct LaneStore {
    std::uint16_t* keys;
    std::uint16_t* values;
    std::uint32_t position;
    cr::Rows layout;
};

inline constexpr unsigned kMaxStoreLanes = 16U;

struct LaneStores {
    LaneStore lane[kMaxStoreLanes];
};

template <unsigned kBlock>
__global__ __launch_bounds__(kBlock) void store_lanes_kernel(const float* keys, const float* values,
                                                             std::size_t src_stride, LaneStores lanes, unsigned* flag) {
    const unsigned l = blockIdx.x;
    const LaneStore& lane = lanes.lane[l];
    const bool second = blockIdx.y != 0U;
    std::uint16_t* dst = (second ? lane.values : lane.keys) + cr::row_offset(lane.layout, lane.position);
    const float* src = (second ? values : keys) + static_cast<std::size_t>(l) * src_stride;
    bool ok = true;
    for (unsigned i = threadIdx.x; i < lane.layout.width; i += kBlock) {
        const float x = src[i];
        const std::uint16_t b = cr::encode(x);
        ok = ok && cr::finite(x) && cr::finite(b);
        dst[i] = b;
    }
    if (!ok && flag != nullptr) atomicOr(flag, 1U);
}

inline hipError_t store_lanes(const float* keys, const float* values, std::size_t src_stride, const LaneStores& lanes,
                              unsigned count, unsigned* flag, hipStream_t stream) {
    if (count == 0U) return hipSuccess;
    if (count > kMaxStoreLanes || keys == nullptr || values == nullptr) return hipErrorInvalidValue;
    for (unsigned l = 0; l < count; ++l) {
        const auto& lane = lanes.lane[l];
        if (lane.keys == nullptr || lane.values == nullptr || lane.layout.width == 0U ||
            src_stride < lane.layout.width) return hipErrorInvalidValue;
    }
    constexpr unsigned kBlock = 128U;
    store_lanes_kernel<kBlock><<<dim3(count, 2U), dim3(kBlock), 0, stream>>>(keys, values, src_stride, lanes, flag);
    return hipGetLastError();
}

}  // namespace strix::attn_core::cache_rows
