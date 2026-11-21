#pragma once

#include "strix/attn/cache_rows.hpp"

namespace strix::attn_core {

struct PreparedFiniteRows {
    const float* queries;
    const float* keys;
    const float* values;
    const float* bias;
    const float* sinks;
    const float* output;
    unsigned rows;
    unsigned heads;
    unsigned kv_heads;
    unsigned width;
    unsigned bias_extent;
    unsigned* failure;
};

// Scan the original prepared operands as well as the result. A nonfinite
// score operand can otherwise be masked or yield a finite softmax output.
static __global__ void validate_prepared_finite(PreparedFiniteRows a) {
    namespace cr = strix::attn::cache_rows;
    const unsigned head = blockIdx.x;
    for (std::size_t row = blockIdx.y; row < a.rows; row += gridDim.y) {
        const auto query_head = row * a.heads + head;
        bool valid = threadIdx.x != 0U || !a.sinks || cr::finite(a.sinks[query_head]);
        for (unsigned i = threadIdx.x; i < a.width; i += blockDim.x) {
            const auto q = query_head * a.width + i;
            valid = valid && cr::encodes_finite(a.queries[q]) && cr::finite(a.output[q]);
            if (head < a.kv_heads) {
                const auto kv = (row * a.kv_heads + head) * a.width + i;
                valid = valid && cr::encodes_finite(a.keys[kv]) && cr::encodes_finite(a.values[kv]);
            }
        }
        for (std::size_t i = threadIdx.x; i < a.bias_extent; i += blockDim.x)
            valid = valid && cr::finite(a.bias[query_head * a.bias_extent + i]);
        if (!valid) atomicOr(a.failure, 1U);
    }
}

// The caller has already admitted the prepared attention geometry.
template <class Stream>
inline void enqueue_validate_prepared_finite(PreparedFiniteRows a, Stream stream) {
    constexpr unsigned max_grid_rows = 65535U;
    const auto grid_rows = a.rows < max_grid_rows ? a.rows : max_grid_rows;
    validate_prepared_finite<<<dim3(a.heads, grid_rows), 128, 0, stream>>>(a);
}

}  // namespace strix::attn_core
