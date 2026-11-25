#pragma once

#include "strix/head_argmax.hpp"

#include <cstdint>

// Keep View in the namespace used by the incumbent CUDA kernels.
namespace strix::cuda::core::head_argmax {

namespace ha = ::strix::head_argmax;

struct View {
    const float* logits = nullptr;
    std::uint64_t stride = 0;       // floats between rows (0 = count)
    std::uint32_t count = 0;        // logits per row
    std::uint32_t rows = 0;
    std::uint32_t first_index = 0;  // global index of position 0 (vocabulary slices)
    std::uint32_t chunks = 0;       // 0 = ha::default_chunks(count); <= ha::kMaxChunks
    ha::Pick* partial = nullptr;    // ha::workspace_bytes(rows, chunks) of device memory
};

namespace detail {

inline bool prepare_split_view(View& v) {
    if (v.stride == 0U) v.stride = v.count;
    if (v.chunks == 0U) v.chunks = ha::default_chunks(v.count);
    return v.logits && v.partial && v.count != 0U && v.chunks <= ha::kMaxChunks && v.stride >= v.count;
}
}  // namespace detail

}  // namespace strix::cuda::core::head_argmax
