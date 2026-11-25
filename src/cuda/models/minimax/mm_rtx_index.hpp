#pragma once

#include "mm_rtx_pack.hpp"
#include "core/dense_f32_input_major_plan.hpp"

namespace strix::minimax::rtxpf {

inline constexpr unsigned kIndexInput = 6144;

namespace index_detail {

// Weight admission plans one row. Match the CUDA form's K and output policy
// here so this header can be used without the CUDA SDK.
struct AdmissionTile {
    static constexpr unsigned rows = 1, outputs = 128, k = 64;
};
inline constexpr unsigned kMaxOutputTiles = 65535;

}  // namespace index_detail

struct IndexProjection {
    const char* tensor;
    unsigned outputs, stride;

    [[nodiscard]] bool admits(const host::TensorInfo& weight) const {
        // IQ has two equal rank slices. IK uses the full tensor on both ranks.
        if (weight.shape.size() != 2 || weight.shape[0] != stride || weight.shape[1] != kIndexInput)
            return false;
        if (weight.dtype != "F32" && weight.dtype != "BF16" && weight.dtype != "F16") return false;
        const unsigned bytes = weight.dtype == "F32" ? 4U : 2U;
        // Admit the transposed weight and fixed output layout here. The shared
        // launcher checks the actual prefill row count when it plans the grid.
        return weight.bytes == std::uint64_t{stride} * kIndexInput * bytes &&
            ::strix::core::plan_f32_input_major_rows<index_detail::AdmissionTile>(
                1, kIndexInput, outputs, stride, 1, index_detail::kMaxOutputTiles).valid();
    }
};

inline constexpr IndexProjection kIndexQuery{"index_q_proj.weight", 256, 512};
inline constexpr IndexProjection kIndexKey{"index_k_proj.weight", 128, 128};

}  // namespace strix::minimax::rtxpf
