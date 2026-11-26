#pragma once

#include "core/residual_mix.hpp"
#include "inkling2_mini_forward.hpp"
#include "inkling2_mini_resident_forward.hpp"
#include "inkling2_mini_resident_weights.hpp"

namespace strix::inkling2 {

// Each cross-pass query tensor holds two rows per core layer (query/moe or key/value).
inline constexpr unsigned kMiniMessageQueryRows = 2U * (kCoreLast - kCoreFirst);

// One batch holds the message rows of both query tensors for the next pass.
inline constexpr std::size_t kMiniMessageBatchElements = 2U * std::size_t{kMiniMessageQueryRows} * kHiddenWidth;

}  // namespace strix::inkling2

#if defined(__HIPCC__) || defined(__CUDACC__)
namespace strix::inkling2 {

inline constexpr core::residual_mix::LowRankController kMiniController{
    kHiddenWidth, kControllerRank, 2U, 1e-6F, true, true,
};

constexpr std::size_t mini_controller_elements(unsigned rows) {
    return kMiniController.scratch_elements(rows);
}

// Operations binds the backend's shared grouped BF16-weight projections and
// RMS launcher. The controller keeps FP32 rows and explicitly rounds only
// the normalized down operands, latent and output norm to BF16.
template <class Operations>
void enqueue_mini_layer_input(
    const MiniResidentWeights& weights, LayerVisit visit, MessagePath path, unsigned rows,
    std::span<const float> raw, std::span<const std::span<const float>> previous_bank,
    std::span<float> output, std::span<float> scratch, Operations& operations) {
    namespace residual = core::residual_mix;
    const auto row_elements = std::size_t{rows} * kHiddenWidth;
    if (!valid_mini_visit(visit) || rows == 0 || rows > kMaxContext ||
        raw.size() != row_elements || output.size() != row_elements)
        throw std::invalid_argument("Inkling 2 Mini: invalid layer input rows");
    const auto scale = weights.bf16(mini_layer_parameter(visit.physical_layer, mini_input_norm_suffix(visit, path)));
    if (visit.pass == 0U) {
        operations.rms(raw.data(), scale.data(), output.data(), rows);
        return;
    }
    const bool key_value = path == MessagePath::key || path == MessagePath::value;
    const auto queries = weights.f32(key_value ? "middle_cycle_cross_pass_kv_queries"
                                                : "middle_cycle_cross_pass_queries");
    const unsigned query = 2U * (visit.physical_layer - kCoreFirst) +
                           (path == MessagePath::value || path == MessagePath::moe ? 1U : 0U);
    if (query >= queries.size() / kHiddenWidth)
        throw std::invalid_argument("Inkling 2 Mini: invalid controller query");
    const auto stem = mini_layer_parameter(
        visit.physical_layer, std::string("controllers.") + std::string(mini_controller_path(path)) + ".");
    const residual::ControllerWeights controller{
        queries.subspan(std::size_t{query} * kHiddenWidth, kHiddenWidth),
        weights.bf16(stem + "down_weight").data(),
        weights.bf16(stem + "forget_weight").data(),
        weights.bf16(stem + "gain_weight").data(),
        scale.data(),
    };
    residual::enqueue_low_rank_controller(kMiniController, controller, rows, raw, previous_bank,
                                          output, scratch, operations);
}

}  // namespace strix::inkling2
#endif
