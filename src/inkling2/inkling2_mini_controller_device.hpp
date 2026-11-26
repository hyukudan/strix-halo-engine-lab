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
// The one-row step program keeps one controller row (message, forget, gain, conditioned, latent) and the message
// rows of one pass after it.
inline constexpr std::size_t kMiniStepControllerElements =
    4U * std::size_t{kHiddenWidth} + kControllerRank + 2U * std::size_t{kMiniMessageQueryRows} * kHiddenWidth;

}  // namespace strix::inkling2

#if defined(__HIPCC__) || defined(__CUDACC__)
namespace strix::inkling2 {

inline constexpr core::residual_mix::LowRankController kMiniController{
    kHiddenWidth, kControllerRank, 2U, 1e-6F, true, true,
};

constexpr std::size_t mini_controller_elements(unsigned rows) {
    return kMiniController.scratch_elements(rows);
}

static_assert(kMiniStepControllerElements ==
              kMiniController.scratch_elements_per_row() + 2U * std::size_t{kMiniMessageQueryRows} * kHiddenWidth);

// Where the message of a pass 1-2 controller comes from in a residual bank batch over the cross-pass queries:
// set 0 = middle_cycle_cross_pass_queries (query 2l, moe 2l + 1), set 1 = middle_cycle_cross_pass_kv_queries
// (key 2l, value 2l + 1), l = physical layer - kCoreFirst. Row index of the batch output [2][rows][width].
struct MiniMessageSlot {
    bool key_value;
    unsigned query;
    [[nodiscard]] constexpr unsigned row() const { return (key_value ? kMiniMessageQueryRows : 0U) + query; }
};

constexpr MiniMessageSlot mini_message_slot(LayerVisit visit, MessagePath path) {
    const bool key_value = path == MessagePath::key || path == MessagePath::value;
    return {key_value, 2U * (visit.physical_layer - kCoreFirst) +
                           (path == MessagePath::value || path == MessagePath::moe ? 1U : 0U)};
}

// The controller weights of a pass 1-2 visit; query is its learned readout row.
inline core::residual_mix::ControllerWeights mini_controller_weights(const MiniResidentWeights& weights, LayerVisit visit,
                                                                     MessagePath path) {
    const auto scale = weights.bf16(mini_layer_parameter(visit.physical_layer, mini_input_norm_suffix(visit, path)));
    const auto slot = mini_message_slot(visit, path);
    const auto queries = weights.f32(slot.key_value ? "middle_cycle_cross_pass_kv_queries"
                                                     : "middle_cycle_cross_pass_queries");
    if (slot.query >= queries.size() / kHiddenWidth)
        throw std::invalid_argument("Inkling 2 Mini: invalid controller query");
    const auto stem = mini_layer_parameter(
        visit.physical_layer, std::string("controllers.") + std::string(mini_controller_path(path)) + ".");
    return {
        queries.subspan(std::size_t{slot.query} * kHiddenWidth, kHiddenWidth),
        weights.bf16(stem + "down_weight").data(),
        weights.bf16(stem + "forget_weight").data(),
        weights.bf16(stem + "gain_weight").data(),
        scale.data(),
    };
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
    if (visit.pass == 0U) {
        const auto scale = weights.bf16(mini_layer_parameter(visit.physical_layer, mini_input_norm_suffix(visit, path)));
        operations.rms(raw.data(), scale.data(), output.data(), rows);
        return;
    }
    residual::enqueue_low_rank_controller(kMiniController, mini_controller_weights(weights, visit, path), rows, raw,
                                          previous_bank, output, scratch, operations);
}

// A pass 1-2 controller whose message readout was already produced (one row of a residual bank batch over the
// pass's bank): the launches after the readout, with the same operands and bits as enqueue_mini_layer_input.
template <class Operations>
void enqueue_mini_layer_input_message(
    const MiniResidentWeights& weights, LayerVisit visit, MessagePath path, unsigned rows,
    std::span<const float> raw, std::span<const float> message, std::span<float> output, std::span<float> scratch,
    Operations& operations) {
    const auto row_elements = std::size_t{rows} * kHiddenWidth;
    if (!valid_mini_visit(visit) || visit.pass == 0U || rows == 0 || rows > kMaxContext ||
        raw.size() != row_elements || output.size() != row_elements || message.size() != row_elements)
        throw std::invalid_argument("Inkling 2 Mini: invalid layer input rows");
    core::residual_mix::enqueue_low_rank_controller_tail(kMiniController, mini_controller_weights(weights, visit, path),
                                                         rows, raw, message, output, scratch, operations);
}

}  // namespace strix::inkling2
#endif
