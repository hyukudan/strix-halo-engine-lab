#pragma once
// Calibration inputs of Inkling 2 Mini (naming map as data) for the shared input statistics
// (strix/weight/calib_moments.hpp): per physical layer the query / key / value attention inputs and the output
// projection input, and the head input. A layer visited by several passes accumulates every visit. The query input also
// feeds the attention output gate (the same rows: attn_core::enqueue_prepared_gqa groups them), so it lists both.
#include "inkling2_mini_checkpoint.hpp"
#include "inkling2_mini_config.hpp"
#include "strix/weight/calib_moments.hpp"

#include <array>
#include <string>

namespace strix::inkling2 {

enum class MiniCalibrationInput { query, key, value, output, head };

inline weight::calib::InputMoments& mini_calibration_input(weight::calib::MomentSet& set, MiniCalibrationInput input,
                                                           unsigned physical_layer = 0U) {
    const std::string tensors(kMiniTextTensorPrefix);
    if (input == MiniCalibrationInput::head) {
        const std::array<std::string, 1> consumers{tensors + "unembed.weight"};
        return set.declare("head_input", kHiddenWidth, consumers);
    }
    const auto layer = "layers." + std::to_string(physical_layer) + ".attn.";
    struct Row {
        const char* input;
        const char* matrix;
        unsigned width;
    };
    static constexpr std::array<Row, 4> rows{{{"query_input", "wq_du.weight", kHiddenWidth},
                                              {"key_input", "wk_dv.weight", kHiddenWidth},
                                              {"value_input", "wv_dv.weight", kHiddenWidth},
                                              {"output_input", "wo_ud.weight", kQueryWidth}}};
    const auto& row = rows[static_cast<unsigned>(input)];
    if (input == MiniCalibrationInput::query) {
        const std::array<std::string, 2> consumers{tensors + layer + row.matrix, tensors + layer + "gate_attn.weight"};
        return set.declare(layer + row.input, row.width, consumers);
    }
    const std::array<std::string, 1> consumers{tensors + layer + row.matrix};
    return set.declare(layer + row.input, row.width, consumers);
}

}  // namespace strix::inkling2
