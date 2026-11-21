#pragma once

#include "inkling2_mini_config.hpp"
#include "strix/weight/formats/microscaled.hpp"

namespace strix::inkling2 {

inline constexpr weight::microscaled::RowOrder mini_expert_row_order(MiniExpertProjection projection) {
    using weight::microscaled::RowOrder;
    return projection == MiniExpertProjection::gate_up ? RowOrder::interleaved_halves : RowOrder::contiguous;
}

inline constexpr weight::microscaled::MatrixFormat mini_expert_format(unsigned physical_layer) {
    using weight::microscaled::MatrixFormat;
    return physical_layer < kMxFp8First ? MatrixFormat::nvfp4 : MatrixFormat::mxfp8;
}

}  // namespace strix::inkling2
