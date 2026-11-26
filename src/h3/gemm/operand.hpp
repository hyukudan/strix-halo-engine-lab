#pragma once

#include "h3/gemm/ref.hpp"
#include "strix/weight/channel_quant_operand.hpp"

namespace strix::h3::gemm {
enum class Operand : std::uint32_t { int8_convrot, w4a8 };

using weight::channel_quant::e4m3fn;
using weight::channel_quant::PackedW4A8;
using weight::channel_quant::PreparedW4A8;
using weight::channel_quant::prepare_w4a8;
using weight::channel_quant::w4a8_code;
} // namespace strix::h3::gemm
