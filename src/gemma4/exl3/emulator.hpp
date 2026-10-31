#pragma once
#include "format.hpp"

namespace strix::gemma4::exl3 {
[[nodiscard]] std::vector<float> emulate_prefill(LinearView w, std::span<const float> input, unsigned rows);
}
