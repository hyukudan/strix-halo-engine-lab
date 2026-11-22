#pragma once

#include <string_view>

namespace strix::exec::detail {

// A native transfer is explicitly declared by the physical lowering and
// carries a backend-readable POD payload instead of a kernel thunk.
inline constexpr std::string_view native_transfer_form = "native-transfer/v1";

// Recipes from this lowering are opt-in capture comparisons. This prefix
// travels in the existing chain identity and participates in X1's identity.
inline constexpr std::string_view cuda_capture_recipe = "cuda-capture-recipe/v1/";

}  // namespace strix::exec::detail
