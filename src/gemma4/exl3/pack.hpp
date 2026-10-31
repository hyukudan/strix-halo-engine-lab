#pragma once
#include "gemma4/model/model.hpp"

namespace strix::gemma4::exl3 {
// Maps arm bytes plus only the mandatory non-quantized checkpoint tensors.
// Verifies every adopted tensor hash. No full FP32 model expansion occurs.
// The pack's arm label (manifest "arm", e.g. "M4"), for identity auto-selection.
[[nodiscard]] std::string pack_arm(const std::filesystem::path& pack);
[[nodiscard]] ModelManifest load_arm(const std::filesystem::path& pack,
                                     const std::filesystem::path& bf16);
} // namespace strix::gemma4::exl3
