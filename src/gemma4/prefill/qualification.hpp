#pragma once
#include "device.hpp"

namespace strix::gemma4::prefill {
// Explicit full-model CLI gates. They do not run in ctest and do not admit defaults.
void check_c4(DevicePrefill& prefill, gpu::DeviceTarget& decoder, const ModelConfig& config,
              std::span<const std::uint32_t> tokens, unsigned steps);
void check_dflash(DevicePrefill& prefill, gpu::DeviceTarget& decoder, const HostWeights& weights,
                  std::span<const std::uint32_t> tokens, unsigned steps,
                  const std::filesystem::path& draft_model, const std::filesystem::path& draft_rope);
} // namespace strix::gemma4::prefill
