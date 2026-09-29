#pragma once

#include "strix/launch_dispatch.hpp"

#include <cstdint>
#include <string>

namespace strix {

enum class HipActivationStorage : std::uint32_t {
    bf16 = 2,
    fp32 = 4,
};

struct HipElementwiseLauncherConfig {
    std::uint32_t hidden_size = 0;
    std::uint32_t hc_count = 0;
    HipActivationStorage activation =
        HipActivationStorage::bf16;
};

[[nodiscard]] std::uint64_t
hip_elementwise_launcher_identity(
    std::uint64_t operation_tag,
    const HipElementwiseLauncherConfig& config) noexcept;

[[nodiscard]] ExecutionLauncher
make_hip_hyper_expand_launcher(
    std::string kernel_id,
    HipElementwiseLauncherConfig* config);

[[nodiscard]] ExecutionLauncher
make_hip_inject_launcher(
    std::string kernel_id,
    HipElementwiseLauncherConfig* config);

[[nodiscard]] ExecutionLauncher
make_hip_moe_combine_launcher(
    std::string kernel_id,
    HipElementwiseLauncherConfig* config);

}  // namespace strix
