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

class HipElementwiseLauncherBundle {
public:
    explicit HipElementwiseLauncherBundle(
        HipElementwiseLauncherConfig config)
        : config_(config) {}

    HipElementwiseLauncherBundle(
        const HipElementwiseLauncherBundle&) = delete;
    HipElementwiseLauncherBundle& operator=(
        const HipElementwiseLauncherBundle&) = delete;
    HipElementwiseLauncherBundle(
        HipElementwiseLauncherBundle&&) = delete;
    HipElementwiseLauncherBundle& operator=(
        HipElementwiseLauncherBundle&&) = delete;

    void register_for_compiled_execution(
        const CompiledSerialExecution& compiled,
        ExecutionLauncherRegistry& registry);

    [[nodiscard]] const HipElementwiseLauncherConfig&
    config() const noexcept {
        return config_;
    }

private:
    HipElementwiseLauncherConfig config_;
};

}  // namespace strix
