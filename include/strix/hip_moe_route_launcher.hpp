#pragma once

#include "strix/execution_binding.hpp"
#include "strix/hip_runtime_launchers.hpp"
#include "strix/launch_dispatch.hpp"

#include <cstdint>
#include <string>
#include <memory>

namespace strix {
class WaveRouterManifestAdmission;

struct HipMoeRouteLauncherConfig {
    std::uint32_t hidden_size = 0;
    std::uint32_t expert_count = 0;
    std::uint32_t top_k = 0;
    HipActivationStorage activation =
        HipActivationStorage::bf16;
    bool normalize_selected = true;
    bool parallel_projection = false;
    bool ranked_topk = false;
    bool wave_topk = false;
    std::shared_ptr<const WaveRouterManifestAdmission> wave_manifest_admission;
};

void register_hip_moe_route_kernel_variant(
    const ModelContract& model,
    const ExecutionBindingRequest& request,
    const HipMoeRouteLauncherConfig& config,
    kernels::KernelRegistry& registry,
    kernels::QualificationStatus qualification =
        kernels::QualificationStatus::experimental);

[[nodiscard]] ExecutionLauncher
make_hip_moe_route_launcher(
    std::string kernel_id,
    HipMoeRouteLauncherConfig* config);

class HipMoeRouteLauncherBundle {
public:
    explicit HipMoeRouteLauncherBundle(
        HipMoeRouteLauncherConfig config)
        : config_(config) {}

    HipMoeRouteLauncherBundle(
        const HipMoeRouteLauncherBundle&) = delete;
    HipMoeRouteLauncherBundle& operator=(
        const HipMoeRouteLauncherBundle&) = delete;
    HipMoeRouteLauncherBundle(
        HipMoeRouteLauncherBundle&&) = delete;
    HipMoeRouteLauncherBundle& operator=(
        HipMoeRouteLauncherBundle&&) = delete;

    void register_for_compiled_execution(
        const CompiledSerialExecution& compiled,
        ExecutionLauncherRegistry& registry);

private:
    HipMoeRouteLauncherConfig config_;
};

// Source-only native MTP entry points; qualification is separate.
struct MtpNativeContext;
void preflight_hip_mtp_route(const MtpNativeContext&);
void launch_hip_mtp_route(const MtpNativeContext&,void* stream);

}  // namespace strix
