#pragma once

#include "strix/execution_binding.hpp"
#include "strix/hip_runtime_launchers.hpp"
#include "strix/launch_dispatch.hpp"

#include <cstdint>
#include <string>
#include <memory>
#include <span>

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
    bool staged_logits = false;
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

// Known-prefix window rows: 2/4/8 C1 row views of one serial parallel router
// step. Logits read each weight tile once for all rows; every row keeps the
// serial ascending fused chain and the configured selection body, so each
// row's logits/indices/weights equal its serial launch bit for bit. Each row
// is admitted exactly as the serial launcher admits it; launch uses the
// current HIP launch stream and is capture safe.
void moe_route_known_window_rows_preflight(std::span<const ExecutionDispatchContext> rows,
    const HipMoeRouteLauncherConfig& config);
void moe_route_known_window_rows_launch(std::span<const ExecutionDispatchContext> rows,
    const HipMoeRouteLauncherConfig& config);
// Prefill: all T rows of a router step (8-row tiles one stride apart).
void moe_route_known_prefill_rows_preflight(std::span<const ExecutionDispatchContext> rows,
    const HipMoeRouteLauncherConfig& config);
void moe_route_known_prefill_rows_launch(std::span<const ExecutionDispatchContext> rows,
    const HipMoeRouteLauncherConfig& config);

// Source-only native MTP entry points; qualification is separate.
struct MtpNativeContext;
void preflight_hip_mtp_route(const MtpNativeContext&);
void launch_hip_mtp_route(const MtpNativeContext&,void* stream);

}  // namespace strix
