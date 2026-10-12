#pragma once

#include "strix/execution_binding.hpp"
#include "strix/hip_runtime_launchers.hpp"
#include "strix/hip_host_upload_backend.hpp"
#include "strix/launch_dispatch.hpp"
#include "strix/model_contract.hpp"
#include "strix/ple_lookup.hpp"

#include <cstdint>
#include <memory>
#include <string>

namespace strix {

struct HipPleLauncherConfig {
    ModelContract model =
        ModelContract::qwen38_flash_next();
    HipActivationStorage activation =
        HipActivationStorage::bf16;
    // Default keeps the existing synchronous copy and creates no upload pool.
    // Opt-in bundles require this fixed additional pinned-host admission budget.
    static constexpr std::uint64_t owned_event_upload_maximum_bytes =
        PleHashContract::maximum_heads * 512;
    bool enable_owned_event_upload = false;
    // Captured path uses a resident BF16 table and stable device n-gram IO.
    // File-backed lookup and event-upload paths remain eager only.
    bool enable_resident_graph = false;
    // An injected pool is an explicit alternative and outlives the launcher.
    HostUploadPool* upload_pool = nullptr;
};

void register_hip_ple_kernel_variant(
    const ModelContract& model,
    const ExecutionBindingRequest& request,
    const HipPleLauncherConfig& config,
    kernels::KernelRegistry& registry,
    kernels::QualificationStatus qualification =
        kernels::QualificationStatus::experimental);

[[nodiscard]] ExecutionLauncher
make_hip_ple_launcher(
    std::string kernel_id,
    HipPleLauncherConfig* config);

class HipPleLauncherBundle {
public:
    explicit HipPleLauncherBundle(
        HipPleLauncherConfig config);
    ~HipPleLauncherBundle();

    HipPleLauncherBundle(
        const HipPleLauncherBundle&) = delete;
    HipPleLauncherBundle& operator=(
        const HipPleLauncherBundle&) = delete;
    HipPleLauncherBundle(
        HipPleLauncherBundle&&) = delete;
    HipPleLauncherBundle& operator=(
        HipPleLauncherBundle&&) = delete;

    void register_for_compiled_execution(
        const CompiledSerialExecution& compiled,
        ExecutionLauncherRegistry& registry);

    [[nodiscard]] std::uint64_t owned_upload_admission_bytes() const noexcept {
        return uploads_ ? HipPleLauncherConfig::owned_event_upload_maximum_bytes : 0;
    }

private:
    struct OwnedUploads;
    std::unique_ptr<OwnedUploads> uploads_;
    HipPleLauncherConfig config_;
};

}  // namespace strix
