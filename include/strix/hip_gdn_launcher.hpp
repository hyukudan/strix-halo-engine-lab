#pragma once

#include "strix/execution_binding.hpp"
#include "strix/hip_runtime_launchers.hpp"
#include "strix/launch_dispatch.hpp"
#include "strix/model_contract.hpp"

#include <cstdint>
#include <string>
#include <span>

namespace strix {

class HipGdnIslands;

struct HipGdnLauncherConfig {
    ModelContract model =
        ModelContract::qwen38_flash_next();
    HipActivationStorage activation =
        HipActivationStorage::bf16;
    bool enable_device_graph_inputs = false;
    bool virtual_q8_input_projections = false;
    bool prefetch_q8_input_projections = false;
    // Explicit eager geometry overlay; requires the two input options above.
    bool wave8_q8_input_projections = false;
    bool virtual_q8_output_projection = false;
    // Explicit eager FP32 output organization; requires virtual Q8 output.
    bool wave8_q8_output_projection = false;
};

// Closed eager R2 window helper. Each context remains an R1 row view. Both
// rows are deeply validated before any enqueue; recurrence updates row0 then
// row1 on the same state. No allocation, synchronization or session publication.
// Caller retains contexts, config, resources and owners through stream quiescence.
void gdn_known_window_r2_preflight(std::span<const ExecutionDispatchContext, 2> rows,
    const HipGdnLauncherConfig& config);
void gdn_known_window_r2_launch(std::span<const ExecutionDispatchContext, 2> rows,
    const HipGdnLauncherConfig& config);

#ifdef STRIX_BUILD_KNOWN_WINDOW_DIAGNOSTIC
enum class GdnKnownWindowProjectionDirection { bf16_to_fp32, fp32_to_bf16 };
enum class GdnKnownWindowProjectionMode { incumbent_row_loop, rows2 };
struct GdnKnownWindowProjectionView { std::uintptr_t address = 0; std::uint64_t bytes = 0; };
// Test-only projection payloads on independent row addresses (contiguous
// or strided). Same incumbent epilogues; no state mutation, allocation or sync.
void gdn_known_window_r2_projection_probe(GdnKnownWindowProjectionDirection,
    GdnKnownWindowProjectionMode, std::span<const GdnKnownWindowProjectionView, 2> input,
    GdnKnownWindowProjectionView weights, std::span<const GdnKnownWindowProjectionView, 2> output,
    std::uint32_t width, std::uint32_t outputs, void* stream);
#endif

void register_hip_gdn_kernel_variant(
    const ModelContract& model,
    const ExecutionBindingRequest& request,
    const HipGdnLauncherConfig& config,
    kernels::KernelRegistry& registry,
    kernels::QualificationStatus qualification =
        kernels::QualificationStatus::experimental);

[[nodiscard]] ExecutionLauncher
make_hip_gdn_launcher(
    std::string kernel_id,
    HipGdnLauncherConfig* config);

class HipGdnLauncherBundle {
public:
    explicit HipGdnLauncherBundle(
        HipGdnLauncherConfig config)
        : config_(config) {}

    HipGdnLauncherBundle(
        const HipGdnLauncherBundle&) = delete;
    HipGdnLauncherBundle& operator=(
        const HipGdnLauncherBundle&) = delete;
    HipGdnLauncherBundle(
        HipGdnLauncherBundle&&) = delete;
    HipGdnLauncherBundle& operator=(
        HipGdnLauncherBundle&&) = delete;

    void register_for_compiled_execution(
        const CompiledSerialExecution& compiled,
        ExecutionLauncherRegistry& registry, HipGdnIslands* islands = nullptr);

private:
    HipGdnLauncherConfig config_;
};

}  // namespace strix
