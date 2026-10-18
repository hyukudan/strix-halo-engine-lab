#pragma once

#include "strix/execution_binding.hpp"
#include "strix/hip_runtime_launchers.hpp"
#include "strix/launch_dispatch.hpp"

#include <cstdint>
#include <atomic>
#include <memory>
#include <span>
#include <string>
#include <string_view>
#include <vector>

namespace strix {

inline constexpr std::string_view hc_projection_next_tile_suffix = ".prefetch-next-tile-v1";

inline constexpr std::string_view splitk_hc_fp32_numerical_identity =
    "hc-down-splitk16x640-r40-fp32-v1";
inline constexpr std::string_view splitk_hc_fp64_numerical_identity =
    "hc-down-splitk16x640-r40-fp64-oracle-v1";
inline constexpr std::string_view deferred_hc_numerical_identity =
    "hc-down-splitk16x640-r40-fp32-deferred-rms-v1";

struct HipHyperMixArithmeticStats {
    // Actual successful host enqueues; graph replays are never inferred.
    std::atomic<std::uint64_t> splitk_f32_submissions{0};
    std::atomic<std::uint64_t> deferred_norm_submissions{0};
    std::atomic<std::uint64_t> materialized_norm_submissions{0};
};

struct HipHyperMixLauncherConfig {
    std::uint32_t hidden_size = 0;
    std::uint32_t hc_count = 0;
    std::uint32_t lowrank = 0;
    float rms_epsilon = 1e-6F;
    bool parallel = false;
    bool staged_projections = false;
    bool fused_injection = false;
    bool typed_projection = false;
    bool prefetch_rows = false;
    bool prefetch_q8_output = false;
    bool lane_chain_output = false;
    // Experimental integration, selected only by a diagnostic host driver.
    bool splitk_projection = false;
    bool splitk_projection_fp64 = false;
    bool splitk_deferred_norm = false;
    // Exact current-weight, eager-only raw next-tile projection pipeline.
    bool prefetch_hc_projection_next_tile = false;
    // fast-prefill-f16wmma-v1: the wide prefill rows run the FP16 WMMA projection/output kernels (prefill only).
    bool fast_prefill_wmma = false;
    HipActivationStorage activation =
        HipActivationStorage::bf16;
    std::shared_ptr<HipHyperMixArithmeticStats> arithmetic_stats;
};

void register_hip_hyper_mix_kernel_variant(
    const ModelContract& model,
    const ExecutionBindingRequest& request,
    const HipHyperMixLauncherConfig& config,
    kernels::KernelRegistry& registry,
    kernels::QualificationStatus qualification =
        kernels::QualificationStatus::experimental);

[[nodiscard]] ExecutionLauncher
make_hip_hyper_mix_launcher(
    std::string kernel_id,
    HipHyperMixLauncherConfig* config);

// Validates every compiled mixer before the first typed launch or capture.
void preflight_typed_hyper_projection_model(const CompiledSerialExecution& compiled,
    const ExecutionRuntimeResources& resources);

void preflight_q8_hyper_output_model(const CompiledSerialExecution& compiled,
    const ExecutionRuntimeResources& resources, const HipHyperMixLauncherConfig& config);

void preflight_splitk_hyper_projection_model(const CompiledSerialExecution& compiled,
    const ExecutionRuntimeResources& resources, const HipHyperMixLauncherConfig& config);

// Known-window rows (2..8 C1 contexts of one attn/mlp/final HC step). Each row
// produces the bytes of its serial typed prefetched Q8 launch (split-K or v8
// projection, lane-chain or Q8 batched output); HC weights are read once.
void hyper_mix_known_window_rows_preflight(std::span<const ExecutionDispatchContext> rows,
    const HipHyperMixLauncherConfig& config);
void hyper_mix_known_window_rows_launch(std::span<const ExecutionDispatchContext> rows,
    const HipHyperMixLauncherConfig& config);
// The inject rows step that produces this HC step's input, folded into the persistent HC rows launch
// (STRIX_HC_PERSISTENT=1 and STRIX_HC_PERSISTENT_INJECT=1). Both steps are admitted as their own launches
// would admit them; the streams written equal the inject launch's bytes. Returns false, with nothing
// enqueued, when the pair is not eligible (the caller then launches both steps itself).
bool hyper_mix_known_window_rows_launch_injected(std::span<const ExecutionDispatchContext> inject_rows,
    const HipElementwiseLauncherConfig& inject_config, std::span<const ExecutionDispatchContext> rows,
    const HipHyperMixLauncherConfig& config);
// Prefill: all T rows of an HC step (T a multiple of 8, tiles one stride apart) in one launch per kernel.
void hyper_mix_known_prefill_rows_preflight(std::span<const ExecutionDispatchContext> rows,
    const HipHyperMixLauncherConfig& config);
void hyper_mix_known_prefill_rows_launch(std::span<const ExecutionDispatchContext> rows,
    const HipHyperMixLauncherConfig& config);

class HipHyperMixLauncherBundle {
public:
    explicit HipHyperMixLauncherBundle(
        HipHyperMixLauncherConfig config)
        : config_(config), fallback_config_(config) {
        fallback_config_.lane_chain_output=false;
        fallback_config_.splitk_deferred_norm=false;
    }

    HipHyperMixLauncherBundle(
        const HipHyperMixLauncherBundle&) = delete;
    HipHyperMixLauncherBundle& operator=(
        const HipHyperMixLauncherBundle&) = delete;

    void register_for_compiled_execution(
        const CompiledSerialExecution& compiled,
        ExecutionLauncherRegistry& registry,
        const ExecutionRuntimeResources* resources = nullptr);

private:
    HipHyperMixLauncherConfig config_;
    HipHyperMixLauncherConfig fallback_config_;
};

// Number of persistent window HC launches so far (STRIX_HC_PERSISTENT=1), for fixtures.
std::uint64_t hip_hyper_mix_testing_persistent_launches();
// Phase timestamps of the last persistent launch ([block][8], wall_clock64; STRIX_HC_PERSISTENT_TRACE).
std::vector<std::uint64_t> hip_hyper_mix_testing_persistent_trace();

}  // namespace strix
