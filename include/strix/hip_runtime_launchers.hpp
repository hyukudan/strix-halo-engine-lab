#pragma once

#include "strix/execution_binding.hpp"
#include "strix/launch_dispatch.hpp"

#include <cstdint>
#include <span>
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

void register_hip_elementwise_kernel_variants(
    const ModelContract& model,
    const ExecutionBindingRequest& request,
    const HipElementwiseLauncherConfig& config,
    kernels::KernelRegistry& registry,
    kernels::QualificationStatus qualification =
        kernels::QualificationStatus::experimental);

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

// Known-prefix window rows: 2/4/8 C1 row views of one serial inject
// (attn/mlp) or MoE-combine step in one launch. Each row is admitted as the
// serial launcher admits it and runs the same per-element arithmetic, so its
// output equals the serial launch bit for bit. Uses the current HIP stream.
void inject_known_window_rows_preflight(std::span<const ExecutionDispatchContext> rows,
    const HipElementwiseLauncherConfig& config);
void inject_known_window_rows_launch(std::span<const ExecutionDispatchContext> rows,
    const HipElementwiseLauncherConfig& config);
// The resolved views of an admitted inject (first = streams, second = hidden, weights = gates) or MoE-combine
// (first = shared, second = routed) rows call: the same admission as the launch, nothing enqueued, for a
// consumer that runs the element arithmetic inside its own kernel.
struct ElementwiseRowsViews {
    const void* first[8];
    const void* second[8];
    const float* weights[8];
    void* output[8];
    std::uint32_t count = 0;
};
ElementwiseRowsViews inject_known_window_rows_views(std::span<const ExecutionDispatchContext> rows,
    const HipElementwiseLauncherConfig& config);
ElementwiseRowsViews moe_combine_known_window_rows_views(std::span<const ExecutionDispatchContext> rows,
    const HipElementwiseLauncherConfig& config);
// Prefill (STRIX_QPF2K_HC): the views of an admitted full-T inject / MoE-combine rows call (the prefill launch's
// admission, nothing enqueued) when every view is one uniform row stride apart: row r of a view at base + r * stride.
// Returns false (views unset) otherwise.
struct ElementwisePrefillViews {
    const void* first = nullptr;
    const void* second = nullptr;
    const float* weights = nullptr;
    void* output = nullptr;
    std::uint64_t first_stride = 0, second_stride = 0, weights_stride = 0, output_stride = 0;
    std::uint32_t count = 0;
};
bool inject_known_prefill_rows_views(std::span<const ExecutionDispatchContext> rows, const HipElementwiseLauncherConfig& config,
    ElementwisePrefillViews& views);
bool moe_combine_known_prefill_rows_views(std::span<const ExecutionDispatchContext> rows, const HipElementwiseLauncherConfig& config,
    ElementwisePrefillViews& views);
void moe_combine_known_window_rows_preflight(std::span<const ExecutionDispatchContext> rows,
    const HipElementwiseLauncherConfig& config);
// The inject rows of several sessions' windows (segments of 2/4/8 rows, each admitted as its window; up to
// 32 rows) in one launch: every row's elements exactly as its own window's launch.
void inject_known_window_cohort_rows_launch(std::span<const std::span<const ExecutionDispatchContext>> segments,
    const HipElementwiseLauncherConfig& config);
// The same for the MoE-combine rows (shared + routed per row).
void moe_combine_known_window_cohort_rows_launch(std::span<const std::span<const ExecutionDispatchContext>> segments,
    const HipElementwiseLauncherConfig& config);
void moe_combine_known_window_rows_launch(std::span<const ExecutionDispatchContext> rows,
    const HipElementwiseLauncherConfig& config);
// Prefill: all T rows of an inject / MoE-combine step (8-row tiles one stride apart) in one launch.
void inject_known_prefill_rows_preflight(std::span<const ExecutionDispatchContext> rows,
    const HipElementwiseLauncherConfig& config);
void inject_known_prefill_rows_launch(std::span<const ExecutionDispatchContext> rows,
    const HipElementwiseLauncherConfig& config);
void moe_combine_known_prefill_rows_preflight(std::span<const ExecutionDispatchContext> rows,
    const HipElementwiseLauncherConfig& config);
void moe_combine_known_prefill_rows_launch(std::span<const ExecutionDispatchContext> rows,
    const HipElementwiseLauncherConfig& config);

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

// Source-only native MTP entry points; qualification is separate.
struct MtpNativeContext;
enum class MtpInjectRole : unsigned { attention, mlp };
void preflight_hip_mtp_inject(const MtpNativeContext&,MtpInjectRole);
void launch_hip_mtp_inject(const MtpNativeContext&,MtpInjectRole,void* stream);
// Multi-session draft batch (at most 8 contexts, on `stream`): every context's inject, and every context's
// publish_next_state copy (wide_next -> wide_current); each row equals its own one-token launch/copy.
void launch_hip_mtp_inject_rows(std::span<const MtpNativeContext> contexts,MtpInjectRole,void* stream);
void launch_hip_mtp_publish_next_rows(std::span<const MtpNativeContext> contexts,void* stream);
void preflight_hip_mtp_moe_combine(const MtpNativeContext&);
void launch_hip_mtp_moe_combine(const MtpNativeContext&,void* stream);

}  // namespace strix
