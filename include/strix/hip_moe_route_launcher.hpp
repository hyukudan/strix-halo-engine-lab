#pragma once

#include "strix/execution_binding.hpp"
#include "strix/hip_runtime_launchers.hpp"
#include "strix/launch_dispatch.hpp"

#include <array>
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
    // fast-prefill-f16wmma-v1, router component: prefill tiles' logits in BF16 WMMA (hip_router_fast_prefill.inc).
    bool router_fast_prefill = false;
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
void moe_route_known_window_cohort_launch(std::span<const std::span<const ExecutionDispatchContext>> segments,
    const HipMoeRouteLauncherConfig& config);
// Prefill: all T rows of a router step (8-row tiles one stride apart).
void moe_route_known_prefill_rows_preflight(std::span<const ExecutionDispatchContext> rows,
    const HipMoeRouteLauncherConfig& config);
void moe_route_known_prefill_rows_launch(std::span<const ExecutionDispatchContext> rows,
    const HipMoeRouteLauncherConfig& config);
// Bit-identity self-test + bench of the prefill selection forms (STRIX_PREFILL_ROUTER_SELECT=1 wave-per-row kernel vs the
// block-per-row kernel) on generated logits (flavor 0 uniform, 1 ties, 2 NaN payloads, 3 +-Inf/+-0/denormals/max, 4 degenerate
// rows, 5 exp underflow). Returns {rows, rows differing, old us/call, new us/call} (bench only when bench_reps > 0).
// Bit-identity self-test + bench of the decode staged router logits with activations staged in LDS (STRIX_ROUTER_STAGE_ACT=1)
// vs the incumbent staged kernel: E512 x 2560 F32 weights, rows 1/2/4/8 (BF16 activations; float for <= 2 rows), flavor 1 adds
// NaN/Inf weights and -Inf activations. Returns {logits, logits differing, old us/call, new us/call} (bench when bench_reps > 0).
std::array<double, 4> router_staged_logits_selftest(std::uint32_t rows_count, bool float_activation, std::uint32_t flavor,
    std::uint32_t seed, std::uint32_t bench_reps);
std::array<double, 4> router_prefill_select_selftest(std::uint32_t flavor, std::uint32_t tiles, std::uint32_t experts,
    std::uint32_t top_k, bool normalize, std::uint32_t seed, std::uint32_t bench_reps);

// Source-only native MTP entry points; qualification is separate.
struct MtpNativeContext;
void preflight_hip_mtp_route(const MtpNativeContext&);
void launch_hip_mtp_route(const MtpNativeContext&,void* stream);
// Multi-session draft batch of launch_hip_mtp_route (staged production branch): each context's routes and
// weights equal its own launch, bit for bit; at most 8 contexts, on `stream` (ordered by the caller).
[[nodiscard]] bool hip_mtp_route_rows_supported();
void launch_hip_mtp_route_rows(std::span<const MtpNativeContext> contexts,void* stream);

}  // namespace strix
