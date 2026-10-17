#pragma once

#include "strix/execution_binding.hpp"
#include "strix/hip_runtime_launchers.hpp"
#include "strix/launch_dispatch.hpp"

#include <cstdint>
#include <string>

namespace strix {

struct HipLmHeadLauncherConfig {
    std::uint32_t hidden_size = 0;
    std::uint32_t vocab_size = 0;
    HipActivationStorage input =
        HipActivationStorage::bf16;
    bool virtual_q8_wave = false;
    // Explicit eager BF16/Q8_0 candidates; wave8 also requires prefetch.
    bool prefetch_q8_wave = false;
    bool wave8_q8_wave = false;
};

void register_hip_lm_head_kernel_variant(
    const ModelContract& model,
    const ExecutionBindingRequest& request,
    const HipLmHeadLauncherConfig& config,
    kernels::KernelRegistry& registry,
    kernels::QualificationStatus qualification =
        kernels::QualificationStatus::experimental);

[[nodiscard]] ExecutionLauncher
make_hip_lm_head_launcher(
    std::string kernel_id,
    HipLmHeadLauncherConfig* config);

class HipLmHeadLauncherBundle {
public:
    explicit HipLmHeadLauncherBundle(
        HipLmHeadLauncherConfig config)
        : config_(config) {}

    HipLmHeadLauncherBundle(
        const HipLmHeadLauncherBundle&) = delete;
    HipLmHeadLauncherBundle& operator=(
        const HipLmHeadLauncherBundle&) = delete;
    HipLmHeadLauncherBundle(
        HipLmHeadLauncherBundle&&) = delete;
    HipLmHeadLauncherBundle& operator=(
        HipLmHeadLauncherBundle&&) = delete;

    void register_for_compiled_execution(
        const CompiledSerialExecution& compiled,
        ExecutionLauncherRegistry& registry);

private:
    HipLmHeadLauncherConfig config_;
};

// Source-only native MTP entry points; qualification is separate.
struct MtpNativeContext;
struct MtpTensorView;
enum class HipMtpLmHeadRecipe : std::uint32_t {
    generic256=0,
    prefetch8_wave8_q8=1,
};
// Pure host recipe admission: no HIP calls or qualification. Unsupported fast
// geometry/format uses the original generic native MTP head. Actual launches
// still require the full retained-owner context and dynamic buffer checks.
[[nodiscard]] HipMtpLmHeadRecipe select_hip_mtp_lm_head_recipe(const ModelContract&,const MtpTensorView&);
[[nodiscard]] std::uint64_t hip_mtp_lm_head_profile_identity(const ModelContract&,const MtpTensorView&);
void preflight_hip_mtp_lm_head(const MtpNativeContext&);
void launch_hip_mtp_lm_head(const MtpNativeContext&,void* stream);
// Raw MTP draft head over Q8_0 rows [rows][hidden]: the prefetch8 wave8 recipe
// above, so each row's logit is bit-identical to the same row of the full head.
void launch_hip_q8_head_rows(const std::uint16_t* hidden,const void* q8_rows,std::uint32_t hidden_size,
    std::uint32_t rows,float* logits,void* stream);

}  // namespace strix
