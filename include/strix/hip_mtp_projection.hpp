#pragma once
#include "strix/hip_mtp_resources.hpp"
#include <string_view>
namespace strix {
inline constexpr std::string_view mtp_hyper_q8_profile_v1 =
    "mtp-hc-r1-staged128-ordered-f32-q8inject-bf16-v1";
inline constexpr std::string_view mtp_routed_q8_profile_v1 =
    "mtp-routed-q8-r1-tree256-slotascending-f32-bf16-v1";
enum class MtpHyperRole : unsigned { attention, mlp, head };
// Native sidecar tensor roles only; no serial IDs/session metadata synthesized.
// Pure all-view/tensor/alias admission, owner remains retained by caller.
void preflight_hip_mtp_hyper_mix(const MtpNativeContext&,MtpHyperRole);
void launch_hip_mtp_hyper_mix(const MtpNativeContext&,MtpHyperRole,void* stream);
void preflight_hip_mtp_routed_experts(const MtpNativeContext&);
// All-Q8 gate/up/down, original ascending-slot FP32 accumulation, final BF16.
// Uses owner's sticky status (never clears it); owner completes/checks before
// publication. No host synchronization or route-index readback is performed.
void launch_hip_mtp_routed_experts(const MtpNativeContext&,void* stream);
} // namespace strix
