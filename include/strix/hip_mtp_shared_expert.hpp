#pragma once
#include "strix/hip_mtp_resources.hpp"
#include <string_view>
namespace strix {
inline constexpr std::string_view mtp_shared_q8_profile_v1=
    "mtp-shared-q8-tree256-f32-scalar-r1-bf16-v1";
// Actual sidecar Q8 gate/up/down and F32 scalar; hidden0 -> hidden1.
// The admitted owner moe_workspace is reused sequentially on ONE stream.
// No fabricated serial contexts, source IDs, allocation or synchronization.
void preflight_hip_mtp_shared_expert(const MtpNativeContext&);
void launch_hip_mtp_shared_expert(const MtpNativeContext&,void* stream);
} // namespace strix
