#pragma once
#include <cstdint>

namespace strix {
// Draft-only MTP MoE for the canonical Q8_0 sidecar (hidden 2560, intermediate
// 640, 512 experts, top-10 plus shared expert). Two launches replace the
// per-slot generic kernels: (1) gate+up+SiLU for the 10 routed slots and the
// shared expert (plus the shared sigmoid gate) in one grid, wave per row;
// (2) grouped down over all 11 slots, route/shared-gate weighted, written once
// as BF16 into the MoE output. Reduction order differs from the exact path, so
// only draft acceptance can change; replay and teacher rows keep exact kernels.
struct HipMtpFastMoeViews {
    const std::uint16_t* input=nullptr;
    const std::uint32_t* routes=nullptr;
    const float* route_weights=nullptr;
    const void *gate=nullptr,*up=nullptr,*down=nullptr;
    const void *shared_gate=nullptr,*shared_up=nullptr,*shared_down=nullptr;
    const float* shared_scalar=nullptr;
    float* workspace=nullptr;          // hip_mtp_fast_moe_workspace_floats
    std::uint16_t* output=nullptr;     // BF16[2560], may alias input
    std::uint32_t* status=nullptr;     // first-error word; invalid routes set bit 0
};
inline constexpr std::uint32_t hip_mtp_fast_moe_workspace_floats=11U*640U+1U;
void launch_hip_mtp_fast_moe(const HipMtpFastMoeViews&,void* stream);
// Raw exact-path sequences behind the MTP routed/shared launchers (no owner
// checks), shared with the draft benchmark/fixture. Routed temp needs 3840 floats,
// shared workspace 1280 floats.
void enqueue_hip_mtp_routed_experts_q8(const std::uint16_t* input,const std::uint32_t* routes,const float* route_weights,
    const void* gate,const void* up,const void* down,float* temp,std::uint32_t* status,std::uint16_t* output,void* stream);
void enqueue_hip_mtp_shared_expert_q8(const std::uint16_t* input,const void* gate,const void* up,const void* down,
    const void* scalar_gate,float* workspace,std::uint16_t* output,void* stream);
} // namespace strix
