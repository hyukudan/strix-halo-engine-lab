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
// Draft-only Q8_0 GEMV y[rows]=W[rows][width]*x, one wave per row (lanes own
// 8-column groups, then a wave tree). Input BF16 or F32, output F32 or BF16 RNE.
// width must be a multiple of 32.
void launch_hip_mtp_draft_q8_gemv(const void* input,bool input_bf16,const void* q8,std::uint32_t width,
    std::uint32_t rows,void* output,bool output_bf16,void* stream);
void launch_hip_mtp_fast_moe(const HipMtpFastMoeViews&,void* stream);
// Raw MTP HC mix (hidden 2560, 4 streams, rank 320, Q8 down/up, F32 norm): exact
// staged path (inject Q8 [4][10240]) or draft split-K/lane-chain path (inject F32
// [4][10240], 16-aligned). Null inject/injection for the head role. Workspace
// holds 10240+320 floats.
void enqueue_hip_mtp_hyper_mix_q8(const std::uint16_t* input,const float* norm,const void* down,const void* inject,const void* up,
    float* workspace,float* injection,std::uint16_t* output,float epsilon,bool draft,void* stream);
// Flat Q8_0 -> F32 (elements % 32 == 0): out[i]=scale*code, the Q8 loader value.
void launch_hip_mtp_dequantize_q8(const void* q8,std::uint64_t elements,float* output,void* stream);
// Raw exact-path sequences behind the MTP routed/shared launchers (no owner
// checks), shared with the draft benchmark/fixture. Routed temp needs 3840 floats,
// shared workspace 1280 floats.
void enqueue_hip_mtp_routed_experts_q8(const std::uint16_t* input,const std::uint32_t* routes,const float* route_weights,
    const void* gate,const void* up,const void* down,float* temp,std::uint32_t* status,std::uint16_t* output,void* stream);
void enqueue_hip_mtp_shared_expert_q8(const std::uint16_t* input,const void* gate,const void* up,const void* down,
    const void* scalar_gate,float* workspace,std::uint16_t* output,void* stream);
} // namespace strix
