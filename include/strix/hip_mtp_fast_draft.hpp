#pragma once
#include <cstdint>
#include <span>
#include <memory>
#include <stdexcept>
#include "strix/model_runtime_binding.hpp"

namespace strix {
struct MtpQ4SourceIdentity {
    std::shared_ptr<const void> owner;
    std::uint64_t generation=0,elements=0;
    ModelContentIdentity content;
};
inline void require_same_mtp_q4_source(const MtpQ4SourceIdentity& retained,const MtpQ4SourceIdentity& requested) {
    if(!requested.owner || !requested.generation || !requested.content.valid() ||
       retained.owner.get()!=requested.owner.get() || retained.generation!=requested.generation ||
       retained.elements!=requested.elements || retained.content!=requested.content)
        throw std::invalid_argument("MTP Q4 source generation/content/owner changed");
}

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
// any_order drops the barrier bit: the caller guarantees independence from the kernel queued before.
void launch_hip_mtp_draft_q8_gemv(const void* input,bool input_bf16,const void* q8,std::uint32_t width,
    std::uint32_t rows,void* output,bool output_bf16,void* stream,bool any_order=false);
void launch_hip_mtp_fast_moe(const HipMtpFastMoeViews&,void* stream);
// Multi-session draft batch (STRIX_MTP_DRAFT_BATCH): at most this many tokens (one per session) per batched step.
inline constexpr std::uint32_t hip_mtp_draft_batch_rows=8U;
// Batched draft GEMV: up to three weight segments sharing one width (outputs first[s]..first[s+1]-1 of the
// launch are rows of weights[s], written to output[s][t] for token t). Each token's output equals
// launch_hip_mtp_draft_q8_gemv on that token's input alone, bit for bit (same Q4 copy rule, same chains).
struct HipMtpDraftGemvRows {
    const void* weights[3]{};
    std::uint32_t first[4]{};
    std::uint32_t segments=0;
    const void* input[hip_mtp_draft_batch_rows]{};
    void* output[3][hip_mtp_draft_batch_rows]{};
};
void launch_hip_mtp_draft_q8_gemv_rows(const HipMtpDraftGemvRows&,std::uint32_t count,bool input_bf16,std::uint32_t width,
    bool output_bf16,void* stream);
// Batched fast MoE: token t runs launch_hip_mtp_fast_moe's Q4 kernels on views[t] (same blocks, same bits).
struct HipMtpFastMoeRows {HipMtpFastMoeViews views[hip_mtp_draft_batch_rows];};
void launch_hip_mtp_fast_moe_rows(std::span<const HipMtpFastMoeViews> views,void* stream);
// Draft-only Q4_0 copy of a Q8_0 tensor (elements % 32 == 0), converted on `stream` with the
// draft head's rule and kept for the process. The draft GEMV, fast MoE and fused input
// projection read the copy whenever every tensor they need has one.
void register_hip_mtp_draft_q4(const void* q8,std::uint64_t elements,void* stream,
    std::shared_ptr<const void> source_owner={},std::uint64_t generation=0,ModelContentIdentity content={});
const void* hip_mtp_draft_q4(const void* q8) noexcept;
// Raw MTP HC mix (hidden 2560, 4 streams, rank 320, Q8 down/up, F32 norm): exact
// staged path (inject Q8 [4][10240]) or draft split-K/lane-chain path (inject F32
// [4][10240], 16-aligned). Null inject/injection for the head role. Workspace
// holds 10240+320 floats.
void enqueue_hip_mtp_hyper_mix_q8(const std::uint16_t* input,const float* norm,const void* down,const void* inject,const void* up,
    float* workspace,float* injection,std::uint16_t* output,float epsilon,bool draft,void* stream);
// Multi-session draft batch of enqueue_hip_mtp_hyper_mix_q8(..., draft=true, ...): row t is one session's draft HC
// mix (its input streams, workspace of 10240+320 floats, 16-aligned, injection or null with inject, output) and
// equals the one-token draft mix bit for bit. Production draft branch only (hip_mtp_hyper_mix_draft_rows_supported).
struct HipMtpHyperDraftRows {
    const std::uint16_t* input[hip_mtp_draft_batch_rows]{};
    std::uint16_t* output[hip_mtp_draft_batch_rows]{};
    float* workspace[hip_mtp_draft_batch_rows]{};
    float* injection[hip_mtp_draft_batch_rows]{};
};
void enqueue_hip_mtp_hyper_mix_q8_rows(const HipMtpHyperDraftRows&,std::uint32_t count,const float* norm,const void* down,
    const float* inject,const void* up,float epsilon,void* stream);
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
