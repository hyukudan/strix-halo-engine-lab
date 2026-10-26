#pragma once
#include "strix/mtp_forward_semantics.hpp"
#include <cstdint>
#include <span>

namespace strix {
struct HipMtpInputFusionView {
    std::uintptr_t address=0;
    std::uint64_t bytes=0,allocation_generation=0;
};
enum class HipMtpInputStorage : std::uint8_t { unresolved, bf16, fp32 };
enum class HipMtpInputRounding : std::uint8_t { unresolved, fp32, bf16_rne };
struct HipMtpInputFusionRequest {
    std::uint32_t hidden_size=0,hc_count=0;
    float epsilon=0.0F;
    HipMtpInputStorage embedding_storage=HipMtpInputStorage::unresolved;
    HipMtpInputStorage hidden_storage=HipMtpInputStorage::unresolved;
    HipMtpInputStorage output_storage=HipMtpInputStorage::unresolved;
    HipMtpInputRounding normalized_rounding=HipMtpInputRounding::unresolved;
    HipMtpInputRounding projection_rounding=HipMtpInputRounding::unresolved;
    MtpNormWeightEncoding norm_weight_encoding=MtpNormWeightEncoding::effective_scale;
    // Caller must resolve checkpoint conversion metadata; no default inference.
    bool norm_encoding_admitted=false;
    MtpInputFusionArithmetic arithmetic=MtpInputFusionArithmetic::split_projections_then_add;
    HipMtpInputFusionView embedding,wide_hidden,embedding_norm,hidden_norm,packed_projection;
    std::uint64_t projection_row_stride_bytes=0;
    // Each scratch is F32[H+HC*H], even when values have explicit BF16 rounding.
    HipMtpInputFusionView normalized_scratch,projection_scratch,output;
};
// Pure checked payload geometry. Scope: H multiple32 within32..4096, HC2/4.
[[nodiscard]] std::uint64_t hip_mtp_input_fusion_scratch_bytes(std::uint32_t hidden,std::uint32_t hc);
// Metadata/extent/overflow/alignment/alias checks ONLY, no HIP calls or writes.
void preflight_hip_mtp_input_fusion(const HipMtpInputFusionRequest&);
// Draft-only variant: same norms, then one fused wave-per-row projection and
// broadcast add (different summation order; replay/teacher rows keep the exact one).
void launch_hip_mtp_input_fusion_draft(const HipMtpInputFusionRequest&,void* stream);
// Multi-session draft batch: per-session addresses of the views that differ between sessions.
struct HipMtpInputFusionRows {
    std::uintptr_t embedding[8]{},wide_hidden[8]{},normalized[8]{},output[8]{};
};
// Batched launch_hip_mtp_input_fusion_draft (production branch: Q4 copy, HC 4): token t's output equals the
// one-token draft launcher on requests[t] alone, bit for bit. Requests may differ only in their per-session
// views (embedding, wide hidden, scratch, output); at most 8 requests.
void launch_hip_mtp_input_fusion_draft_rows(std::span<const HipMtpInputFusionRequest> requests,void* stream);
// Includes geometry, norm encoding, reduction and all declared cast boundaries.
// An identity is not readiness, checkpoint equivalence or native qualification.
[[nodiscard]] std::uint64_t hip_mtp_input_fusion_profile_identity(const HipMtpInputFusionRequest&);
// Three kernels: two whole-vector RMS reductions; H embedding +HC*H hidden
// projection jobs with separate ascending FP32 chains; broadcast-add/output.
// Normalized and projection casts occur BEFORE the next stage, not implicitly.
// Input weights are original tight Q8_0[2H,H]; each column half uses the FULL
// original row stride. No packed single-chain substitution is admitted here.
// Explicit caller-owned stream, eager/noncapture, owner lifetime through drain.
// No allocation, synchronization, readiness activation or host publication.
void launch_hip_mtp_input_fusion(const HipMtpInputFusionRequest&,void* stream);
} // namespace strix
