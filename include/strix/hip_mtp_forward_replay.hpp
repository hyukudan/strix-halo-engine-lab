#pragma once
#include "strix/hip_mtp_runtime.hpp"
#include "strix/mtp_forward_replay.hpp"
#include <span>
#include <vector>
namespace strix {
struct HipMtpForwardGoldenRow {
    std::uint64_t source_position=0;
    std::uint32_t source_input_token=0,next_input_token=0,selected_token=0;
    std::vector<std::uint16_t> target_hidden_bf16,next_hidden_bf16;
    std::vector<float> full_logits;
};
struct HipMtpForwardGolden {
    ModelContentIdentity independent_source_content,target_content,sidecar_content;
    std::uint64_t independent_reference_identity=0,structural_plan_identity=0,semantic_identity=0,numerical_profile_identity=0;
    std::vector<HipMtpForwardGoldenRow> rows;
};
// Framed SHA256 of supplied metadata and ALL source/next hidden/logit bytes.
// This binds payloads, not the truth/independence of a supplied reference source.
[[nodiscard]] ModelContentIdentity hip_mtp_forward_golden_payload_identity(const HipMtpForwardGolden&);
struct HipMtpForwardReplayRequest {
    MtpForwardReplayPolicy policy;
    ModelContentIdentity expected_golden_payload;
    std::uint64_t maximum_rows=0,host_payload_budget_bytes=0;
};
struct HipMtpForwardReplayRowReceipt {
    std::uint64_t source_position=0,seed_position=0;
    std::uint32_t selected_token=0,completed_steps=0;
    ModelContentIdentity actual_hidden,actual_logits;
};
struct HipMtpNativeForwardReplayResult {
    MtpForwardReplayObservation observation;
    MtpForwardReplayAssessment assessment;
    ModelContentIdentity golden_payload;
    std::uint64_t semantic_identity=0,numerical_profile_identity=0;
    std::vector<HipMtpForwardReplayRowReceipt> rows;
    bool fresh_forward_only=true;
    // No qualification/activation object is minted by this diagnostic consumer.
};
// Real candidate execution with independent full-value goldens. Initially fresh
// predictor frontier0 only; no resumed/rejection/reconciliation claim. Every
// actual completed target ticket and golden row is admitted before native work.
// ALL target hidden inputs are copied/checked against goldens before predictor
// mutation. Capture buffers and receipt capacity are reserved before work.
// Native output views are copied immediately after completion, before reuse.
// A reference is externally supplied/provenance-pinned, never the native run.
// Mismatch returns an assessment under the declared policy; transfer/protocol
// failures drain/poison the actual native owner and publish no healthy result.
[[nodiscard]] HipMtpNativeForwardReplayResult replay_hip_mtp_forward(
    HipMtpRuntime&,std::span<const HipMtpReplayRow>,const HipMtpForwardGolden&,
    const HipMtpForwardReplayRequest&);
} // namespace strix
