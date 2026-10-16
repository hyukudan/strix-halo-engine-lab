#pragma once
#include "strix/hip_lm_head_launcher.hpp"
#include "strix/hip_mtp_resources.hpp"
#include <vector>
namespace strix::tests {
struct HipMtpLmHeadGolden {
    ModelContentIdentity independent_source_content,target_content,sidecar_content;
    std::uint64_t independent_reference_identity=0,plan_identity=0,position=0,head_profile_identity=0;
    HipMtpLmHeadRecipe recipe=HipMtpLmHeadRecipe::generic256;
    std::vector<std::uint16_t> input_bf16;
    std::vector<float> full_logits;
};
struct HipMtpLmHeadComponentRequest {
    ModelContentIdentity expected_payload;
    std::uint64_t host_payload_budget_bytes=0;
};
struct HipMtpLmHeadComponentReceipt {
    HipMtpLmHeadRecipe recipe=HipMtpLmHeadRecipe::generic256;
    std::uint64_t profile_identity=0,outputs_compared=0;
    ModelContentIdentity reference_payload,actual_full_logits;
};
[[nodiscard]] ModelContentIdentity hip_mtp_lm_head_golden_identity(const HipMtpLmHeadGolden&);
// Real retained native owner, explicit recipe, independent complete vocabulary.
// No self-oracle, generic/fast equivalence or runtime qualification is asserted.
[[nodiscard]] HipMtpLmHeadComponentReceipt qualify_hip_mtp_lm_head_component(
    std::shared_ptr<HipMtpResources>,const HipMtpLmHeadGolden&,const HipMtpLmHeadComponentRequest&);
} // namespace strix::tests
