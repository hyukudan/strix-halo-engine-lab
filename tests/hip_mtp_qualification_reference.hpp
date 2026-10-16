#pragma once
#include "hip_mtp_greedy_cycle_components.hpp"
#include "hip_mtp_projection_components.hpp"
#include "hip_mtp_shared_expert_components.hpp"
#include "hip_mtp_target_window_components.hpp"
#include "hip_mtp_lm_head_components.hpp"
#include <filesystem>
namespace strix::tests {
enum class MtpQualificationKind:std::uint32_t {cycles=1,history_fallback=2,projection=3,shared=4,window=5,lm_head=6,checkpoint=7};
struct MtpQualificationReference {
    MtpQualificationKind kind=MtpQualificationKind::cycles;
    ModelContentIdentity independent_source_content,target_content,sidecar_content;
    std::uint64_t independent_reference_identity=0,proposal_identity=0,plan_identity=0,semantic_identity=0,native_profile_identity=0;
    HipMtpGreedyCycleGolden cycles;
    testing::MtpProjectionComponentReference projection;
    testing::MtpSharedComponentReference shared;
    testing::MtpTargetWindowReference window;
    HipMtpLmHeadGolden head;
};
// Bounded complete-file SHA pinning binds supplied bytes, not oracle authenticity.
// Encoding never computes expected model values. Only the selected payload is
// serialized; fixed-width little-endian v2 has no native-struct serialization.
[[nodiscard]] std::vector<std::byte> encode_mtp_qualification_reference(const MtpQualificationReference&,std::uint64_t maximum_bytes);
[[nodiscard]] MtpQualificationReference read_mtp_qualification_reference(const std::filesystem::path&,
    const ModelContentIdentity& expected_file_sha256,std::uint64_t maximum_bytes);
} // namespace strix::tests
