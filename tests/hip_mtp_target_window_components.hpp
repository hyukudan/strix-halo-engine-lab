#pragma once
#include "strix/hip_known_prefix_window.hpp"
#include "strix/hip_mtp_target_window.hpp"
#include <vector>
namespace strix::testing {
struct MtpTargetWindowReferenceChunk {
    std::vector<std::uint32_t> inputs,predictions;
    std::vector<std::vector<std::uint16_t>> wide_hidden;
};
struct MtpTargetWindowReference {
    ModelContentIdentity independent_reference_content,target_content;
    std::uint64_t initial_position=0;
    std::vector<MtpTargetWindowReferenceChunk> chunks;
};
// Actual native window/session, independently bound complete row snapshots.
// No target/MTP model loader duplication or fabricated intermediate sessions.
void qualify_hip_mtp_target_window_capture(HipKnownPrefixWindow&,SerialSessionContext&,
    const MtpTargetWindowReference&,HipMtpTargetWindowCapturePolicy);
} // namespace strix::testing
