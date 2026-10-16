#pragma once
#include "strix/hip_mtp_shared_expert.hpp"
#include <cstddef>
#include <string>
#include <vector>
namespace strix::testing {
struct MtpSharedComponentReference {
    ModelContentIdentity independent_reference_content;
    std::uint64_t proposal_identity=0,position=0;
    std::string numerical_profile;
    std::vector<std::uint16_t> input,output;
    std::vector<std::byte> workspace;
};
// Callable only with actual retained native owner and independent full goldens.
// No standalone test/model loader or self-generated expected outputs.
void qualify_hip_mtp_shared_expert_component(std::shared_ptr<HipMtpResources>,const MtpSharedComponentReference&);
} // namespace strix::testing
