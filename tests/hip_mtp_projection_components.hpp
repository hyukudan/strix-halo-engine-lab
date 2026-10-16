#pragma once
#include "strix/hip_mtp_projection.hpp"
#include <array>
#include <cstddef>
#include <string>
#include <vector>
namespace strix::testing {
struct MtpHyperComponentReference {
    MtpHyperRole role=MtpHyperRole::attention;
    std::vector<std::uint16_t> input,output;
    std::vector<float> injection;
    std::vector<std::byte> workspace;
};
struct MtpRoutedComponentReference {
    std::vector<std::uint16_t> input,output;
    std::array<std::uint32_t,10> routes{};
    std::array<float,10> route_weights{};
    std::vector<std::byte> workspace;
};
struct MtpProjectionComponentReference {
    // Full golden payloads must come from the independently bound oracle;
    // repeated native execution alone is never a mathematical reference.
    ModelContentIdentity independent_reference_content;
    std::uint64_t proposal_identity=0,position=0;
    std::string hyper_profile,routed_profile;
    std::array<MtpHyperComponentReference,3> hyper;
    MtpRoutedComponentReference routed;
};
// Callable qualification component, no main/automatic CTest or duplicate model
// loader. Its caller supplies actual owner and immutable independent goldens.
void qualify_hip_mtp_projection_components(std::shared_ptr<HipMtpResources>,
    const MtpProjectionComponentReference&);
} // namespace strix::testing
