#pragma once
#include "strix/weight/resolve.hpp"
#include "source.hpp"
namespace strix::weight::hip {
struct SourceRegistration {
    FormatId storage;
    std::string_view protocol,arithmetic;
    std::uint32_t group_k,alignment,staged_k,paired_k;
    bool input_transform,output_transform;
    Status (*validate)(const WeightView&,const LaunchShape&);
};
std::span<const SourceRegistration> sources() noexcept;
// Explicit immutable table; caller retains the incumbent launch and numeric Site policy.
const SourceRegistration* find_source(FormatId,std::string_view protocol) noexcept;
}  // namespace strix::weight::hip
