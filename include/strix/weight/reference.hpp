#pragma once
#include "registry.hpp"
namespace strix::weight {
Status signed_group_reference(const Registry&, const WeightView&, std::uint64_t,
                              SignedGroup32&);
}
