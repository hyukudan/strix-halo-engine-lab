#pragma once
#include "strix/model_contract.hpp"

// No HIP API and no table-sized allocation: an opaque CPU allocation with the
// exact advertised extent. Used only by compatibility and lifetime tests.
inline std::shared_ptr<const strix::HipQwenRotaryTable> mock_rotary(unsigned factor,
    unsigned char digest = 1) {
    strix::ModelContentIdentity sha; sha.bytes[0] = digest;
    const auto descriptor = strix::RotaryDescriptor::yarn(factor);
    auto storage = std::make_shared<int>(42);
    return std::make_shared<strix::HipQwenRotaryTable>(
        strix::RotaryIdentity(descriptor, sha), storage, descriptor.table_bytes());
}
