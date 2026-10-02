#pragma once

#include "strix/model_runtime_binding.hpp"

#include <cstdint>
#include <filesystem>
#include <vector>

namespace strix {

struct SerialDecodeReferenceLogit {
    std::uint32_t index = 0;
    float value = 0.0F;
};

struct SerialDecodeReferenceStep {
    std::uint32_t input_token = 0;
    std::uint32_t expected_argmax_token = 0;
    std::vector<SerialDecodeReferenceLogit> logits;
};

struct SerialDecodeReference {
    ModelContentIdentity model_content_identity;
    std::uint64_t independent_reference_identity = 0;
    std::vector<SerialDecodeReferenceStep> steps;
};

[[nodiscard]] SerialDecodeReference
read_serial_decode_reference(
    const std::filesystem::path& path);

void validate_serial_decode_reference_vocabulary(
    const SerialDecodeReference& reference,
    std::uint64_t vocab_size);

}  // namespace strix
