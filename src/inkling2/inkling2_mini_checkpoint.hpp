#pragma once

#include "strix/weight/admission.hpp"

namespace strix::inkling2 {

inline constexpr std::string_view kMiniCheckpointRevision = "ed40e7da625f594fc9c4990f5f66be24873c22d7";
inline constexpr std::string_view kMiniCheckpointShard = "model-00001-of-00001.safetensors";

struct MiniTextTensorSpec {
    std::string name;
    std::string_view wire_dtype;
    std::vector<std::uint64_t> extents;
};

struct MiniCheckpointSummary {
    std::size_t text_tensors{};
    std::size_t vision_tensors{};
    std::uint64_t text_bytes{};
};

// Shape and dtype contract for the published target text weights, in file dimension order.
std::vector<MiniTextTensorSpec> mini_text_tensor_schema();

// These checks only use the header parsed by the shared weight reader.
weight::Status validate_mini_text_checkpoint(const weight::TensorIndex&, MiniCheckpointSummary&);
weight::Status validate_mini_checkpoint_index(const weight::TensorIndex&, std::string_view index_json);

}  // namespace strix::inkling2
