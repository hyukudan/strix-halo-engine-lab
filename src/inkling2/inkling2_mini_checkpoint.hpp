#pragma once

#include "inkling2_mini_config.hpp"
#include "strix/weight/formats/microscaled_matrix.hpp"
#include "strix/weight/tensor_access.hpp"

namespace strix::inkling2 {

inline constexpr std::string_view kMiniCheckpointRevision = "ed40e7da625f594fc9c4990f5f66be24873c22d7";
inline constexpr std::string_view kMiniCheckpointShard = "model-00001-of-00001.safetensors";
inline constexpr std::string_view kMiniTextTensorPrefix = "model.llm.";

using MiniTextTensorSpec = weight::TensorSpec;

struct MiniCheckpointSummary {
    std::size_t text_tensors{};
    std::size_t vision_tensors{};
    std::uint64_t text_bytes{};
};

// Shape and dtype contract for the published target text weights, in file dimension order.
std::vector<MiniTextTensorSpec> mini_text_tensor_schema();
weight::microscaled::ExpertSlabNames mini_expert_slab_names(unsigned physical, MiniExpertProjection);

// These checks only use the header parsed by the shared weight reader.
weight::Status validate_mini_text_checkpoint(const weight::TensorIndex&, MiniCheckpointSummary&);
weight::Status validate_mini_checkpoint_index(const weight::TensorIndex&, std::string_view index_json);

using MiniExpertLayerView = weight::microscaled::ExpertSlabView;

class MiniTextWeights {
public:
    weight::Status open(const std::string& path);
    weight::Status bind(weight::TensorIndex index);
    [[nodiscard]] const weight::RawTensor* find(std::string_view full_name) const {
        return tensors_.find(full_name);
    }
    // Read and widen only the requested contiguous F32 or BF16 elements.
    weight::Status read(std::string_view full_name, std::span<float> out, std::size_t first_element = 0) const {
        return tensors_.read_floats(full_name, out, first_element);
    }
    // All experts of one physical projection. Queuing these slices avoids one
    // allocation and upload per expert and lets all logical visits share them.
    weight::Status expert_layer(unsigned physical, MiniExpertProjection, MiniExpertLayerView&) const;
    // Source-backed view of a single physical layer/expert/projection. Binding
    // and taking a view issue no weight-payload reads.
    weight::Status expert_matrix(unsigned physical, unsigned expert, MiniExpertProjection,
                                 weight::microscaled::MatrixView&) const;
private:
    weight::TensorLookup tensors_;
};

}  // namespace strix::inkling2
