#pragma once

#include "strix/weight/admission.hpp"
#include "strix/weight/formats/microscaled_matrix.hpp"

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

enum class MiniExpertProjection { gate_up, down };

struct MiniExpertMatrixView {
    MiniExpertProjection projection{};
    weight::microscaled::MatrixView matrix;
};

class MiniExpertMatrix {
public:
    weight::Status load(const MiniExpertMatrixView&);
    [[nodiscard]] std::size_t rows() const { return matrix_.rows(); }
    [[nodiscard]] std::size_t columns() const { return matrix_.columns(); }
    // The gate/up projection presents all gate rows followed by all up rows.
    weight::Status decode_row(std::size_t row, std::span<float> out) const;
private:
    MiniExpertProjection projection_{};
    weight::microscaled::StagedMatrix matrix_;
};

class MiniTextWeights {
public:
    weight::Status open(const std::string& path);
    weight::Status bind(weight::TensorIndex index);
    [[nodiscard]] const weight::RawTensor* find(std::string_view full_name) const;
    // Read and widen only the requested contiguous F32 or BF16 elements.
    weight::Status read(std::string_view full_name, std::span<float> out,
                        std::size_t first_element = 0) const;
    // Source-backed view of a single physical layer/expert/projection. Binding
    // and taking a view issue no weight-payload reads.
    weight::Status expert_matrix(unsigned physical, unsigned expert, MiniExpertProjection,
                                 MiniExpertMatrixView&) const;
private:
    std::vector<weight::RawTensor> tensors_;
};

}  // namespace strix::inkling2
