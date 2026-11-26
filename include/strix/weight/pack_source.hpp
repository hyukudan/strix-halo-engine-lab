#pragma once

#include "strix/weight/admission.hpp"
#include "strix/weight/formats/microscaled_matrix.hpp"

namespace strix::weight {

// One file, a checkpoint directory, or its safetensors index JSON. Opening
// indexes headers only; content identity is an explicit full checkpoint read.
class PackCheckpoint {
public:
    static Status open(const std::string& path, PackCheckpoint& out);
    [[nodiscard]] const std::vector<RawTensor>& tensors() const noexcept { return tensors_; }
    [[nodiscard]] const RawTensor* find(std::string_view name) const;
    Status digest(Digest256& out) const;

private:
    std::shared_ptr<const Source> manifest_;
    std::vector<std::shared_ptr<const Source>> shards_;
    std::vector<RawTensor> tensors_;  // sorted by name
};

struct PackMatrixSource {
    enum class Encoding { bf16, mxfp4, fp8_block, nvfp4, mxfp8 };
    Encoding encoding{};
    std::size_t rows{}, columns{};
    Slice codes, scales;
    Slice amax{}, denominator{};
    microscaled::RowOrder row_order{};

    static Status bind(const PackCheckpoint&, const RawTensor&, PackMatrixSource&);
    // When requested, hash the exact bytes already staged for this decode.
    Status decode(std::vector<float>& out, Digest256* digest = nullptr) const;
    Status digest(Digest256& out) const;
};

// Bind one rank-3 microscaled expert bank by its checkpoint plane names.
Status bind_pack_bank(const PackCheckpoint&, const RawTensor&, microscaled::RowOrder,
                      microscaled::ExpertSlabView&);
Status pack_expert(const microscaled::ExpertSlabView&, std::size_t expert, PackMatrixSource&);
std::string packed_expert_name(std::string_view bank, std::size_t expert);

}  // namespace strix::weight
