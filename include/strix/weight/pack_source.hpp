#pragma once

#include "strix/weight/admission.hpp"

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
    enum class Encoding { bf16, mxfp4, fp8_block };
    Encoding encoding{};
    std::size_t rows{}, columns{};
    Slice codes, scales;

    static Status bind(const PackCheckpoint&, const RawTensor&, PackMatrixSource&);
    // When requested, hash the exact bytes already staged for this decode.
    Status decode(std::vector<float>& out, Digest256* digest = nullptr) const;
    Status digest(Digest256& out) const;
};

}  // namespace strix::weight
