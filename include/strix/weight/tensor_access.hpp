#pragma once

#include "admission.hpp"

#include <optional>

namespace strix::weight {

// Exact wire schema, in file dimension order. Other tensors are rejected
// unless they are in the explicitly ignored namespace.
struct TensorSpec {
    std::string name;
    std::string_view wire_dtype;
    std::vector<std::uint64_t> extents;
    bool resident = true;
    // An opted-in BF16 matrix role for a family load plan. Other tensors load as stored.
    std::optional<Role> load_matrix_role{};
};

struct TensorSchema {
    Container container;
    std::string_view version;
    std::span<const TensorSpec> tensors;
    std::string_view ignored_prefix;
};

struct TensorSchemaSummary {
    std::size_t tensors{};
    std::size_t ignored_tensors{};
    std::uint64_t bytes{};
};

// Header-only validation. Summary and bound lookup are published on success.
Status validate_tensor_schema(const TensorIndex&, const TensorSchema&, TensorSchemaSummary&);

class TensorLookup {
public:
    Status open_safetensors(const std::string& path, const TensorSchema&);
    Status open_safetensors(std::shared_ptr<const Source>, const TensorSchema&);
    Status bind(TensorIndex index, const TensorSchema&);
    [[nodiscard]] const RawTensor* find(std::string_view name) const;

    // Return an owned source slice without reading the payload.
    Status slice(std::string_view name, Slice&) const;
    // Read only the requested contiguous elements of a little-endian F32 or BF16 tensor.
    Status read_floats(std::string_view name, std::span<float> out, std::size_t first = 0) const;

private:
    std::vector<RawTensor> tensors_;
};

}  // namespace strix::weight
