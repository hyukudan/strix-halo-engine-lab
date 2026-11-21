#include "strix/weight/tensor_access.hpp"
#include "strix/weight/typed_slice.hpp"

#include <algorithm>
#include <limits>
#include <map>
#include <utility>

namespace strix::weight {
namespace {

bool ignored(const RawTensor& tensor, const TensorSchema& schema) {
    return !schema.ignored_prefix.empty() && tensor.name.starts_with(schema.ignored_prefix);
}

}  // namespace

Status validate_tensor_schema(const TensorIndex& index, const TensorSchema& schema, TensorSchemaSummary& out) {
    if (index.container != schema.container || index.schema != schema.version)
        return {Error::unknown_format, "expected tensor schema " + std::string(schema.version)};

    TensorSchemaSummary summary;
    std::map<std::string_view, const RawTensor*> remaining;
    for (const auto& tensor : index.tensors) {
        if (ignored(tensor, schema)) {
            ++summary.ignored_tensors;
        } else if (!remaining.emplace(tensor.name, &tensor).second) {
            return {Error::malformed, "duplicate tensor " + tensor.name};
        }
    }
    for (const auto& spec : schema.tensors) {
        const auto it = remaining.find(spec.name);
        if (it == remaining.end()) return {Error::malformed, "missing tensor " + spec.name};
        const auto& tensor = *it->second;
        if (tensor.wire_dtype != spec.wire_dtype)
            return {Error::unknown_format, "unexpected dtype " + tensor.wire_dtype + " for " + spec.name};
        if (tensor.extents != spec.extents)
            return {Error::unsupported_shape, "unexpected shape for " + spec.name};
        if (tensor.payload.range.bytes > std::numeric_limits<std::uint64_t>::max() - summary.bytes)
            return {Error::malformed, "tensor payload total overflows"};
        summary.bytes += tensor.payload.range.bytes;
        ++summary.tensors;
        remaining.erase(it);
    }
    if (!remaining.empty())
        return {Error::malformed, "unexpected tensor " + std::string(remaining.begin()->first)};
    out = summary;
    return {Error::ok, {}};
}

Status TensorLookup::open_safetensors(const std::string& path, const TensorSchema& schema) {
    std::shared_ptr<const Source> source;
    if (const auto status = open_source(path, source); !status) return status;
    TensorIndex index;
    if (const auto status = index_safetensors(std::move(source), index); !status) return status;
    return bind(std::move(index), schema);
}

Status TensorLookup::bind(TensorIndex index, const TensorSchema& schema) {
    TensorSchemaSummary summary;
    if (const auto status = validate_tensor_schema(index, schema, summary); !status) return status;
    std::erase_if(index.tensors, [&](const RawTensor& tensor) { return ignored(tensor, schema); });
    std::ranges::sort(index.tensors, {}, &RawTensor::name);
    tensors_ = std::move(index.tensors);
    return {Error::ok, {}};
}

const RawTensor* TensorLookup::find(std::string_view name) const {
    const auto it = std::ranges::lower_bound(tensors_, name, {}, &RawTensor::name);
    return it != tensors_.end() && it->name == name ? &*it : nullptr;
}

Status TensorLookup::slice(std::string_view name, Slice& out) const {
    const auto* tensor = find(name);
    if (!tensor) return {Error::malformed, "unknown weight " + std::string(name)};
    return subslice(tensor->payload, {0, tensor->payload.range.bytes}, out);
}

Status TensorLookup::read_floats(std::string_view name, std::span<float> out, std::size_t first) const {
    const auto* tensor = find(name);
    if (!tensor) return {Error::malformed, "unknown weight " + std::string(name)};
    return read_float_slice(*tensor, out, first);
}

}  // namespace strix::weight
