#include "strix/weight/tensor_map.hpp"

#include <unordered_map>

namespace strix::weight::tensor_map {

std::uint64_t Model::resolve(Sym symbol, std::uint32_t layer) const noexcept {
    const auto index = static_cast<std::size_t>(symbol);
    if (index >= value.size()) return 0;
    if (symbol == Sym::one) return 1;
    if (symbol == Sym::two) return 2;
    const auto& layered = per_layer[index];
    if (!layered.empty()) return layer < layered.size() ? layered[layer] : 0;
    return value[index];
}

bool Model::in(Layers set, std::uint32_t layer) const noexcept {
    if (set == Layers::global) return false;
    if (layer >= layer_class.size()) return false;
    return (layer_class[layer] >> static_cast<std::uint32_t>(set) & 1U) != 0;
}

// Families with a naming table (src/weight/adapters/<family>.cpp), one accessor each.
using FamilyAccessor = const FamilyMap& (*)();
constexpr std::array<FamilyAccessor, 0> kFamilies{};

const FamilyMap* find_family(std::string_view architecture) {
    for (const auto accessor : kFamilies)
        if (const auto& family = accessor(); family.architecture == architecture) return &family;
    return nullptr;
}

std::string expand(std::string_view name, std::int64_t layer) {
    std::string out(name);
    const auto at = out.find("{L}");
    if (at != std::string::npos) out.replace(at, 3, std::to_string(layer));
    return out;
}

namespace {

std::string shape_text(std::span<const std::uint64_t> dims) {
    std::string text = "[";
    for (std::size_t i = 0; i < dims.size(); ++i) text += (i ? "," : "") + std::to_string(dims[i]);
    return text + "]";
}

// Returns false when a symbol of the entry is unresolved (0) or a product overflows.
bool resolve_dims(const Entry& entry, const Model& model, std::uint32_t layer, std::vector<std::uint64_t>& dims) {
    dims.clear();
    for (const auto& extent : entry.dims) {
        if (extent.a == Sym::none) break;
        const auto a = model.resolve(extent.a, layer), b = model.resolve(extent.b, layer);
        if (a == 0 || b == 0 || a > UINT64_MAX / b) return false;
        dims.push_back(a * b);
    }
    return !dims.empty();
}

}  // namespace

Result check(std::span<const gguf::TensorInfo> tensors, std::span<const Entry> table, const Model& model) {
    Result result;
    const auto error = [&](std::string message) {
        if (result.errors.size() < 64) result.errors.push_back(std::move(message));
    };
    struct Expected { std::size_t entry; std::int64_t layer; };
    std::unordered_map<std::string, Expected> expected;
    for (std::size_t e = 0; e < table.size(); ++e) {
        const auto& entry = table[e];
        const bool layered = entry.name.find("{L}") != std::string_view::npos;
        if (layered == (entry.layers == Layers::global)) {
            error("table entry " + std::string(entry.name) + ": layer template and layer class disagree");
            continue;
        }
        const auto add = [&](std::int64_t layer) {
            auto name = expand(entry.name, layer);
            if (!expected.emplace(name, Expected{e, layer}).second) error("claimed by two entries: " + name);
        };
        if (!layered) { add(-1); continue; }
        for (std::uint32_t layer = 0; layer < model.blocks; ++layer)
            if (model.in(entry.layers, layer)) add(layer);
    }
    std::unordered_map<std::string, bool> found;
    std::vector<std::uint64_t> dims;
    for (std::size_t t = 0; t < tensors.size(); ++t) {
        const auto& tensor = tensors[t];
        const auto it = expected.find(tensor.name);
        if (it == expected.end()) { error("unclaimed tensor: " + tensor.name); continue; }
        if (!found.emplace(tensor.name, true).second) { error("tensor listed twice: " + tensor.name); continue; }
        const auto& entry = table[it->second.entry];
        const auto layer = it->second.layer < 0 ? 0U : static_cast<std::uint32_t>(it->second.layer);
        if (!resolve_dims(entry, model, layer, dims)) {
            error("unresolved shape symbols for " + tensor.name);
        } else if (dims != tensor.dimensions) {
            error("shape " + shape_text(tensor.dimensions) + " != expected " + shape_text(dims) + ": " + tensor.name);
        }
        if ((entry.types & type_bit(tensor.ggml_type)) == 0)
            error("type " + std::to_string(tensor.ggml_type) + " not admitted for " + tensor.name);
        result.claims.push_back(Claim{t, it->second.entry, it->second.layer});
    }
    for (const auto& [name, where] : expected)
        if (!found.count(name)) error("missing tensor: " + name);
    if (!result.ok()) result.claims.clear();
    return result;
}

}  // namespace strix::weight::tensor_map
