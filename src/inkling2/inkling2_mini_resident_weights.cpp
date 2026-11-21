#include "inkling2_mini_resident_weights.hpp"

#include <algorithm>
#include <limits>
#include <stdexcept>
#include <utility>

namespace strix::inkling2 {
namespace {

constexpr std::string_view kPrefix = "model.llm.";

constexpr std::array kProjections{MiniExpertProjection::gate_up, MiniExpertProjection::down};

std::string expert_name(unsigned layer, MiniExpertProjection projection) {
    return "layers." + std::to_string(layer) + ".mlp.experts." +
           (projection == MiniExpertProjection::gate_up ? "w13_weight" : "w2_weight");
}

void check(weight::Status status) {
    if (!status) throw std::runtime_error("Inkling 2 Mini resident weights: " + status.detail);
}

}  // namespace

MiniResidentWeightPlan MiniResidentWeightPlan::make(const MiniTextWeights& weights) {
    MiniResidentWeightPlan plan;
    const auto schema = mini_text_tensor_schema();
    std::vector<std::pair<std::string_view, const weight::RawTensor*>> tensors;
    tensors.reserve(schema.size());
    for (const auto& spec : schema) {
        if (spec.name.ends_with("_amax") || spec.name.ends_with("_outer_scale_denominator")) continue;
        const auto* raw = weights.find(spec.name);
        if (!raw || !raw->payload.source || raw->payload.range.bytes == 0U)
            throw std::invalid_argument("Inkling 2 Mini: missing resident tensor " + spec.name);
        tensors.emplace_back(std::string_view(spec.name).substr(kPrefix.size()), raw);
    }
    // Standard checkpoints are a single shard. Read in its file order.
    std::ranges::sort(tensors, {}, [](const auto& tensor) { return tensor.second->payload.range.offset; });
    plan.ranges_.reserve(tensors.size());
    for (const auto& [name, raw] : tensors) {
        constexpr auto limit = std::numeric_limits<std::size_t>::max();
        if (plan.bytes_ > limit - (kAlignment - 1U))
            throw std::overflow_error("Inkling 2 Mini: resident alignment exceeds address space");
        const auto offset = (plan.bytes_ + kAlignment - 1U) & ~(kAlignment - 1U);
        if (raw->payload.range.bytes > limit - offset)
            throw std::overflow_error("Inkling 2 Mini: resident weights exceed address space");
        const auto bytes = static_cast<std::size_t>(raw->payload.range.bytes);
        const auto dtype = raw->wire_dtype == "BF16" ? Dtype::bf16 :
                           raw->wire_dtype == "F32" ? Dtype::f32 : Dtype::bytes;
        plan.ranges_.push_back({raw->payload, offset, bytes});
        plan.tensors_.emplace(std::string(name), Tensor{offset, bytes, dtype});
        plan.bytes_ = offset + bytes;
    }
    for (unsigned layer = 0; layer < kPhysicalLayers; ++layer)
        for (unsigned i = 0; i < kProjections.size(); ++i)
            check(weights.expert_layer(layer, kProjections[i], plan.slabs_[layer][i]));
    return plan;
}

MiniResidentWeights MiniResidentWeights::bind(const MiniResidentWeightPlan& plan,
                                              std::span<const std::uint8_t> storage) {
    if (plan.bytes() == 0U || storage.size() < plan.bytes() ||
        reinterpret_cast<std::uintptr_t>(storage.data()) % MiniResidentWeightPlan::kAlignment != 0U)
        throw std::invalid_argument("Inkling 2 Mini: incompatible resident weight storage");
    MiniResidentWeights bound;
    bound.storage_ = storage.first(plan.bytes());
    bound.tensors_ = plan.tensors_;
    bound.experts_.resize(kPhysicalLayers * kExperts);
    for (unsigned layer = 0; layer < kPhysicalLayers; ++layer) {
        for (unsigned i = 0; i < kProjections.size(); ++i) {
            const auto projection = kProjections[i];
            const auto name = expert_name(layer, projection);
            using Dtype = MiniResidentWeightPlan::Dtype;
            const auto codes = bound.tensor(name, Dtype::bytes);
            const auto scales = bound.tensor(name + "_scale", Dtype::bytes);
            std::array<weight::microscaled::PackedMatrix, kExperts> matrices;
            check(plan.slabs_[layer][i].bind_storage(codes, scales, matrices));
            for (unsigned expert = 0; expert < kExperts; ++expert) {
                auto& pair = bound.experts_[layer * kExperts + expert];
                (projection == MiniExpertProjection::gate_up ? pair.gate_up : pair.down) = matrices[expert];
            }
        }
    }
    return bound;
}

std::span<const std::uint8_t> MiniResidentWeights::tensor(
    std::string_view name, MiniResidentWeightPlan::Dtype dtype) const {
    const auto it = tensors_.find(name);
    if (it == tensors_.end() || it->second.dtype != dtype)
        throw std::invalid_argument("Inkling 2 Mini: incompatible resident tensor " + std::string(name));
    return storage_.subspan(it->second.offset, it->second.bytes);
}

std::span<const std::uint16_t> MiniResidentWeights::bf16(std::string_view name) const {
    const auto bytes = tensor(name, MiniResidentWeightPlan::Dtype::bf16);
    return {reinterpret_cast<const std::uint16_t*>(bytes.data()), bytes.size() / sizeof(std::uint16_t)};
}

std::span<const float> MiniResidentWeights::f32(std::string_view name) const {
    const auto bytes = tensor(name, MiniResidentWeightPlan::Dtype::f32);
    return {reinterpret_cast<const float*>(bytes.data()), bytes.size() / sizeof(float)};
}

std::span<const MiniMoeExpert> MiniResidentWeights::experts(unsigned physical) const {
    if (physical >= kPhysicalLayers || experts_.empty())
        throw std::invalid_argument("Inkling 2 Mini: invalid resident layer");
    return std::span(experts_).subspan(physical * kExperts, kExperts);
}

}  // namespace strix::inkling2
