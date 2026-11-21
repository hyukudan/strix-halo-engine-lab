#include "inkling2_mini_resident_weights.hpp"

#include <stdexcept>

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
    std::vector<weight::ResidentSlicePlan::NamedTensor> tensors;
    tensors.reserve(schema.size());
    for (const auto& spec : schema) {
        if (!spec.resident) continue;
        tensors.emplace_back(std::string_view(spec.name).substr(kPrefix.size()), weights.find(spec.name));
    }
    plan.slices_ = weight::ResidentSlicePlan::make(std::move(tensors), kAlignment);
    for (unsigned layer = 0; layer < kPhysicalLayers; ++layer)
        for (unsigned i = 0; i < kProjections.size(); ++i)
            check(weights.expert_layer(layer, kProjections[i], plan.slabs_[layer][i]));
    return plan;
}

MiniResidentWeights MiniResidentWeights::bind(const MiniResidentWeightPlan& plan,
                                              std::span<const std::uint8_t> storage) {
    MiniResidentWeights bound(weight::ResidentSlices::bind(plan.slices_, storage));
    bound.experts_.resize(kPhysicalLayers * kExperts);
    for (unsigned layer = 0; layer < kPhysicalLayers; ++layer) {
        for (unsigned i = 0; i < kProjections.size(); ++i) {
            const auto projection = kProjections[i];
            const auto name = expert_name(layer, projection);
            std::array<weight::microscaled::PackedMatrix, kExperts> matrices;
            check(bound.slices_.bind_expert_slab(plan.slabs_[layer][i], name, name + "_scale", matrices));
            for (unsigned expert = 0; expert < kExperts; ++expert) {
                auto& pair = bound.experts_[layer * kExperts + expert];
                (projection == MiniExpertProjection::gate_up ? pair.gate_up : pair.down) = matrices[expert];
            }
        }
    }
    return bound;
}

std::span<const std::uint16_t> MiniResidentWeights::bf16(std::string_view name) const {
    return slices_.bf16(name);
}

std::span<const float> MiniResidentWeights::f32(std::string_view name) const {
    return slices_.f32(name);
}

std::span<const MiniMoeExpert> MiniResidentWeights::experts(unsigned physical) const {
    if (physical >= kPhysicalLayers || experts_.empty())
        throw std::invalid_argument("Inkling 2 Mini: invalid resident layer");
    return std::span(experts_).subspan(physical * kExperts, kExperts);
}

}  // namespace strix::inkling2
