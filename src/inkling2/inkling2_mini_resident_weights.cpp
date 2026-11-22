#include "inkling2_mini_resident_weights.hpp"
#include "inkling2_mini_attention_prepare.hpp"

#include <stdexcept>

namespace strix::inkling2 {
namespace {

constexpr std::array kProjections{MiniExpertProjection::gate_up, MiniExpertProjection::down};

void check(weight::Status status) {
    if (!status) throw std::runtime_error("Inkling 2 Mini resident weights: " + status.detail);
}

}  // namespace

MiniResidentWeightPlan MiniResidentWeightPlan::make(const MiniTextWeights& weights, weight::FormatId storage) {
    if (storage != weight::kBf16WeightStorage && storage != weight::kQ8WeightStorage)
        throw std::invalid_argument("invalid Inkling 2 Mini resident weight storage");
    MiniResidentWeightPlan plan;
    plan.storage_ = storage;
    const auto schema = mini_text_tensor_schema();
    std::vector<weight::ResidentSlicePlan::NamedTensor> tensors;
    tensors.reserve(schema.size());
    for (const auto& spec : schema) {
        if (!spec.resident) continue;
        const auto compact = storage == weight::kQ8WeightStorage && spec.load_matrix_role;
        tensors.push_back({std::string_view(spec.name).substr(kMiniTextTensorPrefix.size()), weights.find(spec.name),
                           compact ? std::optional(storage) : std::nullopt});
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
    if (plan.storage_ == weight::kQ8WeightStorage)
        bound.head_ = bound.slices_.matrix("unembed.weight");
    bound.experts_.resize(kPhysicalLayers * kExperts);
    for (unsigned layer = 0; layer < kPhysicalLayers; ++layer) {
        if (bound.compact()) {
            auto& attention = bound.attention_[layer];
            const auto prefix = "layers." + std::to_string(layer) + ".";
            using Role = MiniAttentionProjection;
            for (unsigned i = 0; i < attention.matrix.size(); ++i) {
                const auto role = static_cast<Role>(i);
                if (!is_full_attention(layer) && (role == Role::key_taps || role == Role::value_taps)) continue;
                const auto name = prefix + std::string(mini_attention_projection_spec(role).tensor_suffix);
                switch (role) {
                    case Role::query: case Role::key: case Role::value: case Role::output:
                        attention.matrix[i] = bound.slices_.matrix(name);
                        break;
                    default:
                        attention.matrix[i] = bound.bf16(name).data();
                }
            }
            attention.query_gain = bound.bf16(prefix + "attn.q_l2_clip.scale").data();
            attention.relative_gain = bound.bf16(prefix + "attn.rel_l2_clip.scale").data();
        }
        for (unsigned i = 0; i < kProjections.size(); ++i) {
            const auto projection = kProjections[i];
            const auto names = mini_expert_slab_names(layer, projection);
            const auto codes = std::string_view(names.codes).substr(kMiniTextTensorPrefix.size());
            const auto scales = std::string_view(names.block_scales).substr(kMiniTextTensorPrefix.size());
            std::array<weight::microscaled::PackedMatrix, kExperts> matrices;
            check(bound.slices_.bind_expert_slab(plan.slabs_[layer][i], codes, scales, matrices));
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
