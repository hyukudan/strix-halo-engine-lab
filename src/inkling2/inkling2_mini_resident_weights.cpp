#include "inkling2_mini_resident_weights.hpp"
#include "inkling2_mini_attention_prepare.hpp"

#include <optional>
#include <stdexcept>
#include <string>

namespace strix::inkling2 {
namespace {

constexpr std::array kProjections{MiniExpertProjection::gate_up, MiniExpertProjection::down};

void check(weight::Status status) {
    if (!status) throw std::runtime_error("Inkling 2 Mini resident weights: " + status.detail);
}

using AttentionRole = MiniAttentionProjection;
constexpr unsigned kAttentionRoles = attn_core::PreparedGqaWeights::role_count;

// The attention projection role of a schema tensor ('layers.<n>.<projection suffix>'), if it is one.
std::optional<AttentionRole> attention_role(std::string_view name) {
    constexpr std::string_view layers = "layers.";
    if (!name.starts_with(layers)) return std::nullopt;
    const auto dot = name.find('.', layers.size());
    if (dot == std::string_view::npos || dot == layers.size()) return std::nullopt;
    const auto suffix = name.substr(dot + 1U);
    for (unsigned i = 0; i < kAttentionRoles; ++i) {
        const auto role = static_cast<AttentionRole>(i);
        if (mini_attention_projection_spec(role).tensor_suffix == suffix) return role;
    }
    return std::nullopt;
}

// Whether the compact image of `set` stores this schema tensor as Q8_0.
bool compact_tensor(MiniCompactSet set, const MiniTextTensorSpec& spec, std::string_view name) {
    if (set == MiniCompactSet::attention_head) return spec.load_matrix_role.has_value();
    const auto role = attention_role(name);
    return role && mini_compact_role(set, *role);
}

}  // namespace

MiniCompactSet mini_compact_set(weight::Id policy) {
    if (weight::same_id(policy, weight::kAttentionHeadQ8Policy.id)) return MiniCompactSet::attention_head;
    if (weight::same_id(policy, weight::kAttentionQ8Policy.id)) return MiniCompactSet::attention;
    throw std::invalid_argument("Inkling 2 Mini does not bind weight-load policy " + std::string(policy.name));
}

MiniResidentWeightPlan MiniResidentWeightPlan::make(const MiniTextWeights& weights, weight::FormatId storage,
                                                    weight::RowPitchRule rule, MiniCompactSet set) {
    if (storage != weight::kBf16WeightStorage && storage != weight::kQ8WeightStorage)
        throw std::invalid_argument("invalid Inkling 2 Mini resident weight storage");
    if (set != MiniCompactSet::attention_head && set != MiniCompactSet::attention)
        throw std::invalid_argument("invalid Inkling 2 Mini compact set");
    if (!weight::valid_row_pitch_rule(rule)) throw std::invalid_argument("Inkling 2 Mini: invalid resident row pitch rule");
    const auto output_pitch = weight::pitched_row_bytes(kQueryWidth * sizeof(std::uint16_t), rule);
    MiniResidentWeightPlan plan;
    plan.storage_ = storage;
    plan.set_ = set;
    const auto schema = mini_text_tensor_schema();
    std::vector<weight::ResidentSlicePlan::NamedTensor> tensors;
    tensors.reserve(schema.size());
    for (const auto& spec : schema) {
        if (!spec.resident) continue;
        const auto name = std::string_view(spec.name).substr(kMiniTextTensorPrefix.size());
        const auto compact = storage == weight::kQ8WeightStorage && compact_tensor(set, spec, name);
        const auto pitch = !compact && spec.load_matrix_role == weight::Role::attention_o ? output_pitch : 0U;
        tensors.push_back({name, weights.find(spec.name), compact ? std::optional(storage) : std::nullopt, pitch});
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
    bound.compact_ = plan.storage_ == weight::kQ8WeightStorage;
    if (bound.compact_ && plan.set_ == MiniCompactSet::attention_head)
        bound.head_ = bound.slices_.matrix("unembed.weight");
    bound.experts_.resize(kPhysicalLayers * kExperts);
    for (unsigned layer = 0; layer < kPhysicalLayers; ++layer) {
        if (bound.compact()) {
            auto& attention = bound.attention_[layer];
            const auto prefix = "layers." + std::to_string(layer) + ".";
            for (unsigned i = 0; i < attention.matrix.size(); ++i) {
                const auto role = static_cast<AttentionRole>(i);
                if (!is_full_attention(layer) && (role == AttentionRole::key_taps || role == AttentionRole::value_taps))
                    continue;
                const auto name = prefix + std::string(mini_attention_projection_spec(role).tensor_suffix);
                if (mini_compact_role(plan.set_, role)) attention.matrix[i] = bound.slices_.matrix(name);
                else attention.matrix[i] = bound.bf16(name).data();
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

weight::ResidentBf16Rows MiniResidentWeights::bf16_rows(std::string_view name) const {
    return slices_.bf16_rows(name);
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
