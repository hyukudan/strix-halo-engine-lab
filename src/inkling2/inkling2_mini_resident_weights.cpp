#include "inkling2_mini_resident_weights.hpp"
#include "inkling2_mini_attention_prepare.hpp"

#include <algorithm>
#include <stdexcept>
#include <string_view>

namespace strix::inkling2 {
namespace {

constexpr std::array kProjections{MiniExpertProjection::gate_up, MiniExpertProjection::down};

void check(weight::Status status) {
    if (!status) throw std::runtime_error("Inkling 2 Mini resident weights: " + status.detail);
}

}  // namespace

MiniResidentWeightPlan MiniResidentWeightPlan::make(const MiniTextWeights& weights, weight::FormatId storage,
                                                    weight::RowPitchRule rule) {
    if (storage != weight::kBf16WeightStorage && storage != weight::kQ8WeightStorage)
        throw std::invalid_argument("invalid Inkling 2 Mini resident weight storage");
    if (!weight::valid_row_pitch_rule(rule)) throw std::invalid_argument("Inkling 2 Mini: invalid resident row pitch rule");
    const auto output_pitch = weight::pitched_row_bytes(kQueryWidth * sizeof(std::uint16_t), rule);
    MiniResidentWeightPlan plan;
    plan.storage_ = storage;
    const auto schema = mini_text_tensor_schema();
    std::vector<weight::ResidentSlicePlan::NamedTensor> tensors;
    tensors.reserve(schema.size());
    for (const auto& spec : schema) {
        if (!spec.resident) continue;
        const auto compact = storage == weight::kQ8WeightStorage && spec.load_matrix_role;
        const auto pitch = !compact && spec.load_matrix_role == weight::Role::attention_o ? output_pitch : 0U;
        tensors.push_back({std::string_view(spec.name).substr(kMiniTextTensorPrefix.size()), weights.find(spec.name),
                           compact ? std::optional(storage) : std::nullopt, pitch});
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

void MiniResidentWeights::certify_attention(
    const std::function<void(std::span<const std::uint16_t>, std::span<std::uint16_t>)>& read) {
    if (compact()) return;
    constexpr std::array roles{MiniAttentionProjection::query, MiniAttentionProjection::key,
                               MiniAttentionProjection::value, MiniAttentionProjection::key_taps,
                               MiniAttentionProjection::value_taps, MiniAttentionProjection::relative,
                               MiniAttentionProjection::relative_bias, MiniAttentionProjection::sink,
                               MiniAttentionProjection::output_gate, MiniAttentionProjection::output};
    MiniAttentionCertificate certificate;
    std::vector<std::uint16_t> host;
    for (const auto& spec : mini_text_tensor_schema()) {
        const std::string_view name(spec.name);
        if (!spec.resident || name.find(".layers.") == std::string_view::npos) continue;
        // Every layer norm gain (attn_norm, q/k/v input norms, ...): a superset of the attention input gains.
        const bool gain = name.ends_with("norm.weight");
        const bool projection = std::ranges::any_of(roles, [&](MiniAttentionProjection role) {
            return name.ends_with(mini_attention_projection_spec(role).tensor_suffix);
        }) || name.ends_with("attn.k_sconv.weight") || name.ends_with("attn.v_sconv.weight");  // static channel taps
        if (!gain && !projection) continue;
        const auto staged = bf16(name.substr(kMiniTextTensorPrefix.size()));
        host.resize(staged.size());
        read(staged, host);
        if (gain) certificate.add_gain(host);
        else certificate.add_projection(host);
    }
    attention_certificate_ = certificate;
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
