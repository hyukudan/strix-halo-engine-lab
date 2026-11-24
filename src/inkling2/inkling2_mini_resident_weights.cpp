#include "inkling2_mini_resident_weights.hpp"
#include "inkling2_mini_attention_prepare.hpp"

#include <deque>
#include <stdexcept>
#include <string>

namespace strix::inkling2 {
namespace {

constexpr std::array kProjections{MiniExpertProjection::gate_up, MiniExpertProjection::down};

void check(weight::Status status) {
    if (!status) throw std::runtime_error("Inkling 2 Mini resident weights: " + status.detail);
}

}  // namespace

MiniResidentWeightPlan MiniResidentWeightPlan::make(const MiniTextWeights& weights, weight::RoleStorage storage,
                                                    weight::RowPitchRule rule, const weight::LowbitPack* pack,
                                                    bool keep_bf16) {
    const auto stored = [](weight::FormatId format) {
        return format == weight::kBf16WeightStorage || format == weight::kQ8WeightStorage ||
               format == weight::kQ4G64WeightStorage || format == weight::kQ6G64WeightStorage;
    };
    if (!stored(storage.attention) || !stored(storage.head))
        throw std::invalid_argument("invalid Inkling 2 Mini resident weight storage");
    if (!weight::valid_row_pitch_rule(rule)) throw std::invalid_argument("Inkling 2 Mini: invalid resident row pitch rule");
    const auto output_pitch = weight::pitched_row_bytes(kQueryWidth * sizeof(std::uint16_t), rule);
    MiniResidentWeightPlan plan;
    plan.storage_ = storage;
    plan.keep_bf16_ = keep_bf16 && (storage.attention != weight::kBf16WeightStorage || storage.head != weight::kBf16WeightStorage);
    const auto schema = mini_text_tensor_schema();
    std::deque<std::string> copy_names;  // the plan copies names; these outlive make() only as long as needed
    std::vector<weight::ResidentSlicePlan::NamedTensor> tensors;
    tensors.reserve(schema.size());
    for (const auto& spec : schema) {
        if (!spec.resident) continue;
        const auto role_storage = spec.load_matrix_role == weight::Role::head ? storage.head : storage.attention;
        const auto compact = role_storage != weight::kBf16WeightStorage && spec.load_matrix_role;
        const auto pitch = !compact && spec.load_matrix_role == weight::Role::attention_o ? output_pitch : 0U;
        const auto* raw = weights.find(spec.name);
        std::optional<weight::Slice> prepacked;
        if (pack && compact && raw &&
            (role_storage == weight::kQ4G64WeightStorage || role_storage == weight::kQ6G64WeightStorage)) {
            check(pack->find(spec.name, role_storage, *raw, prepacked));
            if (prepacked) ++plan.prepacked_;
        }
        const auto name = std::string_view(spec.name).substr(kMiniTextTensorPrefix.size());
        if (compact && plan.keep_bf16_) {
            const auto bf16_pitch = spec.load_matrix_role == weight::Role::attention_o ? output_pitch : 0U;
            tensors.push_back({name, raw, std::nullopt, bf16_pitch, std::nullopt});
            tensors.push_back({copy_names.emplace_back(std::string(kMiniLowBitImage) + std::string(name)), raw,
                               role_storage, 0U, std::move(prepacked)});
            continue;
        }
        tensors.push_back({name, raw, compact ? std::optional(role_storage) : std::nullopt, pitch, std::move(prepacked)});
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
    const auto compact_name = [&](std::string_view name) {
        return plan.keep_bf16_ ? std::string(kMiniLowBitImage) + std::string(name) : std::string(name);
    };
    if (plan.storage_.head != weight::kBf16WeightStorage) bound.head_ = bound.slices_.matrix(compact_name("unembed.weight"));
    bound.compact_attention_ = plan.storage_.attention != weight::kBf16WeightStorage;
    bound.bf16_kept_ = plan.keep_bf16_;
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
                        attention.matrix[i] = bound.slices_.matrix(compact_name(name));
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
