#include "inkling2_mini_resident_weights.hpp"
#include "inkling2_mini_attention_prepare.hpp"

#include <algorithm>
#include <set>
#include <stdexcept>
#include <string>
#include <string_view>
#include <vector>

namespace strix::inkling2 {
namespace {

constexpr std::array kProjections{MiniExpertProjection::gate_up, MiniExpertProjection::down};

void check(weight::Status status) {
    if (!status) throw std::runtime_error("Inkling 2 Mini resident weights: " + status.detail);
}

}  // namespace

MiniResidentWeightPlan MiniResidentWeightPlan::make(const MiniTextWeights& weights, weight::FormatId storage,
                                                    weight::RowPitchRule rule, bool narrow_experts) {
    if (storage != weight::kBf16WeightStorage && storage != weight::kQ8WeightStorage)
        throw std::invalid_argument("invalid Inkling 2 Mini resident weight storage");
    if (!weight::valid_row_pitch_rule(rule)) throw std::invalid_argument("Inkling 2 Mini: invalid resident row pitch rule");
    const auto output_pitch = weight::pitched_row_bytes(kQueryWidth * sizeof(std::uint16_t), rule);
    MiniResidentWeightPlan plan;
    plan.storage_ = storage;
    const auto schema = mini_text_tensor_schema();
    plan.narrow_experts_ = narrow_experts;
    std::set<std::string, std::less<>> narrowed_tensors;
    if (narrow_experts)
        for (unsigned layer = 0; layer < kPhysicalLayers; ++layer)
            if (narrowed(layer))
                for (const auto projection : kProjections) {
                    const auto names = mini_expert_slab_names(layer, projection);
                    narrowed_tensors.emplace(names.codes);
                    narrowed_tensors.emplace(names.block_scales);
                }
    std::vector<weight::ResidentSlicePlan::NamedTensor> tensors;
    tensors.reserve(schema.size());
    for (const auto& spec : schema) {
        if (!spec.resident || narrowed_tensors.contains(spec.name)) continue;
        const auto compact = storage == weight::kQ8WeightStorage && spec.load_matrix_role;
        const auto pitch = !compact && spec.load_matrix_role == weight::Role::attention_o ? output_pitch : 0U;
        tensors.push_back({std::string_view(spec.name).substr(kMiniTextTensorPrefix.size()), weights.find(spec.name),
                           compact ? std::optional(storage) : std::nullopt, pitch});
    }
    plan.slices_ = weight::ResidentSlicePlan::make(std::move(tensors), kAlignment);
    for (unsigned layer = 0; layer < kPhysicalLayers; ++layer)
        for (unsigned i = 0; i < kProjections.size(); ++i)
            check(weights.expert_layer(layer, kProjections[i], plan.slabs_[layer][i]));
    if (narrow_experts) {
        // NVFP4 bytes of each narrowed slab: codes then scales, each on the plan's alignment.
        const auto align = [](std::size_t bytes) { return (bytes + kAlignment - 1U) / kAlignment * kAlignment; };
        for (unsigned layer = 0; layer < kPhysicalLayers; ++layer) {
            if (!narrowed(layer)) continue;
            for (unsigned i = 0; i < kProjections.size(); ++i) {
                const auto& slab = plan.slabs_[layer][i];
                const auto elements = slab.experts * slab.rows * slab.columns;
                auto& range = plan.narrow_ranges_[layer][i];
                range.codes = plan.narrow_bytes_;
                range.scales = range.codes + align(elements / 2U);
                plan.narrow_bytes_ = range.scales + align(elements / weight::microscaled::kNvfp4BlockColumns);
            }
        }
    }
    return plan;
}

MiniResidentWeights MiniResidentWeights::bind(const MiniResidentWeightPlan& plan,
                                              std::span<const std::uint8_t> storage,
                                              const MiniReadControllerBits& read_controller_bits) {
    MiniResidentWeights bound(weight::ResidentSlices::bind(plan.slices_, storage));
    bound.narrow_ = plan.narrow_experts_;
    if (read_controller_bits)
        bound.certify_controller(read_controller_bits);
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
        if (plan.narrow_experts_ && MiniResidentWeightPlan::narrowed(layer)) continue;  // attach_narrow binds these
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

MiniAttentionCertificate certify_mini_attention(core::Bf16StagedScan& scan, const MiniStagedLookup& staged,
                                                const MiniStagedRead& read) {
    // No scan scratch: the empty (refusing) certificate, three planes; the checkpoint still loads.
    if (!scan.available()) return {};
    constexpr std::array roles{MiniAttentionProjection::query, MiniAttentionProjection::key,
                               MiniAttentionProjection::value, MiniAttentionProjection::key_taps,
                               MiniAttentionProjection::value_taps, MiniAttentionProjection::relative,
                               MiniAttentionProjection::relative_bias, MiniAttentionProjection::sink,
                               MiniAttentionProjection::output_gate, MiniAttentionProjection::output};
    MiniAttentionCertificate certificate;
    for (const auto& spec : mini_text_tensor_schema()) {
        const std::string_view name(spec.name);
        if (!spec.resident || name.find(".layers.") == std::string_view::npos) continue;
        // Every layer norm gain (attn_norm, q/k/v input norms, ...): a superset of the attention input gains.
        const bool gain = name.ends_with("norm.weight");
        const bool projection = std::ranges::any_of(roles, [&](MiniAttentionProjection role) {
            return name.ends_with(mini_attention_projection_spec(role).tensor_suffix);
        }) || name.ends_with("attn.k_sconv.weight") || name.ends_with("attn.v_sconv.weight");  // static channel taps
        if (!gain && !projection) continue;
        const auto tensor = staged(name.substr(kMiniTextTensorPrefix.size()));
        const float top = scan.magnitude_max(tensor.size(), [&](std::size_t offset, std::span<std::uint16_t> host) {
            read(tensor.subspan(offset, host.size()), host);
        });
        if (gain) certificate.add_gain(top);
        else certificate.add_projection(top);
    }
    return certificate;
}

void MiniResidentWeights::certify_attention(core::Bf16StagedScan& scan, const MiniStagedRead& read) {
    if (compact()) return;
    attention_certificate_ = certify_mini_attention(scan, [this](std::string_view name) { return bf16(name); }, read);
}

void MiniResidentWeights::attach_narrow(unsigned physical, MiniExpertProjection projection,
                                        const weight::microscaled::Nvfp4Slab& slab, std::span<const std::uint8_t> codes,
                                        std::span<const std::uint8_t> scales) {
    if (!narrow_ || physical >= kPhysicalLayers || !MiniResidentWeightPlan::narrowed(physical))
        throw std::invalid_argument("Inkling 2 Mini: narrow expert slab for a layer the image does not narrow");
    std::array<weight::microscaled::PackedMatrix, kExperts> matrices;
    check(slab.bind_storage(codes, scales, matrices));
    for (unsigned expert = 0; expert < kExperts; ++expert) {
        auto& pair = experts_[physical * kExperts + expert];
        (projection == MiniExpertProjection::gate_up ? pair.gate_up : pair.down) = matrices[expert];
    }
}

MiniControllerCertificate read_mini_controller_certificate(const weight::ResidentSlices& slices,
                                                          const MiniReadControllerBits& read) {
    MiniControllerCertificate certificate;
    std::vector<std::uint16_t> host;
    for (const auto& spec : mini_text_tensor_schema()) {
        const std::string_view name(spec.name);
        if (name.find(".controllers.") == std::string_view::npos) continue;
        const bool down = name.ends_with("down_weight");
        if (!down && !name.ends_with("forget_weight") && !name.ends_with("gain_weight")) continue;
        const auto staged = slices.bf16(name.substr(kMiniTextTensorPrefix.size()));
        host.resize(staged.size());
        read(staged, host);
        if (down) certificate.add_down(host);
        else certificate.add_gate(host);
    }
    return certificate;
}

void MiniResidentWeights::certify_controller(const MiniReadControllerBits& read) {
    controller_certificate_ = read_mini_controller_certificate(slices_, read);
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
