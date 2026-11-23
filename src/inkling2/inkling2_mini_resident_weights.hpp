#pragma once

#include "inkling2_mini_attention_prepare.hpp"
#include "inkling2_mini_checkpoint.hpp"
#include "inkling2_mini_moe_dispatch.hpp"
#include "attn_core/prepared_gqa.hpp"

#include "strix/weight/levers.hpp"
#include "strix/weight/resident_slices.hpp"
#include "strix/weight/row_pitch.hpp"

#include <utility>

namespace strix::inkling2 {

using MiniResidentRange = weight::ResidentSliceRange;

// The matrices a compact (Q8_0) resident image covers. The resolved weight-load policy selects it
// (mini_compact_set); BF16 storage ignores it.
enum class MiniCompactSet : std::uint8_t {
    // weights.attention_head.q8_0@1: query, key, value and output projections and the head.
    attention_head,
    // weights.attention.q8_0@1: every attention input projection (query, key, value, output, output gate,
    // relative, sink and the full-layer KV taps); the head and the relative-bias table stay BF16.
    attention,
};

// The attention projection roles the set stores as Q8_0.
[[nodiscard]] constexpr bool mini_compact_role(MiniCompactSet set, MiniAttentionProjection role) {
    using Role = MiniAttentionProjection;
    if (role == Role::query || role == Role::key || role == Role::value || role == Role::output) return true;
    return set == MiniCompactSet::attention && role != Role::relative_bias;
}

// The set of a resolved weight-load policy id; throws for a policy this model does not bind.
[[nodiscard]] MiniCompactSet mini_compact_set(weight::Id policy);

// The resident image a weight-load plan selects. A bare storage id names the incumbent attention_head set.
struct MiniWeightImage {
    weight::FormatId storage = weight::kBf16WeightStorage;
    MiniCompactSet set = MiniCompactSet::attention_head;
    MiniWeightImage() = default;
    MiniWeightImage(weight::FormatId format) : storage(format) {}  // NOLINT(google-explicit-constructor)
    MiniWeightImage(weight::FormatId format, MiniCompactSet compact) : storage(format), set(compact) {}
};

// BF16 storage keeps the incumbent set; a compact plan takes the set of its policy.
[[nodiscard]] inline MiniWeightImage mini_weight_image(const weight::WeightLoadPlan& plan) {
    if (weight::same_id(plan.storage, weight::kBf16WeightStorage)) return {plan.storage};
    return {plan.storage, mini_compact_set(plan.policy)};
}

class MiniResidentWeightPlan {
public:
    static constexpr std::size_t kAlignment = 256U;

    // The input has already passed the Mini text schema. NVFP4 host-only
    // sidecars are fetched separately when the resident expert views bind.
    // rule: channel-safe pitch for the BF16 attention-output rows (0 = packed); Q8_0 compact rows stay packed.
    [[nodiscard]] static MiniResidentWeightPlan make(
        const MiniTextWeights& weights, weight::FormatId storage = weight::kBf16WeightStorage,
        weight::RowPitchRule rule = {}, MiniCompactSet set = MiniCompactSet::attention_head);
    [[nodiscard]] MiniCompactSet compact_set() const { return set_; }
    [[nodiscard]] const weight::ResidentSlicePlan& slices() const { return slices_; }
    [[nodiscard]] std::size_t bytes() const { return slices_.bytes(); }
    [[nodiscard]] std::span<const MiniResidentRange> ranges() const { return slices_.ranges(); }

private:
    friend class MiniResidentWeights;
    weight::ResidentSlicePlan slices_;
    weight::FormatId storage_ = weight::kBf16WeightStorage;
    MiniCompactSet set_ = MiniCompactSet::attention_head;
    std::array<std::array<MiniExpertLayerView, 2>, kPhysicalLayers> slabs_;
};

// Borrow immutable resident bytes. Once bound, the views are independent of
// the checkpoint file and valid for as long as the storage is allocated.
class MiniResidentWeights {
public:
    [[nodiscard]] static MiniResidentWeights bind(const MiniResidentWeightPlan& plan,
                                                  std::span<const std::uint8_t> storage);
    [[nodiscard]] std::span<const std::uint16_t> bf16(std::string_view name) const;
    [[nodiscard]] weight::ResidentBf16Rows bf16_rows(std::string_view name) const;
    [[nodiscard]] std::span<const float> f32(std::string_view name) const;
    [[nodiscard]] std::span<const MiniMoeExpert> experts(unsigned physical) const;
    // True when the attention projections are resident Q8_0 (either compact set).
    [[nodiscard]] bool compact() const { return compact_; }
    // True only for the attention_head set: the head is Q8_0. Otherwise the head stays BF16 (full or screened).
    [[nodiscard]] bool has_compact_head() const { return head_.has_value(); }
    [[nodiscard]] const weight::ResidentMatrixView& compact_head() const { return head_.value(); }
    [[nodiscard]] const attn_core::ResidentPreparedGqaWeights& compact_attention(unsigned physical) const {
        return attention_.at(physical);
    }

private:
    explicit MiniResidentWeights(weight::ResidentSlices slices) : slices_(std::move(slices)) {}

    weight::ResidentSlices slices_;
    std::vector<MiniMoeExpert> experts_;
    bool compact_{};
    std::optional<weight::ResidentMatrixView> head_;
    std::array<attn_core::ResidentPreparedGqaWeights, kPhysicalLayers> attention_{};
};

}  // namespace strix::inkling2
