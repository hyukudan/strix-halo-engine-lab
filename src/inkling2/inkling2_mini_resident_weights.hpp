#pragma once

#include "inkling2_mini_checkpoint.hpp"
#include "inkling2_mini_moe_dispatch.hpp"
#include "attn_core/prepared_gqa.hpp"

#include "strix/weight/resident_slices.hpp"
#include "strix/weight/row_pitch.hpp"

#include <utility>

namespace strix::inkling2 {

using MiniResidentRange = weight::ResidentSliceRange;

class MiniResidentWeightPlan {
public:
    static constexpr std::size_t kAlignment = 256U;

    // The input has already passed the Mini text schema. NVFP4 host-only
    // sidecars are fetched separately when the resident expert views bind.
    // rule: channel-safe pitch for the BF16 attention-output rows (0 = packed); Q8_0 compact rows stay packed.
    [[nodiscard]] static MiniResidentWeightPlan make(
        const MiniTextWeights& weights, weight::FormatId storage = weight::kBf16WeightStorage,
        weight::RowPitchRule rule = {});
    [[nodiscard]] const weight::ResidentSlicePlan& slices() const { return slices_; }
    [[nodiscard]] std::size_t bytes() const { return slices_.bytes(); }
    [[nodiscard]] std::span<const MiniResidentRange> ranges() const { return slices_.ranges(); }

private:
    friend class MiniResidentWeights;
    weight::ResidentSlicePlan slices_;
    weight::FormatId storage_ = weight::kBf16WeightStorage;
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
    [[nodiscard]] bool compact() const { return head_.has_value(); }
    [[nodiscard]] const weight::ResidentMatrixView& compact_head() const { return head_.value(); }
    [[nodiscard]] const attn_core::ResidentPreparedGqaWeights& compact_attention(unsigned physical) const {
        return attention_.at(physical);
    }

private:
    explicit MiniResidentWeights(weight::ResidentSlices slices) : slices_(std::move(slices)) {}

    weight::ResidentSlices slices_;
    std::vector<MiniMoeExpert> experts_;
    std::optional<weight::ResidentMatrixView> head_;
    std::array<attn_core::ResidentPreparedGqaWeights, kPhysicalLayers> attention_{};
};

}  // namespace strix::inkling2
