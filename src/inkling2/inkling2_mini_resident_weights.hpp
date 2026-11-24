#pragma once

#include "inkling2_mini_checkpoint.hpp"
#include "inkling2_mini_moe_dispatch.hpp"
#include "attn_core/prepared_gqa.hpp"

#include "strix/weight/levers.hpp"
#include "strix/weight/lowbit_pack.hpp"
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
    // rule: channel-safe pitch for the BF16 attention-output rows (0 = packed); compact rows stay packed.
    // storage: per role class (attention q / k / v / o, output head): BF16, Q8_0, q4_g64 or q6_g64.
    // pack: optional calibrated low-bit sidecar (strix/weight/lowbit_pack.hpp); a q4_g64 / q6_g64 matrix it holds is
    // loaded from it instead of the load-time conversion (same format and bytes count, other codes).
    // keep_bf16: the compact roles are added beside their BF16 rows (kMiniLowBitImage names) instead of replacing them,
    // so a step can read either (a decision row's decode band); false: one image, the compact rows replace BF16.
    [[nodiscard]] static MiniResidentWeightPlan make(
        const MiniTextWeights& weights, weight::RoleStorage storage = {}, weight::RowPitchRule rule = {},
        const weight::LowbitPack* pack = nullptr, bool keep_bf16 = false);
    [[nodiscard]] unsigned prepacked() const noexcept { return prepacked_; }
    [[nodiscard]] const weight::ResidentSlicePlan& slices() const { return slices_; }
    [[nodiscard]] std::size_t bytes() const { return slices_.bytes(); }
    [[nodiscard]] std::span<const MiniResidentRange> ranges() const { return slices_.ranges(); }

private:
    friend class MiniResidentWeights;
    weight::ResidentSlicePlan slices_;
    weight::RoleStorage storage_;
    bool keep_bf16_{};
    unsigned prepacked_{};
    std::array<std::array<MiniExpertLayerView, 2>, kPhysicalLayers> slabs_;
};

// Name prefix of the compact copy of a role kept beside its BF16 rows.
inline constexpr std::string_view kMiniLowBitImage = "lowbit:";

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
    // Compact (non-BF16) attention projections and head are bound independently (a policy may keep either BF16).
    // bf16_bound(): the BF16 rows of the compact roles stay resident too (two images; the caller picks per step).
    [[nodiscard]] bool compact() const { return compact_attention_; }
    [[nodiscard]] bool bf16_bound() const { return bf16_kept_; }
    [[nodiscard]] bool compact_head_bound() const { return head_.has_value(); }
    [[nodiscard]] const weight::ResidentMatrixView& compact_head() const { return head_.value(); }
    [[nodiscard]] const attn_core::ResidentPreparedGqaWeights& compact_attention(unsigned physical) const {
        return attention_.at(physical);
    }

private:
    explicit MiniResidentWeights(weight::ResidentSlices slices) : slices_(std::move(slices)) {}

    weight::ResidentSlices slices_;
    std::vector<MiniMoeExpert> experts_;
    std::optional<weight::ResidentMatrixView> head_;
    bool compact_attention_ = false, bf16_kept_ = false;
    std::array<attn_core::ResidentPreparedGqaWeights, kPhysicalLayers> attention_{};
};

}  // namespace strix::inkling2
