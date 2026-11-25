#pragma once

#include "inkling2_mini_checkpoint.hpp"
#include "inkling2_mini_moe_dispatch.hpp"
#include "inkling2_mini_attention_operands.hpp"
#include "attn_core/prepared_gqa.hpp"

#include "strix/weight/resident_slices.hpp"
#include "strix/weight/row_pitch.hpp"

#include <functional>
#include <span>
#include <utility>

namespace strix::inkling2 {

using MiniResidentRange = weight::ResidentSliceRange;

// The staged BF16 tensor of a text tensor (name without the text prefix), and the copy of a slice of it into host
// scratch (a device copy on CUDA).
using MiniStagedLookup = std::function<std::span<const std::uint16_t>(std::string_view name)>;
using MiniStagedRead = std::function<void(std::span<const std::uint16_t> staged, std::span<std::uint16_t> host)>;

// The attention certificate over staged tensors: the shared scan (core/weight_certificate.hpp) over every attention
// projection and layer norm gain the schema lists; empty (refusing) when the scan has no scratch.
[[nodiscard]] MiniAttentionCertificate certify_mini_attention(core::Bf16StagedScan& scan, const MiniStagedLookup& staged,
                                                              const MiniStagedRead& read);

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
    // Certifies the attention operand class from the attention tensors this object holds (BF16 storage only; compact
    // storage keeps the empty, refusing certificate), through the shared bounded scan (core::Bf16StagedScan): `read`
    // copies a slice of a staged BF16 tensor into the scan's scratch (a device copy on CUDA; its exceptions
    // propagate). Without scratch the certificate stays empty. The certificate is a function of these bytes only.
    void certify_attention(core::Bf16StagedScan& scan, const MiniStagedRead& read);
    [[nodiscard]] const MiniAttentionCertificate& attention_certificate() const { return attention_certificate_; }
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
    MiniAttentionCertificate attention_certificate_;
};

}  // namespace strix::inkling2
