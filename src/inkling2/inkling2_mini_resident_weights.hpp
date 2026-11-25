#pragma once

#include "inkling2_mini_checkpoint.hpp"
#include "inkling2_mini_moe_dispatch.hpp"
#include "inkling2_mini_attention_operands.hpp"
#include "inkling2_mini_controller_operands.hpp"
#include "attn_core/prepared_gqa.hpp"

#include "strix/weight/formats/microscaled_encode.hpp"
#include "strix/weight/levers.hpp"
#include "strix/weight/lowbit_pack.hpp"
#include "strix/weight/resident_slices.hpp"
#include "strix/weight/row_pitch.hpp"

#include <functional>
#include <span>
#include <utility>

namespace strix::inkling2 {

using MiniResidentRange = weight::ResidentSliceRange;
using MiniReadControllerBits = std::function<void(std::span<const std::uint16_t>, std::span<std::uint16_t>)>;

// The certificate describes the exact BF16 bytes of these resident slices. The reader transfers those slices to host.
[[nodiscard]] MiniControllerCertificate read_mini_controller_certificate(const weight::ResidentSlices& slices,
                                                                         const MiniReadControllerBits& read);

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
    // rule: channel-safe pitch for the BF16 attention-output rows (0 = packed); compact rows stay packed.
    // storage: per role class (attention q / k / v / o, output head): BF16, Q8_0, q4_g64 or q6_g64.
    // pack: optional calibrated low-bit sidecar (strix/weight/lowbit_pack.hpp); a q4_g64 / q6_g64 matrix it holds is
    // loaded from it instead of the load-time conversion (same format and bytes count, other codes).
    // keep_bf16: the compact roles are added beside their BF16 rows (kMiniLowBitImage names) instead of replacing them,
    // so a step can read either (a decision row's decode band); false: one image, the compact rows replace BF16.
    [[nodiscard]] static MiniResidentWeightPlan make(
        const MiniTextWeights& weights, weight::RoleStorage storage = {}, weight::RowPitchRule rule = {},
        bool narrow_experts = false, const weight::LowbitPack* pack = nullptr, bool keep_bf16 = false);
    [[nodiscard]] unsigned prepacked() const noexcept { return prepacked_; }
    [[nodiscard]] const weight::ResidentSlicePlan& slices() const { return slices_; }
    [[nodiscard]] std::size_t bytes() const { return slices_.bytes(); }
    [[nodiscard]] std::span<const MiniResidentRange> ranges() const { return slices_.ranges(); }

    // Narrow expert image (lever moe.experts.mxfp8-nvfp4@1): the checkpoint's MXFP8 expert slabs stay out of the resident
    // slices; the load transcodes each one to NVFP4 (expert_slab is its source) into one separate device allocation of
    // narrow_bytes(), at narrow_range(). EXACT and every load without the lever keep the checkpoint slabs.
    struct NarrowRange {
        std::size_t codes{}, scales{};
    };
    [[nodiscard]] bool narrow_experts() const { return narrow_experts_; }
    [[nodiscard]] static bool narrowed(unsigned physical) {
        return mini_expert_format(physical) != weight::microscaled::MatrixFormat::nvfp4;
    }
    [[nodiscard]] const MiniExpertLayerView& expert_slab(unsigned physical, MiniExpertProjection projection) const {
        return slabs_.at(physical).at(projection == MiniExpertProjection::gate_up ? 0U : 1U);
    }
    [[nodiscard]] NarrowRange narrow_range(unsigned physical, MiniExpertProjection projection) const {
        return narrow_ranges_.at(physical).at(projection == MiniExpertProjection::gate_up ? 0U : 1U);
    }
    [[nodiscard]] std::size_t narrow_bytes() const { return narrow_bytes_; }

private:
    friend class MiniResidentWeights;
    weight::ResidentSlicePlan slices_;
    weight::RoleStorage storage_;
    bool keep_bf16_{};
    unsigned prepacked_{};
    std::array<std::array<MiniExpertLayerView, 2>, kPhysicalLayers> slabs_;
    bool narrow_experts_ = false;
    std::array<std::array<NarrowRange, 2>, kPhysicalLayers> narrow_ranges_{};
    std::size_t narrow_bytes_ = 0;
};

// Name prefix of the compact copy of a role kept beside its BF16 rows.
inline constexpr std::string_view kMiniLowBitImage = "lowbit:";

// Borrow immutable resident bytes. Once bound, the views are independent of
// the checkpoint file and valid for as long as the storage is allocated.
class MiniResidentWeights {
public:
    [[nodiscard]] static MiniResidentWeights bind(const MiniResidentWeightPlan& plan,
                                                  std::span<const std::uint8_t> storage,
                                                  const MiniReadControllerBits& read_controller_bits = {});
    [[nodiscard]] std::span<const std::uint16_t> bf16(std::string_view name) const;
    [[nodiscard]] weight::ResidentBf16Rows bf16_rows(std::string_view name) const;
    [[nodiscard]] std::span<const float> f32(std::string_view name) const;
    [[nodiscard]] std::span<const MiniMoeExpert> experts(unsigned physical) const;
    // Compact (non-BF16) attention projections and head are bound independently (a policy may keep either BF16).
    // bf16_bound(): the BF16 rows of the compact roles stay resident too (two images; the caller picks per step).
    [[nodiscard]] bool compact() const { return compact_attention_; }
    [[nodiscard]] bool bf16_bound() const { return bf16_kept_; }
    [[nodiscard]] bool compact_head_bound() const { return head_.has_value(); }
    // Certifies the attention operand class from the attention tensors this object holds (BF16 storage only; compact
    // storage keeps the empty, refusing certificate), through the shared bounded scan (core::Bf16StagedScan): `read`
    // copies a slice of a staged BF16 tensor into the scan's scratch (a device copy on CUDA; its exceptions
    // propagate). Without scratch the certificate stays empty. The certificate is a function of these bytes only.
    void certify_attention(core::Bf16StagedScan& scan, const MiniStagedRead& read);
    [[nodiscard]] const MiniAttentionCertificate& attention_certificate() const { return attention_certificate_; }
    // True when the expert bank is the narrow image: every physical layer is NVFP4 (expert_format) and the MXFP8
    // layers are bound by attach_narrow, never from the resident slices.
    [[nodiscard]] bool narrow_experts() const { return narrow_; }
    [[nodiscard]] weight::microscaled::MatrixFormat expert_format(unsigned physical) const {
        return narrow_ ? weight::microscaled::MatrixFormat::nvfp4 : mini_expert_format(physical);
    }
    // Binds the device copy (`codes`, `scales`: the bytes of `slab`) of a narrowed layer's projection.
    void attach_narrow(unsigned physical, MiniExpertProjection projection, const weight::microscaled::Nvfp4Slab& slab,
                       std::span<const std::uint8_t> codes, std::span<const std::uint8_t> scales);
    void certify_controller(const MiniReadControllerBits& read);
    [[nodiscard]] const MiniControllerCertificate& controller_certificate() const { return controller_certificate_; }
    [[nodiscard]] const weight::ResidentMatrixView& compact_head() const { return head_.value(); }
    [[nodiscard]] const attn_core::ResidentPreparedGqaWeights& compact_attention(unsigned physical) const {
        return attention_.at(physical);
    }

private:
    explicit MiniResidentWeights(weight::ResidentSlices slices) : slices_(std::move(slices)) {}

    weight::ResidentSlices slices_;
    std::vector<MiniMoeExpert> experts_;
    std::optional<weight::ResidentMatrixView> head_;
    bool narrow_ = false;
    bool compact_attention_ = false, bf16_kept_ = false;
    std::array<attn_core::ResidentPreparedGqaWeights, kPhysicalLayers> attention_{};
    MiniAttentionCertificate attention_certificate_;
    MiniControllerCertificate controller_certificate_;
};

}  // namespace strix::inkling2
