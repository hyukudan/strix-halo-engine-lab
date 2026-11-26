#pragma once
// GLM wire ABI and schedule, extracted without changing their values.
#include "glm/rtxpf/import/glm_dims.hpp"
#include "xport/half_sum_wire.hpp"

namespace strix::glm::xfer {
inline constexpr std::size_t kRowF32 = std::size_t{strix::glm::kHidden} * sizeof(float);
static_assert(kRowF32 == 16384);  // one hidden-state partial row, FP32: 16384 B

// The GLM decode schedule (design notes section 1): every one of the 45 layers has two sublayers (attention: KDA or MLA/DSA; FFN:
// dense L0-2 or MoE + shared), each `X' = post * (p0 + p1) + C X` with an mHC pre-mix that is nonlinear in the whole
// 4 x 4096 stream state, so exactly one reduction of FP32 partial rows per sublayer (90), plus one 8-byte (max, id)
// exchange per step for the vocabulary-split head: 91. N users / verify rows ride in one message (N rows). Router,
// routing, mHC mixes and the indexer selection are replicated: no metadata is exchanged. The MTP draft layer adds 3
// (attention, FFN, its own head argmax) per draft step.
enum class Kind : std::uint32_t { attn_sum = 1, ffn_sum = 2, argmax = 3, control = 4, gather = 5,
                                shard_prepare = 6, shard_partial = 7, test = 15 };
struct Tag {
    Kind kind = Kind::test;
    std::uint32_t layer = 0;
    std::uint64_t step = 0;
};
struct GlmSchedule : strix::xport::HalfSumSchedule<strix::glm::kLayers, strix::glm::kHidden, Kind, Tag> {
    static_assert(kLayers == 45);
    static constexpr unsigned kPerDraftStep = 3;
};

inline constexpr std::uint64_t kTrailerMagic = 0x31524546584d4c47ULL;  // "GLMXFER1"
using Trailer = strix::xport::HalfSumTrailer<kTrailerMagic>;
inline constexpr auto kTrailerDeferred = strix::xport::kHalfSumTrailerDeferred;
inline constexpr auto kTrailerPrev = strix::xport::kHalfSumTrailerPrevious;
inline constexpr auto kInlineCrcMax = strix::xport::kHalfSumInlineCrcMax;

struct GlmTraits : GlmSchedule {
    static constexpr unsigned kStagingSlots = 2;
    using Payload = float;
    using Schedule = GlmSchedule;
    using Tag = xfer::Tag;
    using Trailer = xfer::Trailer;
    using Kind = xfer::Kind;
    static constexpr std::size_t kWidth = strix::glm::kHidden;
    static_assert(kWidth == 4096);
    static constexpr std::uint64_t kPlan = 0x584d4c47ULL;
    static constexpr std::uint64_t kMagic = kTrailerMagic;
    static constexpr std::size_t kInlineCrc = kInlineCrcMax;
    static constexpr std::uint16_t kDeferred = kTrailerDeferred, kPrevious = kTrailerPrev;
    static constexpr std::uint32_t kHeadLayer = 60; // standalone argmax helper's historical tag
    static constexpr const char* kEnvironment = "STRIX_GLMX_";
    static constexpr bool kCrc = true, kCrcDeferred = true;
    static constexpr unsigned kFoldRows = 0;  // xport row fold of the sublayer boundary sum (STRIX_GLMX_FOLD=rows)
    static constexpr const char* kMemoryChoices = "host or device";
    [[nodiscard]] static constexpr const char* crc_description(bool enabled, bool) {
        return enabled ? "on" : "off";
    }
};
} // namespace strix::glm::xfer
