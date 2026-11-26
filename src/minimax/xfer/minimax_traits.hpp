#pragma once
#include "../../xport/transport_backend.hpp"
#include "../../xport/half_sum_wire.hpp"
// MiniMax wire ABI, schedule and existing policy defaults.
namespace strix::minimax::xfer {
inline constexpr std::size_t kHidden = 6144;
inline constexpr std::size_t kRowF32 = kHidden * sizeof(float);  // one hidden-state partial row, FP32: 24576 B

// The MiniMax decode schedule (design and proof in design notes): per layer exactly two reductions of FP32
// partial rows (after o_proj, after the FFN) and one 8-byte (max, id) exchange per step for the vocabulary-split
// head; N users ride in one message (N rows). Router, routing and indexer are replicated: no metadata is exchanged.
enum class Kind : std::uint32_t { attn_sum = 1, ffn_sum = 2, argmax = 3, control = 4, gather = 5, test = 15 };
struct Tag {
    Kind kind = Kind::test;
    std::uint32_t layer = 0;
    std::uint64_t step = 0;
};
using MmSchedule = strix::xport::HalfSumSchedule<60, kHidden, Kind, Tag>;

inline constexpr std::uint64_t kTrailerMagic = 0x3146584d58494d4dULL;  // "MMIXMXF1"
using Trailer = strix::xport::HalfSumTrailer<kTrailerMagic>;
inline constexpr auto kTrailerDeferred = strix::xport::kHalfSumTrailerDeferred;
inline constexpr auto kTrailerPrev = strix::xport::kHalfSumTrailerPrevious;
inline constexpr auto kInlineCrcMax = strix::xport::kHalfSumInlineCrcMax;
struct MinimaxTraits : MmSchedule {
    using Payload = float;
    using Schedule = MmSchedule;
    using Tag = xfer::Tag;
    using Trailer = xfer::Trailer;
    using Kind = xfer::Kind;
    static constexpr std::size_t kWidth = 6144;
    static constexpr std::uint64_t kPlan = 0x46584d4dULL;
    static constexpr std::uint64_t kMagic = kTrailerMagic;
    static constexpr std::size_t kInlineCrc = kInlineCrcMax;
    static constexpr std::uint16_t kDeferred = kTrailerDeferred, kPrevious = kTrailerPrev;
    static constexpr std::uint32_t kHeadLayer = 60; // standalone argmax helper's historical tag
    static constexpr const char* kEnvironment = "STRIX_MMX_";
    static constexpr bool kCrc = true, kCrcDeferred = true;
    static constexpr unsigned kFoldRows = 0;  // xport row fold of the decode all-reduce sum (STRIX_MMX_FOLD=rows)
    static constexpr const char* kMemoryChoices = "host or device";
    [[nodiscard]] static constexpr const char* crc_description(bool enabled, bool deferred) {
        return enabled ? (deferred ? "deferred" : "inline") : "off";
    }
};
} // namespace strix::minimax::xfer
