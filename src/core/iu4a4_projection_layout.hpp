#pragma once
// Column layout of an A4W4 projection call (core/iu4a4_projections.hpp): the projections of one call that read the same
// input rows are ONE weight image of the shared A4W4 dense class (core/iu4a4_dense.hpp), their outputs side by side in
// call order. Segment s owns image columns [first_col[s], first_col[s + 1]); every segment is a whole number of 16-column
// tiles, so a column tile belongs to exactly one segment and stores into that segment's own row-major output.
// Host and device; no HIP calls (the host tests check the plan).
#include <cstddef>
#include <cstdint>
#include <span>

#if defined(__HIPCC__)
#define STRIX_IU4A4_LAYOUT_HD __host__ __device__
#else
#define STRIX_IU4A4_LAYOUT_HD
#endif

namespace strix::core::iu4a4_projections {

inline constexpr unsigned kMaxSegments = 8U;
inline constexpr unsigned kTileColumns = 16U;
inline constexpr unsigned kBlockInputs = 128U;      // iu4a4_rot::kBlock (the rotation block = weight group)
inline constexpr unsigned kMaxInputs = 131072U;     // iu4a4_rot::kMaxInputs

struct Layout {
    unsigned count = 0;  // segments; 0 = not admitted
    unsigned in = 0;     // inputs per row
    unsigned n = 0;      // image columns = sum of the segments' outputs
    unsigned first_col[kMaxSegments + 1U]{};

    [[nodiscard]] STRIX_IU4A4_LAYOUT_HD constexpr bool admitted() const noexcept { return count != 0U; }
    [[nodiscard]] STRIX_IU4A4_LAYOUT_HD constexpr unsigned outputs(unsigned s) const noexcept {
        return first_col[s + 1U] - first_col[s];
    }
    // The segment owning image column c (c < n): the last segment whose first column is <= c.
    [[nodiscard]] STRIX_IU4A4_LAYOUT_HD constexpr unsigned segment(unsigned c) const noexcept {
        unsigned s = 0U;
        while (s + 1U < count && c >= first_col[s + 1U]) ++s;
        return s;
    }
    [[nodiscard]] constexpr std::size_t weight_code_bytes() const noexcept { return std::size_t(n) * in / 2U; }
    [[nodiscard]] constexpr std::size_t weight_step_bytes() const noexcept {
        return std::size_t(n) * (in / kBlockInputs) * 4U;
    }
};

// The layout of a call, or a not-admitted layout: 1..kMaxSegments segments, in a positive multiple of 128 up to
// kMaxInputs, every segment's outputs a positive multiple of 16, and the column total within unsigned range.
[[nodiscard]] constexpr Layout make_layout(unsigned in, std::span<const unsigned> outputs) noexcept {
    Layout layout{};
    if (outputs.empty() || outputs.size() > kMaxSegments || in == 0U || in % kBlockInputs != 0U || in > kMaxInputs)
        return {};
    unsigned n = 0U;
    for (std::size_t s = 0; s < outputs.size(); ++s) {
        const unsigned o = outputs[s];
        if (o == 0U || o % kTileColumns != 0U || o > 0xFFFFFFFFU - n) return {};
        layout.first_col[s] = n;
        n += o;
    }
    layout.count = static_cast<unsigned>(outputs.size());
    layout.in = in;
    layout.n = n;
    for (std::size_t s = outputs.size(); s <= kMaxSegments; ++s) layout.first_col[s] = n;
    return layout;
}

// Store rounding of a call (core::Bf16RowProjection round_bf16 / round_separate): none = the FP32 value, incumbent =
// RoundStore::incumbent (round_bf16), separate = RoundStore::separate (round_bf16 + round_separate).
enum class RoundPolicy : unsigned char { none, incumbent, separate };
[[nodiscard]] constexpr bool round_policy(bool round_bf16, bool round_separate, RoundPolicy& policy) noexcept {
    if (round_separate && !round_bf16) return false;  // not a policy (core::bf16_rows_path refuses it too)
    policy = !round_bf16 ? RoundPolicy::none : (round_separate ? RoundPolicy::separate : RoundPolicy::incumbent);
    return true;
}

// What an image was built from, sealed with it at load and checked at every match: the input width, each segment's
// effective BF16 row pitch in bytes (the builder reads row r of segment s at weights[s] + r * pitch), the rounding
// policy of the image's calls and the weight image rule (the integer value of iu4a4_rot::WeightRule). The weight
// pointer and the output count alone do not name the image: the same base pointer and outputs at another pitch are
// other weights after the first row.
struct Seal {
    unsigned in = 0U;
    std::size_t row_bytes[kMaxSegments]{};
    RoundPolicy round = RoundPolicy::none;
    unsigned rule = 0U;
};

// Which image a call runs on. Built at load: one key per segment (the segment's BF16 weights, its image, its index).
// A call of `input_width` inputs under weight rule `rule` matches image i when every segment of i is one projection of
// the call, in segment order, with the segment's weights, outputs and effective row pitch (row_bytes, 0 = the natural
// input_width * 2 bytes), and every matched projection has the image's rounding policy, its input width and rule are
// the image's (Seal); index[s] = the call position of segment s. The call's other projections (no key, or keys of
// another image) keep their incumbent route. kNoImage: no match (any sealed field differs: refused, not adapted).
struct SegmentKey {
    const void* weights = nullptr;
    unsigned image = 0U, segment = 0U;
};
inline constexpr unsigned kNoImage = 0xFFFFFFFFU;

template <class Projection>
[[nodiscard]] constexpr unsigned match_call(std::span<const SegmentKey> keys, std::span<const Layout> layouts,
                                            std::span<const Seal> seals, unsigned input_width, unsigned rule,
                                            std::span<const Projection> call, unsigned (&index)[kMaxSegments]) noexcept {
    const std::size_t natural = std::size_t(input_width) * 2U;
    const auto key_of = [&](const void* weights) -> const SegmentKey* {
        if (!weights) return nullptr;
        for (const auto& key : keys)
            if (key.weights == weights) return &key;
        return nullptr;
    };
    for (std::size_t first = 0; first < call.size(); ++first) {
        const SegmentKey* lead = key_of(call[first].weights);
        if (!lead || lead->image >= layouts.size() || lead->image >= seals.size() || lead->segment != 0U) continue;
        const Layout& layout = layouts[lead->image];
        const Seal& seal = seals[lead->image];
        unsigned next = 0U;  // next segment to find
        bool ok = layout.admitted() && layout.in == input_width && seal.in == input_width && seal.rule == rule;
        for (std::size_t p = first; ok && p < call.size(); ++p) {
            const SegmentKey* key = key_of(call[p].weights);
            if (!key || key->image != lead->image) continue;
            RoundPolicy round = RoundPolicy::none;
            const std::size_t pitch = call[p].row_bytes != 0U ? std::size_t(call[p].row_bytes) : natural;
            ok = key->segment == next && next < layout.count && call[p].outputs == layout.outputs(next) &&
                 pitch == seal.row_bytes[next] && round_policy(call[p].round_bf16, call[p].round_separate, round) &&
                 round == seal.round;
            if (ok) index[next++] = static_cast<unsigned>(p);
        }
        if (ok && next == layout.count) return lead->image;
    }
    return kNoImage;
}

}  // namespace strix::core::iu4a4_projections
