#pragma once
// a8g32-v1: the backend-neutral quantized-activation contract (int8 codes per value, one FP32 delta and one int32 code
// sum per 32 consecutive values of a row). It is the operand class that producer epilogues emit (graph chain forms
// store_a8g32 / quant_a8g32) and that the Q8_0 / K-quant / IQ x A8 rows and tall families read. This header is its ONE
// CPU reference; backend bodies (HIP src/ops/a8, CUDA src/cuda/core/dense/a8_helpers.cuh and the chain quantiser) must
// reproduce these bits for finite inputs and say so next to their code.
//
// Per group of 32 values x[0..31]:
//   amax  = max |x[j]| from +0 (a NaN value is ignored: std::max keeps the running maximum)
//   tiny  = 0 < amax < 2^-119
//   delta = amax == 0 || tiny ? 1 : amax / 127          (IEEE division, round to nearest even)
//   code  = tiny ? 0 : clamp(nearbyint(x[j] / delta), -127, 127)   (IEEE division; ties to even; NaN -> -127)
//   sum   = sum of the 32 codes (int32)
// Layout of R rows: codes [r][pitch], deltas and sums [r][pitch / 32]; pitch >= width, both multiples of 32. The pitch
// only moves addresses (strix/a8_row_pitch.hpp picks the non-aliasing one); it never changes a bit.
//
// Tier rule: an A8 edge is part of an engine's EXACT identity only when that engine's declared reference reads A8 at
// that edge (its reference was defined with these bits). Anywhere else, replacing an FP32 activation by A8 changes bits
// and is a FAST / TURBO lever that needs gate v2.
#include "strix/a8_row_pitch.hpp"

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <span>
#include <stdexcept>
#include <string_view>

namespace strix::core::a8g32 {

inline constexpr std::string_view identity = "a8g32-v1";
inline constexpr std::uint32_t group = 32U;
inline constexpr float code_max = 127.0F;
inline constexpr float tiny_below = 0x1p-119F;

struct GroupResult {
    float delta;
    std::int32_t sum;
};

// One group: x[0..31] -> codes[0..31], returns its delta and code sum.
[[nodiscard]] inline GroupResult quantize_group(const float* x, std::int8_t* codes) noexcept {
    float amax = 0.0F;
    for (std::uint32_t j = 0; j < group; ++j) amax = std::max(amax, std::fabs(x[j]));
    const bool tiny = amax > 0.0F && amax < tiny_below;
    const float delta = amax == 0.0F || tiny ? 1.0F : amax / code_max;
    std::int32_t sum = 0;
    for (std::uint32_t j = 0; j < group; ++j) {
        const float rounded = tiny ? 0.0F : std::nearbyint(x[j] / delta);
        const auto code = static_cast<std::int8_t>(std::min(code_max, std::max(-code_max, rounded)));
        codes[j] = code;
        sum += code;
    }
    return {delta, sum};
}

// One row of x.size() values (a multiple of 32): codes[x.size()], deltas / sums [x.size() / 32]; sums may be null for
// consumers that do not read them.
inline void quantize_row(std::span<const float> x, std::int8_t* codes, float* deltas, std::int32_t* sums) {
    if (x.size() % group != 0U) throw std::invalid_argument("a8g32 row width must be a multiple of 32");
    for (std::size_t g = 0; g < x.size() / group; ++g) {
        const GroupResult r = quantize_group(x.data() + g * group, codes + g * group);
        deltas[g] = r.delta;
        if (sums != nullptr) sums[g] = r.sum;
    }
}

// R rows of `width` values at a codes pitch of `pitch` (deltas / sums pitch = pitch / 32).
struct Layout {
    std::uint32_t rows = 0, width = 0, pitch = 0;

    // The layout every A8 producer that owns its buffer allocates (non-aliasing pitch).
    [[nodiscard]] static constexpr Layout padded(std::uint32_t rows, std::uint32_t width) noexcept {
        return {rows, width, a8_row_pitch(width)};
    }
    [[nodiscard]] constexpr bool valid() const noexcept {
        return width % group == 0U && pitch % group == 0U && pitch >= width;
    }
    [[nodiscard]] constexpr std::uint32_t group_pitch() const noexcept { return pitch / group; }
    [[nodiscard]] constexpr std::size_t code_count() const noexcept { return std::size_t{pitch} * rows; }
    [[nodiscard]] constexpr std::size_t group_count() const noexcept { return std::size_t{group_pitch()} * rows; }
};

// Rows of x (row r at x + r * x_stride) into the layout; codes / deltas / sums past `width` in a row are not written.
inline void quantize_rows(const float* x, std::size_t x_stride, const Layout& layout, std::int8_t* codes, float* deltas,
                          std::int32_t* sums) {
    if (!layout.valid()) throw std::invalid_argument("a8g32 layout needs width, pitch multiples of 32 and pitch >= width");
    for (std::uint32_t r = 0; r < layout.rows; ++r)
        quantize_row(std::span<const float>(x + std::size_t{r} * x_stride, layout.width), codes + std::size_t{r} * layout.pitch,
                     deltas + std::size_t{r} * layout.group_pitch(),
                     sums != nullptr ? sums + std::size_t{r} * layout.group_pitch() : nullptr);
}

enum class EdgeClass : std::uint8_t {
    reference,  // the engine's declared reference reads A8 here: part of EXACT, bitwise
    lever,      // FP32 in the reference: a bit-changing FAST / TURBO lever behind gate v2
};
[[nodiscard]] constexpr EdgeClass edge_class(bool reference_reads_a8) noexcept {
    return reference_reads_a8 ? EdgeClass::reference : EdgeClass::lever;
}
[[nodiscard]] constexpr bool admitted(EdgeClass edge, bool lossy_tier) noexcept {
    return edge == EdgeClass::reference || lossy_tier;
}

}  // namespace strix::core::a8g32
