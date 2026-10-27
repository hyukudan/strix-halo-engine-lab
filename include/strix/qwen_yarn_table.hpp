#pragma once

#include "strix/rotary_descriptor.hpp"
#include <array>
#include <functional>
#include <iosfwd>
#include <span>

namespace strix {
struct YarnCoefficient { float cosine, sine; };
using YarnRow = std::array<YarnCoefficient, 32>;
struct YarnFrequencies {
    int low, high;
    std::array<double, 32> omega;
};
// Uses the host C++ double libm (pow/log/cos/sin), then one RNE binary32
// conversion. Tables freeze those actual bytes with SHA256; cross-libm bit
// identity at transcendental rounding ties is not promised.
YarnFrequencies qwen_yarn_frequencies(const RotaryDescriptor&);
YarnRow qwen_yarn_row(const RotaryDescriptor&, std::uint32_t position);
using YarnRowsConsumer = std::function<void(std::uint32_t first, std::span<const YarnRow>)>;
// Bounded staging. Native returns without allocation or callbacks.
void generate_qwen_yarn_rows(const RotaryDescriptor&, std::uint32_t first,
    std::uint32_t end, const YarnRowsConsumer&, std::size_t chunk_rows = 256);

// Portable little-endian float2 payload, with schema/endianness/extent,
// canonical descriptor (including exact double bits), and payload SHA256.
// Two bounded passes avoid buffering or duplicating a full table. The range
// is explicit so small fixtures and staged table segments use the same format.
// Native emits no header and no payload.
void write_qwen_yarn_table(std::ostream&, const RotaryDescriptor&,
    std::uint32_t first, std::uint32_t end, std::size_t chunk_rows = 256);
// Validates metadata, exact extent, payload digest and EOF, using bounded space.
// Native accepts only an empty stream. No unchecked data is published.
void validate_qwen_yarn_table(std::istream&, const RotaryDescriptor&,
    std::uint32_t first, std::uint32_t end);
} // namespace strix
