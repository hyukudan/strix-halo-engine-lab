#pragma once

#include <cstddef>
#include <cstdint>

#if defined(__HIPCC__) || defined(__CUDACC__)
#define STRIX_EXL3_TRANSPORT_HD __host__ __device__
#else
#define STRIX_EXL3_TRANSPORT_HD
#endif

namespace strix::weight::exl3_transport {

// Contiguous payload: input scales, output scales, then input-tile-major
// trellis rings. Both scale vectors store little-endian binary16.
inline constexpr unsigned kRateMask = 0xb8; // 4, 5, 6, 8 bits
inline constexpr unsigned kTransformBlock = 128;

STRIX_EXL3_TRANSPORT_HD constexpr bool supports_rate(unsigned bits) {
    return bits > 0 && bits <= 8 && (kRateMask & (1U << (bits - 1))) != 0;
}

STRIX_EXL3_TRANSPORT_HD constexpr bool aligned_shape(std::uint64_t inputs, std::uint64_t outputs) {
    return inputs % kTransformBlock == 0 && outputs % kTransformBlock == 0;
}

STRIX_EXL3_TRANSPORT_HD constexpr unsigned bits(unsigned inputs, std::size_t row_bytes) {
    return unsigned(row_bytes * 8 / inputs);
}

STRIX_EXL3_TRANSPORT_HD constexpr bool geometry(unsigned inputs, unsigned outputs, std::size_t row_bytes) {
    if (!inputs || !outputs || !aligned_shape(inputs, outputs)) return false;
    const unsigned rate = bits(inputs, row_bytes);
    return supports_rate(rate) && row_bytes == std::size_t(inputs) * rate / 8;
}

constexpr bool finite_scale(std::uint8_t high_byte) {
    return (high_byte & 0x7cU) != 0x7cU;
}

void validate(const std::uint8_t* data, unsigned inputs, unsigned outputs, std::size_t row_bytes);

} // namespace strix::weight::exl3_transport

#undef STRIX_EXL3_TRANSPORT_HD
