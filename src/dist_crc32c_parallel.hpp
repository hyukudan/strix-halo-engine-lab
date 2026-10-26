#pragma once

#include <array>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <span>

namespace strix::dist::detail {
#if defined(__x86_64__)
// Shift a raw reflected CRC by 4096 zero bytes. This is a linear map over GF(2);
// four byte lookup tables apply it without serialising the three CRC chains.
inline const auto& crc32c_shift4096() {
    static const auto table = [] {
        std::array<std::array<std::uint32_t, 256>, 4> out{};
        std::array<std::uint32_t, 32> basis{};
        for (unsigned b = 0; b < 32; ++b) {
            std::uint32_t c = std::uint32_t{1} << b;
            for (unsigned i = 0; i < 4096U * 8U; ++i)
                c = (c >> 1U) ^ ((c & 1U) ? 0x82f63b78U : 0U);
            basis[b] = c;
        }
        for (unsigned k = 0; k < 4; ++k)
            for (unsigned v = 0; v < 256; ++v)
                for (unsigned b = 0; b < 8; ++b)
                    if (v & (1U << b)) out[k][v] ^= basis[k * 8U + b];
        return out;
    }();
    return table;
}

__attribute__((target("sse4.2"))) inline std::uint32_t crc32c_parallel(std::span<const std::byte> bytes) noexcept {
    const auto& table = crc32c_shift4096();
    const auto shift = [&](std::uint32_t c) {
        return table[0][c & 255U] ^ table[1][(c >> 8U) & 255U] ^
               table[2][(c >> 16U) & 255U] ^ table[3][c >> 24U];
    };
    std::uint64_t c = 0xffffffffU;
    std::size_t i = 0;
    for (; bytes.size() - i >= 3U * 4096U; i += 3U * 4096U) {
        std::uint64_t a = c, b = 0, d = 0;
        for (std::size_t j = 0; j < 4096U; j += 8U) {
            std::uint64_t x, y, z;
            std::memcpy(&x, bytes.data() + i + j, 8U);
            std::memcpy(&y, bytes.data() + i + 4096U + j, 8U);
            std::memcpy(&z, bytes.data() + i + 8192U + j, 8U);
            a = __builtin_ia32_crc32di(a, x);
            b = __builtin_ia32_crc32di(b, y);
            d = __builtin_ia32_crc32di(d, z);
        }
        c = shift(shift(static_cast<std::uint32_t>(a)) ^ static_cast<std::uint32_t>(b)) ^ static_cast<std::uint32_t>(d);
    }
    for (; bytes.size() - i >= 8U; i += 8U) {
        std::uint64_t word;
        std::memcpy(&word, bytes.data() + i, 8U);
        c = __builtin_ia32_crc32di(c, word);
    }
    auto tail = static_cast<std::uint32_t>(c);
    for (; i < bytes.size(); ++i) tail = __builtin_ia32_crc32qi(tail, static_cast<unsigned char>(bytes[i]));
    return tail ^ 0xffffffffU;
}
#endif
} // namespace strix::dist::detail
