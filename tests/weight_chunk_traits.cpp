#include "core/weight_decode.hpp"

#include <array>
#include <bit>
#include <cmath>
#include <cstdio>
#include <type_traits>

namespace {
using namespace strix::core::weight_decode;

static_assert(std::is_same_v<Chunk<Format::bf16>, Chunk<Format::bf16, false, HalfScale::hardware>>);
static_assert(std::is_same_v<Chunk<Format::q8_0>, Chunk<Format::q8_0, false, HalfScale::hardware>>);

void put16(std::uint8_t* p, unsigned bits) {
    p[0] = static_cast<std::uint8_t>(bits);
    p[1] = static_cast<std::uint8_t>(bits >> 8U);
}

unsigned get16(const std::uint8_t* p) {
    return unsigned(p[0]) | (unsigned(p[1]) << 8U);
}

float scalar_q8(const std::uint8_t* row, unsigned i) {
    const auto* block = row + (i / 32U) * 34U;
    const unsigned h = get16(block);
    const int exponent = int((h >> 10U) & 31U);
    const unsigned fraction = h & 1023U;
    const float magnitude = std::ldexp(float(exponent == 0 ? fraction : fraction + 1024U),
                                       exponent == 0 ? -24 : exponent - 25);
    const float scale = (h & 0x8000U) == 0 ? magnitude : -magnitude;
    const unsigned q = block[2U + i % 32U];
    return scale * float(q < 128U ? int(q) : int(q) - 256);
}

template <Format F, bool Streaming, HalfScale Scale>
bool check(const std::uint8_t* row, unsigned start) {
    const Chunk<F, Streaming, Scale> chunk(row, start);
    for (unsigned e = 0; e < 16U; ++e) {
        const unsigned i = start + e;
        std::uint32_t expected;
        if constexpr (F == Format::bf16)
            expected = std::uint32_t(get16(row + 2U * i)) << 16U;
        else
            expected = std::bit_cast<std::uint32_t>(scalar_q8(row, i));
        if (const auto actual = std::bit_cast<std::uint32_t>(chunk.at(e)); actual != expected) {
            std::fprintf(stderr, "format %u streaming %u scale %u index %u: got %08x expected %08x\n",
                         unsigned(F), unsigned(Streaming), unsigned(Scale), i, actual, expected);
            return false;
        }
    }
    return true;
}

template <Format F>
bool check_modes(const std::uint8_t* row, unsigned start) {
    return check<F, false, HalfScale::hardware>(row, start) && check<F, true, HalfScale::hardware>(row, start) &&
           check<F, false, HalfScale::software>(row, start) && check<F, true, HalfScale::software>(row, start);
}
} // namespace

int main() {
    alignas(16) std::array<std::uint8_t, 65U> bytes{};
    auto* row = bytes.data() + 1U;
    for (unsigned base = 0; base < 65536U; base += 32U) {
        for (unsigned i = 0; i < 32U; ++i) put16(row + 2U * i, base + i);
        if (!check_modes<Format::bf16>(row, 0U) || !check_modes<Format::bf16>(row, 16U)) return 1;
    }

    std::array<std::uint8_t, 8U * 34U> q8{};
    for (unsigned block = 0; block < 8U; ++block)
        for (unsigned i = 0; i < 32U; ++i)
            q8[block * 34U + 2U + i] = static_cast<std::uint8_t>(block * 32U + i);
    for (unsigned scale = 0; scale < 65536U; ++scale) {
        if ((scale & 0x7c00U) == 0x7c00U) continue;
        for (unsigned block = 0; block < 8U; ++block) {
            put16(q8.data() + block * 34U, scale);
            if (!check_modes<Format::q8_0>(q8.data(), block * 32U) ||
                !check_modes<Format::q8_0>(q8.data(), block * 32U + 16U)) return 1;
        }
    }
    std::puts("BF16: all 65536 bit patterns. Q8_0: all 63488 finite scales and 256 codes. Four policies matched bitwise.");
}
