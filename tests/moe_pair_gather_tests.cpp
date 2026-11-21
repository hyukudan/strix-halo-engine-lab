#include "core/iu4_onfly.hpp"
#include "moe_core/pair_gather.hpp"

#include <algorithm>
#include <array>
#include <cstdint>
#include <cstdio>
#include <vector>

namespace {
using strix::core::onfly::f16_bits_to_f32;
using strix::core::onfly::f32_to_f16_bits;

// Serial transform of one routed input block in natural channel order.
std::array<std::uint16_t, 128> reference(unsigned token, unsigned expert, unsigned matrix, unsigned block) {
    std::array<float, 128> v{};
    for (unsigned c = 0; c < 128U; ++c) {
        const unsigned channel = 128U * block + c;
        const float input = static_cast<float>((channel * 17U + token * 29U) % 97U) / 13.0F;
        const float scale = f16_bits_to_f32(f32_to_f16_bits(
            static_cast<float>((channel * 3U + expert * 11U + matrix * 5U) % 31U) / 7.0F - 2.0F));
        v[c] = f16_bits_to_f32(f32_to_f16_bits(input * scale));
    }
    for (unsigned stride = 1; stride < 128U; stride *= 2U) {
        for (unsigned base = 0; base < 128U; base += 2U * stride) {
            for (unsigned i = base; i < base + stride; ++i) {
                const float a = v[i], b = v[i + stride];
                v[i] = a + b;
                v[i + stride] = a - b;
            }
        }
    }
    std::array<std::uint16_t, 128> bits{};
    for (unsigned c = 0; c < 128U; ++c) bits[c] = f32_to_f16_bits(v[c] * 0x1.6a09e6p-4F);
    return bits;
}

template <unsigned Width> bool check_layout() {
    // A 17-pair expert spans two tiles. The next expert starts at padded row 32.
    constexpr unsigned pairs = 19U, padded = 48U, blocks = Width / 128U;
    struct Group { unsigned pair_begin, tbegin; };
    constexpr Group groups[] = {{0U, 0U}, {17U, 32U}};
    constexpr std::uint16_t pad = 0x7e01U;
    const auto token = [](unsigned pair) { return (7U * pair) % 13U; };
    std::vector<std::uint16_t> image(2U * padded * Width, pad);
    const unsigned waves = pairs * 2U * blocks;
    for (unsigned wave = 0; wave < (waves + 7U) / 8U * 8U; ++wave) {
        const strix::moe_core::PairGatherBlock work = strix::moe_core::pair_gather_block<Width>(wave);
        const auto [pair, matrix, block] = work;
        if (matrix > 1U || block >= blocks) return false;
        if ((pair * 2U + matrix) * blocks + block != wave) return false;
        if (pair >= pairs) continue;
        const unsigned expert = pair < 17U ? 0U : 1U;
        const auto values = reference(token(pair), expert, matrix, block);
        for (unsigned lane = 0; lane < 32U; ++lane) {
            for (unsigned j = 0; j < 4U; ++j) {
                const unsigned c = lane + j * 32U;
                auto& dest = image.at(work.output_offset<Width>(groups[expert], padded, work.channel(lane, j)));
                if (dest != pad) return false;
                dest = values[c];
            }
        }
    }

    // Pack the independent row-major reference by walking physical tiles in storage order.
    std::vector<std::uint16_t> logical(padded * Width);
    std::size_t offset = 0;
    for (unsigned matrix = 0; matrix < 2U; ++matrix) {
        std::fill(logical.begin(), logical.end(), pad);
        for (unsigned pair = 0; pair < pairs; ++pair) {
            const unsigned expert = pair < 17U ? 0U : 1U;
            const unsigned row = pair < 17U ? pair : pair + 15U;
            for (unsigned block = 0; block < blocks; ++block) {
                const auto values = reference(token(pair), expert, matrix, block);
                std::copy(values.begin(), values.end(), logical.data() + row * Width + block * 128U);
            }
        }
        for (unsigned r = 0; r < padded; r += 16U)
            for (unsigned c = 0; c < Width; c += 16U)
                for (unsigned i = 0; i < 16U; ++i)
                    for (unsigned j = 0; j < 16U; ++j)
                        if (image[offset++] != logical[(r + i) * Width + c + j]) return false;
    }
    return true;
}
} // namespace

int main() {
    // The single-block case also exercises a partially occupied final workgroup.
    if (check_layout<128>() && check_layout<4096>() && check_layout<6144>()) return 0;
    std::fputs("pair-major gather differs from the tiled CPU reference\n", stderr);
    return 1;
}
