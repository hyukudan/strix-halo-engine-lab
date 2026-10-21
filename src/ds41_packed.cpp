#include "strix/ds41_packed.hpp"

#include "strix/ds41_formats.hpp"
#include "strix/ds41_layer.hpp"

#include <algorithm>
#include <bit>
#include <cmath>
#include <cstring>
#include <stdexcept>
#include <thread>

namespace strix::ds41 {
namespace {

constexpr std::size_t q8_block = 34;

int code_offset(PackedKind k) { return k == PackedKind::u5 ? 16 : 32; }

// Byte offsets of group g's nibbles, high bits and scale inside one row.
struct GroupPlaces {
    std::size_t nibbles, high, scale;
};
GroupPlaces places(PackedKind k, PackedLayout layout, std::size_t groups, std::size_t g) {
    const std::size_t hb = packed_high_bytes(k);
    if (layout == PackedLayout::interleaved) {
        const std::size_t base = g * packed_group_bytes(k);
        return {base + 2U, base + 18U, base};
    }
    return {16U * g, 16U * groups + hb * g, (16U + hb) * groups + 2U * g};
}

}  // namespace

std::optional<PackedChoice> choose_packing(std::span<const std::uint8_t> q8) {
    std::array<bool, 256> seen{};
    for (std::size_t b = 0; b + q8_block <= q8.size(); b += q8_block)
        for (std::size_t j = 2; j < q8_block; ++j) seen[q8[b + j]] = true;
    std::vector<int> values;
    for (int v = -128; v < 128; ++v)
        if (seen[static_cast<std::uint8_t>(static_cast<std::int8_t>(v))]) values.push_back(v);
    if (values.empty()) return std::nullopt;
    PackedChoice c;
    if (values.size() <= 16U) {
        c.kind = PackedKind::cb4;
        for (std::size_t i = 0; i < 16U; ++i) c.table[i] = static_cast<std::int8_t>(values[std::min(i, values.size() - 1U)]);
        return c;
    }
    if (values.front() >= -16 && values.back() <= 15) c.kind = PackedKind::u5;
    else if (values.front() >= -32 && values.back() <= 31) c.kind = PackedKind::u6;
    else return std::nullopt;
    return c;
}

std::vector<std::uint8_t> pack_rows(std::span<const std::uint8_t> q8, std::size_t width, const PackedChoice& choice,
                                    PackedLayout layout) {
    if (width == 0U || width % 32U != 0U || q8.size() % (width / 32U * q8_block) != 0U)
        throw std::invalid_argument("ds41 pack_rows: Q8_0 bytes are not whole rows of the width");
    const std::size_t groups = width / 32U, rows = q8.size() / (groups * q8_block), row_bytes = packed_row_bytes(choice.kind, width);
    std::array<int, 256> index{};
    index.fill(-1);
    if (choice.kind == PackedKind::cb4)
        for (std::size_t i = 16; i-- > 0;) index[static_cast<std::uint8_t>(choice.table[i])] = static_cast<int>(i);
    std::vector<std::uint8_t> out(rows * row_bytes);
    const unsigned threads = std::max(1U, std::min(16U, std::thread::hardware_concurrency()));
    std::vector<std::thread> pool;
    std::vector<int> bad(threads, 0);
    for (unsigned t = 0; t < threads; ++t)
        pool.emplace_back([&, t] {
            for (std::size_t r = rows * t / threads; r < rows * (t + 1U) / threads; ++r)
                for (std::size_t g = 0; g < groups; ++g) {
                    const std::uint8_t* in = q8.data() + (r * groups + g) * q8_block;
                    std::uint8_t* row = out.data() + r * row_bytes;
                    const auto at = places(choice.kind, layout, groups, g);
                    unsigned v[32];
                    for (unsigned j = 0; j < 32U; ++j) {
                        const int code = static_cast<std::int8_t>(in[2U + j]);
                        const int value = choice.kind == PackedKind::cb4 ? index[in[2U + j]] : code + code_offset(choice.kind);
                        const int limit = choice.kind == PackedKind::cb4 ? 15 : 2 * code_offset(choice.kind) - 1;
                        if (value < 0 || value > limit) { bad[t] = 1; return; }
                        v[j] = static_cast<unsigned>(value);
                    }
                    std::memcpy(row + at.scale, in, 2);
                    for (unsigned j = 0; j < 16U; ++j) row[at.nibbles + j] = static_cast<std::uint8_t>((v[j] & 15U) | ((v[j + 16U] & 15U) << 4U));
                    if (choice.kind != PackedKind::cb4) {
                        std::uint64_t high = 0;
                        const bool u5 = choice.kind == PackedKind::u5;
                        for (unsigned j = 0; j < 32U; ++j) high |= static_cast<std::uint64_t>(v[j] >> 4U) << (u5 ? j : 2U * j);
                        std::memcpy(row + at.high, &high, packed_high_bytes(choice.kind));
                    }
                }
        });
    for (auto& th : pool) th.join();
    if (std::any_of(bad.begin(), bad.end(), [](int b) { return b != 0; }))
        throw std::invalid_argument("ds41 pack_rows: a code is not representable in the chosen packing");
    return out;
}

void unpack_group(const std::uint8_t* row, std::size_t width, std::size_t g, const PackedChoice& choice, PackedLayout layout,
                  std::uint8_t* q8_block_out) {
    const auto at = places(choice.kind, layout, width / 32U, g);
    std::memcpy(q8_block_out, row + at.scale, 2);
    std::uint64_t high = 0;
    if (choice.kind != PackedKind::cb4) std::memcpy(&high, row + at.high, packed_high_bytes(choice.kind));
    for (unsigned j = 0; j < 32U; ++j) {
        const std::uint8_t byte = row[at.nibbles + (j & 15U)];
        unsigned v = j < 16U ? byte & 15U : byte >> 4U;
        int code = 0;
        if (choice.kind == PackedKind::cb4) {
            code = choice.table[v];
        } else {
            const bool u5 = choice.kind == PackedKind::u5;
            v |= static_cast<unsigned>((high >> (u5 ? j : 2U * j)) & (u5 ? 1U : 3U)) << 4U;
            code = static_cast<int>(v) - code_offset(choice.kind);
        }
        q8_block_out[2U + j] = static_cast<std::uint8_t>(static_cast<std::int8_t>(code));
    }
}

std::optional<std::vector<std::uint8_t>> pack_f16_as_f8(std::span<const std::uint16_t> f16, std::size_t width) {
    if (width == 0U || width % 32U != 0U || f16.size() % width != 0U) throw std::invalid_argument("ds41 pack_f16_as_f8: not whole rows");
    const std::size_t rows = f16.size() / width, rb = f8_row_bytes(width);
    std::vector<std::uint8_t> out(rows * rb);
    for (std::size_t r = 0; r < rows; ++r)
        for (std::size_t g = 0; g < width / 32U; ++g) {
            const std::uint16_t* in = f16.data() + r * width + 32U * g;
            float amax = 0.0F;
            for (unsigned j = 0; j < 32U; ++j) amax = std::max(amax, std::fabs(f16_to_float(in[j])));
            int k0 = amax > 0.0F ? static_cast<int>(std::ceil(std::log2(amax / 448.0F))) : 0;
            bool found = false;
            for (int k = k0 - 1; k <= k0 + 8 && !found; ++k) {
                if (k < -127 || k > 127) continue;
                const auto e = static_cast<std::uint8_t>(k + 127);
                const float scale = e8m0_to_float(e);
                std::uint8_t codes[32];
                bool ok = true;
                for (unsigned j = 0; j < 32U && ok; ++j) {
                    const float v = f16_to_float(in[j]);
                    const float u = v / scale;  // exact when v is E4M3 x 2^k
                    codes[j] = float_to_e4m3(u);
                    const float back = e4m3_to_float(codes[j]) * scale;
                    ok = std::bit_cast<std::uint32_t>(back) == std::bit_cast<std::uint32_t>(v);
                }
                if (!ok) continue;
                std::memcpy(out.data() + r * rb + 32U * g, codes, 32);
                out[r * rb + width + g] = e;
                found = true;
            }
            if (!found) return std::nullopt;
        }
    return out;
}

float unpack_f8(const std::uint8_t* row, std::size_t width, std::size_t i) {
    return e4m3_to_float(row[i]) * e8m0_to_float(row[width + i / 32U]);
}

}  // namespace strix::ds41
