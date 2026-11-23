#include "core/tp_join_sum.hpp"

#include <array>
#include <bit>
#include <cstddef>
#include <cstdint>
#include <cstdio>

namespace {

template <std::size_t N>
bool same_bits(const std::array<float, N>& actual, const std::array<float, N>& expected) {
    for (std::size_t i = 0; i < N; ++i) {
        if (std::bit_cast<std::uint32_t>(actual[i]) != std::bit_cast<std::uint32_t>(expected[i])) {
            std::fprintf(stderr, "TP join differs at output index %zu\n", i);
            return false;
        }
    }
    return true;
}

}  // namespace

int main() {
    using namespace strix::core::tp_join;
    constexpr std::uint32_t rows = 2, cols = 3, width = 7, col0 = 2;
    constexpr std::uint32_t count = rows * cols;
    const std::array<float, count> a{1.0F, -0.0F, 0x1p24F, -2.0F, 3.0F, 0.0F};
    const std::array<float, count> b{0x1p-24F, -0.0F, -0x1p24F, 5.0F, -0x1p-22F, -0.0F};

    std::array<float, count + 1> flat{};
    flat.fill(17.0F);
    auto flat_ref = flat;
    for (std::uint32_t i = 0; i <= count; ++i) {
        if (i < count) flat_ref[i] = a[i] + b[i];
        join_sum_at(a.data(), b.data(), flat.data(), i, Sum{count});
    }

    std::array<float, rows * width> chunk{};
    chunk.fill(17.0F);
    auto chunk_ref = chunk;
    for (std::uint32_t i = 0; i <= count; ++i) {
        if (i < rows * cols) chunk_ref[std::size_t{i / cols} * width + col0 + i % cols] = a[i] + b[i];
        join_sum_at(a.data(), b.data(), chunk.data(), i, SumColumns{rows, cols, col0, width});
    }
    // Pack / place of a column slice: the index the private Engram / staged-ready copies used,
    // full[(i / half) * stride + first + i % half], for every element and one past the end (guard).
    bool slices = true;
    const std::array<std::array<std::uint32_t, 4>, 4> shapes{{{1, 1, 0, 1}, {3, 4, 4, 8}, {5, 3, 0, 7}, {2, 7, 9, 16}}};
    for (const auto& s : shapes) {
        const std::uint32_t r = s[0], half = s[1], first = s[2], stride = s[3];
        std::array<float, 128> full{}, packed{}, back{}, back_ref{};
        for (std::size_t k = 0; k < full.size(); ++k) full[k] = static_cast<float>(k) * 0.5F - 3.0F;
        packed.fill(-1.0F);
        back.fill(17.0F);
        back_ref.fill(17.0F);
        for (std::uint32_t i = 0; i <= r * half; ++i) {
            pack_at(full.data(), packed.data(), i, Columns{r, half, first, stride});
            place_at(full.data(), back.data(), i, Columns{r, half, first, stride});
        }
        for (std::uint32_t i = 0; i < r * half; ++i) {
            const std::size_t at = std::size_t{i / half} * stride + first + i % half;
            back_ref[at] = full[i];
            if (std::bit_cast<std::uint32_t>(packed[i]) != std::bit_cast<std::uint32_t>(full[at])) slices = false;
        }
        if (std::bit_cast<std::uint32_t>(packed[r * half]) != std::bit_cast<std::uint32_t>(-1.0F)) slices = false;  // guard
        if (!same_bits(back, back_ref)) slices = false;
    }
    if (!slices) std::fprintf(stderr, "TP pack / place differs from the slice index\n");
    return same_bits(flat, flat_ref) && same_bits(chunk, chunk_ref) && slices ? 0 : 1;
}
