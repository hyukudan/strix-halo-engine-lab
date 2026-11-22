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
    return same_bits(flat, flat_ref) && same_bits(chunk, chunk_ref) ? 0 : 1;
}
