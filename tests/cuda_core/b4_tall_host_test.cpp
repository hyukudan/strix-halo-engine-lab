#include "core/tall_bf16_reference.hpp"
#include "core/tall_chunk_policy.hpp"
#include "strix/weight/formats/scalar_float.hpp"

#include <array>
#include <bit>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <limits>
#include <span>
#include <vector>

namespace tall = strix::core::tall;
namespace scalar = strix::weight::scalar_float;

namespace {

void check(bool condition, const char* message) {
    if (condition) return;
    std::fprintf(stderr, "FAIL B4 %s\n", message);
    std::exit(1);
}

void test_order() {
    using Plan = tall::Bf16ExactPlan;
    static_assert(Plan::rows == 64 && Plan::outputs == 64 && Plan::block_k == 32 && Plan::step_k == 16);

    constexpr std::array blocks{0U, 1U, 2U, 3U, 32U, 4U, 5U, 6U, 7U, 8U, 9U, 10U,
        11U, 12U, 13U, 14U, 15U, 16U, 17U, 18U, 19U, 20U, 21U, 22U, 23U, 24U, 25U,
        26U, 27U, 28U, 29U, 30U, 31U};
    std::size_t call = 0;
    const auto result = tall::bf16_exact_reference(1056, [&](float acc, std::uint32_t k, int plane) {
        check(call < blocks.size() * 6, "unexpected K16 operation");
        const auto expected_k = blocks[call / 6] * 32U + ((call % 6) / 3) * 16U;
        const auto expected_plane = 2 - static_cast<int>(call % 3);
        check(k == expected_k && plane == expected_plane, "K16 or plane order");
        ++call;
        if (plane != 0) return acc;
        if (k == 0) return acc + 0x1p24F;
        if (k == 128) return acc + 1.0F;
        if (k == 1024) return acc - 0x1p24F;
        return acc;
    });
    check(call == blocks.size() * 6 && std::bit_cast<std::uint32_t>(result) == 0x3f800000U,
          "segment fold must preserve the unit across K wrap");
    const auto fold = tall::bf16_exact_reference(384, [](float acc, std::uint32_t k, int plane) {
        if (plane != 0) return acc;
        if (k == 0) return acc + 0x1p24F;
        if (k == 128) return acc - 0x1p24F;
        if (k == 256) return acc + 1.0F;
        return acc;
    });
    check(std::bit_cast<std::uint32_t>(fold) == 0x3f800000U, "segments must fold in ascending order");
    std::puts("PASS B4 fixed K16, plane and segment order");
}

constexpr std::array classes{tall::ChunkClass{128, 40}, tall::ChunkClass{16, 8},
                             tall::ChunkClass{32, 16}, tall::ChunkClass{64, 24}};

void expect_chunk(std::uint64_t remaining, tall::ChunkPolicy policy, std::uint32_t rows, std::uint32_t capacity) {
    const auto chunk = tall::resolve_chunk(remaining, policy, classes);
    check(chunk.rows == rows && chunk.capacity == capacity, "chunk class or mask");
}

void test_chunk_policy() {
    constexpr auto max = std::numeric_limits<std::uint32_t>::max();
    expect_chunk(1000, {8, 17, 4, 40}, 64, 64);  // ceil(136 / 4) = 34.
    expect_chunk(1000, {5, 13, 2, 40}, 64, 64);  // ceil(65 / 2) = 33.
    expect_chunk(1000, {8, 17, 4, 23}, 32, 32);
    expect_chunk(17, {8, 17, 4, 40}, 17, 32);
    expect_chunk(16, {8, 17, 4, 40}, 16, 16);
    expect_chunk(17, {8, 17, 4, 8}, 16, 16);
    expect_chunk(1, {8, 17, 4, 7}, 0, 0);
    expect_chunk(0, {8, 17, 4, 40}, 0, 0);
    expect_chunk(1, {8, 17, 0, 40}, 0, 0);
    expect_chunk(1, {8, 3, 4, 40}, 0, 0);
    expect_chunk(1, {0, 17, 4, 40}, 0, 0);
    expect_chunk(std::numeric_limits<std::uint64_t>::max(), {max, max, 2, 40}, 128, 128);
    std::puts("PASS B4 memory-bounded chunk classes and masked tail");
}

// All operands and all intermediates in this case are small integers, so each
// K16 result is exactly representable independent of its internal dot order.
void test_reference() {
    constexpr std::uint32_t width = 160;
    constexpr std::size_t rows = 147;
    constexpr auto planes = static_cast<std::size_t>(tall::Bf16ExactPlan::planes);
    std::array<float, width> weights{};
    for (std::uint32_t k = 0; k < width; ++k) weights[k] = static_cast<float>(static_cast<int>(k % 5) - 2);
    std::vector<std::array<float, planes * width>> input(rows);
    std::array<std::uint32_t, rows> expected{}, output{};
    for (std::size_t row = 0; row < rows; ++row) {
        float sum = 0.0F;
        for (std::size_t plane = 0; plane < planes; ++plane) {
            for (std::uint32_t k = 0; k < width; ++k) {
                const auto value = static_cast<float>(static_cast<int>((k + 3 * row + plane) % 11) - 5);
                input[row][plane * width + k] = scalar::bf16_round(value);
                sum += value * weights[k];
            }
        }
        expected[row] = std::bit_cast<std::uint32_t>(sum);
    }

    for (const auto budget : {8U, 16U, 24U, 40U}) {
        for (std::size_t first = 0; first < rows;) {
            const auto chunk = tall::resolve_chunk(rows - first, {17, 8, 1, budget}, classes);
            check(chunk.rows != 0, "reference chunk made no progress");
            const std::span piece(input.data() + first, chunk.rows);
            for (std::size_t row = 0; row < piece.size(); ++row) {
                const auto value = tall::bf16_exact_reference(width, [&](float acc, std::uint32_t k, int plane) {
                    const auto base = static_cast<std::size_t>(plane) * width;
                    for (auto i = k; i < k + 16U; ++i) acc += piece[row][base + i] * weights[i];
                    return acc;
                });
                output[first + row] = std::bit_cast<std::uint32_t>(value);
            }
            first += chunk.rows;
        }
        check(output == expected, "reference bits across chunks");
    }
    std::puts("PASS B4 BF16 reference bits across chunk boundaries");
}

}  // namespace

int main() {
    test_order();
    test_chunk_policy();
    test_reference();
}
