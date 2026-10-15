#include "virtual_q8_oracle.hpp"
#include <algorithm>
#include <array>
#include <bit>
#include <cstdint>
#include <iostream>
#include <random>
#include <vector>

namespace {
std::uint32_t bf16_bits(float value) {
    const auto bits = std::bit_cast<std::uint32_t>(value);
    return (bits + 0x7fffU + ((bits >> 16U) & 1U)) & 0xffff0000U;
}
}
int main() {
    // The prefetch loader's group predicate is safe for every admitted tail.
    for (std::uint64_t width = 32; width <= 8192; width += 32) {
        for (std::uint64_t chunk = 0; chunk < width; chunk += 256) {
            for (unsigned lane = 0; lane < 32; ++lane) {
                const auto base = chunk + 8U * lane;
                for (unsigned local = 0; local < 8; ++local)
                    if ((base + local < width) != (base < width)) return 3;
                if (base < width && (base / 32U != (base + 7U) / 32U || base + 7U >= width)) return 4;
            }
        }
    }
    constexpr std::uint64_t maximum_aligned_width = 0xffffffe0ULL;
    const auto final_chunk = ((maximum_aligned_width - 1U) / 256U) * 256U;
    for (unsigned lane = 0; lane < 32; ++lane) {
        const auto base = final_chunk + 8U * lane;
        for (unsigned local = 0; local < 8; ++local)
            if ((base + local < maximum_aligned_width) != (base < maximum_aligned_width)) return 5;
    }
    // Host-executed coefficient reassociation mutant. The changed arithmetic
    // differs even after the production BF16 rounding, rather than only FP32.
    {
        constexpr std::size_t width = 2560;
        const float activation = std::ldexp(1.0F, -126);
        const float scale = std::ldexp(1.0F, -24);
        std::vector<float> inputs(width, activation), decoded(width, scale * 127.0F);
        volatile float premature_product = activation * scale;
        const float premature_rounded = premature_product;
        std::vector<float> reassociated_inputs(width, premature_rounded), codes(width, 127.0F);
        const auto correct = bf16_bits(virtual_q8_test::legacy_oracle(inputs, decoded));
        const auto mutant = bf16_bits(virtual_q8_test::legacy_oracle(reassociated_inputs, codes));
        if (correct != 0x00020000U || mutant != 0U) return 6;
        std::cout << "Decode-first underflow mutant detected after BF16 rounding: "
                  << correct << " vs " << mutant << '\n';
    }
    {
        std::vector<float> inputs(2560, 1.0F), weights(2560, -0.0F), wrong_weights(2560, 0.0F);
        for (unsigned feature = 0; feature < 256; ++feature) {
            inputs[feature] = -std::ldexp(1.0F, -133);
            weights[feature] = std::ldexp(1.0F, -24);
            wrong_weights[feature] = weights[feature];
        }
        if (bf16_bits(virtual_q8_test::legacy_oracle(inputs, weights)) != 0x80000000U ||
            bf16_bits(virtual_q8_test::legacy_oracle(inputs, wrong_weights)) != 0U) return 7;
        std::cout << "Signed-zero coefficient mutant detected after BF16 rounding\n";
    }
    using virtual_q8_test::Mutation;
    constexpr std::array mutants{Mutation::local_first, Mutation::cross_reverse,
        Mutation::local_reverse, Mutation::wrong_mapping, Mutation::descending_leaves,
        Mutation::negative_zero, Mutation::first_product, Mutation::unfused};
    std::array<bool, mutants.size()> detected{};
    std::mt19937 random(0x6d6e256U);
    for (const std::size_t width : {128U, 288U, 640U, 2560U}) {
        std::vector<float> input(width), weights(width);
        const auto assess = [&] {
            const auto expected = bf16_bits(virtual_q8_test::legacy_oracle(input, weights));
            if (std::bit_cast<std::uint32_t>(virtual_q8_test::legacy_oracle(input, weights)) !=
                std::bit_cast<std::uint32_t>(virtual_q8_test::legacy_oracle(input, weights, Mutation::correct_mapping)))
                throw std::runtime_error("correct virtual register tree disagrees with original tree");
            for (std::size_t m = 0; m < mutants.size(); ++m)
                detected[m] = detected[m] || bf16_bits(virtual_q8_test::legacy_oracle(input, weights, mutants[m])) != expected;
        };
        // One activation vector is shared by every row in each batch.
        for (unsigned batch = 0; batch < 200; ++batch) {
            for (auto& x : input) x = std::bit_cast<float>(bf16_bits(std::ldexp(
                1.0F + static_cast<float>(random() % 128U) / 128.0F,
                static_cast<int>(random() % 21U) - 10))) * ((random() & 1U) ? 1.0F : -1.0F);
            for (unsigned row = 0; row < 16; ++row) {
                for (std::size_t block = 0; block < width / 32U; ++block) {
                    // Exact finite half scales, signed Q8 codes including -128.
                    const float scale = std::ldexp(1.0F + static_cast<float>(random() % 1024U) / 1024.0F,
                        static_cast<int>(random() % 16U) - 10);
                    for (unsigned local = 0; local < 32; ++local)
                        weights[block * 32U + local] = scale * static_cast<float>(static_cast<int>(random() % 256U) - 128);
                }
                assess();
            }
        }
        std::fill(input.begin(), input.end(), -1.0F);
        std::fill(weights.begin(), weights.end(), 0.0F); assess();
        if (width >= 257U) {
            std::fill(input.begin(), input.end(), 0.0F);
            std::fill(weights.begin(), weights.end(), 0.0F);
            // Canonical half scale 32768 in the shared block of features 0/16.
            input[0] = 8.0F; weights[0] = 32768.0F * 64.0F;
            input[128] = 4.0F; weights[128] = 32768.0F * -128.0F;
            input[16] = std::ldexp(1.0F, -15); weights[16] = 32768.0F;
            assess();
            std::fill(input.begin(), input.end(), 0.0F);
            std::fill(weights.begin(), weights.end(), 0.0F);
            input[0] = 1.8515625F; input[256] = 1.921875F;
            weights[0] = 1.5849609375F * 107.0F;
            weights[256] = 1.4638671875F * -117.0F;
            assess();
            if (bf16_bits(virtual_q8_test::legacy_oracle(input, weights)) !=
                    std::bit_cast<std::uint32_t>(-15.1875F) ||
                bf16_bits(virtual_q8_test::legacy_oracle(input, weights, Mutation::unfused)) !=
                    std::bit_cast<std::uint32_t>(-15.125F)) return 2;
        }
    }
    for (std::size_t m = 0; m < detected.size(); ++m)
        if (!detected[m]) { std::cerr << "BF16-invisible mutant " << m << '\n'; return 1; }
    std::cout << "All eight virtual Q8 mutants detected after GDN BF16 rounding\n";
    return 0;
}
