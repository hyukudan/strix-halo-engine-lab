#include "virtual_q8_oracle.hpp"

#include <array>
#include <algorithm>
#include <bit>
#include <cstdint>
#include <iostream>
#include <random>
#include <vector>

int main() {
    using virtual_q8_test::Mutation;
    constexpr std::array mutants{Mutation::local_first, Mutation::cross_reverse,
        Mutation::local_reverse, Mutation::wrong_mapping, Mutation::descending_leaves,
        Mutation::negative_zero, Mutation::first_product, Mutation::unfused};
    std::array<bool, mutants.size()> detected{};
    std::mt19937 random(0x256032U);
    std::vector<float> input(768), weights(768);
    const auto assess = [&] {
        const auto expected = std::bit_cast<std::uint32_t>(virtual_q8_test::legacy_oracle(input, weights));
        for (std::size_t index = 0; index < mutants.size(); ++index)
            detected[index] = detected[index] || std::bit_cast<std::uint32_t>(
                virtual_q8_test::legacy_oracle(input, weights, mutants[index])) != expected;
    };
    for (unsigned trial = 0; trial < 200; ++trial) {
        for (std::size_t feature = 0; feature < input.size(); ++feature) {
            const auto sign = (random() & 1U) ? 1.0F : -1.0F;
            input[feature] = sign * std::ldexp(1.0F + static_cast<float>(random() % 1024U) / 1024.0F,
                static_cast<int>(random() % 41U) - 20);
            const float scale = std::ldexp(1.0F + static_cast<float>(random() % 1024U) / 1024.0F,
                static_cast<int>(random() % 16U) - 10);
            weights[feature] = scale * static_cast<float>(static_cast<int>(random() % 256U) - 128);
        }
        assess();
    }
    std::fill(input.begin(), input.end(), -1.0F);
    std::fill(weights.begin(), weights.end(), 0.0F);
    assess();
    if (std::bit_cast<std::uint32_t>(virtual_q8_test::legacy_oracle(input, weights)) != 0U) return 2;
    std::fill(input.begin(), input.end(), 0.0F);
    std::fill(weights.begin(), weights.end(), 1.0F);
    input[0] = -(1.0F + std::ldexp(1.0F, -10) + std::ldexp(1.0F, -23));
    input[256] = 1.0F + std::ldexp(1.0F, -23);
    weights[256] = 1.0F + std::ldexp(1.0F, -10);
    assess();
    for (std::size_t index = 0; index < detected.size(); ++index) {
        if (!detected[index]) {
            std::cerr << "Undetected oracle mutant " << index << '\n';
            return 1;
        }
    }
    std::cout << "CPU legacy Q8 oracle: all eight mutation controls detected\n";
    return 0;
}
