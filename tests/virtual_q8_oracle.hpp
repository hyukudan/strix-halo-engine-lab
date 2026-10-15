#pragma once

#include <array>
#include <cmath>
#include <cstddef>
#include <span>
#include <stdexcept>

namespace virtual_q8_test {
enum class Mutation { none, local_first, cross_reverse, local_reverse,
    wrong_mapping, descending_leaves, negative_zero, first_product, unfused, correct_mapping };

// Independent CPU statement of the measured legacy contract: 256 leaves,
// ascending feature order, fused updates, then the 128..1 binary tree.
inline float legacy_oracle(std::span<const float> input, std::span<const float> weight,
    Mutation mutation = Mutation::none) {
    if (input.size() != weight.size()) throw std::invalid_argument("oracle extent mismatch");
    std::array<float, 256> leaves{};
    for (std::size_t lane = 0; lane < leaves.size(); ++lane) {
        float sum = mutation == Mutation::negative_zero ? -0.0F : 0.0F;
        bool first = true;
        const auto count = lane < input.size() ? (input.size() - 1U - lane) / 256U + 1U : 0U;
        for (std::size_t term = 0; term < count; ++term) {
            const auto selected = mutation == Mutation::descending_leaves ? count - 1U - term : term;
            const auto feature = lane + selected * 256U;
            if (mutation == Mutation::unfused || (mutation == Mutation::first_product && first)) {
                volatile float product = input[feature] * weight[feature];
                sum = mutation == Mutation::first_product && first ? product : sum + product;
            } else sum = std::fmaf(input[feature], weight[feature], sum);
            first = false;
        }
        leaves[lane] = sum;
    }
    if (mutation == Mutation::none || mutation == Mutation::descending_leaves ||
        mutation == Mutation::negative_zero || mutation == Mutation::first_product ||
        mutation == Mutation::unfused) {
        for (std::size_t stride = 128; stride; stride >>= 1U)
            for (std::size_t lane = 0; lane < stride; ++lane)
                leaves[lane] += leaves[lane + stride];
        return leaves[0];
    }
    std::array<std::array<float, 8>, 32> registers{};
    for (std::size_t lane = 0; lane < 32; ++lane)
        for (std::size_t local = 0; local < 8; ++local)
            registers[lane][local] = leaves[mutation == Mutation::wrong_mapping ? lane + 32U * local : 8U * lane + local];
    const auto local_reduce = [&](bool reversed) {
        for (auto& values : registers) {
            if (reversed) {
                for (std::size_t stride = 1; stride <= 4; stride <<= 1U)
                    for (std::size_t local = 0; local < stride; ++local)
                        values[local] += values[local + stride];
            } else {
                for (std::size_t stride = 4; stride; stride >>= 1U)
                    for (std::size_t local = 0; local < stride; ++local)
                        values[local] += values[local + stride];
            }
        }
    };
    if (mutation == Mutation::local_first) local_reduce(false);
    if (mutation == Mutation::cross_reverse) {
        for (std::size_t distance = 1; distance <= 16; distance <<= 1U)
            for (std::size_t lane = 0; lane < distance; ++lane)
                for (std::size_t local = 0; local < 8; ++local)
                    registers[lane][local] += registers[lane + distance][local];
    } else {
        for (std::size_t distance = 16; distance; distance >>= 1U)
            for (std::size_t lane = 0; lane < distance; ++lane)
                for (std::size_t local = 0; local < 8; ++local)
                    registers[lane][local] += registers[lane + distance][local];
    }
    if (mutation != Mutation::local_first) local_reduce(mutation == Mutation::local_reverse);
    return registers[0][0];
}
} // namespace virtual_q8_test
