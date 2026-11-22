#pragma once

#include "core/residual_mix.hpp"

#include <array>
#include <bit>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <vector>

namespace strix::tests::mix {

using core::residual_mix::MixArgs;

// Independent CPU emulation of the incumbent CUDA's virtual lanes, including
// the explicit FMA chains and the order of the 256-lane tree. The host exp
// result is bit-comparable to native CUDA only for the fixture's equal logits.
template <class Product>
float lane_reduce(unsigned width, Product product) {
    std::array<float, 256> sums{};
    for (std::size_t lane = 0; lane < sums.size(); ++lane)
        for (std::size_t feature = lane; feature < width; feature += sums.size())
            sums[lane] = product(feature, sums[lane]);
    for (std::size_t stride = sums.size() / 2U; stride > 0U; stride /= 2U)
        for (std::size_t lane = 0; lane < stride; ++lane) sums[lane] += sums[lane + stride];
    return sums[0];
}

// Output layout is [query][row][width]. No output pointers in args are read.
inline std::vector<float> readout_oracle(const MixArgs& args) {
    const auto image = std::size_t{args.rows} * args.width;
    std::vector<float> output(args.query_count * image);
    const auto width = static_cast<float>(args.width);
    const float score_scale = 1.0F / std::sqrt(width);
    std::array<std::vector<float>, core::residual_mix::kMaxBanks> normalized;
    for (unsigned b = 0; b < args.bank_count; ++b) normalized[b].resize(args.width);

    for (unsigned row = 0; row < args.rows; ++row) {
        const auto offset = std::size_t{row} * args.width;
        for (unsigned b = 0; b < args.bank_count; ++b) {
            const float* bank = args.banks[b] + offset;
            if (args.normalized) {
                normalized[b].assign(bank, bank + args.width);
                continue;
            }
            const float squares = lane_reduce(args.width, [bank](std::size_t i, float sum) {
                return std::fma(bank[i], bank[i], sum);
            });
            const float inverse = 1.0F / std::sqrt(squares / width + args.epsilon);
            for (unsigned i = 0; i < args.width; ++i) normalized[b][i] = bank[i] * inverse;
        }

        for (unsigned q = 0; q < args.query_count; ++q) {
            std::array<float, core::residual_mix::kMaxBanks> scores{};
            const float* query = args.queries + std::size_t{q} * args.width;
            for (unsigned b = 0; b < args.bank_count; ++b) {
                scores[b] = lane_reduce(args.width, [&](std::size_t i, float sum) {
                    return std::fma(query[i], normalized[b][i], sum);
                }) * score_scale;
            }
            float highest = scores[0];
            for (unsigned b = 1; b < args.bank_count; ++b) highest = std::fmax(scores[b], highest);
            float total = 0.0F;
            for (unsigned b = 0; b < args.bank_count; ++b) {
                scores[b] = std::exp(scores[b] - highest);
                total += scores[b];
            }
            float* result = output.data() + q * image + offset;
            for (unsigned b = 0; b < args.bank_count; ++b) {
                const float probability = scores[b] / total;
                for (unsigned i = 0; i < args.width; ++i)
                    result[i] = std::fma(probability, normalized[b][i], result[i]);
            }
        }
    }
    return output;
}

inline std::uint32_t bits(float x) { return std::bit_cast<std::uint32_t>(x); }

} // namespace strix::tests::mix
