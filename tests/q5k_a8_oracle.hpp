#pragma once
// Independent original-byte Q5_K oracle. No production decoder/dot/evaluator calls.
// Reuses only the existing independent binary32 activation and half/factor oracles.
#include "q4k_a8_oracle.hpp"
#include <limits>

namespace q5k_a8_oracle {
enum class Mutation { none, omit_qh, wrong_qh_plane, signed_weight, reverse_blocks, wrong_tree, fused };
inline unsigned code(std::span<const std::byte> block, unsigned feature, Mutation mutation = Mutation::none) {
    if (block.size() != 176 || feature >= 256) throw std::invalid_argument("Q5 oracle block/feature");
    const auto group = feature/32U, lane = feature%32U;
    const auto packed = std::to_integer<unsigned>(block[48U+(feature/64U)*32U+lane]);
    unsigned result = (packed>>(4U*(group%2U)))&15U;
    const auto plane = mutation == Mutation::wrong_qh_plane ? group^1U : group;
    if (mutation != Mutation::omit_qh)
        result += ((std::to_integer<unsigned>(block[16U+lane])>>plane)&1U)*16U;
    return result;
}
inline float row(std::span<const std::byte> weights, const strix::moe_q4k_a8::QuantizedRow& a,
    Mutation mutation = Mutation::none) {
    if (a.codes.empty() || a.codes.size()%256U || weights.size() != a.codes.size()/256U*176U ||
        a.deltas.size() != a.codes.size()/32U || a.sums.size() != a.deltas.size())
        throw std::invalid_argument("Q5 oracle geometry");
    std::array<float,4> partial{};
    for (std::size_t iteration = 0; iteration < a.codes.size()/256U; ++iteration) {
        const auto block = mutation == Mutation::reverse_blocks ? a.codes.size()/256U-1U-iteration : iteration;
        const auto bytes = weights.subspan(block*176U,176U);
        const auto at = [&](unsigned i) {return std::to_integer<unsigned>(bytes[i]);};
        const float d = q4k_a8_oracle::half(static_cast<std::uint16_t>(at(0)|(at(1)<<8U)));
        const float dm = q4k_a8_oracle::half(static_cast<std::uint16_t>(at(2)|(at(3)<<8U)));
        if (!std::isfinite(d) || !std::isfinite(dm)) throw std::invalid_argument("Q5 oracle nonfinite header");
        std::array<float,8> contribution{};
        for (unsigned group = 0; group < 8; ++group) {
            std::int64_t dot = 0, sum = 0;
            for (unsigned lane = 0; lane < 32; ++lane) {
                const auto value = code(bytes,group*32U+lane,mutation);
                const auto weight = mutation == Mutation::signed_weight && value >= 16 ?
                    static_cast<int>(value)-32 : static_cast<int>(value);
                const int activation = a.codes[block*256U+group*32U+lane];
                dot += weight*activation; sum += activation;
            }
            const auto [scale,minimum] = q4k_a8_oracle::factors(bytes.subspan(4,12),group);
            const float scaled = static_cast<float>(dot*scale), correction = static_cast<float>(sum*minimum);
            const float value = mutation == Mutation::fused ? std::fma(d,scaled,-q4k_a8_oracle::mul(dm,correction)) :
                q4k_a8_oracle::sub(q4k_a8_oracle::mul(d,scaled),q4k_a8_oracle::mul(dm,correction));
            contribution[group] = q4k_a8_oracle::mul(a.deltas[block*8U+group],value);
            if (!std::isfinite(contribution[group])) throw std::invalid_argument("Q5 oracle nonfinite contribution");
        }
        for (unsigned pair = 0; pair < 4; ++pair)
            partial[pair] = q4k_a8_oracle::add(partial[pair],
                q4k_a8_oracle::add(contribution[2U*pair],contribution[2U*pair+1U]));
    }
    const auto& v = partial;
    const float result = mutation == Mutation::wrong_tree ?
        q4k_a8_oracle::add(q4k_a8_oracle::add(v[0],v[2]),q4k_a8_oracle::add(v[1],v[3])) :
        q4k_a8_oracle::add(q4k_a8_oracle::add(v[0],v[1]),q4k_a8_oracle::add(v[2],v[3]));
    if (!std::isfinite(result)) throw std::invalid_argument("Q5 oracle nonfinite reduction");
    return result;
}
} // namespace q5k_a8_oracle
