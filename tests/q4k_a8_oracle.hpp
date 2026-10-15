#pragma once
// Independent test oracle: no production quantizer, factor or combine calls.
#include "strix/moe_q4k_a8_contract.hpp"
#include <algorithm>
#include <array>
#include <bit>
#include <cmath>
#include <cstdint>
#include <span>
#include <stdexcept>

namespace q4k_a8_oracle {
inline std::uint64_t rne_ratio(std::uint64_t numerator, std::uint64_t denominator) {
    auto q = numerator/denominator; const auto r = numerator%denominator;
    if (r*2U > denominator || (r*2U == denominator && (q&1U))) ++q;
    return q;
}
// Correct binary32 division from integer significands, independent of native /.
// Finite dividend and finite nonzero divisor. Handles subnormals and signed0.
inline std::uint32_t divide(std::uint32_t a, std::uint32_t b) {
    const auto sign = (a^b)&0x80000000U; a &= 0x7fffffffU; b &= 0x7fffffffU;
    if (a >= 0x7f800000U || b == 0 || b >= 0x7f800000U) throw std::invalid_argument("oracle division domain");
    if (a == 0) return sign;
    const auto unpack = [](std::uint32_t bits) {
        const auto exp = bits>>23U; std::uint32_t m = (bits&0x7fffffU)|(exp ? 0x800000U : 0U);
        int e = exp ? static_cast<int>(exp)-150 : -149;
        const auto shift = std::countl_zero(m)-8; m <<= static_cast<unsigned>(shift); e -= shift;
        return std::pair{m,e};
    };
    const auto [ma,ea] = unpack(a); const auto [mb,eb] = unpack(b); const int difference = ea-eb;
    int exponent = difference+(ma < mb ? -1 : 0);
    if (exponent < -126) {
        const int shift = difference+149;
        if (shift < -1) return sign;
        const auto n = static_cast<std::uint64_t>(ma) << static_cast<unsigned>(std::max(shift,0));
        const auto d = static_cast<std::uint64_t>(mb) << static_cast<unsigned>(std::max(-shift,0));
        return sign|static_cast<std::uint32_t>(rne_ratio(n,d));
    }
    if (exponent > 127) return sign|0x7f800000U;
    const auto shift = static_cast<unsigned>(difference-exponent+23);
    auto m = rne_ratio(static_cast<std::uint64_t>(ma)<<shift,mb);
    if (m == 0x1000000U) { m >>= 1U; ++exponent; }
    if (exponent > 127) return sign|0x7f800000U;
    return sign|(static_cast<std::uint32_t>(exponent+127)<<23U)|(static_cast<std::uint32_t>(m)-0x800000U);
}
inline strix::moe_q4k_a8::QuantizedRow quantize(std::span<const float> values) {
    if (values.empty() || values.size()%32U) throw std::invalid_argument("oracle groups");
    strix::moe_q4k_a8::QuantizedRow result; result.codes.resize(values.size());
    for (std::size_t group = 0; group < values.size()/32U; ++group) {
        std::uint32_t maximum = 0;
        for (unsigned l = 0; l < 32; ++l) {
            const auto bits = std::bit_cast<std::uint32_t>(values[group*32U+l])&0x7fffffffU;
            if (bits >= 0x7f800000U) throw std::invalid_argument("oracle nonfinite input");
            maximum = std::max(maximum,bits);
        }
        const auto delta_bits = maximum == 0 ? 0x3f800000U : divide(maximum,0x42fe0000U);
        if (delta_bits == 0 || delta_bits >= 0x7f800000U) throw std::invalid_argument("oracle delta invalid");
        result.deltas.push_back(std::bit_cast<float>(delta_bits)); std::int32_t sum = 0;
        for (unsigned l = 0; l < 32; ++l) {
            const auto bits = divide(std::bit_cast<std::uint32_t>(values[group*32U+l]),delta_bits);
            const double magnitude = std::fabs(static_cast<double>(std::bit_cast<float>(bits)));
            int code = static_cast<int>(std::floor(magnitude)); const double fraction = magnitude-code;
            if (fraction > 0.5 || (fraction == 0.5 && (code&1))) ++code;
            code = std::min(code,127); if (bits&0x80000000U) code = -code;
            result.codes[group*32U+l] = static_cast<std::int8_t>(code); sum += code;
        }
        result.sums.push_back(sum);
    }
    return result;
}
inline float half(std::uint16_t value) {
    const auto sign = static_cast<std::uint32_t>(value&0x8000U)<<16U;
    const auto exponent = (value>>10U)&31U; std::uint32_t fraction = value&1023U;
    if (exponent == 31U) return std::bit_cast<float>(sign|0x7f800000U|(fraction<<13U));
    if (exponent == 0) {
        if (fraction == 0) return std::bit_cast<float>(sign);
        unsigned shift = 0; while ((fraction&1024U) == 0) { fraction <<= 1U; ++shift; }
        return std::bit_cast<float>(sign|((113U-shift)<<23U)|((fraction&1023U)<<13U));
    }
    return std::bit_cast<float>(sign|((exponent+112U)<<23U)|(fraction<<13U));
}
inline std::pair<int,int> factors(std::span<const std::byte> packed, unsigned group) {
    const auto at = [&](unsigned i) { return static_cast<int>(std::to_integer<unsigned char>(packed[i])); };
    if (group < 4) return {at(group)&63,at(group+4)&63};
    return {(at(group+4)&15)|((at(group-4)>>6)<<4),(at(group+4)>>4)|((at(group)>>6)<<4)};
}
inline float mul(float a,float b) { volatile float value = a*b; return value; }
inline float add(float a,float b) { volatile float value = a+b; return value; }
inline float sub(float a,float b) { volatile float value = a-b; return value; }
enum class Mutation { none, fused, reverse_blocks, wrong_tree };
inline float row(std::span<const std::byte> weights, const strix::moe_q4k_a8::QuantizedRow& a, Mutation mutation = Mutation::none) {
    std::array<float,4> accum{};
    for (std::size_t iteration = 0; iteration < a.codes.size()/256U; ++iteration) {
        const auto block = mutation == Mutation::reverse_blocks ? a.codes.size()/256U-1U-iteration : iteration;
        std::array<float,8> contributions{};
        const auto bytes = weights.subspan(block*144U,144U);
        const auto at = [&](unsigned i) { return std::to_integer<unsigned char>(bytes[i]); };
        const float d = half(static_cast<std::uint16_t>(at(0)|(static_cast<unsigned>(at(1))<<8U)));
        const float dm = half(static_cast<std::uint16_t>(at(2)|(static_cast<unsigned>(at(3))<<8U)));
        if (!std::isfinite(d) || !std::isfinite(dm)) throw std::invalid_argument("oracle nonfinite header");
        for (unsigned j = 0; j < 8; ++j) {
            std::int64_t dot = 0, sum = 0;
            for (unsigned l = 0; l < 32; ++l) {
                const auto packed = at(16U+(j/2U)*32U+l); const int q = j&1U ? packed>>4U : packed&15U;
                const int code = a.codes[block*256U+j*32U+l]; dot += q*code; sum += code;
            }
            const auto [scale,minimum] = factors(bytes.subspan(4,12),j);
            const float i = static_cast<float>(scale*dot), k = static_cast<float>(minimum*sum);
            const float t = mutation == Mutation::fused ? std::fma(d,i,-mul(dm,k)) : sub(mul(d,i),mul(dm,k));
            contributions[j] = mul(a.deltas[block*8U+j],t);
        }
        for (unsigned pair = 0; pair < 4; ++pair)
            accum[pair] = add(accum[pair],add(contributions[pair*2U],contributions[pair*2U+1U]));
    }
    const float value = mutation == Mutation::wrong_tree ? add(add(accum[0],accum[2]),add(accum[1],accum[3])) :
        add(add(accum[0],accum[1]),add(accum[2],accum[3]));
    if (!std::isfinite(value)) throw std::invalid_argument("oracle combine overflow");
    return value;
}
}  // namespace q4k_a8_oracle
