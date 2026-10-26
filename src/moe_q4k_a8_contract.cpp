#include "strix/moe_q4k_a8_contract.hpp"
#if defined(__FAST_MATH__) || (defined(__FINITE_MATH_ONLY__) && __FINITE_MATH_ONLY__)
#error "Q4K A8G32 requires nonfinite checks and strict floating-point semantics"
#endif
#include <algorithm>
#include <array>
#include <bit>
#include <span>
#include <cfenv>
#include <cfloat>
#include <cmath>
#include <limits>
#include <cstdlib>
#include <stdexcept>
#include <string_view>

namespace strix::moe_q4k_a8 {
namespace {
void check(bool value, const char* message) { if (!value) throw std::invalid_argument(message); }
std::uint8_t byte(std::byte value) { return std::to_integer<std::uint8_t>(value); }
std::uint16_t le16(const std::byte* source) {
    return static_cast<std::uint16_t>(byte(source[0]) | (static_cast<unsigned>(byte(source[1])) << 8U));
}
void check_row(const QuantizedRow& row) {
    check(!row.codes.empty() && row.codes.size() <= max_hidden && row.codes.size()%256U == 0,
        "Q4K A8 row width must divide256 within256..2560");
    check(row.deltas.size() == row.codes.size()/32U && row.sums.size() == row.deltas.size(),
        "Q4K A8 group metadata size mismatch");
    for (std::size_t group = 0; group < row.deltas.size(); ++group) {
        check(std::isfinite(row.deltas[group]) && row.deltas[group] > 0, "Q4K A8 delta must be finite positive");
        std::int32_t sum = 0;
        for (unsigned local = 0; local < 32; ++local) {
            const auto code = row.codes[group*32U+local]; check(code != -128, "Q4K A8 code-128 is forbidden"); sum += code;
        }
        check(sum == row.sums[group], "Q4K A8 sum differs from codes");
    }
}
float contribution(std::span<const std::byte> block, const QuantizedRow& row, std::size_t group,
    unsigned local_group) {
    std::array<std::uint8_t,32> q{};
    for (unsigned lane = 0; lane < 32; ++lane) {
        const auto packed = byte(block[16U+(local_group/2U)*32U+lane]);
        q[lane] = static_cast<std::uint8_t>(local_group%2U == 0U ? packed&15U : packed>>4U);
    }
    const auto ints = integer_group(q, std::span<const std::int8_t>(row.codes.data()+group*32U,32),
        decode_factors(block.subspan(4,12),local_group));
    const float d = half_bits_to_float(le16(block.data())), dm = half_bits_to_float(le16(block.data()+2));
    check(std::isfinite(d) && std::isfinite(dm), "Q4K A8 header must be finite");
    const float u = d*static_cast<float>(ints.scaled);
    const float v = dm*static_cast<float>(ints.minimum);
    const float t = u-v;
    const float result = row.deltas[group]*t;
    check(std::isfinite(u) && std::isfinite(v) && std::isfinite(t) && std::isfinite(result),
        "Q4K A8 combine intermediate outside qualified finite scope");
    return result;
}
}
void require_f32_environment() {
    check(std::numeric_limits<float>::is_iec559 && std::numeric_limits<float>::digits == 24 &&
        FLT_EVAL_METHOD == 0 && std::fegetround() == FE_TONEAREST,
        "Q4K A8 requires binary32 nearest-even without excess precision");
    volatile float tiny = std::numeric_limits<float>::denorm_min(); volatile float two = 2.0F;
    volatile float normal = std::numeric_limits<float>::min(); volatile float half = 0.5F;
    check(normal*half != 0 && tiny*two != 0, "Q4K A8 requires gradual binary32 subnormals/noDAZ/noFTZ");
}
QuantizedRow quantize_f32(std::span<const float> values) {
    require_f32_environment();
    check(!values.empty() && values.size()%32U == 0 && values.size() <= max_hidden,
        "A8G32 input must contain exact32-element groups within2560");
    QuantizedRow result; result.codes.resize(values.size());
    result.deltas.resize(values.size()/32U); result.sums.resize(values.size()/32U);
    for (std::size_t group = 0; group < result.deltas.size(); ++group) {
        float amax = 0;
        for (unsigned lane = 0; lane < 32; ++lane) {
            const float value = values[group*32U+lane]; check(std::isfinite(value), "A8G32 rejects nonfinite activation");
            amax = std::max(amax,std::fabs(value));
        }
        const float delta = amax == 0 ? 1.0F : amax/127.0F;
        check(std::isfinite(delta) && delta > 0, "A8G32 delta underflow/nonfinite");
        std::int32_t sum = 0;
        for (unsigned lane = 0; lane < 32; ++lane) {
            const float divided = values[group*32U+lane]/delta;
            const float rounded = std::nearbyint(divided);
            const auto code = static_cast<std::int8_t>(std::clamp(rounded,-127.0F,127.0F));
            result.codes[group*32U+lane] = code; sum += code;
        }
        result.deltas[group] = delta; result.sums[group] = sum;
    }
    return result;
}
QuantizedRow quantize_bf16(std::span<const std::uint16_t> values) {
    std::vector<float> expanded; expanded.reserve(values.size());
    for (const auto bits : values) expanded.push_back(std::bit_cast<float>(static_cast<std::uint32_t>(bits)<<16U));
    return quantize_f32(expanded);
}
Factors decode_factors(std::span<const std::byte> packed, std::uint32_t group) {
    check(packed.size() == 12 && group < 8, "Q4K factors require12bytes/group0..7");
    const auto lo = group&3U;
    const auto a = byte(packed[lo]), b = byte(packed[lo+4U]), c = byte(packed[lo+8U]);
    return group < 4U ? Factors{static_cast<std::uint32_t>(a&63U),static_cast<std::uint32_t>(b&63U)} :
        Factors{static_cast<std::uint32_t>((c&15U)|((a>>6U)<<4U)),static_cast<std::uint32_t>((c>>4U)|((b>>6U)<<4U))};
}
float half_bits_to_float(std::uint16_t bits) {
    const auto exponent = (bits>>10U)&31U, fraction = bits&1023U;
    const float magnitude = exponent == 0 ? std::ldexp(static_cast<float>(fraction),-24) :
        exponent == 31 ? (fraction == 0 ? std::numeric_limits<float>::infinity() :
            std::numeric_limits<float>::quiet_NaN()) :
        std::ldexp(1.0F+static_cast<float>(fraction)/1024.0F,static_cast<int>(exponent)-15);
    return (bits&0x8000U) != 0 ? -magnitude : magnitude;
}
IntegerGroup integer_group(std::span<const std::uint8_t> q, std::span<const std::int8_t> a, Factors factors) {
    check(q.size() == 32 && a.size() == 32 && factors.scale <= 63 && factors.minimum <= 63,
        "Q4K integer group shape/factors invalid");
    std::int64_t raw = 0, sum = 0;
    for (unsigned lane = 0; lane < 32; ++lane) {
        check(q[lane] <= 15 && a[lane] != -128, "Q4K/A8 code out of contract");
        raw += static_cast<std::int64_t>(q[lane])*a[lane]; sum += a[lane];
    }
    const auto scaled = raw*factors.scale, minimum = sum*factors.minimum;
    check(std::abs(raw) <= 60960 && std::abs(sum) <= 4064 && std::abs(scaled) <= 3840480 &&
        std::abs(minimum) <= 256032, "Q4K integer bound violated");
    return {static_cast<std::int32_t>(raw),static_cast<std::int32_t>(sum),
        static_cast<std::int32_t>(scaled),static_cast<std::int32_t>(minimum)};
}
float evaluate_row(std::span<const std::byte> weights, const QuantizedRow& row) {
    require_f32_environment(); check_row(row);
    check(weights.size() == row.codes.size()/256U*144U, "Q4K packed row size mismatch");
    std::array<float,4> accum{};
    for (std::size_t block = 0; block < row.codes.size()/256U; ++block)
        for (unsigned pair = 0; pair < 4; ++pair) {
            const auto packed = weights.subspan(block*144U,144U);
            const float c0 = contribution(packed,row,block*8U+pair*2U,pair*2U);
            const float c1 = contribution(packed,row,block*8U+pair*2U+1U,pair*2U+1U);
            const float paired = c0+c1; accum[pair] = accum[pair]+paired;
            check(std::isfinite(paired) && std::isfinite(accum[pair]), "Q4K A8 accumulation outside qualified scope");
        }
    const float left = accum[0]+accum[1], right = accum[2]+accum[3]; const float value = left+right;
    check(std::isfinite(left) && std::isfinite(right) && std::isfinite(value), "Q4K A8 combine overflow/nonfinite"); return value;
}
std::string_view numerical_identity_for_gate_type(std::uint32_t gate_type) {
    if (gate_type == 12) return numerical_identity;
    if (gate_type == 13) return q5k_numerical_identity;
    if (gate_type == 42 && q20_a8_enabled()) return q20_numerical_identity;
    if (q20_a8_enabled()) {
        if (gate_type == 22) return iq2s_numerical_identity;
        if (gate_type == 18) return iq3xxs_numerical_identity;
        if (gate_type == 21) return iq3s_numerical_identity;
        if (gate_type == 23) return iq4xs_numerical_identity;
    }
    throw std::invalid_argument("A8 gate/up requires type12 or type13 (or type42 with STRIX_GSQ_Q20_A8=1)");
}
std::array<std::uint8_t,32> decode_q5k_codes(std::span<const std::byte> block, std::uint32_t group) {
    check(block.size() == 176 && group < 8, "Q5K block/group invalid");
    std::array<std::uint8_t,32> result{};
    for (unsigned lane = 0; lane < 32; ++lane) {
        const auto packed = byte(block[48U+(group/2U)*32U+lane]);
        const auto low = group%2U == 0 ? packed&15U : packed>>4U;
        const auto high = (byte(block[16U+lane])>>group)&1U;
        result[lane] = static_cast<std::uint8_t>(low|(high<<4U));
    }
    return result;
}
IntegerGroup integer_group_q5k(std::span<const std::uint8_t> q,
    std::span<const std::int8_t> a, Factors factors) {
    check(q.size() == 32 && a.size() == 32 && factors.scale <= 63 && factors.minimum <= 63,
        "Q5K integer group shape/factors invalid");
    std::int64_t raw = 0, sum = 0;
    for (unsigned lane = 0; lane < 32; ++lane) {
        check(q[lane] <= 31 && a[lane] != -128, "Q5K/A8 code out of contract");
        raw += static_cast<std::int64_t>(q[lane])*a[lane]; sum += a[lane];
    }
    const auto scaled = raw*factors.scale, minimum = sum*factors.minimum;
    check(std::abs(raw) <= 125984 && std::abs(sum) <= 4064 && std::abs(scaled) <= 7936992 &&
        std::abs(minimum) <= 256032, "Q5K integer bound violated");
    return {static_cast<std::int32_t>(raw),static_cast<std::int32_t>(sum),
        static_cast<std::int32_t>(scaled),static_cast<std::int32_t>(minimum)};
}
float evaluate_q5k_row(std::span<const std::byte> weights, const QuantizedRow& row) {
    require_f32_environment(); check_row(row);
    check(weights.size() == row.codes.size()/256U*176U, "Q5K packed row size mismatch");
    std::array<float,4> accum{};
    for (std::size_t block = 0; block < row.codes.size()/256U; ++block)
        for (unsigned pair = 0; pair < 4; ++pair) {
            const auto packed = weights.subspan(block*176U,176U);
            std::array<float,2> contributions{};
            for (unsigned half = 0; half < 2; ++half) {
                const auto local_group = pair*2U+half;
                const auto group = block*8U+local_group;
                const auto codes = decode_q5k_codes(packed,local_group);
                const auto ints = integer_group_q5k(codes,
                    std::span<const std::int8_t>(row.codes.data()+group*32U,32),
                    decode_factors(packed.subspan(4,12),local_group));
                const float d = half_bits_to_float(le16(packed.data())),
                    dm = half_bits_to_float(le16(packed.data()+2));
                check(std::isfinite(d) && std::isfinite(dm), "Q5K A8 header must be finite");
                const float u = d*static_cast<float>(ints.scaled);
                const float v = dm*static_cast<float>(ints.minimum);
                const float t = u-v;
                contributions[half] = row.deltas[group]*t;
                check(std::isfinite(u) && std::isfinite(v) && std::isfinite(t) &&
                    std::isfinite(contributions[half]), "Q5K A8 combine outside qualified finite scope");
            }
            const float paired = contributions[0]+contributions[1];
            accum[pair] = accum[pair]+paired;
            check(std::isfinite(paired) && std::isfinite(accum[pair]), "Q5K A8 accumulation outside qualified scope");
        }
    const float left = accum[0]+accum[1], right = accum[2]+accum[3];
    const float result = left+right;
    check(std::isfinite(left) && std::isfinite(right) && std::isfinite(result), "Q5K A8 combine overflow/nonfinite");
    return result;
}
float evaluate_q20_row(std::span<const std::byte> weights, const QuantizedRow& row) {
    require_f32_environment(); check_row(row);
    check(weights.size() == row.codes.size()/64U*18U, "Q2_0 packed row size mismatch");
    std::array<float,8> lanes{};
    for (std::size_t group = 0; group < row.deltas.size(); ++group) {
        const auto* block = weights.data() + (group/2U)*18U;
        std::int64_t integer = 0;
        for (unsigned j = 0; j < 32; ++j) {
            const auto element = (group%2U)*32U + j;
            const auto q = (byte(block[2U + element/4U]) >> (2U*(element%4U))) & 3U;
            integer += static_cast<std::int64_t>(q)*row.codes[group*32U+j];
        }
        const auto difference = integer - row.sums[group];
        check(std::abs(difference) <= 16256, "Q2_0 integer bound violated");
        const float d = half_bits_to_float(le16(block));
        check(std::isfinite(d), "Q2_0 A8 header must be finite");
        const float scaled = d*static_cast<float>(difference);
        const float value = row.deltas[group]*scaled;
        lanes[group%8U] = lanes[group%8U]+value;
        check(std::isfinite(scaled) && std::isfinite(value) && std::isfinite(lanes[group%8U]),
            "Q2_0 A8 accumulation outside qualified scope");
    }
    const float a01 = lanes[0]+lanes[1], a23 = lanes[2]+lanes[3], a45 = lanes[4]+lanes[5], a67 = lanes[6]+lanes[7];
    const float left = a01+a23, right = a45+a67;
    const float result = left+right;
    check(std::isfinite(result), "Q2_0 A8 combine overflow/nonfinite");
    return result;
}
namespace {
constexpr std::uint64_t gsq_iq2s_grid[1024] = {
#include "strix/qwen_iq2_s_grid.inc"
};
constexpr std::uint32_t gsq_iq3xxs_grid[256] = {
#include "strix/qwen_iq3_xxs_grid.inc"
};
constexpr std::uint32_t gsq_iq3s_grid[512] = {
#include "strix/qwen_iq3_s_grid.inc"
};
constexpr std::array<std::int8_t,16> gsq_kvalues{-127, -104, -83, -65, -49, -35, -22, -10, 1, 13, 25, 38, 53, 69, 89, 113};
std::uint32_t u8(std::span<const std::byte> b, std::size_t at) { return byte(b[at]); }
}
GsqGroup gsq_iq_group(std::uint32_t type, std::span<const std::byte> row, std::uint32_t group) {
    const std::uint32_t be = type == 20 ? 32U : 256U;
    const std::uint32_t bb = type == 20 ? 18U : type == 22 ? 82U : type == 18 ? 98U : type == 21 ? 110U : type == 23 ? 136U : 0U;
    check(bb != 0, "GSQ IQ group type invalid");
    const auto x = row.subspan(static_cast<std::size_t>(group * 32U / be) * bb, bb);
    const std::uint32_t ib = (group * 32U % be) / 32U;
    const float d = half_bits_to_float(le16(x.data()));
    GsqGroup r{};
    if (type == 20 || type == 23) {
        const std::size_t qs = type == 20 ? 2U : 8U + ib * 16U;
        if (type == 20) r.s0 = d;
        else {
            const std::uint32_t scales_h = u8(x, 2) | (u8(x, 3) << 8U);
            const int ls = static_cast<int>(((u8(x, 4U + ib / 2U) >> (4U * (ib % 2U))) & 0xFU) | (((scales_h >> (2U * ib)) & 3U) << 4U));
            r.s0 = d * static_cast<float>(ls - 32);
        }
        for (unsigned j = 0; j < 16; ++j) {
            r.v[j] = gsq_kvalues[u8(x, qs + j) & 15U];
            r.v[j + 16U] = gsq_kvalues[u8(x, qs + j) >> 4U];
        }
    } else if (type == 21) {
        const std::uint32_t nib = ib % 2U == 0U ? (u8(x, 106U + ib / 2U) & 15U) : (u8(x, 106U + ib / 2U) >> 4U);
        r.s0 = d * static_cast<float>(1 + 2 * static_cast<int>(nib));
        const std::uint32_t qh = u8(x, 66U + ib);
        for (unsigned l = 0; l < 4; ++l) for (unsigned second = 0; second < 2; ++second) {
            const auto grid = gsq_iq3s_grid[u8(x, 2U + ib * 8U + 2U * l + second) | ((qh << (8U - 2U * l - second)) & 256U)];
            const auto signs = u8(x, 74U + ib * 4U + l);
            for (unsigned j = 0; j < 4; ++j) {
                const auto e = second * 4U + j;
                const int g = static_cast<int>((grid >> (8U * j)) & 255U);
                r.v[l * 8U + e] = static_cast<std::int8_t>((signs >> e) & 1U ? -g : g);
            }
        }
    } else if (type == 18) {
        const std::uint32_t aux = u8(x, 66U + 4U * ib) | (u8(x, 67U + 4U * ib) << 8U) | (u8(x, 68U + 4U * ib) << 16U) | (u8(x, 69U + 4U * ib) << 24U);
        r.s0 = (d * (0.5F + static_cast<float>(aux >> 28U))) * 0.5F;
        for (unsigned l = 0; l < 4; ++l) {
            const std::uint32_t s7 = (aux >> (7U * l)) & 127U;
            const std::uint32_t signs = s7 | ((static_cast<std::uint32_t>(std::popcount(s7)) & 1U) << 7U);
            for (unsigned second = 0; second < 2; ++second) {
                const auto grid = gsq_iq3xxs_grid[u8(x, 2U + ib * 8U + 2U * l + second)];
                for (unsigned j = 0; j < 4; ++j) {
                    const auto e = second * 4U + j;
                    const int g = static_cast<int>((grid >> (8U * j)) & 255U);
                    r.v[l * 8U + e] = static_cast<std::int8_t>((signs >> e) & 1U ? -g : g);
                }
            }
        }
    } else {  // 22
        const std::uint32_t sc = u8(x, 74U + ib), qh = u8(x, 66U + ib);
        r.s0 = (d * (0.5F + static_cast<float>(sc & 15U))) * 0.25F;
        r.s1 = (d * (0.5F + static_cast<float>(sc >> 4U))) * 0.25F;
        for (unsigned l = 0; l < 4; ++l) {
            const auto grid = gsq_iq2s_grid[u8(x, 2U + ib * 4U + l) | ((qh << (8U - 2U * l)) & 0x300U)];
            const auto signs = u8(x, 34U + ib * 4U + l);
            for (unsigned j = 0; j < 8; ++j) {
                const int g = static_cast<int>((grid >> (8U * j)) & 255U);
                r.v[l * 8U + j] = static_cast<std::int8_t>((signs >> j) & 1U ? -g : g);
            }
        }
    }
    return r;
}
float evaluate_gsq_row(std::uint32_t type, std::span<const std::byte> weights, const QuantizedRow& row) {
    if (type == 42) return evaluate_q20_row(weights, row);
    require_f32_environment(); check_row(row);
    std::array<float,8> lanes{};
    for (std::size_t group = 0; group < row.deltas.size(); ++group) {
        const auto g = gsq_iq_group(type, weights, static_cast<std::uint32_t>(group));
        std::int64_t i0 = 0, i1 = 0;
        for (unsigned j = 0; j < 32; ++j) (j < 16 ? i0 : i1) += static_cast<std::int64_t>(g.v[j]) * row.codes[group*32U+j];
        float value;
        if (type == 22) {
            const float u0 = g.s0*static_cast<float>(i0), u1 = g.s1*static_cast<float>(i1);
            const float t = u0+u1;
            value = row.deltas[group]*t;
        } else {
            const float scaled = g.s0*static_cast<float>(i0+i1);
            value = row.deltas[group]*scaled;
        }
        lanes[group%8U] = lanes[group%8U]+value;
        check(std::isfinite(value) && std::isfinite(lanes[group%8U]), "GSQ IQ A8 accumulation outside qualified scope");
    }
    const float a01 = lanes[0]+lanes[1], a23 = lanes[2]+lanes[3], a45 = lanes[4]+lanes[5], a67 = lanes[6]+lanes[7];
    const float left = a01+a23, right = a45+a67;
    const float result = left+right;
    check(std::isfinite(result), "GSQ IQ A8 combine overflow/nonfinite");
    return result;
}
void validate(const Request& request, const Bindings& b) {
    check(request.enabled && request.approved_numerical_identity == numerical_identity_for_gate_type(request.gate_type),
        "Q4K A8 probe requires explicit matching numerical identity approval");
    const auto s = request.shape;
    check(s.hidden >= 256 && s.hidden <= max_hidden && s.hidden%256U == 0 && s.rows > 0 &&
        s.rows <= max_rows && s.rows%32U == 0 && s.experts > 0 && s.experts <= max_experts &&
        s.top_k > 0 && s.top_k <= max_top_k && s.logical_rows == 1, "Q4K A8 probe admits bounded C1 dimensions only");
    check(request.input == InputKind::bf16 || request.input == InputKind::fp32, "Q4K A8 input storage unsupported");
    check(request.gate_type == request.up_type && !request.fused_gate_up,
        "A8 prototype requires separate matching type12 or type13 gate/up tensors");
    const std::uint64_t block_bytes = request.gate_type == 13 ? 176U : request.gate_type == 42 ? 72U :  // per 256
        request.gate_type == 22 ? 82U : request.gate_type == 18 ? 98U : request.gate_type == 21 ? 110U :
        request.gate_type == 23 ? 136U : 144U;
    const std::uint64_t weight_bytes = static_cast<std::uint64_t>(s.experts)*s.rows*(s.hidden/256U)*block_bytes;
    const std::uint64_t output_bytes = static_cast<std::uint64_t>(s.top_k)*2U*s.rows*4U;
    const std::array<DeviceBytes,9> views{b.input,b.gate,b.up,b.routes,b.codes,b.deltas,b.sums,b.status,b.output};
    const std::array<std::uint64_t,9> required{static_cast<std::uint64_t>(s.hidden)*(request.input==InputKind::bf16?2U:4U),
        weight_bytes,weight_bytes,s.top_k*4ULL,s.hidden,s.hidden/32U*4ULL,s.hidden/32U*4ULL,4,output_bytes};
    const std::array<std::uint64_t,9> alignments{16,16,16,16,16,16,16,16,16};
    for (std::size_t i = 0; i < views.size(); ++i) {
        check(views[i].address != 0 && views[i].address%alignments[i] == 0 && views[i].bytes == required[i] &&
            views[i].bytes <= std::numeric_limits<std::uintptr_t>::max()-views[i].address,
            "Q4K A8 device view address/alignment/exact byte size invalid");
        for (std::size_t j = 0; j < i; ++j)
            if (i >= 4 || j >= 4 || (i == 2 && j == 1))
                check(views[i].address+views[i].bytes <= views[j].address ||
                    views[j].address+views[j].bytes <= views[i].address,
                    "Q4K A8 mutable views and separate gate/up must not overlap");
    }
}
}  // namespace strix::moe_q4k_a8
