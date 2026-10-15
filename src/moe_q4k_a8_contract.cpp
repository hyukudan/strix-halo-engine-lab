#include "strix/moe_q4k_a8_contract.hpp"
#include <algorithm>
#include <array>
#include <bit>
#include <cfenv>
#include <cfloat>
#include <cmath>
#include <limits>
#include <stdexcept>

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
void validate(const Request& request, const Bindings& b) {
    check(request.enabled && request.approved_numerical_identity == numerical_identity,
        "Q4K A8 probe requires explicit matching numerical identity approval");
    const auto s = request.shape;
    check(s.hidden >= 256 && s.hidden <= max_hidden && s.hidden%256U == 0 && s.rows > 0 &&
        s.rows <= max_rows && s.rows%32U == 0 && s.experts > 0 && s.experts <= max_experts &&
        s.top_k > 0 && s.top_k <= max_top_k && s.logical_rows == 1, "Q4K A8 probe admits bounded C1 dimensions only");
    check(request.input == InputKind::bf16 || request.input == InputKind::fp32, "Q4K A8 input storage unsupported");
    check(request.gate_type == 12 && request.up_type == 12 && !request.fused_gate_up,
        "Q4K A8 prototype requires separate type12 gate/up tensors");
    const std::uint64_t weight_bytes = static_cast<std::uint64_t>(s.experts)*s.rows*(s.hidden/256U)*144U;
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
