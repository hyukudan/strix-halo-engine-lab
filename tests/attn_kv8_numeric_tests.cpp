// Domain of the two 8-bit KV forms on adversarial inputs (NUM-10): kv-int8-tok-v1 (strix/attn/kv_int8_token.hpp) and
// kv-shadow-i8g16-v1 (strix/attn/cache_shadow.hpp).
//   1. Finite rows: the kv-int8-tok-v1 steps store the same bytes as the v1 statements they replace (fmaxf tree,
//      max / 127, fminf/fmaxf clamp), for every reduction order the HIP writer can use.
//   2. Non-finite rows (NaN, +-Inf, NaN payloads, finite values whose BF16 rounding overflows): scale NaN, codes 0,
//      status raised, every feature decodes to NaN; the v1 statements are shown to encode a finite wrong value.
//   4. The split-K row plan as data and the KV bytes one decode step reads per form (the slot / bandwidth arithmetic).
//   3. Extreme finite scales and long accumulations: per-feature error bounds of both forms and the FP32 error of a
//      64K-position value sum read through each form.
#include "strix/attn/cache_shadow.hpp"
#include "strix/attn/kv_int8_token.hpp"
#include "strix/attn/split_decode_plan.hpp"

#include <cmath>
#include <cstdint>
#include <cstdio>
#include <limits>
#include <random>
#include <vector>

namespace k8 = strix::attn::kv_int8_token;
namespace cs = strix::attn::cache_shadow;

namespace {

int failures = 0;
void expect(bool ok, const char* what, double detail = 0.0) {
    if (!ok) {
        ++failures;
        std::printf("FAIL %s (%g)\n", what, detail);
    }
}
std::uint32_t bits(float x) { return __builtin_bit_cast(std::uint32_t, x); }
float from_bits(std::uint32_t b) { return __builtin_bit_cast(float, b); }
constexpr float inf = std::numeric_limits<float>::infinity();
const float qnan = std::numeric_limits<float>::quiet_NaN();

// The v1 statements of the HIP writer before this change (reference for the finite identity and for the defect).
struct V1Row {
    std::vector<std::int8_t> codes;
    float scale = 0.0F;
};
V1Row v1_encode(const std::vector<float>& src, unsigned threads) {
    std::vector<float> x(threads, 0.0F), m(threads, 0.0F);
    for (unsigned i = 0; i < threads; ++i) {
        x[i] = i < src.size() ? k8::round_bf16(src[i]) : 0.0F;
        m[i] = std::fabs(x[i]);
    }
    for (unsigned stride = threads / 2U; stride > 0U; stride >>= 1U)
        for (unsigned f = 0; f < stride; ++f)
            m[f] = std::fmax(m[f], m[f + stride]);
    V1Row r;
    r.scale = m[0] / 127.0F;
    for (std::size_t i = 0; i < src.size(); ++i) {
        const float c = r.scale > 0.0F ? std::fmin(std::fmax(std::rint(x[i] / r.scale), -127.0F), 127.0F) : 0.0F;
        r.codes.push_back(static_cast<std::int8_t>(static_cast<int>(c)));
    }
    return r;
}
// The new steps folded as the HIP tree folds them (threads a power of two, zero padding).
float tree_max(const std::vector<float>& src, unsigned threads) {
    std::vector<float> m(threads, 0.0F);
    for (unsigned i = 0; i < threads; ++i)
        m[i] = i < src.size() ? k8::magnitude(src[i], k8::round_bf16(src[i])) : 0.0F;
    for (unsigned stride = threads / 2U; stride > 0U; stride >>= 1U)
        for (unsigned f = 0; f < stride; ++f)
            m[f] = k8::max_step(m[f], m[f + stride]);
    return m[0];
}

void finite_identity(std::mt19937& rng) {
    std::uniform_real_distribution<float> unit(-1.0F, 1.0F);
    std::uniform_int_distribution<int> exponent(-140, 120);
    const unsigned dims[] = {64U, 128U, 256U, 512U};
    unsigned rows = 0;
    for (unsigned dim : dims)
        for (int trial = 0; trial < 400; ++trial) {
            std::vector<float> src(dim);
            const float row_scale = std::ldexp(1.0F, exponent(rng));
            for (auto& x : src)
                x = unit(rng) * row_scale * (trial % 7 == 0 ? std::ldexp(1.0F, exponent(rng) / 8) : 1.0F);
            if (trial % 11 == 0)
                for (auto& x : src)
                    x = 0.0F;
            if (trial % 13 == 0)
                src[static_cast<unsigned>(trial) % dim] = std::numeric_limits<float>::max() * 0.99F;  // rounds to a finite BF16
            if (trial % 17 == 0)
                src[static_cast<unsigned>(trial) % dim] = from_bits(0x00000001U);                   // FP32 denormal
            const auto v1 = v1_encode(src, 256U >= dim ? 256U : dim);
            std::vector<std::int8_t> codes(dim);
            float s = 0.0F;
            const unsigned status = k8::encode_row_ref(src.data(), dim, codes.data(), s);
            const float t = tree_max(src, 256U >= dim ? 256U : dim);
            const bool finite_row = k8::finite(v1.scale);
            if (!finite_row)
                continue;  // only rows whose BF16 values are all finite belong to the identity statement
            ++rows;
            expect(status == 0U, "finite row: no status");
            expect(bits(s) == bits(v1.scale), "finite row: scale bits == v1", s);
            expect(bits(k8::scale(t)) == bits(s), "finite row: tree order == serial order", t);
            expect(codes == v1.codes, "finite row: codes == v1");
        }
    expect(rows > 1400U, "finite identity coverage", rows);
}

void nonfinite_rows() {
    struct Case {
        const char* what;
        float value;
    };
    const Case cases[] = {{"NaN", qnan},
                          {"-NaN", -qnan},
                          {"+Inf", inf},
                          {"-Inf", -inf},
                          {"NaN payload 0x7fffffff (BF16 formula wraps it to -0)", from_bits(0x7fffffffU)},
                          {"NaN payload 0xffffffff (BF16 formula wraps it to +0)", from_bits(0xffffffffU)},
                          {"FLT_MAX (BF16 rounding overflows to +Inf)", std::numeric_limits<float>::max()}};
    for (const auto& c : cases) {
        for (unsigned at : {0U, 7U, 255U}) {
            std::vector<float> src(256);
            for (unsigned i = 0; i < 256; ++i)
                src[i] = 0.25F * float(int(i % 9) - 4);
            src[at] = c.value;
            std::vector<std::int8_t> codes(256, 5);
            float s = 1.0F;
            const unsigned status = k8::encode_row_ref(src.data(), 256U, codes.data(), s);
            expect(status == k8::status_nonfinite, c.what);
            expect(s != s, c.what, s);
            bool zero = true, all_nan = true;
            for (unsigned i = 0; i < 256; ++i) {
                zero = zero && codes[i] == 0;
                const float d = k8::decode(codes[i], s);
                all_nan = all_nan && d != d;
            }
            expect(zero, c.what);
            expect(all_nan, c.what);
            expect(tree_max(src, 256U) != tree_max(src, 256U), c.what);  // NaN at any tree position
        }
    }
    // The v1 statements on the same rows: a finite wrong value (the defect this closes).
    std::vector<float> src(256, 0.5F);
    src[3] = qnan;
    auto v1 = v1_encode(src, 256U);
    expect(k8::finite(v1.scale) && v1.codes[3] == -127, "v1 NaN -> code -127 with a finite scale (defect reproduced)");
    src[3] = inf;
    v1 = v1_encode(src, 256U);
    const float rebuilt = float(v1.codes[3]) * v1.scale;
    expect(std::isinf(rebuilt) && rebuilt < 0.0F, "v1 +Inf -> -Inf (defect reproduced)", rebuilt);
}

void int8g16_domain() {
    float group[16];
    std::int8_t codes[16];
    std::uint16_t scale = 0;
    for (float bad : {qnan, inf, -inf}) {
        for (unsigned j = 0; j < 16; ++j)
            group[j] = 0.1F * float(j);
        group[5] = bad;
        const unsigned f = cs::encode_group(group, codes, scale);
        expect((f & cs::flag_nonfinite) != 0U, "i8g16 non-finite flagged");
        if (bad != bad)
            expect(codes[5] == 0, "i8g16 NaN codes 0 (defined, no UB conversion)");
        else
            expect((f & cs::flag_overflow) != 0U, "i8g16 infinite amax: scale overflow flagged");
    }
    expect(cs::Int8G16::code(qnan, 1.0F) == 0 && cs::Int8G16::code(1.0F, qnan) == 0 && cs::Int8G16::code(inf, inf) == 0,
           "i8g16 NaN quotient codes 0");
    expect(cs::Int8G16::code(inf, 1.0F) == 127 && cs::Int8G16::code(-inf, 1.0F) == -127, "i8g16 infinite x saturates");
    // amax / 127 above the largest half: flagged, never a silent finite scale.
    for (unsigned j = 0; j < 16; ++j)
        group[j] = 9.0e6F;
    expect((cs::encode_group(group, codes, scale) & cs::flag_overflow) != 0U, "i8g16 scale overflow flagged");
}

// Per-feature error bounds over extreme finite scales.
void error_bounds(std::mt19937& rng) {
    std::uniform_real_distribution<float> unit(-1.0F, 1.0F);
    double worst_tok = 0.0, worst_g16 = 0.0, worst_tiny = 0.0;
    for (int e = -40; e <= 22; e += 2) {
        for (int trial = 0; trial < 40; ++trial) {
            float src[16];
            for (auto& x : src)
                x = unit(rng) * std::ldexp(1.0F, e);
            // kv-int8-tok-v1: |decode - bf16(x)| <= s / 2 (s normal FP32 at every tested exponent).
            std::int8_t codes[16];
            float s = 0.0F;
            expect(k8::encode_row_ref(src, 16U, codes, s) == 0U, "tok finite");
            for (unsigned j = 0; j < 16; ++j) {
                const double err = std::fabs(double(k8::decode(codes[j], s)) - double(k8::round_bf16(src[j])));
                if (s > 0.0F)
                    worst_tok = std::fmax(worst_tok, err / double(s));
            }
            // int8-g16: normal-half scale -> error <= d/2 (+ half rounding of the scale); tiny groups -> error <= amax.
            std::uint16_t sb = 0;
            const unsigned f = cs::encode_group(src, codes, sb);
            const float d = cs::Int8G16::scale_value(sb);
            float amax = 0.0F;
            for (float x : src)
                amax = std::fmax(amax, std::fabs(x));
            if (amax / 127.0F > 65504.0F) {
                expect((f & cs::flag_overflow) != 0U, "g16 overflow flagged at large exponents");
                continue;
            }
            expect(f == 0U, "g16 finite in range: no flag");
            for (unsigned j = 0; j < 16; ++j) {
                const double err = std::fabs(double(cs::Int8G16::decode(codes[j], d)) - double(src[j]));
                if (d >= 6.103515625e-5F)
                    worst_g16 = std::fmax(worst_g16, err / double(d));
                else
                    worst_tiny = std::fmax(worst_tiny, err / double(amax));
            }
        }
    }
    expect(worst_tok <= 0.5 + 1e-5, "tok: error <= s/2 (+ product rounding)", worst_tok);
    expect(worst_g16 <= 0.5 + 0.07, "g16 normal scale: error <= d/2 + scale rounding", worst_g16);
    expect(worst_tiny <= 1.0 + 1e-6, "g16 subnormal or zero scale: error <= amax", worst_tiny);
    std::printf("bounds: tok %.4f s, g16 %.4f d (normal half scale), g16 %.4f amax (subnormal/zero scale)\n", worst_tok,
                worst_g16, worst_tiny);
}

// A 64K-position value sum (softmax-like weights, FP32 accumulation in position order) read through each form:
// the error stays at the per-feature bound times the weight mass (no growth from the accumulation order).
void long_accumulation(std::mt19937& rng) {
    constexpr unsigned positions = 65536U, dim = 16U;
    std::normal_distribution<float> normal(0.0F, 1.0F);
    std::uniform_real_distribution<float> unit(0.0F, 1.0F);
    std::vector<float> weight(positions);
    double mass = 0.0;
    for (auto& w : weight) {
        w = std::exp(-12.0F * unit(rng));  // spans P in [6e-6, 1]
        mass += w;
    }
    double ref[dim]{}, tok[dim]{}, g16[dim]{};
    float acc_tok[dim]{}, acc_g16[dim]{}, acc_ref[dim]{};
    double bound_tok = 0.0, bound_g16 = 0.0;
    for (unsigned p = 0; p < positions; ++p) {
        float v[dim];
        const float row_scale = std::ldexp(1.0F, int(p % 9U) - 4);
        for (auto& x : v)
            x = normal(rng) * row_scale;
        std::int8_t c1[dim], c2[dim];
        float s = 0.0F;
        std::uint16_t sb = 0;
        k8::encode_row_ref(v, dim, c1, s);
        cs::encode_group(v, c2, sb);
        const float d = cs::Int8G16::scale_value(sb);
        bound_tok += double(weight[p]) * 0.75 * double(s);  // s/2 code step + 2^-9 amax BF16 rounding (amax = 127 s)
        bound_g16 += double(weight[p]) * 0.57 * double(d);
        for (unsigned j = 0; j < dim; ++j) {
            acc_ref[j] += weight[p] * v[j];
            acc_tok[j] += weight[p] * k8::decode(c1[j], s);
            acc_g16[j] += weight[p] * cs::Int8G16::decode(c2[j], d);
            ref[j] += double(weight[p]) * double(v[j]);
        }
    }
    double worst_tok = 0.0, worst_g16 = 0.0, worst_fp32 = 0.0;
    for (unsigned j = 0; j < dim; ++j) {
        tok[j] = std::fabs(double(acc_tok[j]) - ref[j]);
        g16[j] = std::fabs(double(acc_g16[j]) - ref[j]);
        worst_fp32 = std::fmax(worst_fp32, std::fabs(double(acc_ref[j]) - ref[j]));
        worst_tok = std::fmax(worst_tok, tok[j]);
        worst_g16 = std::fmax(worst_g16, g16[j]);
    }
    // The bound is the sum of per-position bounds plus the FP32 accumulation error of the master itself (x4 margin).
    expect(worst_tok <= bound_tok + 4.0 * worst_fp32 + 1e-3, "64K value sum: tok error within the per-feature bound",
           worst_tok);
    expect(worst_g16 <= bound_g16 + 4.0 * worst_fp32 + 1e-3, "64K value sum: g16 error within the per-feature bound",
           worst_g16);
    std::printf("64K sum (mass %.1f): |err| tok %.3e (bound %.3e), g16 %.3e (bound %.3e), fp32 master %.3e\n", mass,
                worst_tok, bound_tok, worst_g16, bound_g16, worst_fp32);
}

// Split-K plan data (the Gemma global-layer plan: dim 512, 128-key tiles, 256 lanes, 8 heads / 2 KV heads, 64K) and the
// bytes-per-token arithmetic of the forms.
void plan_and_bytes() {
    using strix::attn::SplitDecodePlan;
    using F = cs::Form;
    constexpr SplitDecodePlan plan{.dim = 512, .tile = 128, .threads = 256, .group = 4, .heads = 8, .kv_heads = 2,
                                   .max_positions = 65536};
    static_assert(plan.valid() && plan.partial_floats() == 2056U && plan.max_tiles() == 512U);
    for (std::uint32_t rows : {1U, 3U, 8U})
        for (std::uint32_t stride : {64U, 128U, 8192U, 65536U}) {
            const std::size_t tiles = (std::size_t(stride) + 127U) / 128U;
            const std::size_t old_formula = std::size_t(rows) * (8U * 512U + 2U * tiles * 2056U) * sizeof(float);
            expect(plan.scratch_bytes(rows, stride) == old_formula, "plan scratch == the kernel's formula", stride);
        }
    SplitDecodePlan bad = plan;
    bad.tile = 96;
    expect(!bad.valid(), "plan rejects a tile that is not a multiple of 64");
    // 7 global layers read 2 KV heads x 512 per position: FP32 57,344 B, kv8 16,128 B, f16 28,672 B.
    expect(strix::attn::kv_read_bytes(F::none, 1U, 2U, 512U, 7U) == 57344U, "FP32 bytes per position");
    expect(strix::attn::kv_read_bytes(F::kv8, 1U, 2U, 512U, 7U) == 16128U, "kv8 bytes per position");
    expect(strix::attn::kv_read_bytes(F::f16, 1U, 2U, 512U, 7U) == 28672U, "f16 bytes per position");
    expect(strix::attn::kv_read_bytes(F::none, 65280U, 2U, 512U, 7U) == 3743416320ULL, "FP32 bytes at 65,280");
    expect(strix::attn::kv_read_bytes(F::kv8, 65280U, 2U, 512U, 7U) == 1052835840ULL, "kv8 bytes at 65,280");
    // Resident shadow of a 64K slot: 4 global producer layers.
    expect(4U * cs::shadow_bytes(F::kv8, 65536U, 1024U) == 603979776ULL, "kv8 shadow of a 64K slot");
    expect(cs::shadow_bytes(F::none, 65536U, 1024U) == 0U, "no shadow for the master form");
}

}  // namespace

int main() {
    std::mt19937 rng(20261126U);
    finite_identity(rng);
    nonfinite_rows();
    int8g16_domain();
    error_bounds(rng);
    long_accumulation(rng);
    plan_and_bytes();
    if (failures != 0) {
        std::printf("attn-kv8-numeric: %d failures\n", failures);
        return 1;
    }
    std::printf("attn-kv8-numeric: PASS\n");
    return 0;
}
