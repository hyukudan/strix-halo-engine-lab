// Host contract of the shared RoPE and gated elementwise ops (src/ops/rope, src/ops/glu):
//  - glu Round::none + DeclaredExp == ds41_moe.hpp swiglu_clamped (the lifted DS41 form) bit for bit;
//  - glu Round::bf16 == the MODEL_B reference expressions (modelb-host-ref-v1: bf(bf(silu(g)) * u), bf(bf(x) * bf(sigmoid(g))));
//  - native table + reference_apply Round::bf16 == the MODEL_B reference rotation (interleaved pairs, theta^(-2i/64) in FP32,
//    cos / sin from double rounded to BF16, BF16 outputs) on the trailing 64 of a 256-wide head;
//  - positions index table rows; inverse undoes the rotation (Round::none, within rounding).
#include "ops/glu/glu.hpp"
#include "ops/rope/rope.hpp"
#include "glm/common/glm_math.hpp"

#include <array>
#include <bit>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <limits>
#include <utility>
#include <vector>

namespace {

int failures = 0;
void check(bool ok, const char* what) {
    if (!ok) {
        std::fprintf(stderr, "FAIL %s\n", what);
        ++failures;
    }
}
bool same_bits(float a, float b) { return std::memcmp(&a, &b, sizeof a) == 0; }
float bf(float x) { return strix::ds41::round_to_bf16(x); }

std::uint32_t lcg = 12345U;
float next_value(float scale) {
    lcg = lcg * 1664525U + 1013904223U;
    return (static_cast<float>(lcg >> 8U) / 16777216.0F - 0.5F) * 2.0F * scale;
}

void glu_tests() {
    using strix::ops::glu::DeclaredExp;
    using strix::ops::glu::Round;
    const float limits[] = {10.0F, 7.0F, strix::ops::glu::no_clamp};
    bool none_ok = true, bf_ok = true, gate_ok = true, gate_none_ok = true;
    for (int i = 0; i < 20000; ++i) {
        const float g = bf(next_value(24.0F)), u = bf(next_value(24.0F)), x = next_value(3.0F);
        for (const float limit : limits) {
            none_ok = none_ok && same_bits(strix::ops::glu::swiglu<DeclaredExp, Round::none>(g, u, limit),
                                           strix::ds41::swiglu_clamped(g, u, limit));
            const float gc = g > limit ? limit : g, uc = u < -limit ? -limit : (u > limit ? limit : u);
            const float silu = gc / (1.0F + strix::ds41::declared_exp(-gc));
            bf_ok = bf_ok && same_bits(strix::ops::glu::swiglu<DeclaredExp, Round::bf16>(g, u, limit), bf(bf(silu) * uc));
        }
        gate_ok = gate_ok && same_bits(strix::ops::glu::sigmoid_gate<DeclaredExp, Round::bf16>(x, g),
                                       bf(bf(x) * bf(strix::ds41::declared_sigmoid(g))));
        gate_none_ok = gate_none_ok && same_bits(strix::ops::glu::sigmoid_gate<DeclaredExp, Round::none>(x, g),
                                                 x * strix::ds41::declared_sigmoid(g));
    }
    check(none_ok, "swiglu none == ds41 swiglu_clamped");
    check(bf_ok, "swiglu bf16 == modelb reference expression");
    check(gate_ok, "sigmoid gate bf16 == modelb reference expression");
    check(gate_none_ok, "sigmoid gate none");
    // Clamp edges: gate clamped from above only, up on both sides, NaN passes through, +inf limit = none.
    check(strix::ops::glu::swiglu<DeclaredExp, Round::none>(50.0F, 1.0F, 10.0F) ==
              strix::ops::glu::swiglu<DeclaredExp, Round::none>(10.0F, 1.0F, 10.0F), "gate clamp");
    check(strix::ops::glu::swiglu<DeclaredExp, Round::none>(1.0F, -50.0F, 10.0F) ==
              strix::ops::glu::swiglu<DeclaredExp, Round::none>(1.0F, -10.0F, 10.0F), "up clamp");
    check(strix::ops::glu::swiglu<DeclaredExp, Round::none>(-50.0F, 1.0F, 10.0F) ==
              strix::ops::glu::swiglu<DeclaredExp, Round::none>(-50.0F, 1.0F, strix::ops::glu::no_clamp), "gate has no lower bound");
    check(std::isnan(strix::ops::glu::swiglu<DeclaredExp, Round::bf16>(std::numeric_limits<float>::quiet_NaN(), 1.0F, 10.0F)),
          "NaN passes");
}

void glm_swiglu_tests() {
    using namespace strix::ops::glu;
    struct GlmExp {
        static float exp(float x) { return strix::glm::declared_exp(x); }
    };
    const float limit = strix::glm::kClamp;
    bool matched = true, distinguished_forms = false;
    const auto compare = [&](float g, float u) {
        const float gg = g > limit ? limit : g;
        const float uu = u > limit ? limit : (u < -limit ? -limit : u);
        const float expected = strix::glm::silu(gg) * uu;
        const float actual = swiglu<GlmExp, Round::none, Form::mul_recip>(g, u, limit);
        matched &= same_bits(actual, expected);
        distinguished_forms |= !same_bits(actual, swiglu<GlmExp, Round::none>(g, u, limit));
    };
    for (int i = 0; i < 20000; ++i) {
        const float g = next_value(24.0F), u = next_value(24.0F);
        compare(g, u);
    }
    const float inf = std::numeric_limits<float>::infinity();
    const float nan = std::numeric_limits<float>::quiet_NaN();
    for (float g : {-0.0F, 0.0F, -inf, inf, -89.0F, -88.0F, limit, nan}) compare(g, 2.0F);
    for (float u : {-0.0F, 0.0F, -inf, inf, -limit, limit, nan}) compare(2.0F, u);
    check(matched, "mul_recip matches GLM private SwiGLU bits");
    check(distinguished_forms, "GLM samples distinguish reciprocal multiply from divide");
}

void swiglu_a8_tests() {
    using namespace strix::ops::glu;
    constexpr unsigned lanes = 32;
    using Row = std::array<float, lanes>;
    bool codes_match = true, deltas_match = true, sums_match = true;
    const auto compare = [&](const Row& gate, const Row& up, float limit) {
        Row act{}, ref_act{}, maxima{};
        for (unsigned i = 0; i < lanes; ++i) {
            act[i] = swiglu<DeclaredExp, Round::none>(gate[i], up[i], limit);
            ref_act[i] = strix::ds41::swiglu_clamped(gate[i], up[i], limit);
            maxima[i] = std::fabs(ref_act[i]);
        }
        // Reproduce the private wave's simultaneous XOR updates at each stride.
        for (unsigned stride = 16; stride != 0; stride /= 2) {
            const Row prev = maxima;
            for (unsigned i = 0; i < lanes; ++i) maxima[i] = std::fmax(prev[i], prev[i ^ stride]);
        }
        std::array<int, lanes> ref_sums{};
        int sum = 0;
        for (unsigned i = 0; i < lanes; ++i) {
            const float amax = maxima[i];
            const bool tiny = amax > 0.0F && amax < 0x1p-119F;
            const float ref_delta = amax == 0.0F || tiny ? 1.0F : amax / 127.0F;
            const float rounded = tiny ? 0.0F : std::rint(ref_act[i] / ref_delta);
            const int ref_code = static_cast<int>(std::fmin(127.0F, std::fmax(-127.0F, rounded)));
            ref_sums[i] = ref_code;

            // Host quantizer for the shared epilogue. The device uses its wave32 reducer instead.
            const auto quantize = [&](float lane_act) {
                float peak = std::fabs(lane_act);
                for (unsigned j = 0; j < lanes; ++j) {
                    if (j != i) peak = std::fmax(peak, std::fabs(act[j]));
                }
                const bool underflow = peak > 0.0F && peak < 0x1p-119F;
                const float delta = peak == 0.0F || underflow ? 1.0F : peak / 127.0F;
                const float code = underflow ? 0.0F : std::nearbyint(lane_act / delta);
                return std::pair{static_cast<int>(std::fmin(127.0F, std::fmax(-127.0F, code))), delta};
            };
            const auto [code, delta] = A8Epilogue::apply<DeclaredExp, Round::none>(gate[i], up[i], limit, quantize);
            codes_match &= code == ref_code;
            deltas_match &= same_bits(delta, ref_delta);
            sum += code;
        }
        for (unsigned stride = 16; stride != 0; stride /= 2) {
            const auto prev = ref_sums;
            for (unsigned i = 0; i < lanes; ++i) ref_sums[i] += prev[i ^ stride];
        }
        sums_match &= sum == ref_sums[0];
    };

    Row gate{}, up{};
    compare(gate, up, 10.0F);
    gate.fill(100.0F);
    up.fill(0x1p-126F);
    compare(gate, up, 10.0F);
    for (float peak : {0x1.fffffep-120F, 0x1p-119F, 127.0F}) {
        gate.fill(128.0F);  // Declared exp returns zero, so the power-of-two scale is exact.
        up.fill(0.0F);
        up[0] = peak / 128.0F;
        up[1] = up[0] * (0.5F / 127.0F);
        up[2] = up[0] * (-1.5F / 127.0F);
        compare(gate, up, no_clamp);
    }
    for (int group = 0; group < 512; ++group) {
        for (unsigned i = 0; i < lanes; ++i) {
            gate[i] = next_value(32.0F);
            up[i] = next_value(32.0F);
        }
        compare(gate, up, group % 2 == 0 ? 10.0F : no_clamp);
    }
    check(codes_match, "fused SwiGLU A8 codes match private arithmetic");
    check(deltas_match, "fused SwiGLU A8 delta bits match private arithmetic");
    check(sums_match, "fused SwiGLU A8 sums match private arithmetic");
}

// Linear epilogue: bf16 = one rounding of (x + bias) (the oracle's bf(dot + bias)), unit bias = bf(x); none = x + bias.
void epilogue_tests() {
    using strix::ops::glu::Round;
    bool ok = true;
    const float bias[3] = {0.25F, -1.5F, 3.0e-3F};
    for (int i = 0; i < 4000; ++i) {
        const float x = next_value(40.0F);
        const auto c = static_cast<std::size_t>(i % 3);
        ok = ok && same_bits(strix::ops::glu::epilogue<Round::bf16>(x, bias, c), bf(x + bias[c]));
        ok = ok && same_bits(strix::ops::glu::epilogue<Round::bf16>(x, nullptr, c), bf(x));
        ok = ok && same_bits(strix::ops::glu::epilogue<Round::none>(x, bias, c), x + bias[c]);
        ok = ok && same_bits(strix::ops::glu::epilogue<Round::none>(x, nullptr, c), x);
    }
    check(ok, "linear epilogue");
    // one rounding, not two: a value whose separate roundings differ from the joint one
    check(same_bits(strix::ops::glu::epilogue<Round::bf16>(1.0F + 0x1.0p-9F, bias, 0), bf(1.0F + 0x1.0p-9F + 0.25F)), "single rounding");
}

// Store epilogue (q8a8 View::round_bf16 / StoreRounded, known_rows PrefillRows::rawdots_bf16): the producer stores
// epilogue<bf16>(total, bias, o) instead of total, and the separate pass y = epilogue<bf16>(y, bias, o) is dropped.
// (1) Every FP32 pattern: the unbiased bf16 epilogue is idempotent, so a configuration that still runs the pass over
//     stored rounded values (a diagnostic path, a mixed binder) gives the same bits as either form alone.
// (2) K parts: the parts are added by the store in the same order, and the epilogue sees the same full total.
void store_epilogue_tests() {
    using strix::ops::glu::Round;
    std::uint64_t bad = 0;
    for (std::uint64_t u = 0; u <= 0xffffffffULL; ++u) {
        const float x = std::bit_cast<float>(static_cast<std::uint32_t>(u));
        const float once = strix::ops::glu::epilogue<Round::bf16>(x, nullptr, 0U);
        const float twice = strix::ops::glu::epilogue<Round::bf16>(once, nullptr, 0U);
        if (std::bit_cast<std::uint32_t>(once) != std::bit_cast<std::uint32_t>(twice)) ++bad;
    }
    check(bad == 0U, "bf16 epilogue idempotent over all 2^32 FP32 patterns");
    const float bias[4] = {0.25F, -1.5F, 3.0e-3F, 0.0F};
    bool ok = true;
    for (int i = 0; i < 20000; ++i) {
        const float s0 = next_value(40.0F), s1 = next_value(40.0F), s2 = next_value(40.0F);
        const auto o = static_cast<std::size_t>(i % 4);
        const float* b = (i & 4) != 0 ? bias : nullptr;
        // unfused: Store writes s0, adds s1, adds s2; then the epilogue pass
        float y = s0;
        y = y + s1;
        y = y + s2;
        const float unfused = strix::ops::glu::epilogue<Round::bf16>(y, b, o);
        // fused (enqueue_parts): plain Store for parts 0..1, StoreRounded on the last part: epilogue(*dst + value)
        float z = s0;
        z = z + s1;
        const float fused = strix::ops::glu::epilogue<Round::bf16>(z + s2, b, o);
        ok = ok && same_bits(unfused, fused);
        // one part: StoreRounded writes epilogue(sum); a later (kept) pass changes nothing when unbiased
        const float one = strix::ops::glu::epilogue<Round::bf16>(s0, b, o);
        if (b == nullptr) ok = ok && same_bits(strix::ops::glu::epilogue<Round::bf16>(one, nullptr, o), one);
    }
    check(ok, "store epilogue == plain store + epilogue pass (parts, bias, no bias)");
    // (3) Batched form (q8a8 kernel_linear, batch > 1, no bias; Hy4 attn_v_b: H products of vd columns, out_stride H * vd,
    //     batch_out vd): product z stores row r column o at r * stride + z * batch_out + o. These indices cover
    //     [0, rows * stride) exactly once, so the per-product store epilogue = the separate pass over the whole buffer.
    for (const auto& shape : {std::array<std::size_t, 3>{7U, 64U, 256U}, std::array<std::size_t, 3>{16U, 3U, 48U}}) {
        const std::size_t rows = shape[0], heads = shape[1], cols = shape[2], stride = heads * cols, n = rows * stride;
        std::vector<float> sums(n), pass(n), fused(n, 0.0F);
        std::vector<unsigned> hits(n, 0U);
        for (std::size_t i = 0; i < n; ++i) sums[i] = next_value(40.0F);
        for (std::size_t i = 0; i < n; ++i) pass[i] = strix::ops::glu::epilogue<Round::bf16>(sums[i], nullptr, 0U);  // separate pass
        for (std::size_t z = 0; z < heads; ++z)
            for (std::size_t r = 0; r < rows; ++r)
                for (std::size_t o = 0; o < cols; ++o) {
                    const std::size_t at = r * stride + z * cols + o;  // StoreRounded::put of product z (out + z * batch_out)
                    ++hits[at];
                    fused[at] = strix::ops::glu::epilogue<Round::bf16>(sums[at], nullptr, o);
                }
        bool cover = true, same = true;
        for (std::size_t i = 0; i < n; ++i) {
            cover = cover && hits[i] == 1U;
            same = same && same_bits(pass[i], fused[i]);
        }
        check(cover && same, "batched store epilogue covers every element once and equals the separate pass");
    }
}

// The MODEL_B reference rotation, written independently of the op (modelb-host-ref-v1).
void modelb_rope(float* x, std::size_t dims, std::uint32_t pos, float theta) {
    for (std::size_t i = 0; i < dims / 2U; ++i) {
        const float inv = 1.0F / std::pow(theta, static_cast<float>(2U * i) / static_cast<float>(dims));
        const float angle = static_cast<float>(pos) * inv;
        const float c = bf(static_cast<float>(std::cos(static_cast<double>(angle))));
        const float s = bf(static_cast<float>(std::sin(static_cast<double>(angle))));
        const float a = x[2U * i], b = x[2U * i + 1U];
        x[2U * i] = bf(a * c - b * s);
        x[2U * i + 1U] = bf(b * c + a * s);
    }
}

void rope_tests() {
    using strix::ops::rope::Round;
    using strix::ops::rope::View;
    constexpr std::uint32_t heads = 4, head = 256, pairs = 32, nope = 192, rows = 3;
    const std::uint32_t positions[rows] = {0U, 511U, 70000U};
    for (const float theta : {1.0e7F, 1.0e4F}) {
        // Table rows 0 .. 70000 would be large: build one table row per used position (positions index rows 0..2).
        std::vector<float> ct(rows * pairs), st(rows * pairs);
        for (std::uint32_t r = 0; r < rows; ++r)
            strix::ops::rope::build_native_table(theta, pairs, positions[r], 1U, Round::bf16, ct.data() + r * pairs,
                                                 st.data() + r * pairs);
        std::vector<float> x(std::size_t{rows} * heads * head);
        for (auto& v : x) v = bf(next_value(4.0F));
        auto want = x;
        for (std::uint32_t r = 0; r < rows; ++r)
            for (std::uint32_t h = 0; h < heads; ++h) modelb_rope(want.data() + (r * heads + h) * head + nope, 2U * pairs, positions[r], theta);
        const std::uint32_t slots[rows] = {0U, 1U, 2U};
        View v;
        v.x = x.data();
        v.rows = rows;
        v.vectors = heads;
        v.row_stride = heads * head;
        v.vector_stride = head;
        v.offset = nope;
        v.pairs = pairs;
        v.cos = ct.data();
        v.sin = st.data();
        v.positions = slots;
        strix::ops::rope::reference_apply(v, Round::bf16);
        check(std::memcmp(x.data(), want.data(), x.size() * sizeof(float)) == 0, "rope bf16 == modelb reference rotation");
        // Inverse with FP32 tables (BF16 cos / sin are not orthonormal) restores the input within FP32 rounding.
        for (std::uint32_t r = 0; r < rows; ++r)
            strix::ops::rope::build_native_table(theta, pairs, positions[r], 1U, Round::none, ct.data() + r * pairs,
                                                 st.data() + r * pairs);
        auto y = x;
        v.x = y.data();
        strix::ops::rope::reference_apply(v, Round::none);
        v.inverse = true;
        strix::ops::rope::reference_apply(v, Round::none);
        float worst = 0.0F;
        for (std::size_t i = 0; i < y.size(); ++i) worst = std::fmax(worst, std::fabs(y[i] - x[i]));
        check(worst < 1e-5F, "rope inverse round trip");
    }
    // Table rows are positions: one table of 4 rows from position 5 equals four one-row tables.
    std::vector<float> c4(4U * pairs), s4(4U * pairs), c1(pairs), s1(pairs);
    strix::ops::rope::build_native_table(1.0e4F, pairs, 5U, 4U, Round::bf16, c4.data(), s4.data());
    strix::ops::rope::build_native_table(1.0e4F, pairs, 7U, 1U, Round::bf16, c1.data(), s1.data());
    check(std::memcmp(c4.data() + 2U * pairs, c1.data(), pairs * sizeof(float)) == 0 &&
              std::memcmp(s4.data() + 2U * pairs, s1.data(), pairs * sizeof(float)) == 0, "table row = position");
    bool threw = false;
    try {
        strix::ops::rope::reference_apply(View{}, Round::bf16);
    } catch (const std::invalid_argument&) {
        threw = true;
    }
    check(threw, "empty view rejected");
}

}  // namespace

int main() {
    glu_tests();
    glm_swiglu_tests();
    swiglu_a8_tests();
    epilogue_tests();
    store_epilogue_tests();
    rope_tests();
    if (failures) return 1;
    std::printf("PASS ops rope/glu host contract\n");
    return 0;
}
