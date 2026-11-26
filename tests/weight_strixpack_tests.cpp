// Strix pack host layout (strix/weight/host_lowbit.hpp sp::) and its calibrated coder (host_lowbit_gptq.hpp
// encode_rows_sp): the 16-byte header round-trips and every rule refuses a bad header; every scale byte round-trips
// through nearest_scale_byte and is exact; hand-written codes land at their documented bits for SP2 / SP3 / SP4;
// pack -> unpack is bit-identical to the reference decode for the default, linear, fitted and random valid tables,
// on tensors with zero groups, outliers and tiny groups; every decoded weight is the nearest table level for its stored
// scale and is exact in BF16; the coder without calibration equals the round-to-nearest coder byte for byte; the fitted
// table is never worse on its sample; not_foldable follows the used scales.
#include "strix/weight/host_lowbit.hpp"
#include "strix/weight/host_lowbit_gptq.hpp"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <iostream>
#include <random>
#include <string_view>
#include <vector>

namespace {

namespace wei = strix::weight;
namespace sp = strix::weight::sp;
namespace gq = strix::weight::lowbit::gptq;

int failures = 0;
void check(bool ok, const char* what) {
    if (!ok) {
        ++failures;
        std::cerr << "FAIL: " << what << '\n';
    }
}

bool same_bits(float a, float b) { return std::memcmp(&a, &b, sizeof a) == 0; }
bool bf16_exact(float v) {
    std::uint32_t u = 0;
    std::memcpy(&u, &v, sizeof u);
    return (u & 0xffffU) == 0U;
}

std::vector<unsigned> magnitudes() {
    std::vector<unsigned> m;
    for (unsigned t = 0; t <= 40U; ++t)
        if (sp::representable_magnitude(t)) m.push_back(t);
    return m;
}

void check_magnitudes() {
    const auto m = magnitudes();
    const std::vector<unsigned> expect{1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 18, 20, 22, 24, 26, 28, 30};
    check(m == expect, "representable magnitudes = integers 1..31 with <= 4 significant bits");
}

void check_header() {
    sp::Header h;
    h.bits = 4;
    h.base = -37;
    h.table = sp::default_table<4>();
    check(*h.invalid() == '\0', "default SP4 header valid");
    std::uint8_t raw[sp::kHeaderBytes];
    h.write(raw);
    check(raw[0] == 1U && raw[1] == 4U && raw[2] == 5U && raw[3] == 0U && raw[4] == static_cast<std::uint8_t>(-37) &&
              raw[8] == 1U && raw[15] == 24U,
          "header bytes at their offsets");
    sp::Header back;
    check(*sp::Header::read(raw, 4U, back) == '\0' && back.base == -37 && back.table == h.table && back.bits == 4U,
          "header round trip");
    for (std::size_t len = 0U; len < sp::kHeaderBytes; ++len) {
        sp::Header out;
        out.base = 99;
        check(*sp::Header::read(std::span(raw).first(len), 4U, out) != '\0' && out.base == 99,
              "short on-disk header is refused before reading");
    }
    check(*sp::Header::read(raw, 3U, back) != '\0' && back.bits == 4U,
          "valid SP4 header cannot describe SP3 storage");
    const auto refused = [&](auto mutate, const char* what) {
        std::uint8_t r[sp::kHeaderBytes];
        h.write(r);
        mutate(r);
        sp::Header out;
        out.base = 99;
        check(*sp::Header::read(r, 4U, out) != '\0' && out.base == 99, what);
    };
    refused([](std::uint8_t* r) { r[0] = 2; }, "version refused");
    refused([](std::uint8_t* r) { r[1] = 5; }, "bits 5 refused");
    refused([](std::uint8_t* r) { r[1] = 6; }, "bits 6 refused (no table width)");
    refused([](std::uint8_t* r) { r[2] = 6; }, "scale group 64 refused");
    refused([](std::uint8_t* r) { r[3] = sp::kFlagFp16Scale64; }, "fp16_scale64 flag refused");
    refused([](std::uint8_t* r) { r[3] = 0x80; }, "unknown flag refused");
    refused([](std::uint8_t* r) { r[3] = sp::kFlagLinear; }, "linear flag with a non-linear table refused");
    refused([](std::uint8_t* r) { r[4] = static_cast<std::uint8_t>(-121); }, "base below range refused");
    refused([](std::uint8_t* r) { r[4] = 91; }, "base above range refused");
    refused([](std::uint8_t* r) { r[6] = 1; }, "reserved byte refused");
    refused([](std::uint8_t* r) { r[9] = 1; }, "table not increasing refused");
    refused([](std::uint8_t* r) { r[15] = 17; }, "magnitude 17 (5 significant bits) refused");
    refused([](std::uint8_t* r) { r[15] = 32; }, "magnitude 32 refused");
    refused([](std::uint8_t* r) { r[8] = 0; }, "magnitude 0 refused");
    sp::Header h3;
    h3.bits = 3;
    h3.table = sp::default_table<3>();
    check(*h3.invalid() == '\0', "default SP3 header valid");
    h3.table[4] = 1;
    check(*h3.invalid() != '\0', "SP3 entry beyond the levels refused");
    sp::Header lin;
    lin.bits = 2;
    lin.table = sp::linear_table<2>();
    lin.flags = sp::kFlagLinear;
    check(*lin.invalid() == '\0', "SP2 linear header valid");
    for (const auto& t : {sp::default_table<4>(), sp::default_table<3>(), sp::default_table<2>(), sp::linear_table<4>(),
                          sp::linear_table<3>()}) {
        bool ok = true;
        for (const auto v : t) ok = ok && (v == 0U || sp::representable_magnitude(v));
        check(ok, "default / linear tables hold representable magnitudes");
    }
}

void check_scales() {
    for (const int base : {sp::kMinBase, -60, -14, 0, sp::kMaxBase - 31}) {
        float prev = 0.0F;
        bool exact = true, round_trip = true, increasing = true, midpoint_low = true;
        for (unsigned b = 0; b < 256U; ++b) {
            const auto byte = static_cast<std::uint8_t>(b);
            const float s = sp::scale_value(byte, base);
            const double want = std::ldexp(8.0 + (b & 7U), static_cast<int>(b >> 3U) + base - 3);
            exact = exact && static_cast<double>(s) == want && std::isnormal(s);
            round_trip = round_trip && sp::nearest_scale_byte(static_cast<double>(s), base) == byte;
            increasing = increasing && s > prev;
            if (b > 0U) {
                const double mid = 0.5 * (static_cast<double>(prev) + static_cast<double>(s));
                midpoint_low = midpoint_low && sp::nearest_scale_byte(mid, base) == static_cast<std::uint8_t>(b - 1U);
            }
            prev = s;
        }
        check(exact, "scale byte value = (1 + m / 8) * 2^(e + base), normal FP32");
        check(round_trip, "nearest_scale_byte(scale_value(b)) == b for all 256 bytes");
        check(increasing, "scale values strictly increase with the byte");
        check(midpoint_low, "a scale midway between two bytes rounds to the lower byte");
        check(sp::nearest_scale_byte(std::ldexp(1.0, base - 1), base) == 0U, "below the lowest scale -> byte 0");
        check(sp::nearest_scale_byte(std::ldexp(1.0, base + 33), base) == 255U, "above the highest scale -> byte 255");
        check(sp::nearest_scale_byte(1.97 * std::ldexp(1.0, base + 31), base) == 255U, "carry past e = 31 -> byte 255");
    }
    check(sp::nearest_scale_byte(0.0, 0) == 0U, "zero scale -> byte 0");
    check(sp::choose_base(1.0F) == -30 && sp::choose_base(1.5F) == -30 && sp::choose_base(0.0F) == 0,
          "base puts the largest scale in the top binades");
    check(sp::choose_base(1e-38F) == sp::kMinBase && sp::choose_base(1e38F) == sp::kMaxBase, "base clamped");
}

// Every magnitude x every scale byte is exact in BF16 and FP16-representable when normal.
void check_bf16_products() {
    bool exact = true;
    for (const int base : {-60, -30, -14, 0})
        for (const auto t : magnitudes())
            for (unsigned b = 0; b < 256U; ++b) {
                const float v = static_cast<float>(t) * sp::scale_value(static_cast<std::uint8_t>(b), base);
                exact = exact && bf16_exact(v) && bf16_exact(-v);
            }
    check(exact, "table magnitude x scale is exact in BF16 for every magnitude and byte");
}

template <unsigned Bits>
void check_layout() {
    using S = sp::Storage<Bits>;
    static_assert(S::bytes == (Bits == 4U ? 68U : Bits == 3U ? 52U : 36U));
    static_assert(S::bytes % 4U == 0U);
    for (unsigned pattern = 0; pattern < 3U; ++pattern) {
        unsigned codes[S::elements];
        std::uint8_t scales[S::groups] = {0, 77, 200, 255};
        for (unsigned j = 0; j < S::elements; ++j) codes[j] = (j * 7U + pattern * 3U + j / 5U) % (1U << Bits);
        std::uint8_t block[S::bytes];
        sp::pack_block<S>(codes, scales, block);
        bool ok = true, bits_ok = true;
        for (unsigned j = 0; j < S::elements; ++j) {
            ok = ok && S::code(block, j) == codes[j];
            unsigned raw = 0;  // the documented bit positions, read independently
            if constexpr (Bits == 4U) raw = (block[j / 2U] >> (4U * (j & 1U))) & 15U;
            else if constexpr (Bits == 2U) raw = (block[j / 4U] >> (2U * (j & 3U))) & 3U;
            else raw = ((block[j / 4U] >> (2U * (j & 3U))) & 3U) | (((block[32U + j / 8U] >> (j & 7U)) & 1U) << 2U);
            bits_ok = bits_ok && raw == codes[j];
        }
        check(ok, "code(j) returns the packed code at every position");
        check(bits_ok, "codes sit at the documented bit positions");
        for (unsigned g = 0; g < S::groups; ++g) check(S::scale_byte(block, g) == scales[g], "scale bytes after the codes");
        sp::Header h;
        h.bits = static_cast<std::uint8_t>(Bits);
        h.base = -40;
        h.table = sp::default_table<Bits>();
        bool dec = true;
        for (unsigned j = 0; j < S::elements; ++j) {
            const unsigned c = codes[j];
            const float mag = static_cast<float>(h.table[c & (S::levels - 1U)]) * sp::scale_value(scales[j / 32U], h.base);
            dec = dec && same_bits(sp::decode<S>(block, j, h), (c >> (Bits - 1U)) != 0U ? -mag : mag);
        }
        check(dec, "decode = sign x table[index] x scale");
    }
}

std::vector<float> make_tensor(std::size_t n, std::size_t k, unsigned seed) {
    std::mt19937 rng(seed);
    std::normal_distribution<float> normal(0.0F, 0.02F);
    std::vector<float> w(n * k);
    for (auto& v : w) v = normal(rng);
    for (std::size_t i = 0; i < 32U; ++i) w[k + i] = 0.0F;  // an all-zero group (row 1, group 0)
    w[5] = 0.9F;                                              // an outlier
    w[2 * k + 64] = -0.7F;
    for (std::size_t i = 0; i < 32U; ++i) w[3 * k + 96 + i] = 1e-9F * static_cast<float>(i);  // a tiny group
    w[4 * k + 7] = -0.0F;
    return w;
}

template <unsigned Bits>
std::array<std::uint8_t, 8> random_table(std::mt19937& rng) {
    const auto m = magnitudes();
    std::array<std::uint8_t, 8> t{};
    std::vector<unsigned> pick;
    while (pick.size() < (1U << (Bits - 1U))) {
        const unsigned v = m[rng() % m.size()];
        bool seen = false;
        for (const auto p : pick) seen = seen || p == v;
        if (!seen) pick.push_back(v);
    }
    std::sort(pick.begin(), pick.end());
    for (std::size_t i = 0; i < (std::size_t{1} << (Bits - 1U)); ++i) t[i] = static_cast<std::uint8_t>(pick[i]);
    return t;
}

// pack -> unpack bit-identical, nearest level for the stored scale, BF16 exact; returns the decoded weights.
template <unsigned Bits>
void check_round_trip(const std::vector<float>& w, std::size_t n, std::size_t k, const sp::Header& h, const char* what) {
    using S = sp::Storage<Bits>;
    check(*h.invalid() == '\0', what);
    std::vector<std::uint8_t> bytes(n * S::row_bytes(k));
    std::vector<float> dq(n * k);
    sp::encode_rows<S>(w.data(), n, k, h, bytes.data(), dq.data());
    bool identical = true, nearest = true, exact = true, sign = true;
    const std::size_t blocks = k / S::elements;
    for (std::size_t r = 0; r < n; ++r)
        for (std::size_t b = 0; b < blocks; ++b) {
            const std::uint8_t* block = bytes.data() + (r * blocks + b) * S::bytes;
            for (unsigned j = 0; j < S::elements; ++j) {
                const std::size_t e = r * k + b * S::elements + j;
                const float v = sp::decode<S>(block, j, h);
                identical = identical && same_bits(v, dq[e]);
                exact = exact && bf16_exact(v);
                const double s = static_cast<double>(sp::scale_value(S::scale_byte(block, j / 32U), h.base));
                bool zero_group = true;
                for (unsigned i = 0; i < 32U; ++i) zero_group = zero_group && w[r * k + b * S::elements + (j / 32U) * 32U + i] == 0.0F;
                if (zero_group) {
                    nearest = nearest && S::code(block, j) == 0U;
                    continue;
                }
                const double a = std::fabs(static_cast<double>(w[e]));
                const double got = std::fabs(static_cast<double>(v));
                for (unsigned i = 0; i < S::levels; ++i)
                    nearest = nearest && std::fabs(a - got) <= std::fabs(a - static_cast<double>(h.table[i]) * s);
                sign = sign && ((w[e] < 0.0F) == (v < 0.0F));
            }
        }
    check(identical, "pack -> unpack bit-identical to the coder's decode");
    check(nearest, "each weight decodes to the nearest table level of its stored scale (zero groups: code 0)");
    check(sign, "sign kept (negative zero coded as positive)");
    check(exact, "every decoded weight exact in BF16");
    // The calibrated coder without calibration = the round-to-nearest coder, byte for byte and value for value.
    std::vector<std::uint8_t> via(bytes.size(), 0xa5);
    std::vector<float> dq2(n * k);
    gq::Options o;
    o.dequant = dq2.data();
    o.threads = 3;
    check(bool(gq::encode_rows_sp<S>(h, w.data(), n, k, via.data(), o)), "gptq sp coder ok");
    check(via == bytes, "gptq coder without calibration == round-to-nearest bytes");
    check(std::memcmp(dq.data(), dq2.data(), dq.size() * sizeof(float)) == 0, "gptq coder decode == RTN decode");
}

template <unsigned Bits>
void check_tables() {
    using S = sp::Storage<Bits>;
    constexpr std::size_t n = 24, k = 256;
    const auto w = make_tensor(n, k, 11U + Bits);
    check_round_trip<Bits>(w, n, k, sp::make_header<S>(w.data(), n, k, sp::default_table<Bits>()), "default table header");
    auto lin = sp::make_header<S>(w.data(), n, k, sp::linear_table<Bits>());
    check((lin.flags & sp::kFlagLinear) != 0U, "linear table sets the linear flag");
    check_round_trip<Bits>(w, n, k, lin, "linear table header");
    const auto fitted = sp::fit_table<S>(w.data(), n, k, 1U);
    const auto hf = sp::make_header<S>(w.data(), n, k, fitted);
    check_round_trip<Bits>(w, n, k, hf, "fitted table header");
    const double e_def = sp::table_error<S>(w.data(), n, k, 1U, sp::make_header<S>(w.data(), n, k, sp::default_table<Bits>()), nullptr);
    check(sp::table_error<S>(w.data(), n, k, 1U, hf, nullptr) <= e_def, "fitted table never worse than the default on its sample");
    std::mt19937 rng(1234U + Bits);
    for (unsigned i = 0; i < 40U; ++i)
        check_round_trip<Bits>(w, n, k, sp::make_header<S>(w.data(), n, k, random_table<Bits>(rng)), "random table header");
    // Extreme tables: the smallest and the largest representable magnitudes.
    std::array<std::uint8_t, 8> low{}, high{};
    const auto m = magnitudes();
    for (unsigned i = 0; i < S::levels; ++i) {
        low[i] = static_cast<std::uint8_t>(m[i]);
        high[i] = static_cast<std::uint8_t>(m[m.size() - S::levels + i]);
    }
    check_round_trip<Bits>(w, n, k, sp::make_header<S>(w.data(), n, k, low), "lowest table header");
    check_round_trip<Bits>(w, n, k, sp::make_header<S>(w.data(), n, k, high), "highest table header");
}

// The largest group of a tensor never needs a byte above 255 (base headroom), and the error feedback decode matches.
template <unsigned Bits>
void check_edges() {
    using S = sp::Storage<Bits>;
    constexpr std::size_t n = 8, k = 128;
    std::vector<float> w(n * k, 0.0F);
    for (std::size_t i = 0; i < n * k; ++i) w[i] = static_cast<float>((static_cast<int>(i * 37U % 101U) - 50)) * 1e-3F;
    for (std::size_t i = 0; i < 32U; ++i) w[i] = (i % 2U ? -1.0F : 1.0F) * 3.0F;  // the largest group, flat
    auto h = sp::make_header<S>(w.data(), n, k, sp::default_table<Bits>());
    std::vector<std::uint8_t> bytes(n * S::row_bytes(k));
    std::vector<float> dq(n * k);
    sp::encode_rows<S>(w.data(), n, k, h, bytes.data(), dq.data());
    check(S::scale_byte(bytes.data(), 0) < 255U, "largest group encodes below the top byte");
    double err = 0.0;
    for (unsigned j = 0; j < 32U; ++j) err += std::fabs(static_cast<double>(w[j] - dq[j]));
    check(err / 32.0 < 3.0 * 0.07, "largest group within one scale step");
    sp::mark_foldable<S>(h, bytes.data(), n, k);
    check((h.flags & sp::kFlagNotFoldable) == 0U, "ordinary tensor is foldable");
    std::vector<float> tiny(n * k);
    for (std::size_t i = 0; i < n * k; ++i) tiny[i] = w[i] * 1e-6F;
    auto ht = sp::make_header<S>(tiny.data(), n, k, sp::default_table<Bits>());
    sp::encode_rows<S>(tiny.data(), n, k, ht, bytes.data(), dq.data());
    sp::mark_foldable<S>(ht, bytes.data(), n, k);
    check((ht.flags & sp::kFlagNotFoldable) != 0U, "products below the FP16 normal range mark not_foldable");
    check(*ht.invalid() == '\0', "not_foldable header valid");
    // Error feedback with a correlated H: the stored bytes still decode to the coder's values.
    const auto big = make_tensor(16, 256, 5U);
    std::vector<double> hm(256 * 256, 0.0);
    std::mt19937 rng(9U);
    std::normal_distribution<float> nd(0.0F, 1.0F);
    std::vector<float> x(64 * 256);
    for (std::size_t r = 0; r < 64U; ++r) {
        const float common = nd(rng);
        for (std::size_t c = 0; c < 256U; ++c) x[r * 256 + c] = nd(rng) + 0.8F * common;
    }
    gq::accumulate_second_moments(x.data(), 64, 256, hm);
    std::vector<std::uint8_t> dead;
    const auto factor = gq::inverse_factor(hm, 256, 0.01, dead);
    std::vector<float> imp(256);
    for (std::size_t i = 0; i < 256U; ++i) imp[i] = static_cast<float>(hm[i * 256 + i]);
    auto hb = sp::make_header<S>(big.data(), 16, 256, sp::fit_table<S>(big.data(), 16, 256, 1U, imp.data()));
    std::vector<std::uint8_t> fb(16 * S::row_bytes(256));
    std::vector<float> fdq(16 * 256);
    gq::Options o;
    o.factor = factor.data();
    o.dead = dead.data();
    o.importance = imp.data();
    o.dequant = fdq.data();
    o.threads = 2;
    check(bool(gq::encode_rows_sp<S>(hb, big.data(), 16, 256, fb.data(), o)), "feedback coder ok");
    bool same = true;
    for (std::size_t r = 0; r < 16U; ++r)
        for (std::size_t j = 0; j < 256U; ++j)
            same = same && same_bits(sp::decode<S>(fb.data() + (r * 2U + j / 128U) * S::bytes, static_cast<unsigned>(j % 128U), hb),
                                     fdq[r * 256 + j]);
    check(same, "feedback bytes decode to the coder's values");
    std::vector<std::uint8_t> fb1(fb.size());
    o.threads = 1;
    o.dequant = nullptr;
    check(bool(gq::encode_rows_sp<S>(hb, big.data(), 16, 256, fb1.data(), o)) && fb1 == fb, "feedback bytes independent of threads");
}

// The codec refactor keeps the q4_g64 coder: encode_rows (no technique) == encode_block, block for block.
void check_q4_unchanged() {
    constexpr std::size_t n = 20, k = 256;
    const auto w = make_tensor(n, k, 3U);
    std::vector<std::uint8_t> a(n * wei::Q4G64Storage::row_bytes(k)), b(a.size());
    check(bool(gq::encode_rows<wei::Q4G64Storage>(w.data(), n, k, a.data())), "q4 rows ok");
    for (std::size_t blk = 0; blk < n * k / 128U; ++blk)
        check(bool(wei::lowbit::encode_block<wei::Q4G64Storage>(w.data() + blk * 128U, b.data() + blk * wei::Q4G64Storage::bytes)),
              "q4 block ok");
    check(a == b, "q4_g64 through the codec policy == encode_block");
}

}  // namespace

int main() {
    check_magnitudes();
    check_header();
    check_scales();
    check_bf16_products();
    check_layout<2>();
    check_layout<3>();
    check_layout<4>();
    check_tables<2>();
    check_tables<3>();
    check_tables<4>();
    check_edges<2>();
    check_edges<3>();
    check_edges<4>();
    check_q4_unchanged();
    if (failures) {
        std::cerr << failures << " failures\n";
        return 1;
    }
    std::cout << "weight_strixpack_tests: PASS\n";
    return 0;
}
