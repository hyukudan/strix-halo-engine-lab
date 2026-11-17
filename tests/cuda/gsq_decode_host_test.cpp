// GSQ decode identity (CPU): the word-level branch-free decode (GsqDecode<T>::word / scales, both the constant-codebook
// one-shot form and the load() + shared-codebook form the kernels use) equals the reference definition (code() per code,
// the former scales with the former branching FP16 conversion) bit for bit, for every GSQ storage type:
//   - FP16 -> FP32: all 65,536 inputs (bits, NaN payloads included);
//   - codebooks: every grid byte is 1..127 (the SWAR sign step needs it), the packed IQ4_NL words equal kvalues;
//   - every byte value at every block position (8 random backgrounds each), every word of every group;
//   - every codebook index for every word slot (qs byte x qh byte, 65,536 blocks for IQ3_S / IQ2_S);
//   - every FP16 scale value for every group.
#include "../../src/cuda/weight/gsq_decode.cuh"

#include <cstdint>
#include <cstdio>
#include <cstring>
#include <random>
#include <vector>

namespace {
namespace w = strix::cuda::weight;
namespace qd = strix::cuda::quant::detail;
int failures = 0;
#define CHECK(c, ...) do { if (!(c)) { if (++failures <= 20) { std::printf("FAIL %s:%d ", __FILE__, __LINE__); std::printf(__VA_ARGS__); std::printf("\n"); } } } while (0)

std::uint32_t bits(float f) { std::uint32_t u; std::memcpy(&u, &f, 4); return u; }
// The former branching conversion, verbatim (reference).
float ref_fp16(std::uint16_t h) {
    const std::uint32_t sign = static_cast<std::uint32_t>(h & 0x8000U) << 16U;
    std::uint32_t exp = (h >> 10U) & 0x1fU, man = h & 0x3ffU, b;
    if (exp == 0U) {
        if (man == 0U) b = sign;
        else {
            exp = 127U - 15U + 1U;
            while ((man & 0x400U) == 0U) { man <<= 1U; --exp; }
            b = sign | (exp << 23U) | ((man & 0x3ffU) << 13U);
        }
    } else if (exp == 31U) b = sign | 0x7f800000U | (man << 13U) | (man ? 0x00400000U : 0U);
    else b = sign | ((exp + 112U) << 23U) | (man << 13U);
    float f; std::memcpy(&f, &b, 4); return f;
}
// The former GsqDecode<T>::scales, verbatim except ref_fp16 (reference).
template <unsigned T> void ref_scales(const std::uint8_t* row, unsigned group, float& a, float& b) {
    unsigned s; const auto* x = w::GsqDecode<T>::block(row, group, s);
    const float d = ref_fp16(std::uint16_t(x[0]) | (std::uint16_t(x[1]) << 8));
    b = 0.0F; a = 0.0F;
    if constexpr (T == 20 || T == 42) a = d;
    if constexpr (T == 21) a = d * float(1 + 2 * int((x[strix::weight::formats::Iq3S::scales + s / 2] >> (4 * (s % 2))) & 15));
    if constexpr (T == 18) {
        const unsigned off = 66 + 4 * s;
        const unsigned aux = unsigned(x[off]) | (unsigned(x[off + 1]) << 8) | (unsigned(x[off + 2]) << 16) | (unsigned(x[off + 3]) << 24);
        a = (d * (0.5F + float(aux >> 28))) * 0.5F;
    }
    if constexpr (T == 22) { a = (d * (0.5F + float(x[74 + s] & 15))) * 0.25F; b = (d * (0.5F + float(x[74 + s] >> 4))) * 0.25F; }
    if constexpr (T == 23) {
        const unsigned high = unsigned(x[2]) | (unsigned(x[3]) << 8);
        const unsigned ls = ((x[4 + s / 2] >> (4 * (s % 2))) & 15) | (((high >> (2 * s)) & 3) << 4);
        a = d * float(int(ls) - 32);
    }
}
template <unsigned T> unsigned ref_word(const std::uint8_t* row, unsigned group, unsigned index) {
    unsigned r = 0;
    for (unsigned j = 0; j < 4; ++j) r |= (unsigned(w::GsqDecode<T>::code(row, group, 4 * index + j)) & 255U) << (8 * j);
    return r;
}

template <unsigned T> struct Checker {
    using D = w::GsqDecode<T>;
    std::vector<std::uint32_t> smem = std::vector<std::uint32_t>(D::grid_words ? D::grid_words : 1U);
    long words = 0, scale_checks = 0;
    Checker() { w::gsq_grid_fill<T>(smem.data(), 0U, 1U); }
    void block(const std::uint8_t* x) {
        const w::GsqSmemGrid grid{smem.data()};
        for (unsigned g = 0; g < D::be / 32U; ++g) {
            for (unsigned i = 0; i < 8U; ++i) {
                const unsigned ref = ref_word<T>(x, g, i);
                CHECK(D::word(x, g, i) == ref, "T%u one-shot word g%u i%u: %08x vs %08x", T, g, i, D::word(x, g, i), ref);
                ++words;
            }
            for (unsigned q = 0; q < 4U; ++q) {  // the FAST lane pair (q, q + 4) through load() + the shared codebook
                const auto raw = D::load(x, g, q, q + 4U);
                CHECK(D::word(raw, 0U, q, grid) == ref_word<T>(x, g, q), "T%u smem word g%u i%u", T, g, q);
                CHECK(D::word(raw, 1U, q + 4U, grid) == ref_word<T>(x, g, q + 4U), "T%u smem word g%u i%u", T, g, q + 4U);
            }
            float a, b, ra, rb;
            D::scales(x, g, a, b);
            ref_scales<T>(x, g, ra, rb);
            CHECK(bits(a) == bits(ra) && bits(b) == bits(rb), "T%u scales g%u: %08x %08x vs %08x %08x", T, g, bits(a), bits(b), bits(ra), bits(rb));
            ++scale_checks;
        }
    }
    void run(std::mt19937& rng) {
        std::vector<std::uint8_t> x(D::bb);
        auto fill = [&] { for (auto& c : x) c = std::uint8_t(rng()); };
        for (unsigned p = 0; p < D::bb; ++p)  // every value at every position
            for (unsigned v = 0; v < 256U; ++v)
                for (unsigned r = 0; r < 8U; ++r) { fill(); x[p] = std::uint8_t(v); block(x.data()); }
        for (unsigned h = 0; h < 65536U; ++h) { fill(); x[0] = std::uint8_t(h); x[1] = std::uint8_t(h >> 8); block(x.data()); }  // every FP16 scale
        if constexpr (T == 21 || T == 22) {  // every codebook index in every word slot of sub-group 0: qs byte x qh byte
            const unsigned qs = 2, nqs = T == 21 ? 8U : 4U, qh = 66;
            for (unsigned v = 0; v < 256U; ++v)
                for (unsigned h = 0; h < 256U; ++h) {
                    fill();
                    for (unsigned k = 0; k < nqs; ++k) x[qs + k] = std::uint8_t(v);
                    x[qh] = std::uint8_t(h);
                    block(x.data());
                }
        }
        std::printf("T%-2u words %ld scales %ld\n", T, words, scale_checks);
    }
};

void test_fp16() {
    int bad = 0;
    for (unsigned h = 0; h < 65536U; ++h)
        if (bits(qd::fp16(std::uint16_t(h))) != bits(ref_fp16(std::uint16_t(h)))) ++bad;
    CHECK(bad == 0, "fp16: %d of 65536 differ", bad);
    std::printf("fp16 65536/65536 checked, %d differ\n", bad);
}
void test_codebooks() {
    namespace g = w::gsq_detail;
    const std::uint32_t kv[4] = {g::kKv0, g::kKv1, g::kKv2, g::kKv3};
    for (unsigned i = 0; i < 16U; ++i) CHECK(std::int8_t((kv[i / 4] >> (8 * (i % 4))) & 255U) == qd::kvalues_iq4nl_h[i], "kvalue word %u", i);
    auto bytes_ok = [](std::uint64_t e, unsigned n) {
        for (unsigned j = 0; j < n; ++j) { const unsigned b = unsigned(e >> (8 * j)) & 255U; if (b < 1U || b > 127U) return false; }
        return true;
    };
    for (unsigned i = 0; i < 512U; ++i) CHECK(bytes_ok(qd::iq3s_grid_h[i], 4), "iq3s grid %u", i);
    for (unsigned i = 0; i < 256U; ++i) CHECK(bytes_ok(qd::iq3xxs_grid_h[i], 4), "iq3xxs grid %u", i);
    for (unsigned i = 0; i < 1024U; ++i) CHECK(bytes_ok(qd::iq2s_grid_h[i], 8), "iq2s grid %u", i);
    for (unsigned n = 0; n < 65536U; ++n) {  // kvalue4 on all 4-index patterns (two bytes x all values, both halves)
        const std::uint32_t idx = (n & 15U) | (((n >> 4) & 15U) << 8) | (((n >> 8) & 15U) << 16) | (((n >> 12) & 15U) << 24);
        std::uint32_t ref = 0;
        for (unsigned j = 0; j < 4U; ++j) ref |= (std::uint32_t(std::uint8_t(qd::kvalues_iq4nl_h[(idx >> (8 * j)) & 15U]))) << (8 * j);
        CHECK(g::kvalue4(idx) == ref, "kvalue4 %08x", idx);
    }
    for (unsigned v = 0; v < 256U; ++v) {
        std::uint32_t ref = 0;
        for (unsigned j = 0; j < 4U; ++j) ref |= (std::uint32_t(int((v >> (2 * j)) & 3U) - 1) & 255U) << (8 * j);
        CHECK(g::crumbs4(v) == ref, "crumbs4 %u", v);
    }
}
}  // namespace

int main() {
    std::mt19937 rng(20261117U);
    test_fp16();
    test_codebooks();
    Checker<18>{}.run(rng);
    Checker<20>{}.run(rng);
    Checker<21>{}.run(rng);
    Checker<22>{}.run(rng);
    Checker<23>{}.run(rng);
    Checker<42>{}.run(rng);
    std::printf("%s: %d failures\n", failures ? "FAIL" : "PASS", failures);
    return failures ? 1 : 0;
}
