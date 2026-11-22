// KV shadow contract (strix/attn/cache_shadow.hpp): the shared writer reference reproduces, bit for bit and flag for
// flag, the scalar form of the engine mirror writers it replaces (FP32 master -> F16 halves; FP32 master -> int8 codes
// with one FP16 scale per 16 features; position-tag check), on edge values and on random rows.
#include "strix/attn/cache_shadow.hpp"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <limits>
#include <random>
#include <vector>

namespace cs = strix::attn::cache_shadow;

namespace {

int failures = 0;
void expect(bool ok, const char* what, std::uint64_t at) {
    if (!ok) {
        ++failures;
        if (failures < 20)
            std::fprintf(stderr, "FAIL %s at %llu\n", what, static_cast<unsigned long long>(at));
    }
}

// --- the engine's mirror writers, transcribed statement by statement (per-thread bodies run serially) ---
unsigned private_mirror_f16(const float* k, const float* v, const std::uint64_t* tags, std::uint64_t from, unsigned rows,
                            unsigned width, _Float16* hk, _Float16* hv) {
    unsigned failure = 0U;
    for (std::size_t index = 0; index < std::size_t(rows) * width; index += 4U) {
        const auto p = from + index / width;
        if (index % width == 0 && tags[p] != p)
            failure |= 2U;
        const auto at = std::size_t(p) * width + index % width;
        bool overflow = false;
        for (unsigned j = 0; j < 4; ++j) {
            const auto x = static_cast<_Float16>(k[at + j]), y = static_cast<_Float16>(v[at + j]);
            overflow |= (__builtin_bit_cast(unsigned short, x) & 0x7c00U) == 0x7c00U ||
                        (__builtin_bit_cast(unsigned short, y) & 0x7c00U) == 0x7c00U;
            hk[at + j] = x;
            hv[at + j] = y;
        }
        if (overflow)
            failure |= 4U;
    }
    return failure;
}
constexpr unsigned kv8_group = 16U;
void kv8_finite(float x, unsigned* failure) {
    if ((__builtin_bit_cast(unsigned, x) & 0x7f800000U) == 0x7f800000U)
        *failure |= 1U;
}
float kv8_abs(float x) { return __builtin_bit_cast(float, __builtin_bit_cast(unsigned, x) & 0x7fffffffU); }
unsigned short kv8_scale_bits(float amax, unsigned* failure) {
    const auto d = static_cast<_Float16>(amax / 127.0F);
    const auto bits = __builtin_bit_cast(unsigned short, d);
    if ((bits & 0x7c00U) == 0x7c00U)
        *failure |= 4U;
    return bits;
}
float kv8_scale_value(unsigned short bits) { return float(__builtin_bit_cast(_Float16, bits)); }
int kv8_code(float x, float d) {
    if (d == 0.0F)
        return 0;
    float c = __builtin_rintf(x / d);
    c = c > 127.0F ? 127.0F : (c < -127.0F ? -127.0F : c);
    return int(c);
}
unsigned private_mirror_kv8(const float* k, const float* v, const std::uint64_t* tags, std::uint64_t from, unsigned rows,
                            unsigned width, std::int8_t* ck, std::int8_t* cv, std::uint16_t* sk, std::uint16_t* sv) {
    unsigned failure = 0U;
    const unsigned groups = width / kv8_group;
    for (unsigned y = 0; y < 2U; ++y)
        for (std::size_t group_index = 0; group_index < std::size_t(rows) * groups; ++group_index) {
            const bool value = y != 0;
            const auto p = from + group_index / groups;
            if (!value && group_index % groups == 0 && tags[p] != p)
                failure |= 2U;
            const auto at = std::size_t(p) * width + (group_index % groups) * kv8_group;
            const float* src = (value ? v : k) + at;
            float x[kv8_group];
            float amax = 0.0F;
            for (unsigned j = 0; j < kv8_group; ++j)
                x[j] = src[j];
            for (unsigned j = 0; j < kv8_group; ++j) {
                kv8_finite(x[j], &failure);
                const float a = kv8_abs(x[j]);
                amax = a > amax ? a : amax;
            }
            const auto bits = kv8_scale_bits(amax, &failure);
            const float d = kv8_scale_value(bits);
            std::int8_t* codes = (value ? cv : ck) + at;
            for (unsigned j = 0; j < kv8_group; ++j)
                codes[j] = std::int8_t(kv8_code(x[j], d));
            (value ? sv : sk)[std::size_t(p) * groups + group_index % groups] = bits;
        }
    return failure;
}
// The FAST reader's conversions of a row the shadow does not hold yet.
float private_rounded(float x, unsigned* failure) {
    const auto h = static_cast<_Float16>(x);
    if ((__builtin_bit_cast(unsigned short, h) & 0x7c00U) == 0x7c00U)
        *failure |= 4U;
    return float(h);
}
float private_half_value(unsigned bits16) { return float(__builtin_bit_cast(_Float16, (unsigned short)bits16)); }

float bits_f(std::uint32_t b) { return __builtin_bit_cast(float, b); }
std::uint32_t f_bits(float x) { return __builtin_bit_cast(std::uint32_t, x); }

void fill_edges(std::vector<float>& x) {
    const float inf = std::numeric_limits<float>::infinity(), nan = std::numeric_limits<float>::quiet_NaN();
    const float edges[] = {0.0F, -0.0F, 1.0F, -1.0F, 65504.0F, 65519.99F, 65520.0F, -65520.0F, 6.1035156e-5F, 5.9604645e-8F,
                           2.9802322e-8F, 8.940697e-8F, 1e-30F, inf, -inf, nan, 8.3e6F, 8.4e6F, 127.0F, 63.5F, 0.5F, 1.5F,
                           bits_f(0x7f7fffffU), bits_f(0x00000001U), bits_f(0x80000001U), 3.0e-3F};
    for (std::size_t i = 0; i < x.size(); ++i)
        x[i] = edges[i % (sizeof(edges) / sizeof(edges[0]))];
}

void f16_case(const std::vector<float>& k, const std::vector<float>& v, const std::vector<std::uint64_t>& tags,
              std::uint64_t from, unsigned rows, unsigned width, const char* what) {
    const std::size_t n = k.size();
    std::vector<_Float16> pk(n), pv(n);
    std::vector<std::uint16_t> sk(n, 0x1212U), sv(n, 0x1212U);  // rows outside the launch: same filler bytes
    std::memset(pk.data(), 0x12, n * sizeof(_Float16));
    std::memset(pv.data(), 0x12, n * sizeof(_Float16));
    const unsigned fp = private_mirror_f16(k.data(), v.data(), tags.data(), from, rows, width, pk.data(), pv.data());
    const unsigned fs = cs::mirror_f16_ref<cs::Tagged>(k.data(), v.data(), tags.data(), from, rows, width, sk.data(),
                                                       sv.data());
    expect(fp == fs, what, fp * 16U + fs);
    for (std::size_t i = 0; i < n; ++i) {
        expect(__builtin_bit_cast(std::uint16_t, pk[i]) == sk[i] && __builtin_bit_cast(std::uint16_t, pv[i]) == sv[i],
               what, i);
        const float a = cs::F16::decode(sk[i]), b = private_half_value(sk[i]);
        expect(f_bits(a) == f_bits(b) || (std::isnan(a) && std::isnan(b)), "f16 decode", i);
    }
}

void i8_case(const std::vector<float>& k, const std::vector<float>& v, const std::vector<std::uint64_t>& tags,
             std::uint64_t from, unsigned rows, unsigned width, const char* what) {
    const std::size_t n = k.size(), ns = n / kv8_group;
    std::vector<std::int8_t> pck(n, 7), pcv(n, 7), sck(n, 7), scv(n, 7);
    std::vector<std::uint16_t> psk(ns, 9U), psv(ns, 9U), ssk(ns, 9U), ssv(ns, 9U);
    const unsigned fp = private_mirror_kv8(k.data(), v.data(), tags.data(), from, rows, width, pck.data(), pcv.data(),
                                           psk.data(), psv.data());
    const unsigned fs = cs::mirror_i8g16_ref<cs::Tagged>(k.data(), v.data(), tags.data(), from, rows, width, sck.data(),
                                                         scv.data(), ssk.data(), ssv.data());
    expect(fp == fs, what, fp * 16U + fs);
    expect(pck == sck && pcv == scv && psk == ssk && psv == ssv, what, 0U);
    for (std::size_t i = 0; i < n; ++i) {
        const float d = cs::Int8G16::scale_value(ssk[i / kv8_group]);
        const float a = cs::Int8G16::decode(sck[i], d), b = d * float(int(pck[i]));
        expect(f_bits(a) == f_bits(b) || (std::isnan(a) && std::isnan(b)), "i8g16 decode", i);
    }
}

void run(unsigned slots, unsigned width, std::uint64_t from, unsigned rows, bool edges, bool bad_tag, std::mt19937& rng,
         const char* what) {
    std::vector<float> k(std::size_t(slots) * width), v(k.size());
    if (edges) {
        fill_edges(k);
        fill_edges(v);
        std::reverse(v.begin(), v.end());
    } else {
        std::normal_distribution<float> nd(0.0F, 1.0F);
        std::uniform_int_distribution<int> ex(-20, 12);
        for (std::size_t i = 0; i < k.size(); ++i) {
            k[i] = std::ldexp(nd(rng), ex(rng));
            v[i] = std::ldexp(nd(rng), ex(rng) / 2);
        }
    }
    std::vector<std::uint64_t> tags(slots);
    for (unsigned p = 0; p < slots; ++p)
        tags[p] = p;
    if (bad_tag)
        tags[from + rows / 2U] += 1U;
    f16_case(k, v, tags, from, rows, width, what);
    i8_case(k, v, tags, from, rows, width, what);
}

void reader_tests(std::mt19937& rng) {
    // The reader's on-the-fly F16 rounding == encode + decode with the same overflow bit.
    std::uniform_int_distribution<std::uint32_t> any;
    for (unsigned i = 0; i < 2000000U; ++i) {
        const float x = bits_f(any(rng));
        unsigned fp = 0U;
        const float a = private_rounded(x, &fp);
        const auto h = cs::F16::encode(x);
        const unsigned fs = cs::half_special(h) ? cs::flag_overflow : 0U;
        const float b = cs::F16::decode(h);
        expect(fp == fs && (f_bits(a) == f_bits(b) || (std::isnan(a) && std::isnan(b))), "reader f16", i);
    }
    // Two-lane combine of half-group maxima (Count 8 reader) == the group maximum of the writer.
    std::normal_distribution<float> nd(0.0F, 3.0F);
    for (unsigned t = 0; t < 20000U; ++t) {
        float x[16];
        for (float& e : x)
            e = nd(rng);
        float lo = 0.0F, hi = 0.0F, all = 0.0F;
        for (unsigned j = 0; j < 8; ++j)
            lo = cs::Int8G16::max_step(lo, cs::Int8G16::magnitude(x[j]));
        for (unsigned j = 8; j < 16; ++j)
            hi = cs::Int8G16::max_step(hi, cs::Int8G16::magnitude(x[j]));
        for (unsigned j = 0; j < 16; ++j)
            all = cs::Int8G16::max_step(all, cs::Int8G16::magnitude(x[j]));
        expect(f_bits(cs::Int8G16::max_step(lo, hi)) == f_bits(all) && f_bits(cs::Int8G16::max_step(hi, lo)) == f_bits(all),
               "two-lane maximum", t);
    }
}

}  // namespace

int main() {
    std::mt19937 rng(20261122U);
    run(8U, 64U, 0U, 8U, true, false, rng, "edges");
    run(12U, 512U, 3U, 7U, true, true, rng, "edges bad tag");
    for (unsigned t = 0; t < 40U; ++t)
        run(20U, t % 2U ? 512U : 256U, t % 5U, 8U + t % 8U, false, t % 7U == 3U, rng, "random");
    reader_tests(rng);
    if (failures != 0) {
        std::fprintf(stderr, "cache_shadow_tests: %d failures\n", failures);
        return EXIT_FAILURE;
    }
    std::puts("cache_shadow_tests: ok");
    return EXIT_SUCCESS;
}
