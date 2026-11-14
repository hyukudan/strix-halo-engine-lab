#pragma once
// Packed grouped W4A8 storage contract: low nibble first, optional 16-level
// codebook, FP8 E4M3FN/F32 relative scales, F32 channel scales. Decode to the
// signed INT8 grid BEFORE the existing convrot GEMM. No new activation policy.
#include "h3/gemm/ref.hpp"
#include <map>
#include <stdexcept>

namespace strix::h3::gemm {
enum class Operand : std::uint32_t { int8_convrot, w4a8 };

H3_HD inline float e4m3fn(std::uint8_t u) {
    const unsigned e = (u >> 3U) & 15U, m = u & 7U;
    if (e == 15U && m == 7U) return std::bit_cast<float>(0x7fc00000U);
    const float v = e ? std::bit_cast<float>((e + 120U) << 23U) * (1.0F + float(m) / 8.0F)
                      : float(m) / 512.0F;
    return u & 128U ? -v : v;
}
// RNE independent of the host floating-point rounding mode; device uses the
// same helper. Clamp before conversion also bounds all integer casts.
H3_HD inline std::int8_t w4a8_code(std::uint8_t packed, unsigned parity, float scale, const float* codebook) {
    const unsigned code = (packed >> (4U * parity)) & 15U;
    float v = (codebook ? codebook[code] : float(int(code) - 8)) * scale;
    v = v < -127.0F ? -127.0F : (v > 127.0F ? 127.0F : v);
    const int base = int(v); const float frac = v - float(base);
    const int sign = frac < 0 ? -1 : 1;
    const float mag = frac < 0 ? -frac : frac;
    return static_cast<std::int8_t>(base + ((mag > 0.5F || (mag == 0.5F && (base % 2))) ? sign : 0));
}
struct PackedW4A8 {
    std::uint32_t group_size = 16;
    std::vector<std::uint8_t> scales_fp8;
    std::vector<float> scales_f32, codebook;
    void validate(std::uint32_t n, std::uint32_t k, std::size_t bytes) const {
        if (!n || !k || k % 256 || group_size < 4 ||
            (16 % group_size && group_size % 16) || k % group_size || bytes != std::size_t(n)*k/2 ||
            (!scales_fp8.empty() && !scales_f32.empty()) ||
            scales_fp8.size()+scales_f32.size() != std::size_t(n)*(k/group_size) ||
            (!codebook.empty() && codebook.size()!=16))
            throw std::invalid_argument("h3 w4a8: packed weight/scale/codebook geometry");
        for (auto u:scales_fp8) { const float s=e4m3fn(u); if (!std::isfinite(s) || s<0) throw std::invalid_argument("h3 w4a8: group scale"); }
        for (auto s:scales_f32) if (!std::isfinite(s) || s<0) throw std::invalid_argument("h3 w4a8: group scale");
        for (auto c:codebook) if (!std::isfinite(c)) throw std::invalid_argument("h3 w4a8: codebook");
    }
    std::int8_t at(const std::int8_t* packed, std::size_t offset) const {
        const auto g=offset/group_size;
        const float s=scales_fp8.empty()?scales_f32[g]:e4m3fn(scales_fp8[g]);
        return w4a8_code(static_cast<std::uint8_t>(packed[offset/2]),unsigned(offset%2),s,codebook.empty()?nullptr:codebook.data());
    }
};

// B-fetch policy preparation: at most 256 distinct scales, with a 256-entry
// packed-byte -> two signed INT8 codes table per scale. No expanded matrix.
// F32 scale sets above this bound retain the general arithmetic fetch policy.
struct PreparedW4A8 {
    std::vector<std::uint8_t> scale_ids;
    std::vector<std::uint16_t> pairs;
    unsigned group_shift = 0;
    std::size_t bytes() const { return scale_ids.size() + pairs.size() * sizeof(std::uint16_t); }
};
inline PreparedW4A8 prepare_w4a8(const PackedW4A8& p) {
    PreparedW4A8 out;
    // Non-power-of-two groups remain supported by the general policy.
    if (!p.group_size || (p.group_size & (p.group_size - 1U))) return out;
    for (auto g = p.group_size; g > 1U; g >>= 1U) ++out.group_shift;
    std::map<float, unsigned> ids;
    std::vector<float> scales;
    const auto insert = [&](float s) {
        if (ids.count(s)) return true;
        if (scales.size() == 256U) return false;
        ids.emplace(s, unsigned(scales.size())); scales.push_back(s); return true;
    };
    // Scan FP8's finite domain rather than every matrix group to build the dictionary.
    if (!p.scales_fp8.empty()) {
        for (unsigned u = 0; u < 256U; ++u) {
            const float s = e4m3fn(std::uint8_t(u));
            if (std::isfinite(s) && s >= 0) (void)insert(s);
        }
    } else {
        for (float s : p.scales_f32) if (!insert(s)) return {};
    }
    out.pairs.resize(scales.size() * 256U);
    const float* cb = p.codebook.empty() ? nullptr : p.codebook.data();
    for (unsigned id = 0; id < scales.size(); ++id)
        for (unsigned u = 0; u < 256U; ++u) {
            const auto lo = std::uint8_t(w4a8_code(std::uint8_t(u), 0, scales[id], cb));
            const auto hi = std::uint8_t(w4a8_code(std::uint8_t(u), 1, scales[id], cb));
            out.pairs[id * 256U + u] = std::uint16_t(unsigned(lo) | (unsigned(hi) << 8U));
        }
    out.scale_ids.reserve(p.scales_fp8.size() + p.scales_f32.size());
    if (!p.scales_fp8.empty()) {
        std::uint8_t map[256]{};
        for (unsigned u = 0; u < 256U; ++u) {
            const float s = e4m3fn(std::uint8_t(u));
            if (std::isfinite(s) && s >= 0) map[u] = std::uint8_t(ids.at(s));
        }
        for (auto u : p.scales_fp8) out.scale_ids.push_back(map[u]);
    } else {
        for (float s : p.scales_f32) out.scale_ids.push_back(std::uint8_t(ids.at(s)));
    }
    return out;
}
} // namespace strix::h3::gemm
