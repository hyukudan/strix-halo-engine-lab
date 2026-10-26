#include "strix/ggml_dequant.hpp"

#include "strix/iq4_nl.hpp"

#include <array>
#include <bit>
#include <cstring>
#include <stdexcept>
#include <string>

// The block decoders below keep llama.cpp's expressions verbatim (implicit int -> float of small,
// exactly representable integers), so -Wconversion is silenced for this file only.
#if defined(__GNUC__)
#pragma GCC diagnostic ignored "-Wconversion"
#pragma GCC diagnostic ignored "-Wsign-conversion"
#endif

namespace strix::ggml_dequant {
namespace {

constexpr std::size_t qk_k = 256;

constexpr std::uint64_t iq2s_grid[1024] = {
#include "strix/qwen_iq2_s_grid.inc"
};
constexpr std::uint32_t iq3xxs_grid[256] = {
#include "strix/qwen_iq3_xxs_grid.inc"
};
constexpr std::uint32_t iq3s_grid[512] = {
#include "strix/qwen_iq3_s_grid.inc"
};
constexpr std::array<std::uint8_t, 8> kmask_iq2xs{1, 2, 4, 8, 16, 32, 64, 128};
constexpr std::array<std::int8_t, 16> kvalues_iq4nl{-127, -104, -83, -65, -49, -35, -22, -10, 1, 13, 25, 38, 53, 69, 89, 113};

// ksigns_iq2xs[i] = i with bit 7 set when popcount(i) is odd (llama.cpp table, recomputed from its parity rule).
constexpr std::uint8_t ksigns(unsigned i) noexcept {
    return static_cast<std::uint8_t>(i | ((std::popcount(i) & 1U) << 7U));
}

std::uint16_t u16(const std::uint8_t* p) noexcept { return static_cast<std::uint16_t>(p[0] | (p[1] << 8)); }
std::uint32_t u32(const std::uint8_t* p) noexcept { std::uint32_t v; std::memcpy(&v, p, 4); return v; }

void get_scale_min_k4(int j, const std::uint8_t* q, std::uint8_t* d, std::uint8_t* m) noexcept {
    if (j < 4) {
        *d = q[j] & 63; *m = q[j + 4] & 63;
    } else {
        *d = (q[j + 4] & 0xF) | ((q[j - 4] >> 6) << 4);
        *m = (q[j + 4] >> 4) | ((q[j - 0] >> 6) << 4);
    }
}

// --- one block each; x = block bytes, y = block outputs ---

void q2_0(const std::uint8_t* x, float* y) {  // fp16 d, 16 bytes of 2-bit codes, value (q - 1) * d
    const float d = fp16(u16(x));
    for (int j = 0; j < 64; ++j) {
        const std::uint8_t q = (x[2 + j / 4] >> ((j % 4) * 2)) & 0x03;
        y[j] = (static_cast<int>(q) - 1) * d;
    }
}
void q4_0(const std::uint8_t* x, float* y) {
    const float d = fp16(u16(x)); const std::uint8_t* qs = x + 2;
    for (int j = 0; j < 16; ++j) {
        const int x0 = (qs[j] & 0x0F) - 8, x1 = (qs[j] >> 4) - 8;
        y[j] = x0 * d; y[j + 16] = x1 * d;
    }
}
void q5_0(const std::uint8_t* x, float* y) {
    const float d = fp16(u16(x)); const std::uint32_t qh = u32(x + 2); const std::uint8_t* qs = x + 6;
    for (int j = 0; j < 16; ++j) {
        const std::uint8_t xh_0 = ((qh >> (j + 0)) << 4) & 0x10, xh_1 = ((qh >> (j + 12))) & 0x10;
        const std::int32_t x0 = ((qs[j] & 0x0F) | xh_0) - 16, x1 = ((qs[j] >> 4) | xh_1) - 16;
        y[j] = x0 * d; y[j + 16] = x1 * d;
    }
}
void q5_1(const std::uint8_t* x, float* y) {
    const float d = fp16(u16(x)), m = fp16(u16(x + 2)); const std::uint32_t qh = u32(x + 4); const std::uint8_t* qs = x + 8;
    for (int j = 0; j < 16; ++j) {
        const std::uint8_t xh_0 = ((qh >> (j + 0)) << 4) & 0x10, xh_1 = ((qh >> (j + 12))) & 0x10;
        const int x0 = (qs[j] & 0x0F) | xh_0, x1 = (qs[j] >> 4) | xh_1;
        y[j] = x0 * d + m; y[j + 16] = x1 * d + m;
    }
}
void q8_0(const std::uint8_t* x, float* y) {
    const float d = fp16(u16(x));
    for (int j = 0; j < 32; ++j) y[j] = static_cast<std::int8_t>(x[2 + j]) * d;
}
void iq4_nl(const std::uint8_t* x, float* y) {
    const float d = fp16(u16(x)); const std::uint8_t* qs = x + 2;
    for (int j = 0; j < 16; ++j) { y[j] = d * kvalues_iq4nl[qs[j] & 0xf]; y[j + 16] = d * kvalues_iq4nl[qs[j] >> 4]; }
}
// block_q3_K: hmask[32], qs[64], scales[12], d
void q3_k(const std::uint8_t* x, float* y) {
    const std::uint32_t kmask1 = 0x03030303, kmask2 = 0x0f0f0f0f;
    std::uint32_t aux[4];
    const std::int8_t* scales = reinterpret_cast<const std::int8_t*>(aux);
    const float d_all = fp16(u16(x + 108));
    const std::uint8_t* q = x + 32; const std::uint8_t* hm = x; std::uint8_t m = 1;
    std::memcpy(aux, x + 96, 12);
    const std::uint32_t tmp = aux[2];
    aux[2] = ((aux[0] >> 4) & kmask2) | (((tmp >> 4) & kmask1) << 4);
    aux[3] = ((aux[1] >> 4) & kmask2) | (((tmp >> 6) & kmask1) << 4);
    aux[0] = (aux[0] & kmask2) | (((tmp >> 0) & kmask1) << 4);
    aux[1] = (aux[1] & kmask2) | (((tmp >> 2) & kmask1) << 4);
    int is = 0; float dl;
    for (int n = 0; n < 256; n += 128) {
        int shift = 0;
        for (int j = 0; j < 4; ++j) {
            dl = d_all * (scales[is++] - 32);
            for (int l = 0; l < 16; ++l) *y++ = dl * (static_cast<std::int8_t>((q[l + 0] >> shift) & 3) - ((hm[l + 0] & m) ? 0 : 4));
            dl = d_all * (scales[is++] - 32);
            for (int l = 0; l < 16; ++l) *y++ = dl * (static_cast<std::int8_t>((q[l + 16] >> shift) & 3) - ((hm[l + 16] & m) ? 0 : 4));
            shift += 2; m <<= 1;
        }
        q += 32;
    }
}
// block_q4_K: d, dmin, scales[12], qs[128]
void q4_k(const std::uint8_t* x, float* y) {
    const std::uint8_t* q = x + 16; const std::uint8_t* sc_bytes = x + 4;
    const float d = fp16(u16(x)), min = fp16(u16(x + 2));
    int is = 0; std::uint8_t sc, m;
    for (int j = 0; j < 256; j += 64) {
        get_scale_min_k4(is + 0, sc_bytes, &sc, &m); const float d1 = d * sc; const float m1 = min * m;
        get_scale_min_k4(is + 1, sc_bytes, &sc, &m); const float d2 = d * sc; const float m2 = min * m;
        for (int l = 0; l < 32; ++l) *y++ = d1 * (q[l] & 0xF) - m1;
        for (int l = 0; l < 32; ++l) *y++ = d2 * (q[l] >> 4) - m2;
        q += 32; is += 2;
    }
}
// block_q5_K: d, dmin, scales[12], qh[32], qs[128]
void q5_k(const std::uint8_t* x, float* y) {
    const std::uint8_t* ql = x + 48; const std::uint8_t* qh = x + 16; const std::uint8_t* sc_bytes = x + 4;
    const float d = fp16(u16(x)), min = fp16(u16(x + 2));
    int is = 0; std::uint8_t sc, m; std::uint8_t u1 = 1, u2 = 2;
    for (int j = 0; j < 256; j += 64) {
        get_scale_min_k4(is + 0, sc_bytes, &sc, &m); const float d1 = d * sc; const float m1 = min * m;
        get_scale_min_k4(is + 1, sc_bytes, &sc, &m); const float d2 = d * sc; const float m2 = min * m;
        for (int l = 0; l < 32; ++l) *y++ = d1 * ((ql[l] & 0xF) + (qh[l] & u1 ? 16 : 0)) - m1;
        for (int l = 0; l < 32; ++l) *y++ = d2 * ((ql[l] >> 4) + (qh[l] & u2 ? 16 : 0)) - m2;
        ql += 32; is += 2; u1 <<= 2; u2 <<= 2;
    }
}
// block_q6_K: ql[128], qh[64], scales[16] (int8), d
void q6_k(const std::uint8_t* x, float* y) {
    const float d = fp16(u16(x + 208));
    const std::uint8_t* ql = x; const std::uint8_t* qh = x + 128; const std::int8_t* sc = reinterpret_cast<const std::int8_t*>(x + 192);
    for (int n = 0; n < 256; n += 128) {
        for (int l = 0; l < 32; ++l) {
            const int is = l / 16;
            const std::int8_t q1 = static_cast<std::int8_t>((ql[l + 0] & 0xF) | (((qh[l] >> 0) & 3) << 4)) - 32;
            const std::int8_t q2 = static_cast<std::int8_t>((ql[l + 32] & 0xF) | (((qh[l] >> 2) & 3) << 4)) - 32;
            const std::int8_t q3 = static_cast<std::int8_t>((ql[l + 0] >> 4) | (((qh[l] >> 4) & 3) << 4)) - 32;
            const std::int8_t q4 = static_cast<std::int8_t>((ql[l + 32] >> 4) | (((qh[l] >> 6) & 3) << 4)) - 32;
            y[l + 0] = d * sc[is + 0] * q1;
            y[l + 32] = d * sc[is + 2] * q2;
            y[l + 64] = d * sc[is + 4] * q3;
            y[l + 96] = d * sc[is + 6] * q4;
        }
        y += 128; ql += 64; qh += 32; sc += 8;
    }
}
// block_iq4_xs: d, scales_h (u16), scales_l[4], qs[128]
void iq4_xs(const std::uint8_t* x, float* y) {
    const float d = fp16(u16(x)); const std::uint16_t scales_h = u16(x + 2); const std::uint8_t* scales_l = x + 4;
    const std::uint8_t* qs = x + 8;
    for (int ib = 0; ib < 8; ++ib) {
        const int ls = ((scales_l[ib / 2] >> 4 * (ib % 2)) & 0xf) | (((scales_h >> 2 * ib) & 3) << 4);
        const float dl = d * (ls - 32);
        for (int j = 0; j < 16; ++j) { y[j + 0] = dl * kvalues_iq4nl[qs[j] & 0xf]; y[j + 16] = dl * kvalues_iq4nl[qs[j] >> 4]; }
        y += 32; qs += 16;
    }
}
// block_iq2_s: d, qs[64] (first 32 = grid low bits, last 32 = signs), qh[8], scales[8]
void iq2_s(const std::uint8_t* x, float* y) {
    const float d = fp16(u16(x));
    const std::uint8_t* qs = x + 2; const std::uint8_t* qh = x + 66; const std::uint8_t* scales = x + 74;
    const std::uint8_t* signs = qs + qk_k / 8;
    float db[2];
    for (int ib32 = 0; ib32 < 8; ++ib32) {
        db[0] = d * (0.5f + (scales[ib32] & 0xf)) * 0.25f;
        db[1] = d * (0.5f + (scales[ib32] >> 4)) * 0.25f;
        for (int l = 0; l < 4; ++l) {
            const float dl = db[l / 2];
            const auto* grid = reinterpret_cast<const std::uint8_t*>(iq2s_grid + (qs[l] | (qh[ib32] << (8 - 2 * l) & 0x300)));
            for (int j = 0; j < 8; ++j) y[j] = dl * grid[j] * (signs[l] & kmask_iq2xs[j] ? -1.f : 1.f);
            y += 8;
        }
        qs += 4; signs += 4;
    }
}
// block_iq3_xxs: d, qs[96] (64 grid indices + 32 bytes of scales_and_signs)
void iq3_xxs(const std::uint8_t* x, float* y) {
    const float d = fp16(u16(x));
    const std::uint8_t* qs = x + 2; const std::uint8_t* scales_and_signs = qs + qk_k / 4;
    for (int ib32 = 0; ib32 < 8; ++ib32) {
        const std::uint32_t aux32 = u32(scales_and_signs + 4 * ib32);
        const float db = d * (0.5f + (aux32 >> 28)) * 0.5f;
        for (int l = 0; l < 4; ++l) {
            const std::uint8_t signs = ksigns((aux32 >> 7 * l) & 127);
            const auto* grid1 = reinterpret_cast<const std::uint8_t*>(iq3xxs_grid + qs[2 * l + 0]);
            const auto* grid2 = reinterpret_cast<const std::uint8_t*>(iq3xxs_grid + qs[2 * l + 1]);
            for (int j = 0; j < 4; ++j) {
                y[j + 0] = db * grid1[j] * (signs & kmask_iq2xs[j + 0] ? -1.f : 1.f);
                y[j + 4] = db * grid2[j] * (signs & kmask_iq2xs[j + 4] ? -1.f : 1.f);
            }
            y += 8;
        }
        qs += 8;
    }
}
// block_iq3_s: d, qs[64], qh[8], signs[32], scales[4]
void iq3_s(const std::uint8_t* x, float* y) {
    const float d = fp16(u16(x));
    const std::uint8_t* qs = x + 2; const std::uint8_t* qh = x + 66; const std::uint8_t* signs = x + 74; const std::uint8_t* scales = x + 106;
    for (int ib32 = 0; ib32 < 8; ib32 += 2) {
        const float db1 = d * (1 + 2 * (scales[ib32 / 2] & 0xf));
        const float db2 = d * (1 + 2 * (scales[ib32 / 2] >> 4));
        for (int l = 0; l < 4; ++l) {
            const auto* grid1 = reinterpret_cast<const std::uint8_t*>(iq3s_grid + (qs[2 * l + 0] | ((qh[0] << (8 - 2 * l)) & 256)));
            const auto* grid2 = reinterpret_cast<const std::uint8_t*>(iq3s_grid + (qs[2 * l + 1] | ((qh[0] << (7 - 2 * l)) & 256)));
            for (int j = 0; j < 4; ++j) {
                y[j + 0] = db1 * grid1[j] * (signs[l] & kmask_iq2xs[j + 0] ? -1.f : 1.f);
                y[j + 4] = db1 * grid2[j] * (signs[l] & kmask_iq2xs[j + 4] ? -1.f : 1.f);
            }
            y += 8;
        }
        qs += 8; signs += 4;
        for (int l = 0; l < 4; ++l) {
            const auto* grid1 = reinterpret_cast<const std::uint8_t*>(iq3s_grid + (qs[2 * l + 0] | ((qh[1] << (8 - 2 * l)) & 256)));
            const auto* grid2 = reinterpret_cast<const std::uint8_t*>(iq3s_grid + (qs[2 * l + 1] | ((qh[1] << (7 - 2 * l)) & 256)));
            for (int j = 0; j < 4; ++j) {
                y[j + 0] = db2 * grid1[j] * (signs[l] & kmask_iq2xs[j + 0] ? -1.f : 1.f);
                y[j + 4] = db2 * grid2[j] * (signs[l] & kmask_iq2xs[j + 4] ? -1.f : 1.f);
            }
            y += 8;
        }
        qh += 2; qs += 8; signs += 4;
    }
}

using BlockFn = void (*)(const std::uint8_t*, float*);

struct Entry { Geometry g; BlockFn fn; std::string_view name; };

std::optional<Entry> entry(std::uint32_t t) noexcept {
    switch (static_cast<Type>(t)) {
    case Type::q2_0: return Entry{{64, 18}, q2_0, "Q2_0"};
    case Type::q4_0: return Entry{{32, 18}, q4_0, "Q4_0"};
    case Type::q5_0: return Entry{{32, 22}, q5_0, "Q5_0"};
    case Type::q5_1: return Entry{{32, 24}, q5_1, "Q5_1"};
    case Type::q8_0: return Entry{{32, 34}, q8_0, "Q8_0"};
    case Type::iq4_nl: return Entry{{32, 18}, iq4_nl, "IQ4_NL"};
    case Type::q3_k: return Entry{{256, 110}, q3_k, "Q3_K"};
    case Type::q4_k: return Entry{{256, 144}, q4_k, "Q4_K"};
    case Type::q5_k: return Entry{{256, 176}, q5_k, "Q5_K"};
    case Type::q6_k: return Entry{{256, 210}, q6_k, "Q6_K"};
    case Type::iq4_xs: return Entry{{256, 136}, iq4_xs, "IQ4_XS"};
    case Type::iq2_s: return Entry{{256, 82}, iq2_s, "IQ2_S"};
    case Type::iq3_xxs: return Entry{{256, 98}, iq3_xxs, "IQ3_XXS"};
    case Type::iq3_s: return Entry{{256, 110}, iq3_s, "IQ3_S"};
    case Type::f32: return Entry{{1, 4}, nullptr, "F32"};
    case Type::f16: return Entry{{1, 2}, nullptr, "F16"};
    case Type::bf16: return Entry{{1, 2}, nullptr, "BF16"};
    default: return std::nullopt;
    }
}

}  // namespace

float fp16(std::uint16_t bits) noexcept { return iq4_nl::fp16_bits_to_float(bits); }
float bf16(std::uint16_t bits) noexcept { return std::bit_cast<float>(static_cast<std::uint32_t>(bits) << 16U); }

std::optional<Geometry> geometry(std::uint32_t t) noexcept {
    const auto e = entry(t);
    if (!e) return std::nullopt;
    return e->g;
}
std::string_view name(std::uint32_t t) noexcept { const auto e = entry(t); return e ? e->name : std::string_view{"?"}; }
bool supported(std::uint32_t t) noexcept { return entry(t).has_value(); }

std::size_t row_bytes(std::uint32_t t, std::size_t elements) {
    const auto e = entry(t);
    if (!e) throw std::invalid_argument("ggml_dequant: unsupported ggml type " + std::to_string(t));
    if (elements % e->g.block_elements != 0)
        throw std::invalid_argument("ggml_dequant: row of " + std::to_string(elements) + " is not a multiple of the " +
                                    std::string(e->name) + " block");
    return elements / e->g.block_elements * e->g.block_bytes;
}

void dequantize_row(std::uint32_t t, std::span<const std::byte> row, std::span<float> out) {
    const auto e = entry(t);
    const std::size_t bytes = row_bytes(t, out.size());
    if (row.size() < bytes) throw std::invalid_argument("ggml_dequant: short row");
    const auto* x = reinterpret_cast<const std::uint8_t*>(row.data());
    switch (static_cast<Type>(t)) {
    case Type::f32: std::memcpy(out.data(), x, bytes); return;
    case Type::f16: for (std::size_t i = 0; i < out.size(); ++i) out[i] = fp16(u16(x + 2 * i)); return;
    case Type::bf16: for (std::size_t i = 0; i < out.size(); ++i) out[i] = bf16(u16(x + 2 * i)); return;
    default: break;
    }
    const std::size_t blocks = out.size() / e->g.block_elements;
    for (std::size_t b = 0; b < blocks; ++b) e->fn(x + b * e->g.block_bytes, out.data() + b * e->g.block_elements);
}

}  // namespace strix::ggml_dequant
