#pragma once
// Low-bit resident row storage for dense linears (FAST weight formats of the shared dense rows family).
// One vendor-free storage contract; the device decoders (core/weight_decode.hpp Chunk) and the host encoder below
// read the same layout. Row-major [N, K] matrix, K % 128 == 0, each row a run of 128-element superblocks:
//   q4_g64 (68 B):  bytes [0, 64)  codes, element j in byte j / 2 (low nibble = even j), signed 4-bit [-8, 7]
//                   bytes [64, 68) FP16 scale of elements [0, 64) then of [64, 128)
//   q6_g64 (100 B): bytes [0, 64)  low 4 bits of every code (same addressing as q4_g64)
//                   bytes [64, 96) high 2 bits, element j in byte 64 + j / 4 at bit 2 * (j % 4); signed 6-bit [-32, 31]
//                   bytes [96, 100) FP16 scales as above
// Weight = float(fp16 scale) * float(code): one FP32 product, exact (11-bit by <= 6-bit significands).
// The code planes keep every 16-element load of the rows sweep 8-byte (codes) and 4-byte (high bits) aligned.
// Encoding: symmetric round-to-nearest-even with one FP16 scale per 64 elements; the scale is chosen per group from
// kClipSteps clip ratios of max(positive / qmax, negative / -qmin) by the smallest squared reconstruction error, and
// the codes are computed with the stored (FP16) scale (exact ties-to-even, independent of the rounding mode).
#include "core/iu4_onfly.hpp"

#include <array>
#include <bit>
#include <cfenv>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <limits>
#include <span>
#include <stdexcept>

#if defined(__HIPCC__) || defined(__CUDACC__)
#define STRIX_LOWBIT_HD __host__ __device__
#else
#define STRIX_LOWBIT_HD
#endif

namespace strix::weight {

template <unsigned Bits>
struct LowBitStorage {
    static_assert(Bits == 4U || Bits == 6U);
    static constexpr std::uint32_t elements = 128, group = 64, groups = elements / group;
    static constexpr std::uint32_t low_bytes = elements / 2U;                       // 4-bit plane
    static constexpr std::uint32_t high_bytes = Bits == 6U ? elements / 4U : 0U;    // 2-bit plane
    static constexpr std::uint32_t scale_offset = low_bytes + high_bytes;
    static constexpr std::uint32_t bytes = scale_offset + groups * 2U;
    static constexpr int qmax = (1 << (Bits - 1U)) - 1, qmin = -(1 << (Bits - 1U));
    static constexpr std::uint64_t row_bytes(std::uint64_t width) noexcept { return width / elements * bytes; }
    STRIX_LOWBIT_HD static std::uint16_t scale_bits(const std::uint8_t* block, unsigned group_index) noexcept {
        const auto* p = block + scale_offset + 2U * group_index;
        return static_cast<std::uint16_t>(p[0] | (std::uint16_t(p[1]) << 8U));
    }
    // Signed code of element j (0..127) of a superblock.
    STRIX_LOWBIT_HD static int code(const std::uint8_t* block, unsigned j) noexcept {
        const unsigned low = (block[j / 2U] >> ((j % 2U) * 4U)) & 15U;
        if constexpr (Bits == 4U) {
            return low < 8U ? int(low) : int(low) - 16;
        } else {
            const unsigned high = (block[low_bytes + j / 4U] >> ((j % 4U) * 2U)) & 3U;
            const unsigned v = low | (high << 4U);
            return v < 32U ? int(v) : int(v) - 64;
        }
    }
};
using Q4G64Storage = LowBitStorage<4>;
using Q6G64Storage = LowBitStorage<6>;

namespace lowbit {

inline constexpr int kClipSteps = 21;
inline constexpr float clip_ratio(int i) { return 1.0F - 0.02F * static_cast<float>(i); }

// The one binary16 codec of the weight packers (core/iu4_onfly.hpp): round-to-nearest-even narrowing, exact widening.
using ::strix::core::onfly::f16_bits_to_f32;
using ::strix::core::onfly::f32_to_f16_bits;

// Result of encoding one superblock. A nonzero group whose unclipped scale max(positive / qmax, negative / -qmin)
// does not fit FP16 is refused (scale_overflow), never stored as an all-zero group. Converts to true only for ok.
struct EncodeStatus {
    enum Code : std::uint8_t { ok, nonfinite, scale_overflow } code;
    constexpr operator bool() const noexcept { return code == ok; }
};

// The clip search runs its FP32 error sums in round-to-nearest whatever the caller's mode, so a staged source and the
// digest of its bytes agree on every thread; the caller's mode is restored afterwards.
class NearestRounding {
public:
    NearestRounding() : saved_(std::fegetround()) {
        if (saved_ != FE_TONEAREST) std::fesetround(FE_TONEAREST);
    }
    ~NearestRounding() {
        if (saved_ != FE_TONEAREST) std::fesetround(saved_);
    }
    NearestRounding(const NearestRounding&) = delete;
    NearestRounding& operator=(const NearestRounding&) = delete;

private:
    int saved_;
};

// Code of w for the stored scale s > 0: w / s rounded to nearest, ties to even, clamped to [qmin, qmax]. Exact: the
// candidate n from the double reciprocal inv ~ 1 / s is corrected against exact products (n * s and (n + 1/2) * s need
// at most 7 + 11 significant bits), so neither the reciprocal's rounding nor the rounding mode can move a tie.
namespace detail {
template <class Storage>
inline int quantise(float w, double s, double inv) {
    const double a = std::fabs(static_cast<double>(w));
    constexpr double saturate = static_cast<double>(-Storage::qmin) + 1.0;
    if (a >= saturate * s) return w < 0.0F ? Storage::qmin : Storage::qmax;
    auto n = static_cast<int>(a * inv);
    if (static_cast<double>(n) * s > a) --n;
    else if (static_cast<double>(n + 1) * s <= a) ++n;
    const double mid = (static_cast<double>(n) + 0.5) * s;
    if (a > mid || (a == mid && (n & 1) != 0)) ++n;
    const int q = w < 0.0F ? -n : n;
    return q > Storage::qmax ? Storage::qmax : q < Storage::qmin ? Storage::qmin : q;
}
}  // namespace detail
template <class Storage>
inline int quantise(float w, float s) {
    const auto sd = static_cast<double>(s);
    return detail::quantise<Storage>(w, sd, 1.0 / sd);
}

namespace detail {
// One 64-element group of finite inputs, with round-to-nearest installed by encode_block.
// An all-zero group gets scale 0 and codes 0. The FP32 error sums choose the clip step, so callers must
// build without FP contraction (strix-weight: -ffp-contract=off or /fp:strict).
// Optional importance weights only the calibrated clip search.
template <class Storage>
inline EncodeStatus::Code encode_group(const float* w, int* q, std::uint16_t& scale, const float* importance = nullptr) {
    float pos = 0.0F, neg = 0.0F;
    for (unsigned j = 0; j < Storage::group; ++j) {
        pos = w[j] > pos ? w[j] : pos;
        neg = -w[j] > neg ? -w[j] : neg;
        q[j] = 0;
    }
    scale = 0;
    const float a = pos / float(Storage::qmax), b = neg / float(-Storage::qmin), base = a > b ? a : b;
    if (!(base > 0.0F)) return EncodeStatus::ok;
    // Every clip candidate is <= base, so all of them fit FP16 exactly when base does.
    if ((f32_to_f16_bits(base) & 0x7c00U) == 0x7c00U) return EncodeStatus::scale_overflow;
    float best = -1.0F;
    for (int i = 0; i < kClipSteps; ++i) {
        const float raw = clip_ratio(i) * base;
        const auto bits = f32_to_f16_bits(raw > 0x1p-24F ? raw : 0x1p-24F);
        const float s = f16_bits_to_f32(bits);
        const auto sd = static_cast<double>(s), inv = 1.0 / sd;
        float err = 0.0F;
        if (importance == nullptr) {
            for (unsigned j = 0; j < Storage::group; ++j) {
                const float d = w[j] - s * static_cast<float>(quantise<Storage>(w[j], sd, inv));
                err += d * d;
            }
        } else {
            for (unsigned j = 0; j < Storage::group; ++j) {
                const float d = w[j] - s * static_cast<float>(quantise<Storage>(w[j], sd, inv));
                err += importance[j] * (d * d);
            }
        }
        if (std::isfinite(err) && (best < 0.0F || err < best)) {
            best = err;
            scale = bits;
        }
    }
    if (best < 0.0F) return EncodeStatus::nonfinite;
    const auto sd = static_cast<double>(f16_bits_to_f32(scale)), inv = 1.0 / sd;
    for (unsigned j = 0; j < Storage::group; ++j) q[j] = quantise<Storage>(w[j], sd, inv);
    return EncodeStatus::ok;
}
}  // namespace detail

// One superblock of Storage::elements floats -> Storage::bytes. On a non-finite input or a scale overflow the output
// is left untouched and the status says which.
template <class Storage>
inline void pack_block(const int* q, const std::uint16_t* scales, std::uint8_t* out);

template <class Storage>
inline EncodeStatus encode_block(const float* w, std::uint8_t* out) {
    for (unsigned j = 0; j < Storage::elements; ++j)
        if (!std::isfinite(w[j])) return {EncodeStatus::nonfinite};
    const NearestRounding nearest;
    int q[Storage::elements];
    std::uint16_t scales[Storage::groups];
    for (unsigned g = 0; g < Storage::groups; ++g)
        if (const auto code = detail::encode_group<Storage>(w + g * Storage::group, q + g * Storage::group, scales[g]);
            code != EncodeStatus::ok)
            return {code};
    pack_block<Storage>(q, scales, out);
    return {EncodeStatus::ok};
}

// Codes (Storage::elements, in [qmin, qmax]) and Storage::groups FP16 scale bits -> one superblock.
template <class Storage>
inline void pack_block(const int* q, const std::uint16_t* scales, std::uint8_t* out) {
    std::memset(out, 0, Storage::bytes);
    for (unsigned g = 0; g < Storage::groups; ++g) {
        out[Storage::scale_offset + 2U * g] = static_cast<std::uint8_t>(scales[g] & 0xffU);
        out[Storage::scale_offset + 2U * g + 1U] = static_cast<std::uint8_t>(scales[g] >> 8U);
    }
    for (unsigned j = 0; j < Storage::elements; ++j) {
        const auto v = static_cast<unsigned>(q[j]);
        out[j / 2U] = static_cast<std::uint8_t>(out[j / 2U] | ((v & 15U) << ((j % 2U) * 4U)));
        if constexpr (Storage::high_bytes != 0U)
            out[Storage::low_bytes + j / 4U] =
                static_cast<std::uint8_t>(out[Storage::low_bytes + j / 4U] | (((v >> 4U) & 3U) << ((j % 4U) * 2U)));
    }
}

// Reference decode of element j of a superblock (host tests, gate tooling).
template <class Storage>
inline float decode(const std::uint8_t* block, unsigned j) {
    return f16_bits_to_f32(Storage::scale_bits(block, j / Storage::group)) * static_cast<float>(Storage::code(block, j));
}

}  // namespace lowbit

// Strix pack (SP2 / SP3 / SP4): the same 128-element row superblock with sign + magnitude-index codes, one scale byte
// per 32 elements and a per-tensor table of 2^(Bits-1) integer magnitudes (no zero level: the grid is centred).
//   SP4 (68 B): bytes [0, 64) codes, element j in byte j / 2 (low nibble = even j), code = sign << 3 | index
//   SP3 (52 B): bytes [0, 32) low 2 bits of every code, element j in byte j / 4 at bit 2 * (j % 4)
//               bytes [32, 48) high bit (the sign), element j in byte 32 + j / 8 at bit j % 8
//   SP2 (36 B): bytes [0, 32) codes as the SP3 low plane, code = sign << 1 | index
//   then 4 scale bytes, one per 32 elements. Scale byte b = e << 3 | m: scale = (1 + m / 8) * 2^(e + base), with the
//   tensor's base from the header.
// Weight = (sign ? -1 : 1) * table[index] * scale. Every weight is exact in BF16 and, within the normal FP16 range,
// in FP16 (a 4-bit by 4-bit significand product). Only the unscaled signed table level is always int8.
// Per-tensor header (16 B): version u8 | bits u8 | log2 scale group u8 (5) | flags u8 | base i8 | 3 zero bytes |
// table u8 x 8 (the first 2^(Bits-1) entries strictly increasing, the rest zero).
// Flags: linear (table = 1, 3, 5, ...: the odd-integer grid), not_foldable (some table x scale product of the tensor is
// not a normal FP16 number); fp16_scale64 is reserved and refused.
namespace sp {

inline constexpr std::uint8_t kVersion = 1;
inline constexpr std::uint8_t kFlagLinear = 1U << 0, kFlagFp16Scale64 = 1U << 1, kFlagNotFoldable = 1U << 2;
inline constexpr unsigned kHeaderBytes = 16;
// Lowest / highest base accepted: every scale (1 + m / 8) * 2^(e + base) is then a normal FP32 number.
inline constexpr int kMinBase = -120, kMaxBase = 90;

template <unsigned Bits>
struct Storage {
    static_assert(Bits == 2U || Bits == 3U || Bits == 4U);
    static constexpr unsigned bits = Bits;
    static constexpr std::uint32_t elements = 128, group = 32, groups = elements / group;
    static constexpr unsigned levels = 1U << (Bits - 1U);
    static constexpr std::uint32_t code_bytes = elements * Bits / 8U;
    static constexpr std::uint32_t scale_offset = code_bytes;
    static constexpr std::uint32_t bytes = code_bytes + groups;
    static constexpr std::uint64_t row_bytes(std::uint64_t width) noexcept { return width / elements * bytes; }
    // Unsigned code of element j (0..127): sign << (Bits - 1) | magnitude index.
    STRIX_LOWBIT_HD static unsigned code(const std::uint8_t* block, unsigned j) noexcept {
        if constexpr (Bits == 4U) {
            return (block[j / 2U] >> ((j % 2U) * 4U)) & 15U;
        } else if constexpr (Bits == 2U) {
            return (block[j / 4U] >> ((j % 4U) * 2U)) & 3U;
        } else {
            const unsigned low = (block[j / 4U] >> ((j % 4U) * 2U)) & 3U;
            const unsigned high = (block[32U + j / 8U] >> (j % 8U)) & 1U;
            return low | (high << 2U);
        }
    }
    STRIX_LOWBIT_HD static std::uint8_t scale_byte(const std::uint8_t* block, unsigned group_index) noexcept {
        return block[scale_offset + group_index];
    }
};
using SP2Storage = Storage<2>;
using SP3Storage = Storage<3>;
using SP4Storage = Storage<4>;

// A magnitude the table may hold: an integer in [1, 31] with at most 4 significant bits.
constexpr bool representable_magnitude(unsigned t) noexcept {
    if (t < 1U || t > 31U) return false;
    while ((t & 1U) == 0U) t >>= 1U;
    return t < 16U;
}

// Default tables (fixed, not fitted): see the v1 format page.
template <unsigned Bits>
constexpr std::array<std::uint8_t, 8> default_table() noexcept {
    if constexpr (Bits == 4U) return {1, 3, 5, 8, 10, 14, 18, 24};
    else if constexpr (Bits == 3U) return {3, 10, 18, 30, 0, 0, 0, 0};
    else return {1, 3, 0, 0, 0, 0, 0, 0};
}
template <unsigned Bits>
constexpr std::array<std::uint8_t, 8> linear_table() noexcept {
    std::array<std::uint8_t, 8> t{};
    for (unsigned i = 0; i < (1U << (Bits - 1U)); ++i) t[i] = static_cast<std::uint8_t>(2U * i + 1U);
    return t;
}

struct Header {
    std::uint8_t version = kVersion, bits = 4, log2_group = 5, flags = 0;
    std::int8_t base = 0;
    std::array<std::uint8_t, 8> table{};

    [[nodiscard]] unsigned levels() const noexcept { return 1U << (bits - 1U); }
    void write(std::uint8_t* out) const noexcept {
        out[0] = version;
        out[1] = bits;
        out[2] = log2_group;
        out[3] = flags;
        out[4] = static_cast<std::uint8_t>(base);
        out[5] = out[6] = out[7] = 0;
        for (unsigned i = 0; i < 8U; ++i) out[8U + i] = table[i];
    }
    // Empty string = valid; otherwise the first violated rule.
    [[nodiscard]] const char* invalid() const noexcept {
        if (version != kVersion) return "unknown version";
        if (bits != 2U && bits != 3U && bits != 4U) return "bits must be 2, 3 or 4";
        if (log2_group != 5U) return "scale group must be 32 (log2 5)";
        if ((flags & ~(kFlagLinear | kFlagNotFoldable)) != 0U) return "unknown or unsupported flag";
        if (base < kMinBase || base > kMaxBase) return "base out of range";
        for (unsigned i = 0; i < 8U; ++i) {
            if (i >= levels()) {
                if (table[i] != 0U) return "table entry beyond the levels is not zero";
                continue;
            }
            if (!representable_magnitude(table[i])) return "table magnitude not an integer in [1, 31] with <= 4 significant bits";
            if (i > 0U && table[i] <= table[i - 1U]) return "table not strictly increasing";
            if ((flags & kFlagLinear) != 0U && table[i] != 2U * i + 1U) return "linear flag with a non-linear table";
        }
        return "";
    }
    // Parse the leading header of a tensor in the declared storage width. Leave out unchanged on failure.
    static const char* read(std::span<const std::uint8_t> in, unsigned storage_bits, Header& out) noexcept {
        if (in.size() < kHeaderBytes) return "short Strix pack header";
        if (in[5] != 0U || in[6] != 0U || in[7] != 0U) return "reserved bytes not zero";
        Header h;
        h.version = in[0];
        h.bits = in[1];
        h.log2_group = in[2];
        h.flags = in[3];
        h.base = static_cast<std::int8_t>(in[4]);
        for (unsigned i = 0; i < 8U; ++i) h.table[i] = in[8U + i];
        if (const char* why = h.invalid(); *why != '\0') return why;
        if (h.bits != storage_bits) return "Strix pack header/storage bit-width mismatch";
        out = h;
        return "";
    }
};

// Value of scale byte b for a tensor base: exact (4 significant bits, normal FP32 for every accepted base).
STRIX_LOWBIT_HD inline std::uint32_t scale_bits(std::uint8_t b, int base) noexcept {
    return (static_cast<std::uint32_t>(base + 127) * 8U + b) << 20U;
}
inline float scale_value(std::uint8_t b, int base) noexcept {
    return std::bit_cast<float>(scale_bits(b, base));
}

// Reference decode of element j of a superblock (host tests, emulation tooling).
template <class S>
inline float decode(const std::uint8_t* block, unsigned j, const Header& h) {
    const unsigned c = S::code(block, j);
    const float magnitude = static_cast<float>(h.table[c & (S::levels - 1U)]) *
                            scale_value(S::scale_byte(block, j / S::group), h.base);
    return (c >> (S::bits - 1U)) != 0U ? -magnitude : magnitude;
}

// Codes (S::elements unsigned codes, sign << (Bits - 1) | index) and S::groups scale bytes -> one superblock.
template <class S>
inline void pack_block(const unsigned* codes, const std::uint8_t* scales, std::uint8_t* out) {
    std::memset(out, 0, S::bytes);
    for (unsigned j = 0; j < S::elements; ++j) {
        const unsigned v = codes[j];
        if constexpr (S::bits == 4U) {
            out[j / 2U] = static_cast<std::uint8_t>(out[j / 2U] | ((v & 15U) << ((j % 2U) * 4U)));
        } else {
            out[j / 4U] = static_cast<std::uint8_t>(out[j / 4U] | ((v & 3U) << ((j % 4U) * 2U)));
            if constexpr (S::bits == 3U)
                out[32U + j / 8U] = static_cast<std::uint8_t>(out[32U + j / 8U] | (((v >> 2U) & 1U) << (j % 8U)));
        }
    }
    for (unsigned g = 0; g < S::groups; ++g) out[S::scale_offset + g] = scales[g];
}

// Tensor base: the largest unclipped group scale amax / table[top] (smax > 0) lands in the top two binades of the
// 32 the scale byte spans, so a group scale may round up one binade and still be encodable. Clamped to the accepted range.
inline int choose_base(double smax) noexcept {
    if (!(smax > 0.0) || !std::isfinite(smax)) return 0;
    const int base = std::ilogb(smax) - 30;
    return base < kMinBase ? kMinBase : base > kMaxBase ? kMaxBase : base;
}

// Nearest scale byte to s > 0 for the base (ties to the lower byte); 0 below the lowest scale, 255 above the highest.
inline std::uint8_t nearest_scale_byte(double s, int base) noexcept {
    if (!(s > 0.0)) return 0;
    if (!std::isfinite(s)) return 255;
    int e = std::ilogb(s) - base;
    if (e < 0) return 0;
    if (e > 31) return 255;
    // s = f * 2^(e + base), f in [1, 2): m = round((f - 1) * 8), m == 8 carries into the next binade.
    const double f = std::ldexp(s, -(e + base));
    const double m8 = (f - 1.0) * 8.0;
    auto m = static_cast<int>(m8);
    if (m8 - m > 0.5) ++m;
    if (m == 8) {
        m = 0;
        if (++e > 31) return 255;
    }
    return static_cast<std::uint8_t>((e << 3) | m);
}

// Magnitude index of |w| for the scale s (a scale_value): nearest table level, ties to the lower level. Exact: the
// midpoints (t_i + t_i+1) / 2 * s are exact doubles (<= 6 + 4 significant bits) compared with the exact |w|.
template <unsigned Levels>
inline unsigned nearest_level(double a, double s, const std::uint8_t* table) noexcept {
    unsigned i = 0;
    while (i + 1U < Levels && a > 0.5 * static_cast<double>(table[i] + table[i + 1U]) * s) ++i;
    return i;
}

template <class S>
inline unsigned quantise(float w, double s, const std::uint8_t* table) noexcept {
    const unsigned idx = nearest_level<S::levels>(std::fabs(static_cast<double>(w)), s, table);
    return (w < 0.0F ? (1U << (S::bits - 1U)) : 0U) | idx;
}

// One group of finite inputs: choose the scale byte with the smallest (optionally weighted) squared error
// over the inclusive nearest-byte endpoints of [amax / top / 2, amax / top * 9 / 8] (ties: lower byte).
// The caller sets round-to-nearest and compiles without contraction for the FP64 error sum.
// An all-zero group gets byte 0 and code 0 for every element (the tensor's smallest magnitude, 2^base * table[0]).
template <class S>
inline void encode_group(const float* w, const Header& h, unsigned* codes, std::uint8_t& byte,
                         const float* importance = nullptr) {
    float amax = 0.0F;
    for (unsigned j = 0; j < S::group; ++j) amax = std::fabs(w[j]) > amax ? std::fabs(w[j]) : amax;
    byte = 0;
    if (amax > 0.0F) {
        const double top = static_cast<double>(h.table[S::levels - 1U]);
        const double s0 = static_cast<double>(amax) / top;
        const std::uint8_t lo = nearest_scale_byte(0.5 * s0, h.base), hi = nearest_scale_byte(1.125 * s0, h.base);
        double best = -1.0;
        for (unsigned b = lo; b <= hi; ++b) {
            const double s = static_cast<double>(scale_value(static_cast<std::uint8_t>(b), h.base));
            double err = 0.0;
            for (unsigned j = 0; j < S::group; ++j) {
                const double a = std::fabs(static_cast<double>(w[j]));
                const double d = a - static_cast<double>(h.table[nearest_level<S::levels>(a, s, h.table.data())]) * s;
                err += (importance ? static_cast<double>(importance[j]) : 1.0) * d * d;
            }
            if (best < 0.0 || err < best) {
                best = err;
                byte = static_cast<std::uint8_t>(b);
            }
        }
    }
    const auto s = static_cast<double>(scale_value(byte, h.base));
    for (unsigned j = 0; j < S::group; ++j) codes[j] = amax > 0.0F ? quantise<S>(w[j], s, h.table.data()) : 0U;
}

// Header for a tensor (n x k floats, k % 128 == 0) and a table: base from the largest group amax, not_foldable when
// some used table x scale product is not a normal FP16 number. The caller sets the linear flag for the linear table.
template <class S>
inline Header header_for(float amax, const std::array<std::uint8_t, 8>& table) {
    Header h;
    h.bits = static_cast<std::uint8_t>(S::bits);
    h.table = table;
    h.base = static_cast<std::int8_t>(choose_base(static_cast<double>(amax) / table[S::levels - 1U]));
    if (table == linear_table<S::bits>()) h.flags = static_cast<std::uint8_t>(h.flags | kFlagLinear);
    return h;
}
inline float tensor_amax(const float* w, std::size_t count) noexcept {
    float amax = 0.0F;
    for (std::size_t e = 0; e < count; ++e) amax = std::fabs(w[e]) > amax ? std::fabs(w[e]) : amax;
    return amax;
}
inline void require_coding_shape(const float* w, std::size_t n, std::size_t k) {
    constexpr auto max_elements = static_cast<std::size_t>(std::numeric_limits<std::ptrdiff_t>::max());
    if (!w || !n || !k || k % 128U != 0 || n > max_elements / k)
        throw std::invalid_argument("Strix pack coder requires a nonempty [N,K] with K divisible by 128");
}
inline bool finite_inputs(const float* w, std::size_t n, std::size_t k, const float* importance = nullptr) {
    for (std::size_t i = 0; i < n * k; ++i)
        if (!std::isfinite(w[i])) return false;
    for (std::size_t i = 0; importance && i < k; ++i)
        if (!std::isfinite(importance[i]) || importance[i] < 0.0F) return false;
    return true;
}
template <class S>
inline Header make_header(const float* w, std::size_t n, std::size_t k, const std::array<std::uint8_t, 8>& table) {
    require_coding_shape(w, n, k);
    if (!finite_inputs(w, n, k)) throw std::invalid_argument("Strix pack header requires finite weights");
    const lowbit::NearestRounding nearest;
    const auto h = header_for<S>(tensor_amax(w, n * k), table);
    if (*h.invalid() != '\0') throw std::invalid_argument("Strix pack header requires a valid table");
    return h;
}

// Sets / clears not_foldable from the scale bytes actually used (rows of S::row_bytes(k) bytes).
template <class S>
inline void mark_foldable(Header& h, const std::uint8_t* bytes, std::size_t n, std::size_t k) {
    unsigned lo = 256, hi = 0;
    const std::size_t blocks = n * (k / S::elements);
    for (std::size_t b = 0; b < blocks; ++b)
        for (unsigned g = 0; g < S::groups; ++g) {
            const unsigned v = S::scale_byte(bytes + b * S::bytes, g);
            lo = v < lo ? v : lo;
            hi = v > hi ? v : hi;
        }
    h.flags = static_cast<std::uint8_t>(h.flags & ~kFlagNotFoldable);
    if (blocks == 0U) return;
    const double smallest = static_cast<double>(h.table[0]) * static_cast<double>(scale_value(static_cast<std::uint8_t>(lo), h.base));
    const double largest = static_cast<double>(h.table[h.levels() - 1U]) *
                           static_cast<double>(scale_value(static_cast<std::uint8_t>(hi), h.base));
    if (smallest < 0x1p-14 || largest > 65504.0) h.flags = static_cast<std::uint8_t>(h.flags | kFlagNotFoldable);
}

enum class ScaleSelection { clip_search, nearest_max };

// Round-to-nearest coding of a whole tensor (no calibration): out = n * S::row_bytes(k) bytes; optional n x k decode.
template <class S>
inline lowbit::EncodeStatus encode_rows(const float* w, std::size_t n, std::size_t k, const Header& h, std::uint8_t* out,
                        float* dequant = nullptr, const float* importance = nullptr,
                        ScaleSelection selection = ScaleSelection::clip_search) {
    require_coding_shape(w, n, k);
    if (!out || h.bits != S::bits || *h.invalid() != '\0')
        throw std::invalid_argument("Strix pack coder requires a matching header and an output buffer");
    if (!finite_inputs(w, n, k, importance)) return {lowbit::EncodeStatus::nonfinite};
    const lowbit::NearestRounding nearest;
    const std::size_t blocks = k / S::elements;
    for (std::size_t r = 0; r < n; ++r)
        for (std::size_t b = 0; b < blocks; ++b) {
            unsigned codes[S::elements];
            std::uint8_t scales[S::groups];
            const float* src = w + r * k + b * S::elements;
            for (unsigned g = 0; g < S::groups; ++g) {
                const auto* group = src + g * S::group;
                auto* indices = codes + g * S::group;
                if (selection == ScaleSelection::clip_search) {
                    encode_group<S>(group, h, indices, scales[g],
                                    importance ? importance + b * S::elements + g * S::group : nullptr);
                } else {
                    const auto amax = tensor_amax(group, S::group);
                    scales[g] = nearest_scale_byte(static_cast<double>(amax) / h.table[S::levels - 1U], h.base);
                    const auto scale = static_cast<double>(scale_value(scales[g], h.base));
                    for (unsigned j = 0; j < S::group; ++j)
                        indices[j] = amax > 0.0F ? quantise<S>(group[j], scale, h.table.data()) : 0U;
                }
            }
            std::uint8_t* block = out + (r * blocks + b) * S::bytes;
            pack_block<S>(codes, scales, block);
            if (dequant)
                for (unsigned j = 0; j < S::elements; ++j) dequant[r * k + b * S::elements + j] = decode<S>(block, j, h);
        }
    return {lowbit::EncodeStatus::ok};
}

// Lossless re-pack when the header and original group scale bytes are retained.
// Several scales can represent the same weights, so recalibrating scale bytes
// from decoded floats alone cannot recover every valid payload.
template <class S>
[[nodiscard]] bool repack_block(const float* decoded, const std::uint8_t* scales, const Header& h, std::uint8_t* out) {
    if (h.bits != S::bits || *h.invalid() != '\0') return false;
    unsigned codes[S::elements];
    std::uint8_t saved[S::groups];
    for (unsigned g = 0; g < S::groups; ++g) {
        saved[g] = scales[g];
        const double s = scale_value(saved[g], h.base);
        for (unsigned j = g * S::group; j < (g + 1U) * S::group; ++j) {
            const auto w = decoded[j];
            if (!std::isfinite(w)) return false;
            const auto c = quantise<S>(w, s, h.table.data());
            if (std::fabs(static_cast<double>(w)) != static_cast<double>(h.table[c % S::levels]) * s) return false;
            codes[j] = c;
        }
    }
    pack_block<S>(codes, saved, out);
    return true;
}

// Weighted squared reconstruction error of a tensor under a header (RTN codes), summed in double.
template <class S>
inline double table_error(const float* w, std::size_t rows, std::size_t k, std::size_t row_stride, const Header& h,
                          const float* importance) {
    double err = 0.0;
    const std::size_t blocks = k / S::elements;
    for (std::size_t r = 0; r < rows; r += row_stride)
        for (std::size_t b = 0; b < blocks * S::groups; ++b) {
            unsigned codes[S::group];
            std::uint8_t byte = 0;
            const float* src = w + r * k + b * S::group;
            const float* imp = importance ? importance + b * S::group : nullptr;
            encode_group<S>(src, h, codes, byte, imp);
            const auto s = static_cast<double>(scale_value(byte, h.base));
            for (unsigned j = 0; j < S::group; ++j) {
                const double d = static_cast<double>(src[j]) -
                                 ((codes[j] >> (S::bits - 1U)) != 0U ? -1.0 : 1.0) *
                                     static_cast<double>(h.table[codes[j] & (S::levels - 1U)]) * s;
                err += (imp ? static_cast<double>(imp[j]) : 1.0) * d * d;
            }
        }
    return err;
}

// Per-tensor table fit (Lloyd on the magnitudes, snapped to representable magnitudes): from the default table,
// alternate (a) RTN codes and scales of the sampled rows (every row_stride-th) and (b) the least-squares magnitude of
// every index, sum(imp * |w| * s) / sum(imp * s^2); each continuous table is snapped for every top magnitude in
// {16, 18, ..., 30} (entries rounded to the nearest representable magnitude, collisions moved up) and the snapped table
// with the smallest sampled error is kept. Returns the best table seen, the default included (never worse on the sample).
template <class S>
inline std::array<std::uint8_t, 8> fit_table(const float* w, std::size_t n, std::size_t k, std::size_t row_stride,
                                             const float* importance = nullptr, unsigned iterations = 4) {
    require_coding_shape(w, n, k);
    if (!finite_inputs(w, n, k, importance)) throw std::invalid_argument("Strix pack table fit requires finite inputs");
    const lowbit::NearestRounding nearest;
    constexpr unsigned L = S::levels;
    if (row_stride == 0U) row_stride = 1U;
    auto best_table = default_table<S::bits>();
    const float amax = tensor_amax(w, n * k);
    Header h = header_for<S>(amax, best_table);
    double best = table_error<S>(w, n, k, row_stride, h, importance);
    const auto snap = [](const double* c, unsigned top) {
        std::array<std::uint8_t, 8> t{};
        unsigned prev = 0;
        for (unsigned i = 0; i < L; ++i) {
            const double v = c[i] / c[L - 1U] * static_cast<double>(top);
            unsigned bestm = 0;
            double bestd = 1e300;
            for (unsigned m = prev + 1U; m <= 31U; ++m) {
                if (!representable_magnitude(m)) continue;
                const double d = std::fabs(static_cast<double>(m) - v);
                if (d < bestd) {
                    bestd = d;
                    bestm = m;
                }
            }
            if (bestm == 0U) return std::array<std::uint8_t, 8>{};  // no room left: invalid candidate
            t[i] = static_cast<std::uint8_t>(bestm);
            prev = bestm;
        }
        return t;
    };
    auto current = best_table;
    for (unsigned it = 0; it < iterations; ++it) {
        Header cur = header_for<S>(amax, current);
        double num[8] = {}, den[8] = {};
        const std::size_t blocks = k / S::elements;
        for (std::size_t r = 0; r < n; r += row_stride)
            for (std::size_t b = 0; b < blocks * S::groups; ++b) {
                unsigned codes[S::group];
                std::uint8_t byte = 0;
                const float* src = w + r * k + b * S::group;
                const float* imp = importance ? importance + b * S::group : nullptr;
                encode_group<S>(src, cur, codes, byte, imp);
                const auto s = static_cast<double>(scale_value(byte, cur.base));
                for (unsigned j = 0; j < S::group; ++j) {
                    const unsigned i = codes[j] & (L - 1U);
                    const double q = imp ? static_cast<double>(imp[j]) : 1.0;
                    num[i] += q * std::fabs(static_cast<double>(src[j])) * s;
                    den[i] += q * s * s;
                }
            }
        double c[8];
        for (unsigned i = 0; i < L; ++i) c[i] = den[i] > 0.0 ? num[i] / den[i] : static_cast<double>(current[i]);
        bool ordered = c[0] > 0.0;
        for (unsigned i = 1; i < L; ++i) ordered = ordered && c[i] > c[i - 1U];
        if (!ordered) break;
        std::array<std::uint8_t, 8> iter_best{};
        double iter_err = -1.0;
        for (unsigned top = 16; top <= 30U; top += 2U) {
            const auto t = snap(c, top);
            if (t[0] == 0U) continue;
            const Header ht = header_for<S>(amax, t);
            const double e = table_error<S>(w, n, k, row_stride, ht, importance);
            if (iter_err < 0.0 || e < iter_err) {
                iter_err = e;
                iter_best = t;
            }
        }
        if (iter_err < 0.0) break;
        if (iter_err < best) {
            best = iter_err;
            best_table = iter_best;
        }
        if (iter_best == current) break;
        current = iter_best;
    }
    return best_table;
}

}  // namespace sp
}  // namespace strix::weight

#undef STRIX_LOWBIT_HD
