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

#include <cfenv>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstring>

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
        if (best < 0.0F || err < best) {
            best = err;
            scale = bits;
        }
    }
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
}  // namespace strix::weight

#undef STRIX_LOWBIT_HD
