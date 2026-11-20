#pragma once
// Wave form of the head-family single-row dense kernel (src/hip_lm_head_launcher.hip, lm_head_kernel) for F32 / F16 / BF16
// weights: one wave32 computes one output and keeps the 256 leaves of the block form (leaf t sums the features t + 256 k in
// k order with the same `sum += activation * weight` expression). Lane l holds the kLocal contiguous leaves
// leaf(l, j) = kLocal * l + j, so one chunk of 256 features is one contiguous kLocal-element load per lane (16 bytes for
// F16 / BF16) instead of one 2-byte load per thread. The block form's LDS tree `s[v] += s[v + d]` (v < d, d = 128 .. 1)
// becomes the cross-lane stages d >= kLocal (lane distance d / kLocal, the low lanes add) and then the local stages
// d < kLocal (element j += element j + d). Leaves, chains and tree pairs are the block form's: the bits are equal.
// The same scheme as virtual_q8_lm_head_kernel (Q8_0) of the same file, here with a per-kind 8-element loader.
// Host-visible so the leaf / tree map is tested on the CPU (tests/dense_rows_wave_tests.cpp).

#include <cstdint>

#if defined(__HIPCC__) || defined(__CUDACC__)
#define STRIX_DRW_HD __host__ __device__
#else
#define STRIX_DRW_HD
#endif

namespace strix::dense_rows_wave {

inline constexpr std::uint32_t kLeaves = 256;              // == lm_head_kernel's block size (kThreads)
inline constexpr std::uint32_t kLanes = 32;                // one wave32 per output
inline constexpr std::uint32_t kLocal = kLeaves / kLanes;  // 8 contiguous leaves per lane
inline constexpr std::uint32_t kWaves = 8;                 // outputs per 256-thread block
static_assert(kLocal * kLanes == kLeaves && (kLocal & (kLocal - 1U)) == 0U);

// Block-form leaf held by (lane, local).
STRIX_DRW_HD constexpr std::uint32_t leaf(std::uint32_t lane, std::uint32_t local) { return kLocal * lane + local; }
// Tree distance d of the block form is a cross-lane stage when d >= kLocal (lane distance d / kLocal).
STRIX_DRW_HD constexpr bool cross_lane(std::uint32_t d) { return d >= kLocal; }

// Preconditions of the contiguous loads: every lane's kLocal features are all in range or all out of range
// (width % kLocal == 0), and every 16-byte load is aligned (base addresses, row width and activation row stride).
constexpr bool admissible(std::uint32_t width, std::uint32_t x_stride, std::uint32_t weight_bytes, std::uintptr_t x,
                          std::uintptr_t weights, std::uint32_t activation_bytes) {
    if (width == 0U || width % kLocal != 0U || x_stride % kLocal != 0U) return false;
    if (weight_bytes != 2U && weight_bytes != 4U) return false;
    if (activation_bytes != 2U && activation_bytes != 4U) return false;
    return x % 16U == 0U && weights % 16U == 0U;
}

}  // namespace strix::dense_rows_wave

#undef STRIX_DRW_HD
