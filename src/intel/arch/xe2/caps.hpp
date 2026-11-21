#pragma once
// Geometry of the discrete Xe2 joint-matrix dense policy. Each device must
// advertise the exact tile and subgroup before the planner can use them.
#include <cstddef>
#include <cstdint>

namespace strix::intel::arch::xe2 {

inline constexpr std::uint32_t kSubgroup = 16;  // native sub-group width (SIMD16)
// joint_matrix FP16 x FP16 -> FP32 on Xe2: A M x K, B K x N, C M x N, one DPAS per tile.
inline constexpr std::uint32_t kMatM = 8, kMatN = 16, kMatK = 16;

// FP16 B tiles interleave two input channels for each output column.
[[nodiscard]] constexpr std::size_t packed_b_offset(std::size_t input, std::size_t output,
                                                        std::size_t outputs) noexcept {
    return (input / 2U) * (2U * outputs) + 2U * output + input % 2U;
}

} // namespace strix::intel::arch::xe2
