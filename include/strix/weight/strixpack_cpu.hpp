#pragma once

#include "strix/weight/host_lowbit.hpp"

namespace strix::weight::sp {

enum class CpuIsa { automatic, scalar, avx2, avx512 };
[[nodiscard]] bool cpu_supports(CpuIsa isa) noexcept;
[[nodiscard]] CpuIsa selected_cpu_isa() noexcept;

// Validated v1 operands only: a matching header, columns divisible by 128 and
// tightly packed [rows, columns] weights. The caller owns buffers and geometry.
// An unavailable requested ISA falls back to scalar.
using DecodeRows = void (*)(const Header&, const std::uint8_t*, std::size_t, std::size_t, float*);
using ProjectRows = void (*)(const Header&, const std::uint8_t*, std::size_t, std::size_t, const float*, float*);

[[nodiscard]] DecodeRows cpu_decode(unsigned bits, CpuIsa isa = CpuIsa::automatic) noexcept;
[[nodiscard]] ProjectRows cpu_project(unsigned bits, CpuIsa isa = CpuIsa::automatic) noexcept;

}  // namespace strix::weight::sp
