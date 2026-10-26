#pragma once
// CRC-32C of the MiniMax prefill exchange module (host only; no HIP headers).
#include <cstddef>
#include <cstdint>

namespace strix::minimax::pfx {
// The vector form (ISA-L folding) when the CPU has it, else strix::dist::crc32c: identical values either way.
std::uint32_t crc32c(const std::byte* data, std::size_t bytes) noexcept;
}  // namespace strix::minimax::pfx
