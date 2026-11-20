#pragma once
// Exact bfloat16 wire for tensor-parallel joins (host / device neutral contract; HIP body src/xport/bf16_wire_hip.hpp).
//
// A TP join (a gather of the two ranks' shares: heads, output rows, expert slots) moves FP32 words. When the producer of a
// share already rounds every value to bfloat16 (a BF16 Linear output, a BF16 GLU / gate epilogue), each FP32 word has 16
// zero low bits: the high half IS the value. The wire then carries the high halves only (2 bytes per value instead of
// 4) and the consumer widens them back by a 16-bit shift: the FP32 words it reads are bitwise the words the FP32 join
// would have delivered. Exact by construction, not a precision choice.
//
// Fail closed: the packer checks every word; a word whose low half is not zero (a share that was NOT rounded to bfloat16,
// i.e. a binding error) raises the writer's flag with kInexactFlag and the pass refuses its result. NaN payloads are
// treated the same way (only the high half travels).
//
// Bound by: Hy4 TP decode step joins and Hy4 verify rows under TP (src/hy4/hip). Partial-sum exchanges (DS41 / MiniMax /
// GLM TP2 half sums, xport/exchange_format.hpp) carry unrounded FP32 partials and cannot use it.
#include <cstddef>
#include <cstdint>
#include <cstring>

namespace strix::pair::bf16_wire {

inline constexpr std::uint32_t kInexactFlag = 2U;  // OR-ed into the writer's flag (bit 0 stays the non-finite bit)

// The wire half of one FP32 word; `exact` false when the low half is not zero.
[[nodiscard]] inline std::uint16_t encode(float x, bool& exact) noexcept {
    std::uint32_t b = 0;
    std::memcpy(&b, &x, 4);
    exact = (b & 0xFFFFU) == 0U;
    return static_cast<std::uint16_t>(b >> 16U);
}
[[nodiscard]] inline float decode(std::uint16_t h) noexcept {
    const std::uint32_t b = std::uint32_t{h} << 16U;
    float x = 0.0F;
    std::memcpy(&x, &b, 4);
    return x;
}
[[nodiscard]] constexpr std::size_t wire_bytes(std::size_t values) noexcept { return 2U * values; }

// Host reference of the packer: n words -> n halves; returns false if any word was not exact (the device packer raises
// kInexactFlag for the same input; the halves written are the same either way).
inline bool pack_ref(const float* src, std::size_t n, std::uint16_t* dst) noexcept {
    bool all = true;
    for (std::size_t i = 0; i < n; ++i) {
        bool e = true;
        dst[i] = encode(src[i], e);
        all = all && e;
    }
    return all;
}
// Host reference of the placer: the swapped wire buffer holds two slots of `slot` halves (slot q = rank q's share);
// slot q's rows [R][count] start at element `at` with row pitch `pitch`; they go to columns [q count, (q + 1) count) of
// the FP32 rows full[R][cols] (the one-node layout of a split by halves).
inline void place_ref(const std::uint16_t* wire, std::size_t slot, std::size_t at, std::size_t pitch, std::size_t count, std::uint32_t R,
                      float* full, std::size_t cols) noexcept {
    for (std::size_t q = 0; q < 2U; ++q)
        for (std::uint32_t r = 0; r < R; ++r)
            for (std::size_t j = 0; j < count; ++j)
                full[std::size_t{r} * cols + q * count + j] = decode(wire[q * slot + at + std::size_t{r} * pitch + j]);
}

}  // namespace strix::pair::bf16_wire
