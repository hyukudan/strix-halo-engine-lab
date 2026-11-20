#pragma once
// Row pitch of an A8 activation operand (int8 codes per value, one FP32 delta and one int32 sum per 32 values) as the
// dense A8 GEMM cores read it (HIP core::f16_dense_tall::q8a8, View::x_stride; deltas pitch = x_stride / 32).
// Backend-neutral layout policy: values and bits do not depend on it, only the addresses of row r.
//
// Why: a CTA of the tall core reads the same 64-byte k slice of 128 rows per stage. When the row pitch is a multiple of
// 8 KiB, those 128 slices (and the 2-4 KiB delta rows) fall on the same cache sets / memory channels and the stage runs
// 3-5x slower (no-model probe, gfx1151: K_b form 5.35 -> 18.4 TF, V_b 7.0 -> 24.4 TF, 16384 -> 3072 at codes pitch
// 2^14 20.6 -> 33.3 TF with a 128-byte pad; 3072-wide rows already run at 33.9 TF). Producers that own the A8 buffer
// allocate rows of a8_row_pitch(width) bytes and pass it as the operand pitch.
#include <cstddef>
#include <cstdint>

namespace strix {

inline constexpr std::uint32_t kA8PitchAliasBytes = 4096U;  // row pitches that are multiples of this alias (T2: 4096 pad 1.30x)
inline constexpr std::uint32_t kA8PitchPadBytes = 128U;     // keeps pitch % 32 == 0 (deltas) and 128-byte alignment

// width: values per row (a multiple of 32). Returns the codes row pitch in bytes (>= width, a multiple of 32).
inline constexpr std::uint32_t a8_row_pitch(std::uint32_t width) {
    return width != 0U && width % kA8PitchAliasBytes == 0U ? width + kA8PitchPadBytes : width;
}
// Bytes of R rows of codes / number of deltas (= sums) of R rows at the pitch above.
inline constexpr std::size_t a8_codes_bytes(std::uint32_t width, std::uint32_t rows) {
    return std::size_t{a8_row_pitch(width)} * rows;
}
inline constexpr std::size_t a8_group_count(std::uint32_t width, std::uint32_t rows) {
    return std::size_t{a8_row_pitch(width) / 32U} * rows;
}

}  // namespace strix
