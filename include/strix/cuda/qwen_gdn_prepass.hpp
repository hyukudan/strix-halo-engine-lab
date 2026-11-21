// Conv prepass split of the CUDA Qwen GDN prefill (src/cuda/models/qwen/qwen_gdn.cu).
//
// The conv runs in place over chunks of kChunk rows, so before any row is overwritten a prepass saves each chunk's three
// preceding raw rows (its boundary taps) and writes the new conv history (the raw rows count-4 .. count-1). The prepass
// grid is (channel blocks, chunks): block y saves chunk y's taps, so no thread walks the chunks serially. Only chunk 0's
// taps reach rows < 0 (the old history), and a new history slot reads the old history only when count < 4, so the block of
// chunk kHistoryChunk alone reads the old history and writes the new one, reads first. Every other block reads workspace
// rows only. Pure copies: each boundary word and history word is the one the one-block-per-channel prepass wrote.
// CUDA-free so the host tests check the rule.
#pragma once

#include <cstdint>

#if defined(__CUDACC__)
#define STRIX_GDN_PREPASS_HD __host__ __device__
#else
#define STRIX_GDN_PREPASS_HD
#endif

namespace strix::cuda::qwen_gdn::prepass {

inline constexpr std::uint32_t kChunk = 32;     // rows per conv block (qwen_gdn.hpp kConvChunk)
inline constexpr std::uint32_t kTaps = 3;       // boundary taps saved per chunk
inline constexpr std::uint32_t kHistory = 4;    // history slots per channel (qwen_gdn.hpp kConvKernel)
inline constexpr std::uint32_t kHistoryChunk = 0;

STRIX_GDN_PREPASS_HD constexpr std::uint32_t chunks(std::uint32_t count) { return (count + kChunk - 1U) / kChunk; }

// Raw row saved as tap `tap` (0..2) of chunk `chunk`: rows chunk * kChunk - 3 .. chunk * kChunk - 1. A row < 0 is old
// history slot row + kHistory.
STRIX_GDN_PREPASS_HD constexpr std::int64_t tap_row(std::uint32_t chunk, std::uint32_t tap) {
    return static_cast<std::int64_t>(chunk) * kChunk + static_cast<std::int64_t>(tap) - static_cast<std::int64_t>(kTaps);
}

// Raw row that becomes history slot `slot` (0..3): rows count - 4 .. count - 1 (< 0: old history slot row + kHistory).
STRIX_GDN_PREPASS_HD constexpr std::int64_t history_row(std::uint32_t count, std::uint32_t slot) {
    return static_cast<std::int64_t>(count) - static_cast<std::int64_t>(kHistory) + static_cast<std::int64_t>(slot);
}

}  // namespace strix::cuda::qwen_gdn::prepass

#undef STRIX_GDN_PREPASS_HD
