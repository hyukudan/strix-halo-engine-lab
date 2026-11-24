#pragma once

#include <algorithm>
#include <optional>

// One resolver for the Inkling prefill row count of every entry point. Host-only, backend-neutral (HIP and CUDA).
//
// The two regimes differ on purpose. The CLI runs one tall chunk (logits are chunk-invariant; fewer weight
// re-streams). The HTTP server keeps small bounded prompt slices so other lanes keep decoding between slices.
// An explicit user value always wins; the bound below only caps it.
namespace strix::inkling2 {

enum class MiniPrefillEntry { cli, http };

inline constexpr unsigned kMiniCliPrefillRows = 8192U;
inline constexpr unsigned kMiniHttpPrefillRows = 128U;
// Largest value --prefill-rows of the server accepts.
inline constexpr unsigned kMiniHttpPrefillRowsMax = 65535U;
// Chunk the CLI falls back to when the implicit default does not fit the device (the previous CLI default).
inline constexpr unsigned kMiniCliPrefillFallbackRows = 2048U;

// requested: the explicit flag value, if any. bound: the cap of the entry point, or 0 for none. CLI: the prompt token
// count (a chunk never exceeds the prompt). HTTP: the context length (a slice never exceeds the context).
[[nodiscard]] constexpr unsigned mini_prefill_rows(MiniPrefillEntry entry, std::optional<unsigned> requested,
                                                   unsigned bound) {
    const unsigned rows = requested.value_or(entry == MiniPrefillEntry::cli ? kMiniCliPrefillRows
                                                                            : kMiniHttpPrefillRows);
    return bound ? std::min(rows, bound) : rows;
}

} // namespace strix::inkling2
