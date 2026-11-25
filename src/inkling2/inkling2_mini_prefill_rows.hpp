#pragma once

#include "strix/engine_profile.hpp"

// Inkling prefill row numbers, as data for the shared selection and retry policy (strix/serve/prefill_chunk_fallback.hpp:
// resolve_prefill_rows, admit_prefill_chunk). Host-only, backend-neutral. The policy (explicit value wins, bound,
// pressure retry) is not here; the CLI and the HTTP server read the numbers only through mini_prefill_rows below.
//
// The two regimes differ on purpose. The CLI runs one tall chunk (fewer weight re-streams). The HTTP server keeps
// small bounded prompt slices so other lanes keep decoding between slices.
namespace strix::inkling2 {

// Largest value --prefill-rows of the server accepts.
inline constexpr unsigned kMiniHttpPrefillRowsMax = 65535U;

enum class MiniPrefillBackend : unsigned char { hip, cuda };

struct MiniPrefillRows {
    unsigned cli;           // implicit CLI / qualification chunk
    unsigned cli_pressure;  // retry chunk after a typed device allocation failure; 0: no retry
    unsigned http;          // implicit HTTP prompt slice
};

inline constexpr unsigned kMiniCliPrefillRows = 8192U;   // the tall chunk
inline constexpr unsigned kMiniHttpPrefillRows = 128U;
inline constexpr unsigned kMiniCliPrefillIncumbentRows = 2048U;  // the previous CUDA default, also the pressure retry

// The one selection. Declared per backend and profile:
//  - hip: the tall chunk for every profile, no pressure retry (an allocation failure reports as before).
//  - cuda: the tall chunk only for the EXACT profile (`standard`), the one whose chunk-invariant logits are measured
//    there, with a retry at the incumbent chunk when the device cannot hold the larger workspaces. Every other CUDA
//    profile keeps the incumbent chunk and no retry until its own schedule is qualified.
constexpr MiniPrefillRows mini_prefill_rows(MiniPrefillBackend backend, profile::Name tier) noexcept {
    if (backend == MiniPrefillBackend::hip) return {kMiniCliPrefillRows, 0U, kMiniHttpPrefillRows};
    if (tier == profile::Name::standard)
        return {kMiniCliPrefillRows, kMiniCliPrefillIncumbentRows, kMiniHttpPrefillRows};
    return {kMiniCliPrefillIncumbentRows, 0U, kMiniHttpPrefillRows};
}

} // namespace strix::inkling2
