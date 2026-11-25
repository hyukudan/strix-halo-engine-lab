#pragma once

#include "strix/engine_profile.hpp"

// Inkling 2 Mini implicit CLI / qualification prefill chunk, as data per backend and profile. Host-only. The shared
// serial-generation code applies it (an explicit --prefill-chunk always wins; the chunk is capped by the prompt). The
// HTTP server keeps its own bounded prompt slice (MiniServeOptions::prefill_rows) and does not read this.
namespace strix::inkling2 {

enum class MiniPrefillBackend : unsigned char { hip, cuda };

inline constexpr unsigned kMiniCliPrefillRows = 8192U;           // the tall chunk
inline constexpr unsigned kMiniCliPrefillIncumbentRows = 2048U;  // the previous CUDA chunk

// - hip: the tall chunk for every profile (unchanged).
// - cuda: the tall chunk for the EXACT profile (`standard`), whose complete generations are measured equal to the
//   2048-row schedule there; every other CUDA profile keeps 2048 rows until its own schedule is measured.
// No profile has a pressure retry: a device that cannot hold the tall workspaces reports the typed allocation failure,
// and --prefill-chunk 2048 selects the previous schedule explicitly.
constexpr unsigned mini_cli_prefill_rows(MiniPrefillBackend backend, profile::Name tier) noexcept {
    if (backend == MiniPrefillBackend::hip || tier == profile::Name::standard) return kMiniCliPrefillRows;
    return kMiniCliPrefillIncumbentRows;
}

} // namespace strix::inkling2
