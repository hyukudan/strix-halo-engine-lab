#pragma once

#include "strix/engine_profile.hpp"

// Inkling 2 Mini implicit CLI / qualification prefill chunk, as data per backend and profile. Host-only. The shared
// serial-generation code applies it (an explicit --prefill-chunk always wins; the chunk is capped by the prompt). The
// HTTP server reads it through the served schedule (inkling2_mini_schedule.hpp: the served route lever, and the
// held-back route); an explicit --prefill-rows always wins there.
namespace strix::inkling2 {

enum class MiniPrefillBackend : unsigned char { hip, cuda };

inline constexpr unsigned kMiniCliPrefillRows = 8192U;           // the tall chunk
inline constexpr unsigned kMiniCliPrefillIncumbentRows = 2048U;  // the previous CUDA chunk

// - hip: the tall chunk for every profile (unchanged).
// - cuda: the tall chunk for the EXACT profile (`standard`), whose complete generations are measured equal to the
//   2048-row schedule there; every other CUDA profile keeps 2048 rows until its own schedule is measured.
constexpr unsigned mini_cli_prefill_rows(MiniPrefillBackend backend, profile::Name tier) noexcept {
    if (backend == MiniPrefillBackend::hip || tier == profile::Name::standard) return kMiniCliPrefillRows;
    return kMiniCliPrefillIncumbentRows;
}

// Pressure fallback of the implicit chunk (serve::SerialGenerationFamily::prefill_chunk_fallback, applied by the shared
// load, prefill_chunk_fallback.hpp): when the row-scaled device workspaces of the implicit chunk are refused with the
// typed device allocation failure, the load is retried once at these rows; zero = no retry.
// - cuda EXACT: the incumbent 2048 rows, so the tall default never shrinks the set of prompts that load (EXACT outputs
//   do not depend on the chunk: the fallback changes no output).
// - every other row: zero (the CUDA profiles already run 2048 rows; HIP keeps its single attempt).
// An explicit --prefill-chunk is never retried.
constexpr unsigned mini_cli_prefill_fallback_rows(MiniPrefillBackend backend, profile::Name tier) noexcept {
    if (backend == MiniPrefillBackend::cuda && tier == profile::Name::standard) return kMiniCliPrefillIncumbentRows;
    return 0U;
}

} // namespace strix::inkling2
