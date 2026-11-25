#pragma once

#include "strix/engine_profile.hpp"

// Inkling 2 Mini implicit CLI / qualification prefill chunk, as data per backend and profile. Host-only. The shared
// serial-generation code applies it (an explicit --prefill-chunk always wins; the chunk is capped by the prompt). The
// HTTP server reads it through the served schedule (inkling2_mini_schedule.hpp: the served route lever, and the
// held-back route of a chunk-invariant backend); an explicit --prefill-rows always wins there.
namespace strix::inkling2 {

enum class MiniPrefillBackend : unsigned char { hip, cuda };

inline constexpr unsigned kMiniCliPrefillRows = 8192U;           // the tall chunk
inline constexpr unsigned kMiniCliPrefillIncumbentRows = 2048U;  // the previous CUDA chunk

// - hip: the tall chunk for every profile (unchanged).
// - cuda: the tall chunk for every profile. The EXACT profile's complete generations are measured equal to the 2048-row
//   schedule there (RTX q15); the CUDA FAST / TURBO columns resolve the same prefill arithmetic (every CUDA dense, MoE
//   and attention prefill form is the EXACT one: no lossy prefill lever is ready on CUDA), so the 2048 rows they kept
//   only made FAST slower than EXACT on the same kernels (served 8K: 8,136 vs 9,642 tok/s, q27).
constexpr unsigned mini_cli_prefill_rows(MiniPrefillBackend, profile::Name) noexcept { return kMiniCliPrefillRows; }

// Pressure fallback of the implicit chunk (serve::SerialGenerationFamily::prefill_chunk_fallback, applied by the shared
// load, prefill_chunk_fallback.hpp): when the row-scaled device workspaces of the implicit chunk are refused with the
// typed device allocation failure, the load is retried once at these rows; zero = no retry.
// - cuda (every profile): the incumbent 2048 rows, so the tall default never shrinks the set of prompts that load (the
//   CUDA prefill outputs do not depend on the chunk: the fallback changes no output).
// - hip: zero (HIP keeps its single attempt).
// An explicit --prefill-chunk is never retried.
constexpr unsigned mini_cli_prefill_fallback_rows(MiniPrefillBackend backend, profile::Name) noexcept {
    return backend == MiniPrefillBackend::cuda ? kMiniCliPrefillIncumbentRows : 0U;
}

} // namespace strix::inkling2
