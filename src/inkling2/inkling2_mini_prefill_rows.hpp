#pragma once

// Inkling prefill row numbers, as data for the shared selection and retry policy (strix/serve/prefill_chunk_fallback.hpp:
// resolve_prefill_rows, admit_prefill_chunk). Host-only, backend-neutral. The policy (explicit value wins, bound,
// pressure retry) is not here.
//
// The two regimes differ on purpose. The CLI runs one tall chunk (fewer weight re-streams). The HTTP server keeps
// small bounded prompt slices so other lanes keep decoding between slices.
namespace strix::inkling2 {

inline constexpr unsigned kMiniCliPrefillRows = 8192U;
inline constexpr unsigned kMiniHttpPrefillRows = 128U;
// Largest value --prefill-rows of the server accepts.
inline constexpr unsigned kMiniHttpPrefillRowsMax = 65535U;
// Chunk a backend that qualified the tall default falls back to when the device cannot hold its workspaces (the
// previous CLI default). A backend without that qualification supplies no fallback.
inline constexpr unsigned kMiniCliPrefillPressureRows = 2048U;

} // namespace strix::inkling2
