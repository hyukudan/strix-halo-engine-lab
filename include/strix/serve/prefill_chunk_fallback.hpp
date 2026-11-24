#pragma once

#include <algorithm>
#include <cstdint>
#include <exception>
#include <optional>
#include <ostream>
#include <utility>

#include "strix/device_allocation_error.hpp"

namespace strix::serve {

// Prefill rows of one entry point. `requested` is the user's explicit value and always wins; otherwise `default_rows`
// (data of the entry point's regime: a tall CLI chunk, or a small bounded HTTP scheduling slice). `bound` caps the
// result (the prompt length for the CLI, the context for HTTP), or 0 for none.
[[nodiscard]] constexpr std::uint32_t resolve_prefill_rows(std::optional<std::uint32_t> requested,
                                                           std::uint32_t default_rows, std::uint32_t bound) {
    const auto rows = requested.value_or(default_rows);
    return bound != 0U ? std::min(rows, bound) : rows;
}

// Admission of a CLI or qualification prefill chunk. `pressure_fallback` is regime data of the backend that qualified
// it (zero: no pressure retry, the resolved chunk is the only attempt). An explicit chunk never falls back.
struct PrefillChunkAdmission {
    std::uint32_t chunk = 0;
    std::uint32_t fallback = 0;
    bool explicit_rows = false;
};

[[nodiscard]] constexpr PrefillChunkAdmission admit_prefill_chunk(std::optional<std::uint32_t> requested,
                                                                  std::uint32_t default_rows,
                                                                  std::uint32_t pressure_fallback,
                                                                  std::uint32_t prompt_rows) {
    return {resolve_prefill_rows(requested, default_rows, prompt_rows),
            requested ? 0U : resolve_prefill_rows(std::nullopt, pressure_fallback, prompt_rows), requested.has_value()};
}

// Loads a decoder at `chunk` rows. When the chunk is the family's implicit default (not set by the user) and the load
// fails with a typed device allocation failure (strix::DeviceAllocationError, thrown by the runtime status check for
// cudaErrorMemoryAllocation / hipErrorOutOfMemory: the row-dependent device buffers), the load is retried once at
// `fallback` rows and one line says so. Nothing else is retried: not a message that merely contains "out of memory",
// and not std::bad_alloc, a host allocation failure that fewer device rows do not address.
// An explicit chunk, or a chunk already at or below the fallback, is never changed: the error propagates.
template <class Load>
auto load_with_chunk_fallback(std::uint32_t chunk, bool explicit_chunk, std::uint32_t fallback, std::ostream& log,
                              Load&& load) {
    if (explicit_chunk || fallback == 0U || chunk <= fallback) return load(chunk);
    try {
        return load(chunk);
    } catch (const DeviceAllocationError&) {
        log << "strix-prefill-chunk " << chunk << " -> " << fallback << " (device memory)\n";
    }
    return load(fallback);
}

// The one entry for a decoder load that sizes row-dependent device buffers: resolves the chunk (admit_prefill_chunk)
// and loads it with the pressure retry. `load` takes the chunk rows.
template <class Load>
auto load_with_prefill_admission(std::optional<std::uint32_t> requested, std::uint32_t default_rows,
                                 std::uint32_t pressure_fallback, std::uint32_t prompt_rows, std::ostream& log,
                                 Load&& load) {
    const auto admission = admit_prefill_chunk(requested, default_rows, pressure_fallback, prompt_rows);
    return load_with_chunk_fallback(admission.chunk, admission.explicit_rows, admission.fallback, log,
                                    std::forward<Load>(load));
}

} // namespace strix::serve
