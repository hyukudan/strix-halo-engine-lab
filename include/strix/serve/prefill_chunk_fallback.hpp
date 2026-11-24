#pragma once

#include <algorithm>
#include <cctype>
#include <cstdint>
#include <exception>
#include <new>
#include <ostream>
#include <string>
#include <string_view>

namespace strix::serve {

// True for a device (or host) allocation failure: std::bad_alloc, or the runtime error text the CUDA and HIP
// check() helpers build from cudaErrorMemoryAllocation / hipErrorOutOfMemory ("<operation>: out of memory").
// A missing file, a bad checkpoint or any other failure is not one.
inline bool is_allocation_failure(const std::exception& error) {
    if (dynamic_cast<const std::bad_alloc*>(&error)) return true;
    std::string text = error.what();
    std::transform(text.begin(), text.end(), text.begin(), [](unsigned char c) { return static_cast<char>(std::tolower(c)); });
    return text.find("out of memory") != std::string_view::npos;
}

// Loads a decoder at `chunk` rows. When the chunk is the family's implicit default (not set by the user) and the load
// fails with an allocation failure (is_allocation_failure: the row-dependent device buffers), the load is retried once at `fallback` rows and one line
// says so. An explicit chunk, or a chunk already at or below the fallback, is never changed: the error propagates.
// Chunk-invariant logits make the fallback a schedule change only.
template <class Load>
auto load_with_chunk_fallback(std::uint32_t chunk, bool explicit_chunk, std::uint32_t fallback, std::ostream& log,
                              Load&& load) {
    if (explicit_chunk || fallback == 0U || chunk <= fallback) return load(chunk);
    try {
        return load(chunk);
    } catch (const std::exception& error) {
        if (!is_allocation_failure(error)) throw;
        log << "strix-prefill-chunk " << chunk << " -> " << fallback << " (device memory)\n";
    }
    return load(fallback);
}

} // namespace strix::serve
