#pragma once

#include "strix/kernel_registry.hpp"

#include <cstddef>
#include <cstdint>
#include <span>
#include <vector>

namespace strix::kernels {

struct OfflineTuningCacheLimits {
    std::uint64_t max_bytes = 16ULL * 1024 * 1024;
    std::uint32_t max_records = 4096;
    std::uint32_t max_string_bytes = 4096;
    std::uint64_t max_total_string_bytes = 8ULL * 1024 * 1024;
};

// Version 1: eight-byte magic, LE u32 version/count, then LE u32-sized
// records. Strings are arbitrary exact bytes with LE u32 lengths (no Unicode
// normalization). Scalar fields and enum tags have fixed widths; booleans are
// single bytes 0/1 and timing is its exact IEEE binary64 encoding.
// Decode validates the COMPLETE input and all limits before allocating records
// or strings. Unknown tags, malformed frames, trailing bytes and nonfinite or
// negative costs reject the entire cache. Duplicate entries remain duplicates;
// select_tuned remains the sole authority for provenance and qualification.
// These functions perform no I/O and do not create or attest measurements.
// Borrowed input bytes/records must remain alive and unchanged until return.
[[nodiscard]] std::vector<std::byte> encode_offline_tuning_cache(
    std::span<const OfflineTuningRecord>, OfflineTuningCacheLimits = {});
[[nodiscard]] std::vector<OfflineTuningRecord> decode_offline_tuning_cache(
    std::span<const std::byte>, OfflineTuningCacheLimits = {});

} // namespace strix::kernels
