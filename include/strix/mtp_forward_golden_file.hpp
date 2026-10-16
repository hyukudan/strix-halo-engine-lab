#pragma once
#include "strix/hip_mtp_forward_replay.hpp"
#include <cstddef>
#include <filesystem>
#include <span>
#include <vector>

namespace strix {
// Independent-reference bytes are caller supplied. Neither encoding nor reading
// supplies evidence that the source is independent or that replay has passed.
[[nodiscard]] HipMtpForwardGolden read_mtp_forward_golden_file(
    const std::filesystem::path&,const ModelContentIdentity& expected_full_file_sha256,
    std::uint64_t maximum_file_bytes);
// Portable producer helper; never obtains reference values from the candidate.
// Returns the complete versioned file bytes, not the native replay payload ID.
[[nodiscard]] std::vector<std::byte> encode_mtp_forward_golden_file(
    const HipMtpForwardGolden&,std::uint64_t maximum_file_bytes);
} // namespace strix
