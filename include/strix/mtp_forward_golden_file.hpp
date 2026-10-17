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
// Encodes, writes a fresh file (refuses to overwrite) and returns its full-file
// SHA256, i.e. the --mtp-golden-sha256 pin of the written golden.
[[nodiscard]] ModelContentIdentity write_mtp_forward_golden_file(
    const std::filesystem::path&,const HipMtpForwardGolden&,std::uint64_t maximum_file_bytes);
// Explicit source marker of a native self-generated (non-independent) golden.
[[nodiscard]] ModelContentIdentity mtp_self_generated_golden_source();
inline constexpr std::uint64_t mtp_self_generated_golden_reference=0x314e4547464c4553ULL; // "SELFGEN1"
} // namespace strix
