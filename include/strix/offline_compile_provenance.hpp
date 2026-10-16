#pragma once
#include <cstdint>
#include <filesystem>
#include <string>
#include <string_view>
namespace strix {
struct OfflineCompileProvenanceLimits {
    std::uint64_t max_manifest_bytes = 16U*1024U*1024U;
    std::uint64_t max_single_file_bytes = 256U*1024U*1024U;
    std::uint64_t max_referenced_bytes = 16ULL*1024U*1024U*1024U;
    std::uint32_t max_files = 8192, max_units = 4096, max_targets = 256;
};
struct OfflineCompileProvenance {
    // Digest of the versioned evaluated-argv/declared-input representation,
    // including heterogeneous arguments and dependency/object fingerprints.
    // NOT transitive runtime/driver-resource closure or build attestation.
    std::string compiler_flags_sha256, payload_sha256;
    std::uint64_t payload_bytes = 0, referenced_bytes = 0;
    std::uint32_t files_checked = 0, units = 0, targets = 0;
};
// Optional setup-only input. Absence is handled by the caller as unknown.
// An explicitly supplied malformed/stale/incomplete manifest is rejected.
// This verifies current files + binary/context binding, NOT compiler execution,
// build attestation, numerical equivalence or kernel qualification.
[[nodiscard]] OfflineCompileProvenance load_offline_compile_provenance(
    const std::filesystem::path&, std::string_view actual_executable_sha256,
    std::string_view embedded_declared_build_context_hex,
    OfflineCompileProvenanceLimits limits = {});
} // namespace strix
