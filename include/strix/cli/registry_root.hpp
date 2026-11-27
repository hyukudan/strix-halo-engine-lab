#pragma once
// The model registry root a server reads: one rule for every serving entry point (strix-serve, the family servers).
// Header-only (no launcher library needed by the family service libraries that call it).
#include <filesystem>

namespace strix::cli {

// `preferred` (the build tree on the build host, or the working directory) when it holds models/registry.toml; else
// the shipped data directory (`shipped`, cli::data_directory of the executable) when one is given; else `preferred`.
[[nodiscard]] inline std::filesystem::path registry_root(const std::filesystem::path& preferred,
                                                         const std::filesystem::path& shipped) {
    if (std::filesystem::exists(preferred / "models/registry.toml") || shipped.empty()) return preferred;
    return shipped;
}

}  // namespace strix::cli
