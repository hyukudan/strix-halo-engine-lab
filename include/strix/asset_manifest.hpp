#pragma once

#include <filesystem>
#include <stdexcept>
#include <string>
#include <string_view>
#include <vector>

namespace strix {

class AssetMismatch : public std::runtime_error {
public:
    using std::runtime_error::runtime_error;
    // Only schema-owned public keys are emitted; other keys use a short SHA-256.
    // Closure failures have no failing entry.
    std::string role, key, key_digest;
    std::size_t index = 0;
};

// JSON input is capped at 1 MiB. Returned JSON is canonical, with no newline.
[[nodiscard]] std::string read_asset_json(const std::filesystem::path& path);
[[nodiscard]] std::string create_asset_manifest(std::string_view spec,
                                               const std::filesystem::path& root);
[[nodiscard]] std::string asset_manifest_digest(std::string_view manifest);
// Returns one bounded summary; throws AssetMismatch for failed asset checks.
[[nodiscard]] std::string verify_asset_manifest(std::string_view manifest,
    const std::filesystem::path& root, const std::vector<std::string>& roles = {});

}  // namespace strix
