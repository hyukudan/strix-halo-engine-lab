#pragma once
// Shard names read from a model.safetensors.index.json weight_map are data from the model directory: every reader
// joins them to that directory only after this check, so a name can only designate a file inside the directory.
#include <cstdint>
#include <string_view>

namespace strix::weight {

// A plain file name: non-empty, not "." or "..", no directory separator, drive colon or NUL.
[[nodiscard]] inline bool shard_basename_ok(std::string_view name) noexcept {
    return !name.empty() && name != "." && name != ".." &&
           name.find_first_of(std::string_view("/\\:\0", 4)) == std::string_view::npos;
}

// A plain file name ending in .safetensors with a non-empty stem.
[[nodiscard]] inline bool safetensors_shard_name_ok(std::string_view name) noexcept {
    constexpr std::string_view ext = ".safetensors";
    return shard_basename_ok(name) && name.size() > ext.size() && name.substr(name.size() - ext.size()) == ext;
}

// The largest model.safetensors.index.json a reader accepts.
inline constexpr std::uint64_t kSafetensorsManifestMaxBytes = std::uint64_t{64} << 20;

}  // namespace strix::weight
