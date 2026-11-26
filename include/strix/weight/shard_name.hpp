#pragma once
// Shard-name rule of the confined pack reader (PackCheckpoint) and the manifest size limit shared by the
// model.safetensors.index.json readers. Model readers that accept external shards by contract do not apply the rule.
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
