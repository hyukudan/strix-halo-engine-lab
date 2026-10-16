#pragma once

#include "strix/model_runtime_binding.hpp"

#include <filesystem>
#include <span>
#include <string>
#include <string_view>
#include <vector>

namespace strix {

struct ModelContentFile {
    std::string logical_name;
    std::filesystem::path path;
};

[[nodiscard]] ModelContentIdentity sha256_file_content(
    const std::filesystem::path& path);

[[nodiscard]] ModelContentIdentity sha256_bytes(
    std::span<const std::byte> bytes);

[[nodiscard]] ModelContentIdentity build_model_content_identity(
    std::span<const ModelContentFile> files);

// One stable full-file pass yields both the raw file digest and its existing
// single-file STRIX-MODEL-CONTENT-V1 identity. Neither digest aliases the other.
struct SingleModelContentIdentity {
    ModelContentIdentity file_sha256,content_identity;
    std::uint64_t file_bytes=0;
};
[[nodiscard]] SingleModelContentIdentity hash_single_model_content_file(const ModelContentFile&);
[[nodiscard]] std::string model_content_identity_hex(
    const ModelContentIdentity& identity);

}  // namespace strix
