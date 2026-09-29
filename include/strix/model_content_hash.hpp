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

[[nodiscard]] ModelContentIdentity build_model_content_identity(
    std::span<const ModelContentFile> files);

[[nodiscard]] std::string model_content_identity_hex(
    const ModelContentIdentity& identity);

}  // namespace strix
