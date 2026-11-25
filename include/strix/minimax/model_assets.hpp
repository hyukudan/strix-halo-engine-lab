#pragma once
#include <filesystem>
#include "strix/file_snapshot.hpp"
#include <map>
#include <string>
#include <vector>

namespace strix::minimax::assets {
// Loader resolution is also the authority for session compatibility inventory.
std::map<std::string, std::filesystem::path> pack_weight_map(const std::filesystem::path& dir, LoaderSnapshot* snapshot = nullptr);
std::vector<std::filesystem::path> pack_files(const std::filesystem::path& dir, LoaderSnapshot* snapshot = nullptr);

inline std::vector<std::filesystem::path> draft_files(const std::filesystem::path& dir) {
    std::vector<std::filesystem::path> files;
    for (const auto& entry : std::filesystem::directory_iterator(dir))
        if (entry.path().extension() == ".safetensors" && entry.is_regular_file()) files.push_back(entry.path());
    return files;
}

inline std::filesystem::path discover_draft_dir(const std::filesystem::path& model) {
    const auto dir = model / "dspark";
    std::error_code error;
    if (std::filesystem::is_directory(dir, error) && !draft_files(dir).empty()) return dir;
    return {};
}
} // namespace strix::minimax::assets
