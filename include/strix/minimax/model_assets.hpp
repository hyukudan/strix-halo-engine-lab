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
std::vector<std::filesystem::path> draft_files(const std::filesystem::path& dir);
} // namespace strix::minimax::assets
