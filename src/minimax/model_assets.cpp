#include "strix/minimax/model_assets.hpp"
#include "strix/serve/json.hpp"
#include "strix/minimax/asset_input.hpp"
#include <set>
#include <stdexcept>

namespace strix::minimax::assets {
std::map<std::string, std::filesystem::path> pack_weight_map(const std::filesystem::path& dir, LoaderSnapshot* snapshot) {
    const auto index = dir / "model.safetensors.index.json";
    Input in(index, snapshot);
    if (!in) throw std::runtime_error("cannot open " + index.string());
    const std::string text{std::istreambuf_iterator<char>(in), std::istreambuf_iterator<char>()};
    const auto json = serve::parse_json(text);
    const auto* weights = json.find("weight_map");
    if (!weights || !weights->is_object()) throw std::runtime_error("invalid weight_map: " + index.string());
    std::map<std::string, std::filesystem::path> out;
    for (const auto& [name, value] : weights->as_object())
        out.emplace(name, dir / value.as_string());
    return out;
}
std::vector<std::filesystem::path> pack_files(const std::filesystem::path& dir, LoaderSnapshot* snapshot) {
    std::vector<std::filesystem::path> out;
    std::set<std::filesystem::path> seen;
    for (const auto& [name, path] : pack_weight_map(dir, snapshot)) {
        (void)name;
        if (seen.insert(path).second) out.push_back(path);
    }
    return out;
}
std::vector<std::filesystem::path> draft_files(const std::filesystem::path& dir) {
    std::vector<std::filesystem::path> out;
    for (const auto& entry : std::filesystem::directory_iterator(dir))
        if (entry.path().extension() == ".safetensors") out.push_back(entry.path());
    return out;
}
} // namespace strix::minimax::assets
