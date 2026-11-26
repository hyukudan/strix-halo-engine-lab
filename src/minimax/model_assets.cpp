#include "strix/minimax/model_assets.hpp"
#include "strix/serve/json.hpp"
#include "strix/minimax/asset_input.hpp"
#include "strix/minimax/model_description.hpp"
#include "strix/weight/shard_name.hpp"
#include <set>
#include <stdexcept>

namespace strix::minimax::assets {
std::map<std::string, std::filesystem::path> pack_weight_map(const std::filesystem::path& dir, LoaderSnapshot* snapshot) {
    // Optional deployment contract; legacy correct packs need no new file.
    const auto contract = dir / "minimax-m3.1-flash.toml";
    if (description::check_enabled() && std::filesystem::exists(contract)) {
        // Admission data is not part of the execution identity or asset receipt.
        Input model(contract);
        if (!model) throw std::runtime_error("cannot open " + contract.string());
        const std::string text{std::istreambuf_iterator<char>(model), std::istreambuf_iterator<char>()};
        if (model.bad()) throw std::runtime_error("cannot read " + contract.string());
        description::validate(model_description::parse(text));
    }
    const auto index = dir / "model.safetensors.index.json";
    Input in(index, snapshot);
    if (!in) throw std::runtime_error("cannot open " + index.string());
    const std::string text{std::istreambuf_iterator<char>(in), std::istreambuf_iterator<char>()};
    const auto json = serve::parse_json_unique(text);
    const auto* weights = json.find("weight_map");
    if (!weights || !weights->is_object()) throw std::runtime_error("invalid weight_map: " + index.string());
    std::map<std::string, std::filesystem::path> out;
    for (const auto& [name, value] : weights->as_object()) {
        if (!value.is_string()) throw std::runtime_error("weight_map." + name + ": expected string, actual " + value.dump());
        if (!weight::safetensors_shard_name_ok(value.as_string()))
            throw std::runtime_error("weight_map." + name + ": invalid safetensors shard name");
        out.emplace(name, dir / value.as_string());
    }
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
} // namespace strix::minimax::assets
