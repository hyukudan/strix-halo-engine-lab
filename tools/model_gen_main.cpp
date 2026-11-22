// strix-model-gen: derive the constant header of a model from its TOML (strix/model/constants.hpp).
//   strix-model-gen MODEL.toml SOURCE_NAME                 print the header
//   strix-model-gen --write MODEL.toml HEADER SOURCE_NAME  (re)write HEADER
//   strix-model-gen --check MODEL.toml HEADER SOURCE_NAME  exit 1 when HEADER differs from the TOML
// SOURCE_NAME is the repository-relative TOML path written into the banner (for example models/glm-5.3-flash.toml).
// All modes validate model schema tables before emitting or comparing the header.
#include "model/constants_profiles.hpp"
#include "minimax/engine/mm_profile_table.hpp"
#include "strix/deepseek41_pair_profile_table.hpp"
#include "strix/glm_pair_profile_table.hpp"
#include "strix/profiles/ds41_rtx.hpp"
#include "strix/profiles/hy4.hpp"
#include "strix/profiles/inkling2_mini.hpp"
#include "strix/profiles/qwen_cuda.hpp"
#include "strix/profiles/qwen_dense_cuda.hpp"
#include "strix/profiles/qwen_dense_hip.hpp"

#include <exception>
#include <fstream>
#include <iostream>
#include <iterator>
#include <span>
#include <string>
#include <string_view>
#include <variant>
#include <vector>

namespace {

std::vector<std::string_view> profile_lever_names(std::string_view model) {
    using namespace strix;
    std::vector<std::string_view> names;
    const auto append = [&names](std::span<const profile::Lever> rows) {
        for (const auto& row : rows) names.push_back(row.name);
    };

    // A declaration may carry defaults for any of its model's backends.
    if (model == "qwen3.8-flash-next") {
        append(qwen::profile::table);
        append(qwen::cuda_profile::table);
    } else if (model == "qwen3.8-27b") {
        append(qwen35::profile::table);
        append(qwen35::hip_profile::table);
    } else if (model == "deepseek-v4.1-flash") {
        append(deepseek41::pair_defaults::profile_table);
        append(ds41::rtx::profile::shared_rows());
    } else if (model == "glm-5.3-flash") {
        append(glm::rank_defaults::product_profile_table);
    } else if (model == "minimax-m3.1-flash") {
        append(minimax::defaults::profile_table);
    } else if (model == "hy4-flash") {
        append(hy4::profile::table);
    } else if (model == inkling2::kMiniProfile.tag) {
        append(inkling2::kMiniProfile.table);
    }
    return names;
}

std::string generate(const char* path, std::string_view source_name) {
    const auto description = strix::model_description::load(path);
    const auto* field = description.find("name");
    const auto* name = field ? std::get_if<std::string>(&field->data) : nullptr;
    const auto lever_names = profile_lever_names(name ? std::string_view(*name) : std::string_view{});
    return strix::model::emit_constants(description, source_name, lever_names);
}

} // namespace

int main(int argc, char** argv) {
    try {
        if (argc == 3) {
            std::cout << generate(argv[1], argv[2]);
            return 0;
        }
        if (argc == 5 && (std::string_view(argv[1]) == "--write" || std::string_view(argv[1]) == "--check")) {
            const auto text = generate(argv[2], argv[4]);
            if (std::string_view(argv[1]) == "--write") {
                std::ofstream out(argv[3], std::ios::binary);
                out << text;
                if (!out) { std::cerr << "strix-model-gen: cannot write " << argv[3] << "\n"; return 2; }
                return 0;
            }
            std::ifstream in(argv[3], std::ios::binary);
            const std::string have((std::istreambuf_iterator<char>(in)), std::istreambuf_iterator<char>());
            if (have != text) { std::cerr << "strix-model-gen: " << argv[3] << " is stale; regenerate it\n"; return 1; }
            return 0;
        }
        std::cerr << "usage: strix-model-gen [--write|--check] MODEL.toml [HEADER] SOURCE_NAME\n";
        return 2;
    } catch (const std::exception& e) {
        std::cerr << "strix-model-gen: " << e.what() << "\n";
        return 2;
    }
}
