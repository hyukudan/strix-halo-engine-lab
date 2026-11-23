// strix-model-gen: derive the constant header of a model from its TOML (strix/model/constants.hpp).
//   strix-model-gen MODEL.toml SOURCE_NAME                 print the header
//   strix-model-gen --write MODEL.toml HEADER SOURCE_NAME  (re)write HEADER
//   strix-model-gen --check MODEL.toml HEADER SOURCE_NAME  exit 1 when HEADER differs from the TOML
// SOURCE_NAME is the repository-relative TOML path written into the banner (for example models/glm-5.3-flash.toml).
#include "strix/model/constants.hpp"
#include "strix/model_description.hpp"

#include <exception>
#include <fstream>
#include <iostream>
#include <iterator>
#include <string>
#include <string_view>

int main(int argc, char** argv) {
    try {
        if (argc == 3) {
            std::cout << strix::model::emit_constants(strix::model_description::load(argv[1]), argv[2]);
            return 0;
        }
        if (argc == 5 && (std::string_view(argv[1]) == "--write" || std::string_view(argv[1]) == "--check")) {
            const auto text = strix::model::emit_constants(strix::model_description::load(argv[2]), argv[4]);
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
