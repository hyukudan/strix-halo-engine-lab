#include "strix/qwen_model_description.hpp"

#include <exception>
#include <iostream>

int main(int argc, char** argv) {
    if (argc != 2 && argc != 3) {
        std::cerr << "Usage: strix-model-describe DESCRIPTION.toml [MODEL.gguf]\n";
        return 2;
    }
    try {
        const auto description = strix::model_description::load(argv[1]);
        const auto baked = strix::ModelContract::qwen38_flash_next();
        const auto graph = strix::qwen_description::inspect(description);
        std::size_t checked = 0;
        if (argc == 3) {
            const auto info = strix::gguf::read_file(argv[2]);
            const auto keys = strix::qwen_description::assert_gguf_matches(baked, info);
            checked = keys.size();
            for (const auto& key : keys) std::cout << "checked metadata: " << key << '\n';
        }
        std::cout << "Qwen description matches baked contract; layers " << graph.layers.size() << '\n';
        if (argc == 3)
            std::cout << "GGUF recognized metadata keys checked: " << checked
                << "; absent keys and unrecognized metadata are not validated\n";
        else std::cout << "GGUF metadata not checked (no weight file supplied)\n";
        return 0;
    } catch (const std::exception& error) {
        std::cerr << error.what() << '\n';
        return 1;
    }
}
