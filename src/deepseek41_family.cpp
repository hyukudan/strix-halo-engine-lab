#include "strix/deepseek41_family.hpp"

#include "strix/deepseek41_model.hpp"
#include "strix/model_family_registry.hpp"

#include <string>

namespace strix::deepseek41 {

void register_family() {
    ModelFamily family;
    family.name = "deepseek41";
    family.architectures = {"deepseek41"};
    family.validate = [](const gguf::FileInfo& info) { (void)validate_target(info); };
    family.describe = [](const gguf::FileInfo& info) {
        const auto m = validate_target(info);
        return "deepseek41: " + std::to_string(m.config.num_hidden_layers) + " layers, hidden " + std::to_string(m.config.hidden_size) +
               ", " + std::to_string(m.config.n_routed_experts) + " experts top-" + std::to_string(m.config.num_experts_per_tok) + ", " +
               std::to_string(m.manifest.tensors.size()) + " tensors";
    };
    family.reader_limits = [] { return reader_limits(); };
    register_model_family(std::move(family));
}

}  // namespace strix::deepseek41
