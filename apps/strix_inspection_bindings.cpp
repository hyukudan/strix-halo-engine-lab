#include "cli/detail.hpp"

#include "gemma4/engine/deployment.hpp"
#include "minimax/engine/deployment.hpp"
#include "strix/deploy/minimax_layercut.hpp"
#ifdef STRIX_CLI_QWEN_DESCRIPTION
#include "strix/qwen_model_description.hpp"
#endif

namespace strix::cli {

const DeploymentBinding* deployment_binding(std::string_view family) {
    static constexpr DeploymentBinding minimax_binding{
        [](const std::filesystem::path& path) -> std::unique_ptr<deploy::FootprintProvider> {
            return std::make_unique<minimax::engine::FootprintProvider>(minimax::engine::deployment_model(path));
        },
        [](const deploy::FootprintProvider& footprints) -> std::unique_ptr<deploy::RouteProvider> {
            return std::make_unique<deploy::MinimaxLayercutProvider>(footprints);
        }
    };
    static constexpr DeploymentBinding gemma_binding{
        [](const std::filesystem::path& path) -> std::unique_ptr<deploy::FootprintProvider> {
            return std::make_unique<gemma4::engine::FootprintProvider>(gemma4::engine::deployment_model(path));
        },
        nullptr
    };
    if (family == "minimax") return &minimax_binding;
    if (family == "gemma4") return &gemma_binding;
    return nullptr;
}

#ifdef STRIX_CLI_QWEN_DESCRIPTION
void validate_description(const model_registry::Entry& entry, const model_description::ModelDescription& description) {
    if (entry.family == "qwen" && entry.name == "qwen3.8-flash-next")
        (void)qwen_description::inspect(description);
}
#else
void validate_description(const model_registry::Entry&, const model_description::ModelDescription&) {}
#endif

} // namespace strix::cli
