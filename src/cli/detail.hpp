#pragma once

#include "strix/cli/cli.hpp"
#include "strix/serve/json.hpp"

#include <memory>

namespace strix::deploy {
class FootprintProvider;
class RouteProvider;
}

namespace strix::cli {

// Model adapters supplied by the application at the inspection dispatch boundary.
struct DeploymentBinding {
    std::unique_ptr<deploy::FootprintProvider> (*footprints)(const std::filesystem::path&);
    std::unique_ptr<deploy::RouteProvider> (*route)(const deploy::FootprintProvider&);
};

[[nodiscard]] const DeploymentBinding* deployment_binding(std::string_view family);
void validate_description(const model_registry::Entry&, const model_description::ModelDescription&);
[[nodiscard]] std::string_view platform_name(Platform);
[[nodiscard]] std::string support_row(const support::Row&);
[[nodiscard]] serve::Json describe(const model_registry::Entry&, const std::filesystem::path& path);
[[nodiscard]] serve::Json devices();
[[nodiscard]] serve::Json plan(const model_registry::Entry&, const Request&);
[[nodiscard]] std::filesystem::path executable_directory();
[[nodiscard]] bool executable_exists(const std::filesystem::path&);
int execute(const Command&);

} // namespace strix::cli
