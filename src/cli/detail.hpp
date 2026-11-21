#pragma once

#include "strix/cli/cli.hpp"
#include "strix/serve/json.hpp"

namespace strix::cli {

[[nodiscard]] std::string_view platform_name(Platform);
[[nodiscard]] std::string support_row(const support::Row&);
[[nodiscard]] serve::Json describe(const model_registry::Entry&, const std::filesystem::path& path);
[[nodiscard]] serve::Json devices();
[[nodiscard]] serve::Json plan(const model_registry::Entry&, const Request&);
[[nodiscard]] std::filesystem::path executable_directory();
[[nodiscard]] bool executable_exists(const std::filesystem::path&);
int execute(const Command&);

} // namespace strix::cli
