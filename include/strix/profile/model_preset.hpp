#pragma once

#include "strix/engine_profile.hpp"
#include "strix/model_description.hpp"
#include "strix/serve/json.hpp"

#include <filesystem>

namespace strix::profile {

// A model preset selects a native resolver column. The native resolver still
// owns lever readiness, runtime regime, pack admission and drafter discovery.
struct ModelPreset {
    Name requested;
    Name levers;
    std::string kv;
    serve::Json config;

    void append_options(std::vector<std::string>& args) const;
};

[[nodiscard]] std::string_view tier_word(Name);
[[nodiscard]] Name parse_tier(std::string_view);
[[nodiscard]] ModelPreset resolve_model_preset(const model_description::ModelDescription&,
                                               std::optional<Name>, std::string_view backend);
[[nodiscard]] serve::Json description_json(const model_description::Value&);

struct ModelLaunch {
    std::string asset, backend;
    std::vector<std::string> args;
};
[[nodiscard]] ModelLaunch load_model_launch(const std::filesystem::path& directory,
                                            std::string_view model, std::string_view verb);

// Legacy resolvers run both in the shell and in standalone processes.
// Keep this scope alive through validation and startup, including exec.
class PresetScope {
public:
    PresetScope();
    ~PresetScope();
    PresetScope(const PresetScope&) = delete;
    PresetScope& operator=(const PresetScope&) = delete;
private:
    std::optional<std::string> previous_;
};

} // namespace strix::profile
