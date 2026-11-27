#pragma once

#include "strix/engine_profile.hpp"
#include "strix/model_description.hpp"
#include "strix/profile/multi_device.hpp"
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
    multi_device::Decision placement;   // also config["multi_device"]
    std::string weights_pack{};         // presets.turbo[.backend].weights_pack: resident pack beside the model

    void append_options(std::vector<std::string>& args) const;
};

[[nodiscard]] std::string_view tier_word(Name);
[[nodiscard]] Name parse_tier(std::string_view);
[[nodiscard]] Name model_default_tier(const model_description::ModelDescription&);
[[nodiscard]] bool uses_native_resolution(const model_description::ModelDescription&, std::string_view backend,
                                          std::optional<Name> requested);
[[nodiscard]] std::optional<std::string_view> native_profile_option(const std::vector<std::string>& args);
// `placement` carries --device-count / --placement (its tier is replaced by the resolved tier). Every launcher (CLI, server,
// replica router) passes the same query so the effective "multi_device" member is identical for the same request.
// The preset is inspection: a launcher that executes it first calls multi_device::admit(preset.placement, its executor).
[[nodiscard]] ModelPreset resolve_model_preset(const model_description::ModelDescription&,
                                               std::optional<Name>, std::string_view backend,
                                               const multi_device::Query& placement = {});
[[nodiscard]] serve::Json description_json(const model_description::Value&);
// The replica router's admission through the same resolver: `replicas` whole-model endpoints must resolve the
// model's replica form (else std::invalid_argument with the rule and reason). The result's
// config["multi_device"] is what CLI / serve print for --device-count N with the same tier.
[[nodiscard]] ModelPreset resolve_replica_router(const model_description::ModelDescription&, std::optional<Name>,
                                                 std::uint32_t replicas, bool explicit_replica = false);

struct ModelLaunch {
    std::string asset, backend;
    std::vector<std::string> args;
};
[[nodiscard]] ModelLaunch load_model_launch(const std::filesystem::path& directory,
                                            std::string_view model, std::string_view verb);

// Legacy resolvers run both in the shell and in standalone processes.
// Keep this scope alive through validation and startup, including exec.
void append_native_profile(std::vector<std::string>& args, std::optional<Name> requested);

class PresetScope {
public:
    explicit PresetScope(bool enabled = true);
    ~PresetScope();
    PresetScope(const PresetScope&) = delete;
    PresetScope& operator=(const PresetScope&) = delete;
private:
    std::optional<std::string> previous_;
};

} // namespace strix::profile
