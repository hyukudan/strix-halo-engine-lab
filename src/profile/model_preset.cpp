#include "strix/profile/model_preset.hpp"
#include "strix/turbo.hpp"

#include <algorithm>
#include <type_traits>

namespace strix::profile {
namespace {
using model_description::Value;
using serve::Json;

const Value::Table& table(const Value& v, std::string_view path) {
    if (const auto* result = std::get_if<Value::Table>(&v.data)) return *result;
    throw std::invalid_argument(std::string(path) + " must be a table");
}

std::string_view setting(const Value& v, std::string_view path) {
    if (const auto* result = std::get_if<std::string>(&v.data); result && !result->empty()) return *result;
    throw std::invalid_argument(std::string(path) + " must be a nonempty string");
}

void restore(const char* value) {
#ifdef _WIN32
    (void)_putenv_s(kPresetEnv, value ? value : "");
#else
    if (value) (void)::setenv(kPresetEnv, value, 1);
    else (void)::unsetenv(kPresetEnv);
#endif
}
} // namespace

Json description_json(const Value& value) {
    return std::visit([](const auto& field) -> Json {
        using T = std::decay_t<decltype(field)>;
        if constexpr (std::is_same_v<T, Value::Table>) {
            Json object = Json::object();
            for (const auto& [key, child] : field) object.set(key, description_json(child));
            return object;
        } else if constexpr (std::is_same_v<T, Value::Array>) {
            Json array = Json::array();
            for (const auto& child : field) array.push(description_json(child));
            return array;
        } else {
            return Json(field);
        }
    }, value.data);
}

std::string_view tier_word(Name tier) { return tier == Name::standard ? "exact" : name(tier); }

Name parse_tier(std::string_view tier) {
    if (tier == "exact" || tier == "fast" || tier == "turbo") return parse(tier);
    throw std::invalid_argument("--profile requires exact, fast or turbo (got '" + std::string(tier) + "')");
}

Name model_default_tier(const model_description::ModelDescription& description) {
    const auto* presets = description.find("presets");
    if (!presets) return Name::standard;
    (void)table(*presets, "presets");
    const auto* fallback = description.find("presets.default");
    if (!fallback) throw std::invalid_argument("missing presets.default");
    const auto tier = parse_tier(setting(*fallback, "presets.default"));
    if (tier == Name::turbo) throw std::invalid_argument("presets.default cannot select turbo");
    return tier;
}

bool uses_native_resolution(const model_description::ModelDescription& description,
                            std::string_view backend, std::optional<Name> requested) {
    const auto key = "presets.native_resolution." + std::string(backend);
    const auto* migration = description.find(key);
    if (!migration) return false;
    const auto* choices = std::get_if<Value::Array>(&migration->data);
    if (!choices) throw std::invalid_argument(key + " must be an array of tiers");
    const auto selection = requested ? tier_word(*requested) : "implicit";
    bool selected = false;
    for (const auto& item : *choices) {
        const auto word = setting(item, key);
        if (word != "implicit") (void)parse_tier(word);
        selected |= word == selection;
    }
    return selected;
}

std::optional<std::string_view> native_profile_option(const std::vector<std::string>& args) {
    for (std::size_t i = 0; i < args.size(); ++i) {
        if (args[i].starts_with("--profile=")) return std::string_view(args[i]).substr(10);
        if (args[i] == "--profile") return i + 1 < args.size() ? std::string_view(args[i + 1]) : std::string_view{};
    }
    return std::nullopt;
}

ModelPreset resolve_model_preset(const model_description::ModelDescription& description, std::optional<Name> request,
                                 std::string_view backend, const multi_device::Query& placement) {
    const auto* presets = description.find("presets");
    const auto requested = request.value_or(model_default_tier(description));
    const std::string path = "presets." + std::string(tier_word(requested));
    auto column = requested;
    std::string kv = "resolver";
    std::string weights_pack;  // a calibrated pack resident beside the model (TURBO only)
    if (presets) {
        const auto* selected = description.find(path);
        if (!selected) throw std::invalid_argument("missing " + path);
        const auto read = [&](const Value& value, std::string_view field, bool base) {
            for (const auto& [key, item] : table(value, field)) {
                if (key == "levers") column = parse_tier(setting(item, std::string(field) + ".levers"));
                else if (key == "kv") kv = setting(item, std::string(field) + ".kv");
                else if (key == "weights_pack") weights_pack = setting(item, std::string(field) + ".weights_pack");
                else if (base && (key == "hip" || key == "cuda")) (void)table(item, key);
                else throw std::invalid_argument("unknown preset field: " + std::string(field) + "." + key);
            }
        };
        read(*selected, path, true);
        if (!backend.empty()) {
            const auto field = path + "." + std::string(backend);
            if (const auto* over = description.find(field)) read(*over, field, false);
        }
    }
    if ((requested == Name::standard && column != Name::standard) ||
        (requested != Name::turbo && column == Name::turbo) ||
        (requested != Name::turbo && kv == "fp8"))
        throw std::invalid_argument(path + " selects a higher tier");
    if (kv != "resolver" && kv != "native" && kv != "fp8")
        throw std::invalid_argument(path + ".kv must be resolver, native or fp8");
    if (!weights_pack.empty() && (requested != Name::turbo || weights_pack.find_first_of("/\\") != std::string::npos))
        throw std::invalid_argument(path + ".weights_pack is a file name beside the model, in the turbo preset only");

    Json storage_policy = Json::Object{{"policy", kv}};
    if (const auto* storage = description.find("presets.native_kv"))
        storage_policy.set("native", setting(*storage, "presets.native_kv"));
    Json config = Json::Object{{"profile", tier_word(requested)}, {"backend", backend},
        {"levers", Json::Object{{"column", tier_word(column)}, {"admission", "qualified-by-engine"}}},
        {"kv", std::move(storage_policy)}};
    if (const auto* name = description.find("name")) config.set("model", description_json(*name));
    if (const auto* reference = description.find("presets.reference"))
        config.set("reference", setting(*reference, "presets.reference"));
    config.set("drafter", Json());
    for (const auto* field : {"weights", "drafter", "fusion", "schedule"}) {
        if (const auto* value = description.find(field)) config.set(field, description_json(*value));
    }
    auto query = placement;
    query.tier = requested;
    const auto policy = multi_device::parse_policy(description);
    auto decision = multi_device::select(policy, query);
    decision.legacy = multi_device::legacy_overrides(policy);
    config.set("multi_device", decision.json(policy, query));
    if (!weights_pack.empty()) config.set("weights_pack", weights_pack);
    return {requested, column, std::move(kv), std::move(config), std::move(decision), std::move(weights_pack)};
}

ModelPreset resolve_replica_router(const model_description::ModelDescription& description, std::optional<Name> tier,
                                   std::uint32_t replicas, bool explicit_replica) {
    multi_device::Query query;
    query.devices = replicas;
    if (explicit_replica) query.placement = multi_device::Form::replica;
    auto preset = resolve_model_preset(description, tier, {}, query);
    if (preset.placement.form != multi_device::Form::replica)
        throw std::invalid_argument("model does not resolve the replica form for " + std::to_string(replicas) +
                                    " replicas (" + std::string(multi_device::form_name(preset.placement.form)) + ", " +
                                    preset.placement.rule + ": " + preset.placement.reason + ")");
    multi_device::admit(preset.placement, multi_device::Executor::router);
    return preset;
}

ModelLaunch load_model_launch(const std::filesystem::path& directory, std::string_view model, std::string_view verb) {
    if (model.empty() || model == "." || model == ".." ||
        model.find_first_not_of("abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789_.-") != std::string_view::npos)
        throw std::invalid_argument("invalid model name for launch configuration");
    const auto path = directory / "launch" / (std::string(model) + ".toml");
    if (!std::filesystem::exists(path)) return {};
    const auto config = model_description::load(path.string());
    const auto* section = config.find(verb);
    if (!section) return {};
    ModelLaunch result;
    for (const auto& [key, item] : table(*section, verb)) {
        if (key == "asset") result.asset = setting(item, "asset");
        else if (key == "backend") {
            result.backend = setting(item, "backend");
            if (result.backend != "hip" && result.backend != "cuda")
                throw std::invalid_argument("launch backend requires hip or cuda");
        } else if (key == "args") {
            const auto* list = std::get_if<Value::Array>(&item.data);
            if (!list) throw std::invalid_argument("launch args must be an array of strings");
            for (const auto& arg : *list) {
                const auto* s = std::get_if<std::string>(&arg.data);
                if (!s) throw std::invalid_argument("launch args must be an array of strings");
                result.args.push_back(*s);
            }
        } else throw std::invalid_argument("unknown launch field: " + key);
    }
    if (result.asset.empty()) throw std::invalid_argument(path.string() + " requires an asset");
    return result;
}

void ModelPreset::append_options(std::vector<std::string>& args) const {
    const auto has = [&args](std::string_view option) {
        return std::ranges::any_of(args, [option](std::string_view arg) {
            return arg == option || (arg.starts_with(option) && arg.substr(option.size()).starts_with('='));
        });
    };
    if (has("--profile")) throw std::invalid_argument("specify --profile before engine arguments");
    // The engine has the native format and the effective EXACT tier for the KV refusal receipt.
    if (requested != Name::turbo && levers != Name::standard) {
        if (has("--prefill-attn8") || has("--decode-attn8"))
            throw std::invalid_argument("KV attention overrides require the turbo preset");
        for (auto it = args.begin(); it != args.end(); ++it) {
            if (*it == "--kv=fp8" || (*it == "--kv" && it + 1 != args.end() && *(it + 1) == "fp8"))
                throw std::invalid_argument("--kv fp8 requires the turbo preset");
        }
    }
    args.emplace_back("--profile");
    args.emplace_back(tier_word(levers));
    if (kv != "resolver" && !has("--kv") && !has("--prefill-attn8") && !has("--decode-attn8")) {
        args.emplace_back("--kv");
        args.push_back(kv);
    }
    // The preset's resident pack is a default, not a request: any explicit weight source, width or pack wins, and the
    // engine keeps native weights (with a note) when the file is absent.
    if (requested == Name::turbo && !weights_pack.empty() && !has(turbo::kWeightsOption) &&
        !has(turbo::kBitsOption) && !has(turbo::kPackOption) && !has(turbo::kProfilePackOption)) {
        args.emplace_back(turbo::kProfilePackOption);
        args.push_back(weights_pack);
    }
}

void append_native_profile(std::vector<std::string>& args, std::optional<Name> requested) {
    if (!requested) return;
    if (std::ranges::any_of(args, [](std::string_view arg) { return arg == "--profile" || arg.starts_with("--profile="); }))
        throw std::invalid_argument("specify --profile once");
    args.emplace_back("--profile");
    args.emplace_back(tier_word(*requested));
}

PresetScope::PresetScope(bool enabled) {
    if (const char* value = environment({}, kPresetEnv)) previous_ = value;
    if (!export_environment(kPresetEnv, enabled ? "1" : "0"))
        throw std::runtime_error("cannot set model preset admission");
}

PresetScope::~PresetScope() { restore(previous_ ? previous_->c_str() : nullptr); }
} // namespace strix::profile
