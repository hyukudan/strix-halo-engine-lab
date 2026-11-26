#include "strix/token_sampling.hpp"
#include "strix/serve/run.hpp"
#include "strix/serve/serve_config.hpp"

#include <charconv>
#include <filesystem>
#include <iostream>
#include <iterator>
#include <set>

// Engine rows come from the generated table (cmake: strix_serve_register_engine); this file names no model.
#define STRIX_SERVE_ENGINE(key, ns) namespace ns { void install_session_service(strix::serve::ServiceFactories&); }
#include "serve_engines.inc"
#undef STRIX_SERVE_ENGINE

namespace strix::serve {
namespace {
constexpr ServeEngine kEngines[] = {
#define STRIX_SERVE_ENGINE(key, ns) {#key, &ns::install_session_service},
#include "serve_engines.inc"
#undef STRIX_SERVE_ENGINE
    {nullptr, nullptr}};  // terminator: keeps the array non-empty in a build without engines
}
std::span<const ServeEngine> built_serve_engines() noexcept { return {kEngines, std::size(kEngines) - 1U}; }
bool install_serve_engine(std::string_view key, ServiceFactories& factories) {
    for (const auto& engine : built_serve_engines())
        if (key == engine.key) { engine.install(factories); return true; }
    return false;
}
void install_service_factories(ServiceFactories& factories) {
    for (const auto& engine : built_serve_engines()) engine.install(factories);
}
namespace {
std::string text(const model_description::ModelDescription& d, const char* key) {
    const auto* value = d.find(key);
    const auto* s = value ? std::get_if<std::string>(&value->data) : nullptr;
    if (!s) throw std::invalid_argument(std::string("description missing string: ") + key);
    return *s;
}
}
int run(int argc, char** argv, const ServiceFactories& factories, std::ostream& out,
        std::ostream& error, const ServeShell& shell) {
    try {
        // --serve-config FILE: the shell options below, validated and expanded in front of the command line
        using config_file::Kind;
        static constexpr config_file::Option shell_options[] = {
            {"model", "--model", Kind::value}, {"registry_root", "--registry-root", Kind::path}, {"registry", "--registry", Kind::path},
            {"description", "--description", Kind::value}, {"bind", "--bind", Kind::value}, {"port", "--port", Kind::value},
            {"profile", "--profile", Kind::value}, {"backend", "--backend", Kind::value}, {"asset", "--asset", Kind::path},
            {"device_count", "--device-count", Kind::value}, {"placement", "--placement", Kind::value}};
        auto file = config_file::expand(argc, argv, shell_options);
        if (file.print) { out << file.effective; return 0; }
        std::vector<char*> expanded;
        for (auto& arg : file.argv) expanded.push_back(arg.data());
        argc = static_cast<int>(expanded.size());
        argv = expanded.data();
        std::string root = shell.registry_root, registry = "models/registry.toml", model, description;
        std::vector<std::string> family_args;
        HttpServerConfig config;
        ServeShell::NativeRequest request;
        bool list = false, inspect = false, family_inspect = false;
        const auto value = [&](int& i) {
            if (++i >= argc) throw std::invalid_argument("missing option value");
            return std::string(argv[i]);
        };
        for (int i = 1; i < argc; ++i) {
            const std::string arg = argv[i];
            if (arg == "--") { while (++i < argc) family_args.emplace_back(argv[i]); break; }
            if (arg == "--help" || arg == "-h") {
                out << "strix-serve --model FAMILY|ID|NAME [--profile exact|fast|turbo] [--asset PATH]\n"
                    "  --backend hip|cuda --native-profile --print-effective-config [-- ENGINE_ARGS]\n"
                    "  --device-count N --placement single|replica|layers|tensor|speculation\n"
                    "  --registry-root DIR --registry PATH --description PATH\n"
                    "  --list-models --inspect --family-inspect --bind ADDRESS --port N\n"
                    "Use -- before family arguments that overlap shell options.\n"
                    "Registry metadata does not imply a built execution adapter.\n";
                return 0;
            }
            if (arg == "--registry-root") root = value(i);
            else if (arg == "--registry") registry = value(i);
            else if (arg == "--model") model = value(i);
            else if (arg == "--description") description = value(i);
            else if (arg == "--profile" || arg.starts_with("--profile=")) {
                if (request.tier) throw std::invalid_argument("duplicate --profile");
                request.tier = profile::parse_tier(arg == "--profile" ? value(i) : arg.substr(10));
            } else if (arg == "--native-profile") {
                if (request.native_profile) throw std::invalid_argument("duplicate --native-profile");
                request.native_profile = true;
            } else if (arg == "--asset") request.asset = value(i);
            else if (arg == "--backend") {
                request.backend = value(i);
                if (request.backend == "intel")
                    throw std::invalid_argument("--backend intel is a family reference option. Pass it after -- for a family that supports it "
                                                "or use a dedicated model reference executable. strix-serve supports hip or cuda at the shell level.");
                if (request.backend == "vulkan" || request.backend == "metal")
                    throw std::invalid_argument("--backend " + request.backend + " only exposes the RMSNorm operator reference. "
                                                "strix-serve supports hip or cuda at the shell level.");
                if (request.backend != "hip" && request.backend != "cuda")
                    throw std::invalid_argument("--backend requires hip or cuda");
            }
            else if (arg == "--device-count") {
                const auto number = value(i);
                unsigned devices = 0;
                const auto parsed = std::from_chars(number.data(), number.data() + number.size(), devices);
                if (parsed.ec != std::errc{} || parsed.ptr != number.data() + number.size() || devices < 1 || devices > 64)
                    throw std::invalid_argument("--device-count requires an integer in 1..64");
                request.placement.devices = devices;
            } else if (arg == "--placement") {
                const auto form = value(i);
                request.placement.placement = profile::multi_device::parse_form(form);
                if (!request.placement.placement)
                    throw std::invalid_argument("--placement requires single, replica, layers, tensor or speculation");
            }
            else if (arg == "--print-effective-config") request.print = true;
            else if (arg == "--list-models") list = true;
            else if (arg == "--inspect") inspect = true;
            else if (arg == "--family-inspect") family_inspect = true;
            else if (arg == "--bind") config.bind_address = request.bind = value(i);
            else if (arg == "--port") {
                const auto& number = request.port = value(i);
                unsigned port = 0;
                const auto parsed = std::from_chars(number.data(), number.data() + number.size(), port);
                if (parsed.ec != std::errc{} || parsed.ptr != number.data() + number.size() || port > 65535)
                    throw std::invalid_argument("invalid port");
                config.port = static_cast<std::uint16_t>(port);
            } else family_args.push_back(arg);
        }
        namespace fs = std::filesystem;
        const auto entries = model_registry::load((fs::path(root) / registry).string());
        std::set<std::string> names;
        for (const auto& e : entries) {
            if (e.id.empty() || e.name.empty() || e.family.empty() || e.description.empty() || !names.insert(e.name).second)
                throw std::invalid_argument("invalid or duplicate registry model");
            const fs::path path(e.description);
            if (path.is_absolute()) throw std::invalid_argument("registry description must be relative");
            for (const auto& part : path) if (part == "..") throw std::invalid_argument("registry description escapes root");
        }
        const auto load = [&](const model_registry::Entry& e) {
            auto d = model_description::load((fs::path(root) / e.description).string());
            if (text(d, "schema") != "strix-model/v1" || text(d, "name") != e.name)
                throw std::invalid_argument("description schema/name differs from registry");
            return d;
        };
        if (list) {
            for (const auto& e : entries) {
                (void)load(e);
                out << e.id << '\t' << e.name << '\t' << e.family << '\t'
                    << (factories.find(e.family) ? "built" : "unbuilt") << '\n';
            }
            return 0;
        }
        if (model.empty() && description.empty()) throw std::invalid_argument("--model is required");
        std::vector<const model_registry::Entry*> selected;
        // An explicit registry ID/name takes precedence over a family alias.
        bool exact = false;
        for (const auto& e : entries) if (e.id == model || e.name == model) exact = true;
        for (const auto& e : entries) {
            if (!model.empty() && !(exact ? e.id == model || e.name == model : e.family == model)) continue;
            if (!description.empty() && fs::path(e.description).lexically_normal() != fs::path(description).lexically_normal()) continue;
            selected.push_back(&e);
        }
        if (selected.empty()) throw std::invalid_argument("unknown model family or description: " + model);
        if (selected.size() != 1) throw std::invalid_argument("ambiguous model family; choose --description or a registry ID");
        ServiceSelection selection{*selected[0], load(*selected[0]), root, std::move(family_args), {}};
        const auto* factory = factories.find(selection.model.family);
        if (inspect) {
            out << selection.model.id << '\t' << selection.model.name << '\t' << selection.model.family
                << '\t' << selection.model.description << '\t' << (factory ? "built" : "unbuilt") << '\n';
            return 0;
        }
        if (request.asset.empty() && selection.arguments.empty() && selection.description.find("presets")) {
            auto local = profile::load_model_launch(".", selection.model.name, "serve");
            request.asset = std::move(local.asset);
            selection.arguments = std::move(local.args);
            if (request.backend.empty()) request.backend = std::move(local.backend);
        }
        // A placement query alone does not apply a model preset: it is validated (below) against the effective tier,
        // so a family-native --profile after `--` keeps its passthrough (same design as strix run/serve).
        const bool preset_requested = request.native_profile || request.tier || request.print ||
            !request.asset.empty() || !request.backend.empty();
        const bool placement_requested = request.placement.placement || request.placement.devices != 1;
        std::optional<std::string> native_word;
        for (std::size_t i = 0; i < selection.arguments.size() && !native_word; ++i) {
            const auto& a = selection.arguments[i];
            if (a.starts_with("--profile=")) native_word = a.substr(10);
            else if (a == "--profile" && i + 1 < selection.arguments.size()) native_word = selection.arguments[i + 1];
        }
        const bool native_profile = !request.tier && !request.print && native_word.has_value();
        // The effective placement, resolved once: the native tier when one is given, else the shell resolver's.
        const auto placement_for = [&](std::string_view backend_name) {
            if (request.native_profile || native_profile) {
                auto query = request.placement;
                if (request.tier) query.tier = *request.tier;
                else if (native_word) {
                    try { query.tier = profile::parse_tier(*native_word); } catch (const std::invalid_argument&) {}   // the engine rejects it
                }
                return profile::multi_device::resolve(selection.description, query);
            }
            return profile::resolve_model_preset(selection.description, request.tier, backend_name, request.placement).placement;
        };
        const std::string_view backend = request.backend.empty() ? "hip" : std::string_view(request.backend);
        if (request.print && request.asset.empty() && selection.arguments.empty()) {
            if (request.native_profile) throw std::invalid_argument("--native-profile inspection requires an engine asset");
            out << profile::resolve_model_preset(selection.description, request.tier, backend, request.placement).config.dump() << '\n';
            return 0;
        }
        const bool external_cuda = backend == "cuda" && !(factory && factory->serves_cuda);
        if ((preset_requested || placement_requested) && (!factory || external_cuda) && shell.launch) {
            // The native handoff carries the whole query (device count, placement) to the backend launcher, which
            // resolves it again; refuse here first so no launch starts for a form one engine does not execute.
            if (!request.print) profile::multi_device::admit(placement_for(backend), profile::multi_device::Executor::engine);
            return shell.launch(selection, request, out);
        }
        if (!factory) {
            error << "known family '" << selection.model.family << "' has no built shared serve adapter; legacy entry points remain separate\n";
            return 3;
        }
        if (external_cuda) throw std::invalid_argument("no CUDA launcher in this build");
        std::optional<profile::PresetScope> scope;
        if (placement_requested && !preset_requested)
            profile::multi_device::admit(placement_for(backend), profile::multi_device::Executor::engine);
        if (preset_requested) {
            if (!request.asset.empty()) {
                selection.arguments.insert(selection.arguments.begin(), request.asset);
                if (!factory->asset_option.empty())
                    selection.arguments.emplace(selection.arguments.begin(), factory->asset_option);
            }
            if (request.native_profile) {
                if (!request.print) profile::multi_device::admit(placement_for(backend), profile::multi_device::Executor::engine);
                profile::append_native_profile(selection.arguments, request.tier);
                if (request.print) selection.arguments.emplace_back("--print-effective-config");
                scope.emplace(false);
            } else if (native_profile) {
                // The native --profile is the tier: no preset injection, the placement is admitted at that tier.
                profile::multi_device::admit(placement_for(backend), profile::multi_device::Executor::engine);
            } else {
                const auto preset = profile::resolve_model_preset(selection.description, request.tier, backend, request.placement);
                // This process serves one worker: only `single` executes here (replicas run behind strix-router).
                if (!request.print) profile::multi_device::admit(preset.placement, profile::multi_device::Executor::engine);
                preset.append_options(selection.arguments);
                if (request.print) {
                    out << preset.config.dump() << '\n' << std::flush;
                    selection.arguments.emplace_back("--print-effective-config");
                }
                scope.emplace();
            }
        }
        selection.sampler_options = validate_sampling_startup_environment(factory->sampling_default);
        factory->validate(selection);
        if(factory->early_exit) if(auto result=factory->early_exit(selection,out)) return *result;
        if (family_inspect) { factory->inspect(selection, out); return 0; }
        const auto serve_leader = [&](SessionService& service, bool owner_on_caller) {
            service.configure_listener(config);
            if (!shell.serve) { serve_session(config, service, owner_on_caller); return; }
            auto& runtime = service.runtime();
            const HttpHandler handler = [&](const HttpRequest& http_request, HttpResponseWriter& writer) {
                service.handle(http_request, writer);
            };
            try { shell.serve(config, handler, runtime); }
            catch (...) { service.failed(); runtime.shutdown(); service.shutdown(); throw; }
            runtime.shutdown(); service.shutdown(); service.stopped();
        };
        if (factory->create_scoped)
            return factory->create_scoped(selection, [&](SessionService& service) { serve_leader(service, true); });
        auto service = factory->create(selection);
        if (!service) throw std::runtime_error("factory returned no service");
        if (auto result=service->startup()) return *result;
        if (service->role() == ServiceRole::follower) {
            try { const int result=service->follow(); service->shutdown(); return result; }
            catch (...) { service->failed(); service->shutdown(); throw; }
        }
        serve_leader(*service, false);
        return 0;
    } catch (const std::exception& e) { error << "strix-serve: " << e.what() << '\n'; return 2; }
    catch (...) { error << "strix-serve: service failure\n"; return 2; }
}
int run(int argc, char** argv) {
    ServiceFactories factories;
    install_service_factories(factories);
    return run(argc, argv, factories, std::cout, std::cerr);
}
} // namespace strix::serve
