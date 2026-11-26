#include "strix/cli/cli.hpp"
#include "strix/build_info.hpp"
#include "detail.hpp"

#include <algorithm>
#include <charconv>
#include <iostream>
#include <stdexcept>
#include <string_view>

namespace strix::cli {
namespace {
constexpr std::string_view usage =
    "Usage: strix --version | version | doctor [--json] | devices [--json]\n"
    "       strix support [--family F] [--backend B] [--state S] [--json]\n"
    "       strix models list [--json] | inspect MODEL [--json]\n"
    "       strix plan MODEL --context N [--devices FILE] --json\n"
    "       strix run|serve [--model] MODEL [--profile exact|fast|turbo] [--backend hip|cuda]\n"
    "             [--asset PATH] [--device-count N] [--placement single|replica|layers|tensor|speculation]\n"
    "             [--print-effective-config] [--dry-run] [-- ENGINE_ARGS]\n"
    "             (serve also takes --host H and --port P)\n"
    "       strix update | rollback\n";

struct Route {
    std::string_view family, source, binary, backend, function;
    bool serving;
    std::string_view model_flag = "", serve_flag = "", host_flag = "", port_flag = "";
    enum Pair { none, ranked, positional, world } pair = none;
    bool rtx_assets = false;
};
// Use a route only when its source entry occurs in the selected registry row.
constexpr Route routes[] = {
    {"qwen", "apps/hip_decode_token_main.cpp", "strix-hip-decode-token", "hip", "cli-generate", false},
    {"qwen", "apps/hip_decode_token_main.cpp", "strix-hip-decode-token", "hip", "serve", true,
        "", "--http-serve", "--http-bind", "--http-port"},
    {"qwen", "apps/cuda_qwen_serve_main.cu", "strix-cuda-qwen-serve", "cuda", "serve", true,
        "--model", "", "--bind", "--port"},
    {"deepseek41", "apps/deepseek41_serve_main.cpp", "strix-deepseek41-serve", "hip", "cli-generate", false,
        "", "", "", "", Route::ranked},
    {"deepseek41", "apps/deepseek41_serve_main.cpp", "strix-deepseek41-serve", "hip", "serve", true,
        "", "", "--http-bind", "--http-port", Route::ranked},
    {"deepseek41", "apps/cuda_ds41_replica.cu", "strix-cuda-ds41-replica", "cuda", "decode", false,
        "", "", "", "", Route::none, true},
    {"minimax", "apps/minimax_rank_main.cpp", "strix-minimax-rank", "hip", "pair/multi-node", false,
        "", "", "", "", Route::positional},
    {"minimax", "apps/minimax_serve_main.cpp", "strix-minimax-serve", "hip", "serve", true,
        "", "", "--bind", "--port", Route::ranked},
    {"glm5", "apps/glm_rank_main.cpp", "strix-glm5-rank", "hip", "pair/multi-node", false,
        "", "", "", "", Route::world},
    {"glm5", "apps/glm_serve_main.cpp", "strix-glm5-serve", "hip", "serve", true,
        "", "", "--host", "--port", Route::world},
    {"gemma4", "apps/gemma4_decode_main.cpp", "strix-gemma4-decode", "hip", "cli-generate", false},
    {"gemma4", "apps/gemma4_serve_main.cpp", "strix-gemma4-serve", "hip", "serve", true, "", "", "--bind", "--port"},
    {"qwen35", "apps/hip_qwen35_decode_main.hip", "strix-hip-qwen35-decode", "hip", "cli-generate", false, "--model"},
    {"qwen35", "apps/cuda_qwen35_decode_main.cu", "strix-cuda-qwen35-decode", "cuda", "cli-generate", false, "--model"},
    {"hy4_flash", "apps/hy4_rank_main.cpp", "strix-hy4-rank", "hip", "cli-generate", false},
    {"hy4_flash", "apps/strix_serve_main.cpp", "strix-serve", "hip", "serve", true, "--asset", "", "--bind", "--port"},
};

[[noreturn]] void invalid(std::string reason) { throw std::invalid_argument(std::move(reason)); }

bool value_token(std::string_view value) { return !value.empty() && value.front() != '-'; }

// --device-count / --placement as the shared multi-device query (CLI and server build the same one).
profile::multi_device::Query placement_query(const Request& req) {
    profile::multi_device::Query query;
    if (req.device_count) query.devices = req.device_count;
    if (!req.placement.empty()) query.placement = profile::multi_device::parse_form(req.placement);
    return query;
}

bool native_profile_arguments(const std::vector<std::string>& args) {
    return std::ranges::any_of(args, [](std::string_view arg) { return arg == "--profile" || arg.starts_with("--profile="); });
}

std::string_view native_profile_word(const std::vector<std::string>& args) {
    for (std::size_t i = 0; i < args.size(); ++i) {
        if (args[i].starts_with("--profile=")) return std::string_view(args[i]).substr(10);
        if (args[i] == "--profile" && i + 1 < args.size()) return args[i + 1];
    }
    return {};
}

std::uint32_t number(std::string_view value, std::string_view name, std::uint32_t lo, std::uint32_t hi) {
    std::uint32_t result = 0;
    const auto parsed = std::from_chars(value.data(), value.data() + value.size(), result);
    if (parsed.ec != std::errc{} || parsed.ptr != value.data() + value.size() || result < lo || result > hi)
        invalid(std::string(name) + " requires an integer in " + std::to_string(lo) + ".." + std::to_string(hi));
    return result;
}

bool alias(const model_registry::Entry& entry, std::string_view name) {
    return entry.id == name || entry.name == name;
}

const model_registry::Entry& model(const std::vector<model_registry::Entry>& entries, const std::string& name) {
    auto found = std::ranges::find_if(entries, [&](const auto& entry) { return alias(entry, name); });
    if (found == entries.end()) {
        const auto path = std::filesystem::path(name).lexically_normal();
        const auto filename = path.filename().empty() ? path.parent_path().filename() : path.filename();
        const auto base = filename.extension() == ".gguf" ? filename.stem().string() : filename.string();
        found = std::ranges::find_if(entries, [&](const auto& entry) { return alias(entry, base); });
    }
    if (found == entries.end()) invalid("unknown model '" + name + "'; see strix models list");
    return *found;
}

std::string_view option(const std::vector<std::string>& args, std::string_view flag) {
    const auto it = std::ranges::find(args, flag);
    if (it == args.end() || it + 1 == args.end() || !value_token(*(it + 1)))
        invalid(std::string(flag) + " requires a value in engine arguments");
    if (std::find(it + 1, args.end(), flag) != args.end())
        invalid(std::string(flag) + " was given more than once");
    return *(it + 1);
}

void check_route(const Route& route, const std::vector<std::string>& args) {
    if (route.pair == Route::positional) {
        if (args.size() < 3 || (args[0] != "0" && args[0] != "1") ||
            !value_token(args[1]) || !value_token(args[2]))
            invalid("this route requires RANK (0 or 1), TRANSPORT and OUTPUT_DIR after the model asset");
    } else if (route.pair != Route::none) {
        if (route.pair == Route::world && option(args, "--world") != "2")
            invalid("this pair route requires --world 2");
        const auto rank = option(args, "--rank");
        if (rank != "0" && rank != "1") invalid("--rank must be 0 or 1");
        (void)option(args, "--transport");
    }
    if (route.rtx_assets) {
        (void)option(args, "--receipt");
        (void)option(args, "--import-state");
        if (std::ranges::find(args, "--serve") == args.end()) (void)option(args, "--ref");
    }
}

serve::Json entry_json(const model_registry::Entry& entry) {
    return serve::Json::Object{{"id", entry.id}, {"name", entry.name}, {"family", entry.family},
                               {"description", entry.description}};
}

} // namespace

std::string_view platform_name(Platform platform) {
    switch (platform) {
    case Platform::linux_host: return "linux";
    case Platform::windows_host: return "windows";
    case Platform::macos_host: return "macos";
    case Platform::other: return "other";
    }
    return "other";
}

std::string support_row(const support::Row& row) {
    return row.family + '/' + row.backend + '/' + row.arch + '/' + row.os + '/' + row.function +
        " (" + row.format + ") [" + std::string(support::state_name(row.state)) + ']';
}

Request parse(const std::vector<std::string>& args) {
    Request r;
    if (args.empty()) return r;
    const auto& verb = args.front();
    std::size_t i = 1;
    if (verb == "--help" || verb == "-h" || verb == "help") r.verb = Verb::help;
    else if (verb == "--version" || verb == "version") r.verb = Verb::version;
    else if (verb == "doctor") r.verb = Verb::doctor;
    else if (verb == "support") r.verb = Verb::support;
    else if (verb == "devices") r.verb = Verb::devices;
    else if (verb == "models" && i < args.size() && args[i++] == "list") r.verb = Verb::models;
    else if (verb == "inspect") r.verb = Verb::inspect;
    else if (verb == "plan") r.verb = Verb::plan;
    else if (verb == "run") r.verb = Verb::run;
    else if (verb == "serve") r.verb = Verb::serve;
    else if (verb == "update") r.verb = Verb::update;
    else if (verb == "rollback") r.verb = Verb::rollback;
    else invalid("unknown command; see strix --help");

    const bool launch = r.verb == Verb::run || r.verb == Verb::serve;
    const bool allows_json = r.verb == Verb::doctor || r.verb == Verb::devices ||
        r.verb == Verb::models || r.verb == Verb::inspect || r.verb == Verb::plan || r.verb == Verb::support;
    if (launch || r.verb == Verb::inspect || r.verb == Verb::plan) {
        if (launch && i < args.size() && args[i] == "--model") ++i;
        if (i == args.size() || !value_token(args[i])) invalid(verb + " requires MODEL");
        r.model = args[i++];
    }
    while (i < args.size()) {
        const auto& arg = args[i++];
        const auto value = [&]() -> const std::string& {
            if (i == args.size() || !value_token(args[i])) invalid(arg + " requires a value");
            return args[i++];
        };
        if (launch && arg == "--") {
            r.args.assign(args.begin() + static_cast<std::ptrdiff_t>(i), args.end());
            break;
        }
        if (allows_json && arg == "--json" && !r.json) r.json = true;
        else if (r.verb == Verb::support && arg == "--family" && r.support_filter.family.empty())
            r.support_filter.family = value();
        else if (r.verb == Verb::support && arg == "--backend" && r.support_filter.backend.empty())
            r.support_filter.backend = value();
        else if (r.verb == Verb::support && arg == "--state" && !r.support_filter.state) {
            r.support_filter.state = support::parse_state(value());
            if (!r.support_filter.state) invalid("unknown support state");
        }
        else if (launch && arg == "--dry-run" && !r.dry_run) r.dry_run = true;
        else if (launch && arg == "--print-effective-config" && !r.print_effective_config)
            r.print_effective_config = true;
        else if (launch && (arg == "--profile" || arg.starts_with("--profile=")) && !r.profile)
            r.profile = profile::parse_tier(arg == "--profile" ? std::string_view(value()) : std::string_view(arg).substr(10));
        else if (launch && arg == "--asset" && r.asset.empty()) r.asset = value();
        else if (launch && arg == "--backend" && r.backend.empty()) {
            r.backend = value();
            if (r.backend != "hip" && r.backend != "cuda") invalid("--backend requires hip or cuda");
        }
        else if (r.verb == Verb::serve && (arg == "--host" || arg == "--bind") && r.host.empty()) r.host = value();
        else if (r.verb == Verb::serve && arg == "--port" && r.port.empty()) {
            r.port = value();
            (void)number(r.port, "--port", 1, 65535);
        } else if (launch && arg == "--device-count" && !r.device_count)
            r.device_count = number(value(), "--device-count", 1, 64);
        else if (launch && arg == "--placement" && r.placement.empty()) {
            r.placement = value();
            if (!profile::multi_device::parse_form(r.placement))
                invalid("--placement requires single, replica, layers, tensor or speculation");
        } else if (r.verb == Verb::plan && arg == "--devices" && r.devices.empty())
            r.devices = value();
        else if (r.verb == Verb::plan && arg == "--context" && !r.context)
            r.context = number(value(), "--context", 1, UINT32_MAX);
        else invalid("unexpected argument '" + arg + "'; see strix --help");
    }
    if (r.verb == Verb::plan && (!r.context || !r.json)) invalid("plan requires --context N and --json");
    return r;
}

Command resolve(const Request& req, const std::vector<model_registry::Entry>& entries,
                const support::Matrix& matrix, const DoctorProbes& probes, const std::filesystem::path& bin,
                const ExecutableExists& exists, const model_description::ModelDescription* description) {
    const auto& entry = model(entries, req.model);
    const bool serving = req.verb == Verb::serve;
    bool registered = false;
    bool supported = false, installed_refusal = false, refusal_has_row = false;
    std::string refusal;
    const auto os = platform_name(probes.platform());
    for (const auto& route : routes) {
        if (route.family != entry.family || route.serving != serving ||
            (!req.backend.empty() && req.backend != route.backend) ||
            std::ranges::find(entry.apps, route.source) == entry.apps.end()) continue;
        registered = true;
        // A packaged binary alone does not qualify a GPU architecture.
        // Unknown architecture must fail closed, just as doctor() does.
        const auto architectures = probes.architectures(route.backend);
        auto executable = bin / route.binary;
#if defined(_WIN32)
        executable += ".exe";
#endif
        const bool installed = exists(executable);
        const support::Row* best = nullptr;
        for (const auto& row : matrix.rows) {
            if (row.family == entry.family && row.backend == route.backend && row.os == os &&
                row.function == route.function &&
                std::ranges::find(architectures, row.arch) != architectures.end() &&
                (!best || row.state > best->state))
                best = &row;
        }
        if (!best || best->state < support::State::connected) {
            if (refusal.empty() || (installed && !installed_refusal) ||
                (installed == installed_refusal && best && !refusal_has_row)) {
                refusal = best ? "support matrix row " + support_row(*best) +
                    " is below connected" + (best->note.empty() ? "" : "; " + best->note) :
                    "no support matrix row for " + entry.family + '/' + std::string(route.backend) +
                    '/' + std::string(os) + '/' + std::string(route.function);
                installed_refusal = installed;
                refusal_has_row = best != nullptr;
            }
            continue;
        }
        supported = true;
        if (!installed) continue;
        auto rest = req.args;
        auto asset = req.asset.empty() ? req.model : req.asset;
        if (req.asset.empty() && alias(entry, asset)) {
            if (rest.empty() || !value_token(rest.front()))
                invalid("model '" + entry.name + "' requires an asset path as the first argument after --");
            asset = rest.front();
            rest.erase(rest.begin());
        }
        if (!req.print_effective_config) check_route(route, rest);
        const bool hy4_rank = route.binary == "strix-hy4-rank";
        const bool hy4_shell = route.binary == "strix-serve";
        auto query = placement_query(req);
        const bool native_pair = hy4_rank && query.devices == 2 &&
            (!query.placement || query.placement == profile::multi_device::Form::layers ||
             query.placement == profile::multi_device::Form::tensor);
        Command cmd{executable, {}};
        if (hy4_shell) cmd.args.insert(cmd.args.end(), {"--model", entry.id});
        if (!route.model_flag.empty()) cmd.args.emplace_back(route.model_flag);
        cmd.args.push_back(std::move(asset));
        if (!route.serve_flag.empty()) cmd.args.emplace_back(route.serve_flag);
        if (!req.host.empty()) {
            cmd.args.emplace_back(route.host_flag);
            cmd.args.push_back(req.host);
        }
        if (!req.port.empty()) {
            cmd.args.emplace_back(route.port_flag);
            cmd.args.push_back(req.port);
        } else if (serving && entry.family == "deepseek41" && !req.print_effective_config && option(rest, "--rank") == "0" &&
                   std::ranges::find(rest, "--http-port") == rest.end()) {
            cmd.args.insert(cmd.args.end(), {"--http-port", "8080"});
        }
        if (!hy4_shell) cmd.args.insert(cmd.args.end(), rest.begin(), rest.end());
        if (native_pair) {
            cmd.args.emplace_back("--pair-expected");
            if (query.placement) cmd.args.insert(cmd.args.end(), {"--pair-placement", req.placement});
            // Request facts become available in the rank; no shell decision can grant this form.
            query.placement.reset();
        }
        if (hy4_shell) {
            if (req.profile) cmd.args.insert(cmd.args.end(), {"--profile", std::string(profile::tier_word(*req.profile))});
            if (req.device_count) cmd.args.insert(cmd.args.end(), {"--device-count", std::to_string(req.device_count)});
            if (!req.placement.empty()) cmd.args.insert(cmd.args.end(), {"--placement", req.placement});
            if (req.print_effective_config) cmd.args.emplace_back("--print-effective-config");
            cmd.args.emplace_back("--");
            cmd.args.insert(cmd.args.end(), rest.begin(), rest.end());
            return cmd;
        }
        if (description) {
            // A native --profile in the engine arguments (or a model without [presets]) keeps its passthrough (no preset injection), but the explicit
            // placement is still validated: the description stays available independently of argument injection.
            if (!req.profile && !req.print_effective_config && (native_profile_arguments(rest) || !description->find("presets"))) {
                if (const auto word = native_profile_word(rest); !word.empty()) {
                    try { query.tier = profile::parse_tier(word); } catch (const std::invalid_argument&) {}   // the engine rejects it
                }
                cmd.placement = profile::multi_device::resolve(*description, query);
            } else {
                cmd.preset = profile::resolve_model_preset(*description, req.profile, route.backend, query);
                cmd.preset->append_options(cmd.args);
                cmd.placement = cmd.preset->placement;
            }
            if (native_pair) cmd.placement.reset();
        }
        if (req.print_effective_config) cmd.args.emplace_back("--print-effective-config");
        return cmd;
    }
    if (!registered) invalid("no registered " + std::string(serving ? "serve" : "run") + " route for family " + entry.family);
    if (installed_refusal || !supported) invalid(std::move(refusal));
    invalid("family " + entry.family + " has no matching executable next to strix in this package");
}

int launch(const Request& request, const std::vector<model_registry::Entry>& entries,
           const model_description::ModelDescription* description, const std::filesystem::path& bin,
           const std::filesystem::path& data, const DoctorProbes& probes, const ExecutableExists& exists,
           std::ostream& out) {
    auto req = request;
    const auto& entry = model(entries, req.model);
    if (description && req.asset.empty() && req.args.empty() && alias(entry, req.model)) {
        auto local = profile::load_model_launch(".", entry.name, req.verb == Verb::serve ? "serve" : "run");
        req.asset = std::move(local.asset);
        req.args = std::move(local.args);
        if (req.backend.empty()) req.backend = std::move(local.backend);
    }
    if (req.print_effective_config && !req.dry_run && description && req.asset.empty() &&
        req.args.empty() && alias(entry, req.model)) {
        out << profile::resolve_model_preset(*description, req.profile, req.backend.empty() ? "hip" : req.backend,
                                             placement_query(req)).config.dump() << '\n';
        return 0;
    }
    auto command = resolve(req, entries, support::load(data / "support/matrix.toml"), probes, bin, exists, description);
    if (entry.family == "hy4_flash") {
        if (req.verb == Verb::serve) command.args.insert(command.args.begin(), {"--registry-root", data.string()});
        else command.args.insert(command.args.end(), {"--pair-model", (data / entry.description).string()});
    }
    if (command.preset && req.print_effective_config)
        out << command.preset->config.dump() << '\n' << std::flush;
    if (req.dry_run) {
        // Inspection reports the placement and its executor; a command that cannot implement it is the worker command.
        if (command.placement) {
            const auto form = command.placement->form;
            out << "# placement: " << profile::multi_device::form_name(form) << " (" << command.placement->rule
                << ") ranks=" << command.placement->ranks << " executor="
                << profile::multi_device::executed_by(form) << '\n';
            if (profile::multi_device::executed_by(form) != "engine")
                out << "# worker command (this entry point cannot execute the placement):\n";
        }
        out << command_line(command) << '\n';
        return 0;
    }
    // Inspection (--print-effective-config: the engine prints and exits, --dry-run) reports any placement; a launch
    // executes one engine, so it is admitted only for the form one engine executes.
    if (command.placement && !req.print_effective_config)
        profile::multi_device::admit(*command.placement, profile::multi_device::Executor::engine);
    std::optional<profile::PresetScope> scope;
    if (command.preset) scope.emplace();
    return execute(command);
}

int dispatch(const Request& req, const std::filesystem::path& bin, const std::filesystem::path& data,
             const DoctorProbes& probes, const ExecutableExists& exists, std::ostream& out) {
    switch (req.verb) {
    case Verb::help: out << usage; return 0;
    case Verb::version: out << build_info_json() << '\n'; return 0;
    case Verb::doctor: out << doctor(probes, support::load(data / "support/matrix.toml"), req.json); return 0;
    case Verb::support: {
        const auto matrix = support::load(data / "support/matrix.toml");
        const auto rows = support::query(matrix, req.support_filter);
        out << (req.json ? support::json(rows, matrix.consumers) : support::markdown(rows, matrix.consumers));
        return 0;
    }
    case Verb::update:
    case Verb::rollback: out << "not available in this release\n"; return 2;
    case Verb::devices: {
        const auto result = devices();
        out << (req.json ? result.dump() : result.dump_spaced()) << '\n';
        return 0;
    }
    default: break;
    }
    const auto entries = model_registry::load((data / "models/registry.toml").string());
    if (req.verb == Verb::models) {
        serve::Json models = serve::Json::array();
        for (const auto& entry : entries) {
            if (req.json) models.push(entry_json(entry));
            else out << entry.name << "\t" << entry.family << '\n';
        }
        if (req.json) out << models.dump() << '\n';
        return 0;
    }
    if (req.verb == Verb::inspect || req.verb == Verb::plan) {
        const auto& entry = model(entries, req.model);
        if (req.verb == Verb::plan) {
            const auto result = plan(entry, req);
            out << result.dump() << '\n';
            return result.find("status")->as_string() == "planned" ? 0 : 2;
        }
        const auto description = (data / entry.description).lexically_normal();
        if (description.parent_path() != (data / "models").lexically_normal())
            invalid("registry description must be a file under models/");
        auto result = entry_json(entry);
        result.set("metadata", describe(entry, description));
        out << (req.json ? result.dump() : result.dump_spaced()) << '\n';
        return 0;
    }
    const auto& entry = model(entries, req.model);
    const auto path = (data / entry.description).lexically_normal();
    if (path.parent_path() != (data / "models").lexically_normal())
        invalid("registry description must be a file under models/");
    const auto description = model_description::load(path.string());
    const auto* name = description.find("name");
    if (!name || std::get_if<std::string>(&name->data) == nullptr || std::get<std::string>(name->data) != entry.name)
        invalid("description name disagrees with the registry for " + entry.name);
    // The description always reaches launch: placement validation does not depend on preset argument injection.
    return launch(req, entries, &description, bin, data, probes, exists, out);
}

} // namespace strix::cli
