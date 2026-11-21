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
    "       strix models list [--json] | inspect MODEL [--json]\n"
    "       strix plan MODEL --context N [--devices FILE] --json\n"
    "       strix run MODEL [--dry-run] [-- ARGS]\n"
    "       strix serve MODEL [--host H --port P] [--dry-run] [-- ARGS]\n"
    "       strix update | rollback\n";

struct Route {
    std::string_view family, source, binary;
    bool serving;
    std::string_view model_flag = "", serve_flag = "", host_flag = "", port_flag = "";
    enum Pair { none, ranked, positional, world } pair = none;
    bool rtx_assets = false;
};
// Use a route only when its source entry occurs in the selected registry row.
constexpr Route routes[] = {
    {"qwen", "apps/hip_decode_token_main.cpp", "strix-hip-decode-token", false},
    {"qwen", "apps/hip_decode_token_main.cpp", "strix-hip-decode-token", true,
        "", "--http-serve", "--http-bind", "--http-port"},
    {"qwen", "apps/cuda_qwen_serve_main.cu", "strix-cuda-qwen-serve", true, "--model", "", "--bind", "--port"},
    {"deepseek41", "apps/deepseek41_serve_main.cpp", "strix-deepseek41-serve", false,
        "", "", "", "", Route::ranked},
    {"deepseek41", "apps/deepseek41_serve_main.cpp", "strix-deepseek41-serve", true,
        "", "", "--http-bind", "--http-port", Route::ranked},
    {"deepseek41", "apps/cuda_ds41_replica.cu", "strix-cuda-ds41-replica", false,
        "", "", "", "", Route::none, true},
    {"minimax", "apps/minimax_rank_main.cpp", "strix-minimax-rank", false,
        "", "", "", "", Route::positional},
    {"minimax", "apps/minimax_serve_main.cpp", "strix-minimax-serve", true,
        "", "", "--bind", "--port", Route::ranked},
    {"glm5", "apps/glm_rank_main.cpp", "strix-glm5-rank", false, "", "", "", "", Route::world},
    {"glm5", "apps/glm_serve_main.cpp", "strix-glm5-serve", true, "", "", "--host", "--port", Route::world},
    {"gemma4", "apps/gemma4_decode_main.cpp", "strix-gemma4-decode", false},
    {"gemma4", "apps/gemma4_serve_main.cpp", "strix-gemma4-serve", true, "", "", "--bind", "--port"},
    {"qwen35", "apps/hip_qwen35_decode_main.hip", "strix-hip-qwen35-decode", false, "--model"},
    {"qwen35", "apps/cuda_qwen35_decode_main.cu", "strix-cuda-qwen35-decode", false, "--model"},
};

[[noreturn]] void invalid(std::string reason) { throw std::invalid_argument(std::move(reason)); }

bool value_token(std::string_view value) { return !value.empty() && value.front() != '-'; }

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

Request parse(const std::vector<std::string>& args) {
    Request r;
    if (args.empty()) return r;
    const auto& verb = args.front();
    std::size_t i = 1;
    if (verb == "--help" || verb == "-h" || verb == "help") r.verb = Verb::help;
    else if (verb == "--version" || verb == "version") r.verb = Verb::version;
    else if (verb == "doctor") r.verb = Verb::doctor;
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
        r.verb == Verb::models || r.verb == Verb::inspect || r.verb == Verb::plan;
    if (launch || r.verb == Verb::inspect || r.verb == Verb::plan) {
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
        else if (launch && arg == "--dry-run" && !r.dry_run) r.dry_run = true;
        else if (r.verb == Verb::serve && arg == "--host" && r.host.empty()) r.host = value();
        else if (r.verb == Verb::serve && arg == "--port" && r.port.empty()) {
            r.port = value();
            (void)number(r.port, "--port", 1, 65535);
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
                const std::filesystem::path& bin, const ExecutableExists& exists) {
    const auto& entry = model(entries, req.model);
    const bool serving = req.verb == Verb::serve;
    bool registered = false;
    for (const auto& route : routes) {
        if (route.family != entry.family || route.serving != serving ||
            std::ranges::find(entry.apps, route.source) == entry.apps.end()) continue;
        registered = true;
        auto executable = bin / route.binary;
#if defined(_WIN32)
        executable += ".exe";
#endif
        if (!exists(executable)) continue;
        auto rest = req.args;
        auto asset = req.model;
        if (alias(entry, asset)) {
            if (rest.empty() || !value_token(rest.front()))
                invalid("model '" + entry.name + "' requires an asset path as the first argument after --");
            asset = rest.front();
            rest.erase(rest.begin());
        }
        check_route(route, rest);
        Command cmd{executable, {}};
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
        } else if (serving && entry.family == "deepseek41" && option(rest, "--rank") == "0" &&
                   std::ranges::find(rest, "--http-port") == rest.end()) {
            cmd.args.insert(cmd.args.end(), {"--http-port", "8080"});
        }
        cmd.args.insert(cmd.args.end(), rest.begin(), rest.end());
        return cmd;
    }
    if (!registered) invalid("no registered " + std::string(serving ? "serve" : "run") + " route for family " + entry.family);
    invalid("family " + entry.family + " has no matching executable next to strix in this package");
}

int dispatch(const Request& req, const std::filesystem::path& bin, const std::filesystem::path& data,
             const DoctorProbes& probes, const ExecutableExists& exists, std::ostream& out) {
    switch (req.verb) {
    case Verb::help: out << usage; return 0;
    case Verb::version: out << build_info_json() << '\n'; return 0;
    case Verb::doctor: out << doctor(probes, req.json); return 0;
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
    const auto command = resolve(req, entries, bin, exists);
    if (req.dry_run) {
        out << command_line(command) << '\n';
        return 0;
    }
    return execute(command);
}

} // namespace strix::cli
