#pragma once
// Qwen 27B dense HIP profile binding for the decode and speculative CLIs.
// Both use the shared profile resolver with the route their native parser selected.
#include "strix/engine_profile.hpp"
#include "strix/qwen35/prefill_windows.hpp"

#include <cstdio>
#include <span>
#include <string>
#include <string_view>
#include <vector>

namespace strix::qwen35::hip_profile {
namespace sp = ::strix::profile;

enum class App { decode, spec };

inline constexpr std::string_view kAllowEnv = "STRIX_QWEN35_HIP_PROFILE_ALLOW_PENDING";
inline constexpr std::string_view kExactWindow = "engine default (64-row EXACT windows, IU4)";
inline constexpr std::string_view kLayersEnv = "STRIX_DENSE_A4W4_LAYERS";
inline constexpr std::string_view kPending = "final gate v2 content-class verdict and matching timing required";
inline constexpr std::string_view kEvidence = "docs/models/qwen3.8-27b/README.md#a4w4-layer-table-hip-fast";

// The historical 8192-token gate covers one held-out text. The two settings form one
// composite candidate, available through explicit controls. Neither cell has a production grant,
// including with --allow-pending. An empty TURBO column inherits the resolved FAST route.
inline constexpr sp::Lever table[] = {
    {.name = "wide_rows", .env = "", .option = "--wide", .value = {"", "512", "", ""}, .builtin = kExactWindow,
     .kind = sp::Kind::bit_changing, .ready = {true, false, true, false}, .runnable = {true, false, true, false},
     .target = kPending, .off = "16", .source = kEvidence},
    {.name = "a4w4_layers", .env = kLayersEnv, .option = "", .value = {"", "48-63", "", ""}, .builtin = "none",
     .kind = sp::Kind::bit_changing, .ready = {true, false, true, false}, .runnable = {true, false, true, false},
     .target = kPending, .off = "none", .source = kEvidence},
    {.name = "prefill_rows", .env = "", .option = "--prefill-rows", .value = {"", "512", "", ""}, .builtin = kExactWindow,
     .kind = sp::Kind::bit_changing, .ready = {true, false, true, false}, .runnable = {true, false, true, false},
     .target = kPending, .off = "16", .source = kEvidence},
};

inline constexpr sp::Engine kDecode{"qwen35-hip-decode", kAllowEnv, std::span(table).first<2>(), false, "qwen35-hip-turbo-v1"};
inline constexpr sp::Engine kSpec{"qwen35-hip-spec", kAllowEnv, std::span(table).last<2>(), false, "qwen35-hip-turbo-v1"};

[[nodiscard]] inline constexpr const sp::Engine& engine(App app) {
    return app == App::decode ? kDecode : kSpec;
}

[[nodiscard]] inline constexpr std::string_view window_lever(App app) {
    return app == App::decode ? "wide_rows" : "prefill_rows";
}

template <class NextRows>
bool take_window_flag(App app, std::string_view arg, const NextRows& next_rows, PrefillFlags& flags) {
    if (app == App::decode && (arg == "--wide" || arg == "--wide-exact")) {
        flags.width = next_rows();
        flags.exact = arg == "--wide-exact";
        return true;
    }
    if (app == App::spec && arg == "--prefill-rows") {
        flags.width = next_rows();
        return true;
    }
    if (app == App::spec && arg == "--prefill-exact") {
        flags.exact = true;
        return true;
    }
    return false;
}

// Feed the native parser's final width, exact modifier and diagnostic route to the resolver.
// This also handles ordered --wide/--wide-exact overrides and the spec app's valueless --prefill-exact.
[[nodiscard]] inline sp::Resolution resolve_prefill(App app, sp::Inputs inputs, const PrefillFlags& flags,
                                                     std::uint32_t max_rows) {
    const auto window = window_lever(app);
    const auto diagnostic_rows = flags.row_path() ? (flags.rows ? flags.rows : 1U) : 0U;
    inputs.cli.clear();  // the parsed width supersedes the shared option scanner's spelling
    if (flags.width || diagnostic_rows)
        inputs.cli.emplace_back(window, std::to_string(flags.width ? flags.width : diagnostic_rows));

    auto resolution = sp::resolve(engine(app), inputs);
    const char* master = sp::environment(inputs.env, "STRIX_DENSE_A4W4");
    for (auto& entry : resolution.entries) {
        if (entry.lever->name == window) {
            if (diagnostic_rows) entry.reason = "explicit decode row path";
            else if (flags.exact) {
                entry.status = sp::Status::explicit_cli;
                entry.target.clear();
                entry.reason = "explicit exact window";
            } else if (flags.width) entry.reason = "explicit window";
        } else {
            const bool listed = entry.status == sp::Status::explicit_env && !entry.value.empty();
            const bool wide = !flags.exact && flags.width > max_rows;
            if (!wide || !::strix::envcfg::parse_flag(master, listed)) {
                if (entry.status == sp::Status::explicit_env) {
                    entry.value = entry.lever->builtin;
                    entry.status = sp::Status::off;
                }
            } else if (!listed && entry.status != sp::Status::off) {
                entry.value = "all";
                entry.requested = entry.value;
                entry.status = sp::Status::explicit_env;
                entry.target.clear();
            }
        }
    }
    return resolution;
}

struct Command {
    sp::Inputs inputs;
    std::vector<std::string> args;

    Command(App app, int argc, char** argv) {
        for (int i = 1; i < argc; ++i) args.emplace_back(argv[i]);
        sp::take_options(engine(app), args, inputs);
    }
};

[[nodiscard]] inline std::string effective_config(App app, const sp::Resolution& resolution) {
    auto json = resolution.json();
    const auto* window = resolution.find(window_lever(app));
    if (window && !window->reason.empty()) {
        json.pop_back();
        json += ",\"prefill_route\":" + sp::json_string(window->reason) + "}";
    }
    return json;
}

// Complete resolution before loading weights or reading cached arithmetic switches.
inline bool finish(App app, const sp::Resolution& resolution, bool inspect) {
    if (inspect) {
        std::puts(effective_config(app, resolution).c_str());
        return true;
    }
    sp::apply(resolution);
    for (const auto& entry : resolution.entries) {
        if (entry.status == sp::Status::pending)
            std::fprintf(stderr, "profile %s: %s pending (%s); %s runs\n", resolution.word(),
                         std::string(entry.lever->name).c_str(), entry.target.c_str(), entry.value.c_str());
    }
    return false;
}
}  // namespace strix::qwen35::hip_profile
