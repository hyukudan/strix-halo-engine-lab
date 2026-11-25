#include "strix/profile/catalog_levers.hpp"

#include "cuda/models/glm/producer/profile.hpp"
#include "strix/profiles/ds41_rtx.hpp"
#include "strix/profiles/gemma4.hpp"
#include "strix/profiles/qwen_cuda.hpp"
#include "strix/serve_cli_parity.hpp"

#include <algorithm>
#include <array>
#include <iostream>
#include <limits>
#include <set>

namespace {
namespace p = strix::profile;
namespace q = strix::qwen::profile;
namespace qc = strix::qwen::cuda_profile;
namespace g4 = strix::gemma4::profile;
namespace ds = strix::ds41::rtx::profile;
namespace glm = strix::glm::rtxpf::profile;
namespace ex = strix::exec;
using strix::lever::Exactness;

constexpr std::array tier_words{"exact", "fast", "turbo", "quality", "default"};

void check(bool condition, std::string_view message) {
    if (!condition) throw std::runtime_error(std::string(message));
}

p::Lookup env(p::Pairs values = {}) {
    return [values = std::move(values)](const char* key) -> const char* {
        for (const auto& [name, value] : values)
            if (name == key) return value.c_str();
        return nullptr;
    };
}

struct Snapshot {
    std::vector<std::string> fields;
    bool refused = false;
    bool operator==(const Snapshot&) const = default;
};

template <class Resolution>
Snapshot snapshot(const Resolution& resolved) {
    Snapshot out;
    auto& fields = out.fields;
    fields.push_back(resolved.json());
    if constexpr (requires { resolved.notices(); }) {
        const auto notices = resolved.notices();
        fields.insert(fields.end(), notices.begin(), notices.end());
    }
    if constexpr (requires { resolved.fingerprint(); }) {
        fields.push_back(p::hex_identity(resolved.fingerprint()));
        fields.push_back(resolved.cache_namespace());
        fields.push_back(resolved.weights.describe());
    }
    for (const auto& entry : resolved.entries) {
        fields.emplace_back(entry.lever->name);
        fields.push_back(entry.value);
        fields.push_back(std::to_string(static_cast<int>(entry.status)));
        if constexpr (requires { entry.target; }) fields.push_back(entry.target);
        if constexpr (requires { entry.wanted; }) fields.push_back(entry.wanted);
        if constexpr (requires { entry.requested; }) {
            fields.push_back(entry.requested);
            fields.push_back(entry.reason);
            fields.push_back(entry.cli_override ? "cli" : "non-cli");
        }
    }
    return out;
}

template <class Work>
Snapshot capture(Work&& work) {
    try {
        return snapshot(work());
    } catch (const std::invalid_argument& error) {
        return {{error.what()}, true};
    }
}

struct Comparison {
    unsigned resolved = 0, refused = 0;

    template <class Row, class Native>
    void run(std::string_view label, p::Name tier, std::span<const Row> rows, Native&& native) {
        // At this base there are no D4 bindings for any incumbent override.
        const auto expected = capture([&] { return native(rows); });
        std::vector<Row> storage;
        const auto actual = capture([&] {
            return p::resolve_effective_profile(tier, rows, {}, storage, native);
        });
        check(actual == expected, label);
        if (expected.refused) ++refused;
        else ++resolved;
    }

    void report(std::string_view engine) const {
        check(resolved > 0 && refused > 0, "expected both resolutions and explicit refusals");
        std::cout << "profile_table_identity_" << engine << " PASS (" << resolved << " resolutions, "
                  << refused << " refusals)\n";
    }
};

template <class Row>
std::set<std::string_view> switch_values(const Row& row) {
    std::set<std::string_view> values{std::string_view{}, "off", row.builtin};
    for (const auto value : row.value)
        if (!value.empty()) values.insert(value);
    if constexpr (requires { row.off; }) {
        if (!row.off.empty()) values.insert(row.off);
    }
    return values;
}

void qwen(bool cuda) {
    Comparison compare;
    const auto rows = cuda ? qc::levers() : q::levers();
    for (const auto word : tier_words) {
        const auto tier = p::parse(word);
        q::Inputs inputs;
        inputs.env = env();
        if (cuda) inputs.caps = &q::kCudaWeightCaps;
        else inputs.regimes = q::fast_regimes;
        const auto run = [&](std::string_view which, const q::Inputs& in) {
            compare.run(std::string(word) + " " + std::string(which), tier, rows, [&](auto candidate) {
                auto resolved = q::resolve(tier, in, candidate);
                if (cuda) (void)qc::choose(resolved);
                return resolved;
            });
        };
        for (const std::uint64_t prompt : {0U, 1U, 8192U, 8193U}) {
            inputs.max_prompt_tokens = prompt;
            run("prompt regime", inputs);
        }
        inputs.allow_pending = true;
        run("--allow-pending", inputs);
        inputs.allow_pending = false;
        inputs.env = env({{"STRIX_QWEN_PROFILE_ALLOW_PENDING", "1"}});
        run("STRIX_QWEN_PROFILE_ALLOW_PENDING", inputs);
        inputs.env = env();
        for (const auto& row : rows) {
            for (auto value : switch_values(row)) {
                if (!row.env.empty()) {
                    auto in = inputs;
                    in.env = env({{std::string(row.env), std::string(value)}});
                    run(row.env, in);
                }
                if (!row.option.empty()) {
                    auto in = inputs;
                    in.cli.emplace_back(row.name, value);
                    if (!row.env.empty()) in.env = env({{std::string(row.env), "off"}});
                    run(row.option, in);
                }
            }
            if (row.env.empty()) continue;
            const auto column = p::value_column(row, tier);
            const auto value = row.value[column];
            if (value.empty()) continue;
            const auto status = p::admit_column(row.ready[column], row.runnable[column], false, row.provisional[column]);
            auto in = inputs;
            in.env = env({{std::string(row.env), std::string(value)},
                          {q::marker_key(row), q::marker(tier, status, value)}});
            run("inherited profile export", in);
        }
        auto weights = inputs;
        weights.weights = {.source = "onfly", .bits = "4", .pack = {}};
        run("--turbo-weights --turbo-weight-bits", weights);
        weights.weights = {.source = "converted", .bits = "3", .pack = "fixture.pack"};
        run("--turbo-weights-pack", weights);
        weights.weights = {};
        weights.env = env({{std::string(p::turbo::kWeightsEnv), "converted"}, {std::string(p::turbo::kBitsEnv), "3"},
                           {std::string(p::turbo::kPackEnv), "fixture.pack"}});
        run("turbo weight environment", weights);
        weights.weights.source = "invalid";
        run("invalid turbo weight source", weights);
    }
    compare.report(cuda ? "qwen_cuda" : "qwen_hip");
}

void gemma() {
    Comparison compare;
    const auto rows = g4::levers();
    for (const auto word : tier_words) {
        const auto tier = p::parse(word);
        for (const auto app : {g4::App::engine, g4::App::decode_bench, g4::App::prefill_bench}) {
            const auto run = [&](std::string_view label, const g4::Inputs& in) {
                compare.run(label, tier, rows, [&](auto candidate) { return g4::resolve(tier, app, in, candidate); });
            };
            g4::Inputs inputs;
            inputs.env = env();
            run(word, inputs);
            for (const auto& row : rows)
                if (row.asset) inputs.assets.emplace_back(row.name, "fixture.gguf");
            run("discovered assets", inputs);
            inputs.allow_pending = true;
            run("--allow-pending", inputs);
            inputs.allow_pending = false;
            inputs.env = env({{"STRIX_G4_PROFILE_ALLOW_PENDING", "1"}});
            run("STRIX_G4_PROFILE_ALLOW_PENDING", inputs);
            inputs.env = env();
            if (app == g4::App::engine) {
                for (const auto flag : {"--draft", "--draft-eagle", "--no-mtp", "STRIX_MTP"}) {
                    auto in = inputs;
                    const std::string_view spelling = flag;
                    in.cli.emplace_back("draft", spelling == "--draft" || spelling == "--draft-eagle" ?
                        "override.gguf" : "off");
                    run(flag, in);
                }
            }
            for (const auto& row : rows) {
                if (!(row.apps & static_cast<unsigned>(app))) continue;
                for (auto value : switch_values(row)) {
                    if (!row.env.empty()) {
                        auto in = inputs;
                        in.env = env({{std::string(row.env), std::string(value)}});
                        run(row.env, in);
                    }
                    if (!row.option.empty()) {
                        auto in = inputs;
                        in.cli.emplace_back(row.name, row.asset && value != "off" ? "override.gguf" : std::string(value));
                        if (!row.env.empty()) in.env = env({{std::string(row.env), "off"}});
                        run(row.option, in);
                    }
                }
            }
        }
    }
    compare.report("gemma4");
}

template <class Inputs, class Row, class App, class Resolve>
void rtx(std::string_view engine, std::span<const Row> rows, std::span<const App> apps,
         Resolve native, std::string_view tier_env, std::string_view pending_env, std::string_view baseline_env = {}) {
    Comparison compare;
    for (const auto word : tier_words) {
        const auto tier = p::parse(word);
        for (const auto app : apps) {
            const auto run = [&](std::string_view label, const Inputs& in) {
                compare.run(label, tier, rows, [&](auto candidate) { return native(app, in, candidate); });
            };
            const Inputs cli{{"--profile", word}, env()};
            run(word, cli);
            run(tier_env, {{}, env({{std::string(tier_env), word}})});
            auto in = cli;
            in.args.emplace_back("--allow-pending");
            run("--allow-pending", in);
            in = cli;
            in.env = env({{std::string(pending_env), "1"}});
            run(pending_env, in);
            if (!baseline_env.empty()) {
                in = cli;
                in.args.emplace_back("--baseline");
                run("--baseline", in);
                in = cli;
                in.env = env({{std::string(baseline_env), "1"}});
                run(baseline_env, in);
            }
            for (const auto& row : rows) {
                if (!(row.apps & static_cast<unsigned>(app))) continue;
                if (!row.env.empty()) {
                    for (const auto value : switch_values(row)) {
                        in = cli;
                        in.env = env({{std::string(row.env), std::string(value)}});
                        run(row.env, in);
                    }
                }
                for (const auto& option : row.options) {
                    if (option.flag.empty()) continue;
                    in = cli;
                    in.args.emplace_back(option.flag);
                    if (option.value == "*") in.args.emplace_back(row.builtin.empty() ? "1" : row.builtin);
                    if (!row.env.empty()) in.env = env({{std::string(row.env), "off"}});
                    run(option.flag, in);
                    if (baseline_env.empty()) continue;
                    auto earlier = in;
                    earlier.args.insert(earlier.args.begin(), "--baseline");
                    run("baseline before alias", earlier);
                    in.args.emplace_back("--baseline");
                    run("baseline after alias", in);
                }
            }
        }
    }
    compare.report(engine);
}

void ds41() {
    constexpr std::array apps{ds::App::replica, ds::App::prefill_service};
    rtx<ds::Inputs>("ds41_rtx", ds::levers(), std::span<const ds::App>(apps),
        [](auto app, const auto& in, auto rows) { return ds::resolve(app, in, rows); },
        "STRIX_DS41_PROFILE", "STRIX_DS41_PROFILE_ALLOW_PENDING", "STRIX_DS41_RTX_BASELINE");
}

void glm_cuda() {
    constexpr std::array apps{glm::App::decode_service};
    rtx<glm::Inputs>("glm_cuda", glm::levers(), std::span<const glm::App>(apps),
        [](auto app, const auto& in, auto rows) { return glm::resolve(app, in, rows); },
        "STRIX_GLM_RTX_PROFILE", "STRIX_GLM_RTX_PROFILE_ALLOW_PENDING");
}

void catalogue() {
    const auto builtins = ex::Catalog::builtin_entries(ex::Backend::hip);
    check(!builtins.empty(), "expected D4 HIP entries");
    auto reference = builtins.front(), fast = reference, turbo = reference, unrelated = reference;
    fast.key.prologue = 1;
    fast.identity = "fast-v1";
    fast.exact = Exactness::bit_changing;
    turbo.key.prologue = 2;
    turbo.identity = "turbo-v1";
    turbo.exact = Exactness::bit_changing;
    unrelated.key.prologue = 3;
    unrelated.identity = "unrelated-v1";
    const std::array entries{reference, fast, turbo, unrelated};
    ex::Catalog kernels;
    kernels.add(entries);
    strix::device::Profile device{};
    device.name = "fixture";
    device.arch = "gfx1151";
    const std::array sites{
        p::CatalogSite{"projection", reference.key, reference.identity},
        p::CatalogSite{"projection", fast.key, reference.identity},
        p::CatalogSite{"projection", turbo.key, reference.identity},
        p::CatalogSite{"projection", unrelated.key, reference.identity},
        p::CatalogSite{"projection", fast.key, reference.identity},
    };
    const std::array turbo_entries{turbo.identity};
    const auto table = p::levers_of(kernels, device, sites, turbo_entries);
    check(table.rows.size() == 3, "expected reference and two distinct variants");
    check(p::levers_of(ex::Catalog::builtin(ex::Backend::hip), device, sites).rows.size() == 1,
          "real D4 catalogue has no tier-changing fixture variants");
    auto wrong_device = device;
    wrong_device.arch = "sm_120";
    check(p::levers_of(kernels, wrong_device, sites).rows.empty(), "architecture filter");

    constexpr std::string_view fast_name = "projection/fast-v1", turbo_name = "projection/turbo-v1";
    std::array classes{p::ClassVerdict{"code", "short", p::GateVerdict::pass, 800}};
    std::array gates{
        p::GateRecord{fast_name, "fixture-model", "held-out", p::GateVerdict::pass, classes},
        p::GateRecord{turbo_name, "fixture-model", "held-out", p::GateVerdict::pass, classes, p::Name::turbo},
    };
    std::array regimes{
        p::RegimeRecord{fast_name, device.name, {strix::lever::Axis::rows, 8, false, true}, 0.001, "fast-receipt"},
        p::RegimeRecord{turbo_name, device.name, {strix::lever::Axis::rows, 8, false, true}, 0.002, "turbo-receipt"},
    };
    p::CatalogInputs inputs{.options = {}, .regime = {.rows = 8}, .model = "fixture-model",
                            .gate_set = "held-out", .content_class = "code", .band = "short"};
    const auto run = [&] { return p::resolve(table, inputs, gates, regimes); };
    const auto expect = [&](bool fast_on, bool turbo_on, std::string_view label) {
        const auto result = run();
        check(result.entries[0].on && result.entries[0].reason == p::CatalogReason::reference, "reference class");
        check(result.entries[1].on == fast_on && result.entries[2].on == turbo_on, label);
    };
    expect(false, false, "exact selects reference");
    inputs.options.cli.emplace_back(fast_name, "1");
    expect(false, false, "explicit request cannot change EXACT");
    inputs.options.cli.clear();
    inputs.options.profile = p::Name::fast;
    expect(true, false, "FAST needs gate and a measured gain");
    const auto qualified = run();
    check(qualified.entries[1].receipt == "fast-receipt", "qualification provenance");

    inputs.options.profile = p::Name::turbo;
    expect(true, true, "TURBO extends FAST");
    gates[1].tier = p::Name::fast;
    expect(true, false, "FAST-limit record does not qualify TURBO");
    check(run().describe().find("\"gate\":\"not-run\"") != std::string::npos, "missing TURBO record reported");
    constexpr std::array turbo_provisional{turbo_name};
    inputs.provisional = turbo_provisional;
    inputs.options.cli.emplace_back(turbo_name, "1");
    expect(true, true, "missing TURBO record allows a declared provisional opt-in");
    check(run().entries[2].provisional && run().entries[2].gate_not_run, "explicit TURBO gate pending receipt");
    gates[1].tier = p::Name::turbo;
    for (const auto verdict : {p::GateVerdict::fail, p::GateVerdict::hold}) {
        gates[1].verdict = verdict;
        expect(true, false, "TURBO FAIL and HOLD block even provisional opt-ins");
    }
    inputs.provisional = {};
    inputs.options.cli.clear();
    gates[1].verdict = p::GateVerdict::unresolved;
    expect(true, false, "TURBO UNRESOLVED leaves FAST in effect");
    gates[1].verdict = p::GateVerdict::pass;
    regimes[1].gain = 0;
    expect(true, false, "TURBO needs a gain over FAST");
    regimes[1].gain = std::numeric_limits<double>::infinity();
    expect(true, false, "TURBO needs a finite gain");
    regimes[1].gain = 0.002;
    inputs.options.profile = p::Name::fast;
    inputs.regime.rows = 7;
    expect(false, false, "FAST outside measured regime");
    inputs.regime.rows = 8;
    regimes[0].device = "other";
    expect(false, false, "receipt device must match");
    regimes[0].device = device.name;
    inputs.band = "long";
    expect(false, false, "gate band must match");
    inputs.band = "short";
    inputs.gate_set = "other";
    expect(false, false, "gate set must match");
    inputs.gate_set = "held-out";
    classes[0].rows = 799;
    expect(false, false, "small content class remains provisional");
    constexpr std::array provisional{fast_name};
    inputs.provisional = provisional;
    inputs.options.cli.emplace_back(fast_name, "1");
    expect(true, false, "declared provisional opt-in");
    check(run().entries[1].provisional, "provisional result is stamped");
    gates[0].verdict = p::GateVerdict::hold;
    expect(false, false, "HOLD refuses explicit provisional");
    gates[0].verdict = p::GateVerdict::fail;
    expect(false, false, "FAIL refuses explicit provisional");
    gates[0].verdict = p::GateVerdict::pass;
    classes[0].rows = 800;
    inputs.regime.rows = 7;
    expect(true, false, "provisional opt-in can measure a new regime");
    check(run().entries[1].provisional, "unmeasured regime is stamped provisional");
    inputs.regime.rows = 8;
    inputs.provisional = {};
    inputs.force_off = provisional;
    expect(false, false, "model forced off");
    inputs.force_off = {};

    constexpr std::array<std::string_view, 1> request_options{"--prompt"}, device_options{"--device"};
    const auto options = p::options_table(table, request_options, device_options);
    check(options.size() == 9 && options.back().source == p::OptionSource::device, "generated option sources");
    const auto option = std::ranges::find(options, fast_name, &p::CatalogOption::lever);
    check(option != options.end() && option->source == p::OptionSource::lever, "generated lever option");
    const auto& fast_option = option->option;
    std::vector<std::string> cli{"--profile", "fast", fast_option, "1", fast_option + "=0", "--prompt", "text"};
    const auto server_result = [&] {
        inputs.options.cli.emplace_back(fast_name, "0");
        return run();
    }();
    inputs.options = {};
    p::take_catalog_options(options, cli, inputs.options);
    const auto cli_result = run();
    const auto parity = strix::parity::compare({{"profile", cli_result.describe()}},
                                              {{"profile", server_result.describe()}}, {});
    check(parity.ok() && !cli_result.entries[1].on,
          "CLI and structured request use the same decision, with last explicit switch winning");
    check(cli_result.describe().find("\"off\":\"0\"") != std::string::npos, "describe prints the off switch");

    // Only the fixture has a catalogue-to-native binding. Exercise both arms
    // with today's effective profile function and verify an untouched row too.
    constexpr std::array bindings{p::CatalogBinding{fast_name, "index_proj_wmma", "1", "0"}};
    p::LadderRow fixture_timing{"index_proj_wmma", "1", p::Name::fast, p::Phase::prefill, "fixture",
                                8, 8, 1, 1, 1.01, 0.005, "fixture"};
    q::Inputs native_inputs;
    native_inputs.env = env();
    native_inputs.regime = {p::Phase::prefill, 8, 1, "fixture", "fixture-exec"};
    const auto native = [&](auto rows) { return q::resolve(p::Name::fast, native_inputs, rows); };
    std::vector<q::Lever> storage;
    const auto off_cells = p::effective_cells(cli_result, bindings);
    const auto on_cells = p::effective_cells(qualified, bindings);
    // The row is measured over the execution that runs under the catalogue's FAST cell (LadderArms::over).
    const auto unmeasured = p::resolve_effective_profile(p::Name::fast, q::levers(), on_cells, storage, native);
    const std::string native_reference =
        unmeasured.find("index_proj_wmma") ? unmeasured.find("index_proj_wmma")->ladder_reference : std::string();
    check(native_reference.starts_with("ladder-arm/v4;"), "the native adapter reports the reference of the cell");
    fixture_timing.over = native_reference;
    native_inputs.ladder = std::span(&fixture_timing, 1);
    const auto off = p::resolve_effective_profile(p::Name::fast, q::levers(), off_cells, storage, native);
    check(off.find("index_proj_wmma") && off.find("index_proj_wmma")->value == "0", "catalogue off reaches native");
    const auto on = p::resolve_effective_profile(p::Name::fast, q::levers(), on_cells, storage, native);
    check(on.find("index_proj_wmma") && on.find("index_proj_wmma")->value == "1", "catalogue on reaches native");
    const auto incumbent = native(q::levers());
    const auto* projected_budget = on.find("qsa_budget");
    const auto* legacy_budget = incumbent.find("qsa_budget");
    check(projected_budget && legacy_budget && projected_budget->value == legacy_budget->value,
          "unbound native row preserved");
    std::cout << "profile_catalog_serve_cli_parity PASS\n";
}

} // namespace

int main(int argc, char** argv) {
    try {
        check(argc == 2, "expected profile fixture name");
        const std::string_view name = argv[1];
        if (name == "qwen_hip") qwen(false);
        else if (name == "qwen_cuda") qwen(true);
        else if (name == "gemma4") gemma();
        else if (name == "ds41_rtx") ds41();
        else if (name == "glm_cuda") glm_cuda();
        else if (name == "catalogue") catalogue();
        else throw std::invalid_argument("unknown profile fixture");
        return 0;
    } catch (const std::exception& error) {
        std::cerr << error.what() << '\n';
        return 1;
    }
}
