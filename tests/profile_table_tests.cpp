#include "cuda/models/glm/producer/profile.hpp"
#include "strix/device/profiles.hpp"
#include "strix/deepseek41_pair_defaults.hpp"
#include "strix/glm_pair_profile_table.hpp"
#include "strix/model_registry.hpp"
#include "strix/profiles/ds41_rtx.hpp"
#include "strix/profiles/ernie.hpp"
#include "strix/profiles/gemma4.hpp"
#include "strix/profiles/hy4.hpp"
#include "strix/profiles/inkling2_mini.hpp"
#include "strix/profiles/qwen_cuda.hpp"
#include "strix/profiles/qwen_dense_cuda.hpp"
#include "strix/profiles/qwen_dense_hip.hpp"
#include "minimax/engine/mm_defaults.hpp"

#include <vector>
#include <algorithm>
#include <array>
#include <iomanip>
#include <iostream>
#include <set>

namespace {
namespace p = strix::profile;
namespace q = strix::qwen::profile;
namespace qc = strix::qwen::cuda_profile;
namespace g4 = strix::gemma4::profile;
namespace ds = strix::ds41::rtx::profile;
namespace glm = strix::glm::rtxpf::profile;

constexpr std::array tier_words{"exact", "fast", "turbo", "quality", "default"};
bool dump = false;

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
        if constexpr (requires { entry.ladder_reference; }) fields.push_back(entry.ladder_reference);
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
        const auto observed = capture([&] { return native(rows); });
        if (dump) {
            std::cout << resolved + refused << '\t' << p::name(tier) << '\t' << std::quoted(label) << '\t'
                      << (observed.refused ? "refused" : "resolved");
            for (const auto& field : observed.fields) std::cout << '\t' << std::quoted(field);
            std::cout << '\n';
        }
        if (observed.refused) ++refused;
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
    // The producer currently leaves its bit-changing rows inaccessible to CLI/environment overrides.
    // Add switches to a copied table to exercise the binding without exposing new production options.
    const auto declared = glm::levers();
    const auto candidate = std::ranges::find(declared, "prefill_tile_large", &glm::Lever::name);
    check(candidate != declared.end(), "GLM producer pending tile row");
    auto row = *candidate;
    row.env = "STRIX_FIXTURE_TILE";
    row.options[0] = {"--fixture-tile", "*"};
    const std::array fixture{row};
    const auto unqualified = [&](const glm::Inputs& in, glm::Status source) {
        const auto resolution = glm::resolve(glm::App::decode_service, in, fixture);
        const auto* entry = resolution.find(row.name);
        check(entry && entry->value == "8192" && entry->status == source &&
            resolution.qualification() == "unqualified: prefill_tile_large" &&
            resolution.json().find("\"unqualified\":[\"prefill_tile_large\"]") != std::string::npos,
            "GLM producer explicit pending tile is unqualified and retains its source");
    };
    unqualified({{"--fixture-tile", "8192"}, env()}, glm::Status::explicit_cli);
    unqualified({{}, env({{std::string(row.env), "8192"}})}, glm::Status::explicit_env);
    check(glm::resolve(glm::App::decode_service, {{}, env({{std::string(row.env), "off"}})}, fixture)
              .qualification() == "qualified", "GLM producer explicit off is qualified");

    constexpr std::array apps{glm::App::decode_service};
    rtx<glm::Inputs>("glm_cuda", glm::levers(), std::span<const glm::App>(apps),
        [](auto app, const auto& in, auto rows) { return glm::resolve(app, in, rows); },
        "STRIX_GLM_RTX_PROFILE", "STRIX_GLM_RTX_PROFILE_ALLOW_PENDING");
}

// Registered product families, plus native sibling entry points reading the same model.
// The device catalogue supplies the product and architecture for each backend.
struct Binding { std::string_view tag, backend, route; };
constexpr Binding bindings[]{
    {"qwen", "hip", "decode"}, {"qwen", "cuda", "serve"},
    {"deepseek41-pair", "hip", "pair"}, {"deepseek41-pair", "cuda", "replica"},
    {"deepseek41-pair", "cuda", "prefill-service"},
    {"glm-pair", "hip", "pair"}, {"glm-pair", "cuda", "producer"},
    {"minimax-pair", "hip", "pair"},
    {"gemma4", "hip", "engine"},
    {"qwen35", "hip", "decode"}, {"qwen35", "hip", "spec"},
    {"qwen35", "cuda", "decode"}, {"qwen35", "cuda", "spec"},
    {"hy4", "hip", "rank"},
    {"inkling2-mini", "cpu", "decode"}, {"inkling2-mini", "hip", "decode"},
    {"inkling2-mini", "cuda", "decode"},
    {"ernie", "hip", "inspection"},
};

void matrix(const char* registry) {
    namespace qd = strix::qwen35;
    const auto no_env = env();
    for (const auto& model : strix::model_registry::load(registry)) {
        bool covered = false;
        for (const auto& binding : bindings) {
            if (model.profile_tag != binding.tag) continue;
            covered = true;
            for (const auto& device : strix::device::catalogue()) {
                if ((binding.backend == "hip" && !device.vendor.empty() && device.vendor != "amd") ||
                    (binding.backend == "cuda" && !device.vendor.empty() && device.vendor != "nvidia") ||
                    (binding.backend == "cpu" && !device.vendor.empty())) continue;
                for (const auto tier : {p::Name::standard, p::Name::fast, p::Name::turbo}) {
                    // Prefill regimes for every binding; the Gemma family also resolves its decode regimes (the TURBO
                    // decode-attention bands are timed at one lane, so four lanes at 32K must stay exact).
                    struct Case { p::Phase phase; unsigned context, lanes; };
                    std::vector<Case> cases;
                    for (const auto context : {0U, 2048U, 4096U, 8192U, 8193U, 16384U, 32768U})
                        cases.push_back({p::Phase::prefill, context, binding.route == "pair" ? 2U : 1U});
                    if (binding.tag == "gemma4") {
                        for (const auto context : {2048U, 4096U, 8191U, 8192U, 32768U, 65280U, 65536U, 65537U})
                            cases.push_back({p::Phase::decode, context, 1U});
                        cases.push_back({p::Phase::decode, 32768U, 4U});
                    }
                    for (const auto& one : cases) {
                        const auto context = one.context;
                        const auto lanes = one.lanes;
                        p::Inputs in;
                        in.profile = tier;
                        in.env = no_env;
                        in.backend = binding.backend;
                        in.regime = {one.phase, context, lanes, device.arch, "matrix"};
                        const auto emit = [&](const auto& resolution) {
                            std::cout << '\t' << std::quoted(resolution.qualification());
                            for (const auto& entry : resolution.entries)
                                std::cout << '\t' << entry.lever->name << '=' << std::quoted(entry.value)
                                          << '/' << static_cast<unsigned>(entry.status);
                        };
                        std::cout << model.name << '\t' << binding.backend << '\t'
                                  << (one.phase == p::Phase::decode ? std::string(binding.route) + "-decode" : std::string(binding.route)) << '\t'
                                  << device.name << '\t' << (tier == p::Name::standard ? "exact" : p::name(tier))
                                  << '\t' << context;
                        try {
                            if (binding.tag == "qwen") {
                                q::Inputs qi;
                                qi.env = no_env;
                                qi.regime = in.regime;
                                qi.max_prompt_tokens = context;
                                if (binding.backend == "hip") {
                                    qi.regimes = q::fast_regimes;
                                    emit(q::resolve(tier, qi));
                                } else {
                                    qi.caps = &q::kCudaWeightCaps;
                                    auto resolved = q::resolve(tier, qi, qc::levers(), qc::ladder_rows);
                                    static_cast<void>(qc::choose(resolved));
                                    emit(resolved);
                                }
                            } else if (binding.tag == "deepseek41-pair") {
                                if (binding.backend == "hip")
                                    emit(p::resolve(strix::deepseek41::pair_defaults::profile_engine(), in));
                                else
                                    emit(ds::resolve(tier, false,
                                        binding.route == "replica" ? ds::App::replica : ds::App::prefill_service,
                                        {{}, no_env, in.regime}));
                            } else if (binding.tag == "glm-pair") {
                                if (binding.backend == "hip")
                                    emit(p::resolve(strix::glm::rank_defaults::product_profile_engine, in));
                                else
                                    emit(glm::resolve(glm::App::decode_service,
                                        {{"--profile", p::name(tier)}, no_env, in.regime}));
                            } else if (binding.tag == "minimax-pair") {
                                emit(p::resolve(strix::minimax::defaults::profile_engine(), in));
                            } else if (binding.tag == "gemma4") {
                                g4::Inputs gi;
                                gi.env = no_env;
                                gi.regime = in.regime;
                                emit(g4::resolve(tier, g4::App::engine, gi));
                            } else if (binding.tag == "qwen35") {
                                const bool spec = binding.route == "spec";
                                if (binding.backend == "hip") {
                                    namespace hp = qd::hip_profile;
                                    emit(hp::resolve_prefill(spec ? hp::App::spec : hp::App::decode, in, {}, 64,
                                                             {true, true}));
                                } else {
                                    namespace cp = qd::profile;
                                    cp::Regime regime;
                                    regime.prompt_tokens = context;
                                    regime.device = device.arch;
                                    regime.arch = device.arch;
                                    regime.cost_device = device.name;
                                    regime.route = spec ? cp::Route::windows : cp::Route::large_prefill;
                                    emit(cp::resolve(tier, regime, {})->resolution);
                                }
                            } else if (binding.tag == "hy4") {
                                namespace hp = strix::hy4::profile;
                                emit(hp::resolve({hp::device_of(device.arch), hp::Topology::one_node, context, "matrix"},
                                                 in)->resolution);
                            } else if (binding.tag == "inkling2-mini") {
                                emit(p::resolve(strix::inkling2::kMiniProfile, in));
                            } else {
                                emit(p::resolve(strix::ernie::kProfileEngine, in));
                            }
                        } catch (const std::invalid_argument& error) {
                            std::cout << "\trefused\t" << std::quoted(error.what());
                        }
                        std::cout << '\n';
                    }
                }
            }
        }
        check(covered, model.name + " has no profile matrix binding");
    }
}

} // namespace

int main(int argc, char** argv) {
    try {
        if (argc == 3 && std::string_view(argv[1]) == "--matrix") {
            matrix(argv[2]);
            return 0;
        }
        check(argc == 2 || (argc == 3 && std::string_view(argv[2]) == "--dump"),
              "expected profile fixture name [--dump]");
        dump = argc == 3;
        const std::string_view name = argv[1];
        if (name == "qwen_hip") qwen(false);
        else if (name == "qwen_cuda") qwen(true);
        else if (name == "gemma4") gemma();
        else if (name == "ds41_rtx") ds41();
        else if (name == "glm_cuda") glm_cuda();
        else throw std::invalid_argument("unknown profile fixture");
        return 0;
    } catch (const std::exception& error) {
        std::cerr << error.what() << '\n';
        return 1;
    }
}
