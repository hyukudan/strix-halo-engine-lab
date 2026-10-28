#include "strix/qpf2k_defaults.hpp"
#include "strix/dispatch_fuse.hpp"
#include "strix/known_prompt_chunks.hpp"
#include "hip_decode_options.hpp"

#include <cstdlib>
#include <functional>
#include <iostream>
#include <map>
#include <optional>
#include <stdexcept>
#include <string>
#include <thread>
#include <vector>

namespace {
using namespace strix;
void expect(bool ok, const char* why) { if (!ok) throw std::runtime_error(why); }
using Environment = std::map<std::string, std::string>;
unsigned initialize(Environment& env, unsigned allowed = ~0U) {
    return qpf2k::initialize([&](const char* name) -> const char* {
        const auto it = env.find(name);
        return it == env.end() ? nullptr : it->second.c_str();
    }, [&](const char* name, const char* value) { return env.emplace(name, value).second; }, allowed);
}
void policy_tests() {
    Environment automatic;
    const auto all = initialize(automatic);
    expect(all == (1U << qpf2k::defaults.size()) - 1U, "every injected item retains provenance");
    expect(automatic.at("STRIX_QPF2K_MOE_DEFAULTED") == "3", "MoE provenance is preserved");
    expect(automatic.at("STRIX_QPF2K_SCORE_ROWS") == "auto" && automatic.at("STRIX_PREFILL_PLE_BATCH") == "1" &&
        automatic.at("STRIX_QPF2K_SHARED_PRE") == "1" && automatic.at("STRIX_QPF2K_MOE_MC") == "1" &&
        automatic.at("STRIX_QPF2K_MOE_HDR") == "1" && !automatic.contains("STRIX_QPF2K_HC_LOWBAND"),
        "exact prefill defaults use auto scoring and MC mode 1; LOWBAND stays opt-in");
    Environment startup;
    expect(initialize(startup, qpf2k::startup_items) == qpf2k::startup_items &&
        !(qpf2k::startup_items & qpf2k::exact_prefill_items), "exact prefill defaults wait for decoder admission");
    for (const auto& entry : qpf2k::defaults)
        if (qpf2k::exact_prefill_items & qpf2k::bit(entry.item))
            expect(!startup.contains(entry.name), "non-decoder clients retain exact prefill opt-ins");
    qpf2k::defaulted_items = all;
    for (const auto& entry : qpf2k::defaults) {
        expect(automatic.at(entry.name) == entry.value, "non-GSQ default values are unchanged");
        expect(!qpf2k::suppressed(entry.item), "ordinary model keeps every default");
        const qpf2k::ModelScope gsq(true);
        expect(qpf2k::suppressed(entry.item), "GSQ excludes every injected item");
    }
    {
        const qpf2k::ModelScope gsq(true);
        { const qpf2k::ModelScope ordinary(false);
          expect(!qpf2k::suppressed(qpf2k::Item::dense), "nested ordinary owner keeps its defaults"); }
        expect(qpf2k::suppressed(qpf2k::Item::dense), "nested scope restores the GSQ owner");
        bool other_thread = true;
        std::thread worker([&] { other_thread = qpf2k::suppressed(qpf2k::Item::dense); });
        worker.join();
        expect(!other_thread, "model exclusion does not leak to another worker");
    }
    for (const auto& entry : qpf2k::defaults) for (const auto* value : {"0", entry.value}) {
        Environment explicit_env{{entry.name, value}};
        qpf2k::defaulted_items = initialize(explicit_env);
        const qpf2k::ModelScope gsq(true);
        expect(explicit_env.at(entry.name) == value && !qpf2k::suppressed(entry.item),
            "explicit settings, including ROUTE/DA16, reach their original qualification checks");
        if (entry.item == qpf2k::Item::qsa || entry.item == qpf2k::Item::hc)
            expect(!explicit_env.contains(entry.item == qpf2k::Item::qsa ? "STRIX_QPF2K_QSA_MASK" : "STRIX_QPF2K_HC_MASK"),
                "an explicit parent does not acquire a default mask");
    }
    for (const auto* value : {"0", "00", "0disabled"}) {
        Environment disabled{{"STRIX_QPF2K_DEFAULTS", value}, {"STRIX_QPF2K_MOE_ROUTE", "14"}};
        expect(initialize(disabled) == 0U && disabled.size() == 2U, "DEFAULTS=0* injects nothing and preserves explicit settings");
    }
    Environment baseline{{"STRIX_BASELINE", "1"}, {"STRIX_QPF2K_MOE_MC", "0"}};
    expect(initialize(baseline) == 0U && baseline.size() == 2U, "environment baseline injects nothing and preserves explicit zero");
    ModelManifest manifest;
    ManifestTensor gate;
    gate.name = "blk.0.ffn_gate_exps.weight";
    for (const auto type : {12U, 13U, 18U, 21U, 22U, 23U, 42U}) {
        gate.ggml_type = type;
        manifest.tensors = {gate};
        expect(qpf2k::has_gsq_gate(manifest) == gsq_a8::gate_type(type), "qualification uses the file's actual gate type");
    }
    gate.ggml_type = 12U;
    manifest.tensors.insert(manifest.tensors.begin(), gate);
    expect(qpf2k::has_gsq_gate(manifest), "a mixed manifest excludes defaults for the entire GSQ owner");
    qpf2k::defaulted_items = all;
}

class Rows final : public KnownPromptWindowBackend, public SerialGreedyStepBackend {
public:
    explicit Rows(std::uint32_t width, bool gsq) : width_(width), gsq_(gsq) {}
    bool supports_tiled_prefill() const noexcept override { return true; }
    bool gsq_model() const noexcept override { return gsq_; }
    std::uint32_t logical_rows() const noexcept override { return width_; }
    KnownWindowOutputPolicy output_policy() const noexcept override { return KnownWindowOutputPolicy::terminal_row; }
    std::vector<float> execute(std::uint32_t, std::uint64_t) override { throw std::logic_error("unexpected serial row"); }
    std::vector<float> execute_window(std::array<std::uint32_t, 2>, std::uint64_t) override { throw std::logic_error("unexpected pair"); }
    std::vector<float> execute_rows(std::span<const std::uint32_t> tokens, std::uint64_t) override {
        calls.push_back(static_cast<unsigned>(tokens.size())); return std::vector<float>(16U, 0.0F);
    }
    std::vector<float> execute_rows_partial(std::span<const std::uint32_t> tokens, std::uint64_t position) override {
        return execute_rows(tokens, position);
    }
    void execute_state_rows(std::span<const std::uint32_t> tokens, std::uint64_t position) override {
        static_cast<void>(execute_rows(tokens, position));
    }
    std::vector<unsigned> calls;
private:
    std::uint32_t width_;
    bool gsq_;
};
void schedule_tests() {
#ifdef _WIN32
    expect(_putenv_s("STRIX_QPF2K_SCHED", "12") == 0, "set schedule fixture");
#else
    expect(::setenv("STRIX_QPF2K_SCHED", "12", 1) == 0, "set schedule fixture");
#endif
    const auto all = qpf2k::defaulted_items;
    // Exercise the real cursor and CLI tail planner, alternating models after their
    // environment caches have initialized. A 1536-row tail merges only for Q4K or explicit SCHED.
    for (const bool gsq : {true, false, true}) for (const bool explicit_sched : {false, true}) {
        qpf2k::defaulted_items = explicit_sched ? all & ~qpf2k::bit(qpf2k::Item::sched) : all;
        const bool merge = !gsq || explicit_sched;
        const qpf2k::ModelScope model(gsq);
        decode_cli::Options options;
        options.known_prefix_window = options.known_prefix_prefill_explicit = options.known_prefix_full_rows = true;
        options.known_prefix_prefill_rows = 8192U;
        options.known_prefix_output_policy = KnownWindowOutputPolicy::terminal_row;
        options.tokens.assign(1536U, 1U);
        decode_cli::apply_default_known_prefill_tails(options);
        expect(options.known_prefix_tail_rows == (merge ? std::vector<std::uint32_t>{} :
            std::vector<std::uint32_t>{1024U, 256U, 64U, 16U}), "GSQ retains the DEFAULTS=0 tail owners");
        Rows main(8192U, gsq), tail1024(1024U, gsq), tail256(256U, gsq);
        KnownPromptChunkRequest request{options.tokens, 0U, 16U, 8192U, 0U, 15U, 8192U,
            KnownWindowOutputPolicy::terminal_row, true};
        request.schedule_extensions = true;
        KnownPromptChunkCursor cursor(request, main, main);
        cursor.set_tail_backend(tail1024); cursor.set_tail_backend(tail256);
        while (cursor.advance()) {}
        expect(main.calls == (merge ? std::vector<unsigned>{1536U} : std::vector<unsigned>{}) &&
            tail1024.calls == (merge ? std::vector<unsigned>{} : std::vector<unsigned>{1024U}) &&
            tail256.calls == (merge ? std::vector<unsigned>{} : std::vector<unsigned>{256U, 256U}),
            "cached SCHED follows each owner's default provenance without changing explicit settings");
    }
    qpf2k::defaulted_items = all;
}

class EnvSetting {
public:
    explicit EnvSetting(const char* name) : name_(name) {
        if (const char* value = std::getenv(name)) previous_ = value;
    }
    ~EnvSetting() { set(previous_ ? previous_->c_str() : nullptr); }
    void set(const char* value) const {
#ifdef _WIN32
        expect(_putenv_s(name_, value ? value : "") == 0, "set environment fixture");
#else
        expect((value ? ::setenv(name_, value, 1) : ::unsetenv(name_)) == 0, "set environment fixture");
#endif
    }
private:
    const char* name_;
    std::optional<std::string> previous_;
};

void ple_reclaim_tests() {
    using ple_reclaim::Mode;
    const EnvSetting flag("STRIX_PLE_RECLAIMABLE"), defaults("STRIX_QPF2K_DEFAULTS");
    const auto all = qpf2k::defaulted_items;
    defaults.set(nullptr);
    flag.set(nullptr);
    expect(ple_reclaim::requested() == Mode::automatic, "unset L1 selects auto");
    defaults.set("0");
    expect(ple_reclaim::requested() == Mode::off, "DEFAULTS=0 disables unset L1 without startup injection");
    defaults.set("00");
    expect(ple_reclaim::requested() == Mode::off, "DEFAULTS=00 disables unset L1 without startup injection");
    defaults.set(nullptr);
    flag.set("auto");
    // Alternate owners after the environment has been read; default provenance must remain owner-local.
    for (const bool gsq : {true, false, true}) {
        const qpf2k::ModelScope model(gsq);
        expect(ple_reclaim::requested() == (gsq ? Mode::off : Mode::automatic), "GSQ suppresses defaulted L1");
    }
    qpf2k::defaulted_items = all & ~qpf2k::bit(qpf2k::Item::ple_reclaim);
    {
        const qpf2k::ModelScope gsq(true);
        for (const auto* value : {"auto", "1", "0"}) {
            flag.set(value);
            expect(ple_reclaim::requested() == (value[0] == 'a' ? Mode::automatic :
                value[0] == '1' ? Mode::required : Mode::off), "explicit L1 survives GSQ suppression");
        }
        flag.set(nullptr);
        expect(ple_reclaim::requested() == Mode::off, "GSQ also suppresses implicit auto without startup injection");
    }
    for (const auto* value : {"", "0", "01", "11", "invalid"}) {
        flag.set(value);
        // An empty Windows environment value removes the variable.
#ifdef _WIN32
        if (!*value) continue;
#endif
        expect(ple_reclaim::requested() == Mode::off, "non-auto/non-1 settings retain the old off behavior");
    }
    defaults.set("0");
    flag.set("1");
    expect(ple_reclaim::requested() == Mode::required, "DEFAULTS=0 preserves explicit strict L1");
    defaults.set(nullptr);

    decode_cli::Options options;
    options.residency = HipModelResidencyMode::registered_file_mapping;
    options.known_prefix_prefill_rows = 8192U;
    options.tokens = {1U, 2U, 3U};
    HipModelResidencyPolicy policy;
    expect(decode_cli::ple_reclaim_mode(options, policy, false) == Mode::required, "strict L1 admits plain tiled prefill");
    flag.set("auto");
#if defined(__linux__)
    constexpr auto automatic = Mode::automatic;
#else
    constexpr auto automatic = Mode::off;
#endif
    expect(decode_cli::ple_reclaim_mode(options, policy, false) == automatic, "auto admits only on Linux");
    flag.set(nullptr);
    expect(decode_cli::ple_reclaim_mode(options, policy, false) == automatic, "unset and explicit auto share admission");
    defaults.set("0");
    expect(decode_cli::ple_reclaim_mode(options, policy, false) == Mode::off, "DEFAULTS=0 skips L1 on the supported route");
    defaults.set(nullptr);

    using Exclude = std::function<void(decode_cli::Options&, HipModelResidencyPolicy&, bool&)>;
    const std::vector<Exclude> exclusions{
        [](auto& o, auto&, auto&) { o.residency = HipModelResidencyMode::resident_copy; },
        [](auto&, auto& p, auto&) { p.copy_tier = HipModelCopyTier::all; },
        [](auto&, auto& p, auto&) { p.promote_file_backed_candidates = true; },
        [](auto&, auto&, auto& graph) { graph = true; },
        [](auto& o, auto&, auto&) { o.known_prefix_prefill_rows = 0; },
        [](auto& o, auto&, auto&) { o.tokens.clear(); },
        [](auto& o, auto&, auto&) { o.http_serve = true; },
        [](auto& o, auto&, auto&) { o.mtp.cohort.emplace_back(); },
        [](auto& o, auto&, auto&) { o.serving_prefix_files.emplace_back("prefix"); },
        [](auto& o, auto&, auto&) { o.mtp.enabled = true; },
        [](auto& o, auto&, auto&) { o.planned_ple_io = true; },
        [](auto& o, auto&, auto&) { o.session_load = "checkpoint"; },
        [](auto& o, auto&, auto&) { o.session_live_control = true; },
        [](auto& o, auto&, auto&) { o.session_then_prompt = "suffix.ids"; },
    };
    for (const auto& exclude : exclusions) {
        auto excluded = options;
        auto excluded_policy = policy;
        bool graph = false;
        exclude(excluded, excluded_policy, graph);
        expect(!decode_cli::ple_reclaim_route_supported(excluded, excluded_policy, graph),
            "unsupported route (including session load) cannot build an L1 prompt plan");
        for (const auto* value : std::array<const char*, 4>{nullptr, "auto", "0", "1"}) {
            flag.set(value);
            bool refused = false;
            try {
                expect(decode_cli::ple_reclaim_mode(excluded, excluded_policy, graph) == Mode::off,
                    "unsupported auto/off routes skip L1");
            } catch (const std::invalid_argument& error) {
                refused = true;
                expect(std::string(error.what()).starts_with("STRIX_PLE_RECLAIMABLE=1 requires"),
                    "strict route refusal text is preserved");
            }
            expect(refused == (value && value[0] == '1'), "only explicit =1 refuses unsupported routes");
        }
    }
    qpf2k::defaulted_items = all;
}
} // namespace
int main(int argc, char** argv) {
    if (argc == 2) {
        const bool first_gsq = std::string(argv[1]) == "--gsq-first";
        const EnvSetting parent("STRIX_DISPATCH_FUSE"), inject("STRIX_WINDOW_INJECT_NORM"),
            commit("STRIX_WINDOW_COMMIT_COMBINE"), hc("STRIX_HC_PERSISTENT"), hc_inject("STRIX_HC_PERSISTENT_INJECT");
        parent.set("1"); inject.set(nullptr); commit.set(nullptr); hc.set(nullptr); hc_inject.set(nullptr);
        qpf2k::defaulted_items = qpf2k::bit(qpf2k::Item::dispatch_fuse);
        for (bool gsq : {first_gsq, !first_gsq, first_gsq, !first_gsq}) {
            const qpf2k::ModelScope owner(gsq);
            expect(strix_dispatch_fuse_enabled() == !gsq && strix_window_inject_norm_enabled() == !gsq &&
                strix_window_commit_combine_enabled() == !gsq && strix_hc_persistent_enabled() == !gsq &&
                strix_hc_persistent_inject_enabled() == !gsq && strix_hc_inject_fusion_enabled() == !gsq,
                "actual fusion readers filter after their raw caches for every owner");
        }
        qpf2k::defaulted_items = 0U;
        const qpf2k::ModelScope owner(true);
        expect(strix_hc_inject_fusion_enabled(), "explicit fusion survives GSQ filtering");
        return 0;
    }
    expect(qpf2k::defaults.size() == 28U, "all scoped prompt defaults are part of the injected-item contract");
    qpf2k::baseline = true;
    std::map<std::string,std::string> baseline_env;
    expect(initialize(baseline_env) == 0U && baseline_env.empty(), "baseline injects no defaults");
    for (const auto& entry : qpf2k::defaults)
        expect(qpf2k::suppressed(entry.item, false), "baseline suppresses cached controlled settings for ordinary owners too");
    qpf2k::baseline = false;
    try { policy_tests(); schedule_tests(); ple_reclaim_tests(); }
    catch (const std::exception& error) { std::cerr << error.what() << '\n'; return 1; }
    std::cout << "qpf2k default qualification tests passed\n";
}
