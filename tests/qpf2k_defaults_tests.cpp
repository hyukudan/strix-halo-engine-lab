#include "strix/qpf2k_defaults.hpp"
#include "strix/dispatch_fuse.hpp"
#include "strix/known_prompt_chunks.hpp"
#include "strix/q51_known_rows.hpp"
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
        automatic.at("STRIX_QPF2K_MOE_HDR") == "1" && automatic.at("STRIX_QPF2K_MOE_XOR64") == "1" && automatic.at("STRIX_QPF2K_HC_LOWBAND") == "3",
        "exact prefill defaults use auto scoring and MC mode 1; LOWBAND uses exact layout mode 3");
    expect(automatic.at("STRIX_QPF2K_QSA_MASK")=="0x2257",
        "attend occupancy bit 13 stays enabled in the production mask");
    Environment historical_mask{{"STRIX_QPF2K_QSA_MASK","0x257"}};
    initialize(historical_mask);
    expect(historical_mask.at("STRIX_QPF2K_QSA_MASK")=="0x257",
        "explicit historical mask keeps the occupancy off switch");
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
class RowClasses final : public KnownPromptWindowBackend, public SerialGreedyStepBackend {
public:
    RowClasses(unsigned rows, std::vector<unsigned>& classes) : width(rows), row_class(classes) {}
    unsigned width;
    std::vector<unsigned>& row_class;
    bool supports_tiled_prefill() const noexcept override { return true; }
    unsigned logical_rows() const noexcept override { return width; }
    KnownWindowOutputPolicy output_policy() const noexcept override { return KnownWindowOutputPolicy::terminal_row; }
    std::vector<float> execute(std::uint32_t, std::uint64_t position) override {
        row_class.at(static_cast<std::size_t>(position)) = 1U; return std::vector<float>(16U, 0.0F);
    }
    std::vector<float> execute_window(std::array<std::uint32_t,2>, std::uint64_t) override { throw std::logic_error("unexpected pair"); }
    std::vector<float> execute_rows_partial(std::span<const std::uint32_t> tokens, std::uint64_t position) override {
        const auto count = static_cast<unsigned>(tokens.size());
        const auto kind = count % 128U == 0U ? 128U : count;
        for (unsigned r = 0; r < count; ++r) row_class.at(static_cast<std::size_t>(position + r)) = kind;
        return std::vector<float>(16U, 0.0F);
    }
    std::vector<float> execute_rows(std::span<const std::uint32_t> tokens, std::uint64_t position) override {
        return execute_rows_partial(tokens, position);
    }
    void execute_state_rows(std::span<const std::uint32_t> tokens, std::uint64_t position) override {
        static_cast<void>(execute_rows_partial(tokens, position));
    }
    void execute_state_rows_partial(std::span<const std::uint32_t> tokens, std::uint64_t position, bool) override {
        execute_state_rows(tokens, position);
    }
};
void schedule_tests() {
    expect(known_prefill_max_rows == 16384U, "known prefill caps stop at T16384");
    for (const auto rows : {16U,1024U,8192U,8193U,16384U}) {
        expect(known_prefill_pool_rows(rows) == (rows <= 8192U ? 8192U : (rows + 511U) / 512U * 512U) &&
            known_prefill_stage_groups(rows) == (known_prefill_pool_rows(rows) / 64U),
            "native pools and stage tables retain historical capacities until a larger call");
    }
    const auto csr = q51_known_rows::scratch_layout(16384U,512U,10U,true);
    expect(csr.slots - csr.rows == UINT64_C(16384) * 10U * 4U && csr.bytes > csr.batch_begins,
        "T16384 CSR reserves all 163840 row/slot assignments");
    bool rejected = false;
    try { (void)q51_known_rows::scratch_layout(32768U,512U,10U,true); }
    catch (const std::invalid_argument&) { rejected = true; }
    expect(rejected, "T32768 remains outside the capacity contract");
#ifdef _WIN32
    expect(_putenv_s("STRIX_QPF2K_SCHED", "12") == 0, "set schedule fixture");
#else
    expect(::setenv("STRIX_QPF2K_SCHED", "12", 1) == 0, "set schedule fixture");
#endif
    const auto all = qpf2k::defaulted_items;
    for (const auto count : {16383U,16384U,16385U,32767U,32768U,32895U,32896U,131072U,262143U}) {
        std::vector<unsigned> old(count), widened(count);
        for (const auto t : {8192U,16384U}) {
            auto& classes = t == 8192U ? old : widened;
            decode_cli::Options options;
            options.known_prefix_window = options.known_prefix_prefill_explicit = options.known_prefix_full_rows = true;
            options.known_prefix_prefill_rows = t;
            options.known_prefix_output_policy = KnownWindowOutputPolicy::terminal_row;
            options.tokens.assign(count,1U);
            decode_cli::apply_default_known_prefill_tails(options,"12");
            RowClasses main(t,classes), tail1024(1024U,classes), tail256(256U,classes), tail64(64U,classes), tail16(16U,classes);
            KnownPromptChunkRequest request{options.tokens,0U,16U,count,0U,15U,t,KnownWindowOutputPolicy::terminal_row,true};
            request.schedule_extensions = true;
            KnownPromptChunkCursor cursor(request,main,main);
            for (auto* tail : {&tail1024,&tail256,&tail64,&tail16}) cursor.set_tail_backend(*tail);
            while (cursor.advance()) {}
        }
        expect(old == widened && std::find(old.begin(),old.end(),0U) == old.end(),
            "T16384 retains every absolute 128/64/16/serial row class through merged tails and context suffix");
        if (count == 262143U)
            expect(old[262016U] == 64U && old[262080U] == 16U && old.back() == 1U,
                "maximum prompt reaches the identical 64+16+16+16+15 serial suffix at 262016");
    }
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

    {
        const EnvSetting rolling("STRIX_PLE_ROLLING"), exact("STRIX_CTX_EXACT_MEMORY"), controller("STRIX_MTP_CONTROLLER");
        rolling.set(nullptr); exact.set(nullptr); controller.set(nullptr); flag.set("auto");
        auto mtp = options; mtp.mtp.enabled = true;
        expect(decode_cli::ple_reclaim_mode(mtp,policy,false) == automatic,"single fixed eager MTP route is admitted");
        mtp.mtp.window_rows_set = {2,4,8}; mtp.mtp.window_rows = 8;
        mtp.mtp.window_prefix = true; mtp.mtp.draft_chain = true; mtp.mtp.spec_min_p = 0.7;
        const EnvSetting cohort_rows("STRIX_COHORT_ROWS"), cohort_wide("STRIX_COHORT_WIDE"), lean("STRIX_LM_HEAD_COHORT_LEAN");
        cohort_rows.set("32"); cohort_wide.set("all"); lean.set("1"); controller.set("minp");
        expect(decode_cli::ple_reclaim_mode(mtp,policy,false) == automatic,"default adaptive min-p CLI admits leased rolling");
        expect(decode_cli::ple_reclaim_inactive_reason(mtp,policy,false,true).empty(),"adaptive row sets and min-p are supported");
        mtp.mtp.spec_min_p = -1.0;
        expect(decode_cli::mtp_draft_min_probability(mtp.mtp)>0.0F &&
            decode_cli::ple_reclaim_mode(mtp,policy,false)==automatic,"implicit adaptive min-p admits the same route");

        rolling.set("0");
        expect(decode_cli::ple_reclaim_mode(mtp,policy,false) == Mode::off,"MTP rolling off keeps warm zero-credit route");
        rolling.set(nullptr); exact.set("0");
        expect(decode_cli::ple_reclaim_mode(mtp,policy,false) == Mode::off,"MTP exact-memory off keeps warm zero-credit route");
        exact.set(nullptr); controller.set("depth");
        expect(decode_cli::ple_reclaim_mode(mtp,policy,false) == Mode::off,"timed controller stays warm");
        expect(decode_cli::ple_reclaim_inactive_reason(mtp,policy,false)=="mtp-depth-controller","depth inactive reason");
        controller.set("survival");
        expect(decode_cli::ple_reclaim_mode(mtp,policy,false)==Mode::off &&
            decode_cli::ple_reclaim_inactive_reason(mtp,policy,false)=="mtp-survival-controller","survival inactive reason");
        controller.set("minp");
        using MtpExclude = std::function<void(decode_cli::Options&)>;
        const std::vector<std::pair<MtpExclude,std::string_view>> cases{
            {[](auto& o) {o.mtp.copy_wide_rows={16};}, "mtp-copy-wide-rows"},
            {[](auto& o) {o.mtp.golden_dump="golden";}, "mtp-golden-dump"},
            {[](auto& o) {o.mtp.sample_timing_schedule=true;}, "mtp-sample-timing-schedule"},
            {[](auto& o) {o.mtp.metadata_only=true;}, "mtp-metadata-only"},
            {[](auto& o) {o.raw_recording_path="record";}, "raw-recording"},
            {[](auto& o) {o.validation_final_state_path="state";}, "validation-final-state"},
            {[](auto& o) {o.prefix_cache_host_bytes=1;}, "prefix-cache-restore"},
            {[](auto& o) {o.prefix_cache_dir="cache";}, "prefix-cache-restore"},
            {[](auto& o) {o.rtx_state_import=true;}, "rtx-state-import"},
        };
        for (const auto& [exclude, reason] : cases) {
            auto excluded=mtp; exclude(excluded); flag.set("auto");
            expect(decode_cli::ple_reclaim_mode(excluded,policy,false)==Mode::off,"unsafe MTP policy remains warm");
            expect(decode_cli::ple_reclaim_inactive_reason(excluded,policy,false,true)==reason,"specific inactive witness reason");
            expect(decode_cli::ple_reclaim_inactive_witness(excluded,policy,false)==
                "ple-rolling: inactive; "+std::string(reason)+"\n","exact runtime inactive witness bytes");
            flag.set("1"); bool refused=false;
            try {(void)decode_cli::ple_reclaim_mode(excluded,policy,false);}
            catch (const std::invalid_argument& error) {
                refused=true;
                expect(std::string(error.what()).ends_with(reason),"strict refusal names the same inactive reason");
            }
            expect(refused,"required unsafe MTP policy refuses");
        }
        flag.set("auto");

    }
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
        [](auto& o, auto&, auto&) { o.mtp.enabled = true; o.mtp.sample_timing_schedule = true; },
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
    expect(qpf2k::defaults.size() == 30U, "all scoped prompt defaults are part of the injected-item contract");
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
