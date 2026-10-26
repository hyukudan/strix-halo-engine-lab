#include "strix/qpf2k_defaults.hpp"
#include "strix/known_prompt_chunks.hpp"
#include "hip_decode_options.hpp"

#include <cstdlib>
#include <iostream>
#include <map>
#include <stdexcept>
#include <string>
#include <thread>
#include <vector>

namespace {
using namespace strix;
void expect(bool ok, const char* why) { if (!ok) throw std::runtime_error(why); }
using Environment = std::map<std::string, std::string>;
unsigned initialize(Environment& env) {
    return qpf2k::initialize([&](const char* name) -> const char* {
        const auto it = env.find(name);
        return it == env.end() ? nullptr : it->second.c_str();
    }, [&](const char* name, const char* value) { return env.emplace(name, value).second; });
}
void policy_tests() {
    Environment automatic;
    const auto all = initialize(automatic);
    expect(all == (1U << qpf2k::defaults.size()) - 1U, "every injected item retains provenance");
    expect(automatic.at("STRIX_QPF2K_MOE_DEFAULTED") == "3", "MoE provenance is preserved");
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
    Environment disabled{{"STRIX_QPF2K_DEFAULTS", "0"}, {"STRIX_QPF2K_MOE_ROUTE", "14"}};
    expect(initialize(disabled) == 0U && disabled.size() == 2U, "DEFAULTS=0 injects nothing and preserves explicit settings");
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
} // namespace
int main() {
    try { policy_tests(); schedule_tests(); }
    catch (const std::exception& error) { std::cerr << error.what() << '\n'; return 1; }
    std::cout << "qpf2k default qualification tests passed\n";
}
