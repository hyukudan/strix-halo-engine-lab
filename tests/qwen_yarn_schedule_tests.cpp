#include "../apps/qwen_yarn_owner_inventory.hpp"
#include "strix/known_prompt_chunks.hpp"
#include "rotary_mock.hpp"
#include <iostream>
#include <memory>
#include <set>

using namespace strix;
using namespace strix::decode_cli;
namespace {
struct Window final : KnownPromptWindowBackend, SerialGreedyStepBackend {
    std::uint32_t rows;
    std::set<std::uint32_t>& used;
    Window(std::uint32_t n, std::set<std::uint32_t>& u) : rows(n), used(u) {}
    std::uint32_t logical_rows() const noexcept override { return rows; }
    bool supports_tiled_prefill() const noexcept override { return true; }
    KnownWindowOutputPolicy output_policy() const noexcept override { return KnownWindowOutputPolicy::terminal_row; }
    std::vector<float> execute_window(std::array<std::uint32_t,2>,std::uint64_t) override { throw std::logic_error("pair"); }
    std::vector<float> execute(std::uint32_t,std::uint64_t) override { return {0,1}; }
    void execute_state(std::uint32_t,std::uint64_t) override {}
    std::vector<float> execute_rows(std::span<const std::uint32_t> t,std::uint64_t) override {
        if (t.size() != rows) throw std::logic_error("owner width");
        used.insert(rows); return {0,1};
    }
    void execute_state_rows(std::span<const std::uint32_t> t,std::uint64_t p) override { (void)execute_rows(t,p); }
    std::vector<float> execute_rows_partial(std::span<const std::uint32_t> t,std::uint64_t) override {
        used.insert(static_cast<std::uint32_t>(t.size())); return {0,1};
    }
    void execute_state_rows_partial(std::span<const std::uint32_t> t,std::uint64_t p,bool) override { (void)execute_rows_partial(t,p); }
};
}
int main() {
    const std::string schedule = std::getenv("STRIX_QPF2K_SCHED");
    const bool yarn = schedule == "12";
    const auto refusal = yarn_environment_refusal({{"STRIX_QPF2K_SCHED",schedule}},2);
    if (refusal.empty() != yarn || (!yarn && refusal.find("STRIX_QPF2K_SCHED") == std::string::npos)) return 1;
    Options o; o.context_tokens = 300000; o.tokens.resize(299000,1); o.known_prefix_prefill_rows = 8192;
    o.known_prefix_output_policy = KnownWindowOutputPolicy::terminal_row;
    o.known_prefix_window = o.known_prefix_full_rows = o.automatic_prefill = true;
    // Native still uses its effective schedule. Only admitted YaRN uses its pruned owners.
    apply_default_known_prefill_tails(o,schedule.c_str());
    std::set<std::uint32_t> used;
    Window primary(8192,used);
    KnownPromptChunkRequest r{o.tokens,0,2,300000,1,0,8192,KnownWindowOutputPolicy::terminal_row,true};
    r.schedule_extensions = true; r.ladder_tail_rows = o.known_prefix_ladder_tail_rows;
    KnownPromptChunkCursor cursor(r,primary,primary);
    std::vector<std::unique_ptr<Window>> tails;
    for (auto rows : o.known_prefix_tail_rows) {
        tails.push_back(std::make_unique<Window>(rows,used)); cursor.set_tail_backend(*tails.back());
    }
    while (cursor.advance()) {}
    if (cursor.progress().consumed_prompt_tokens != o.tokens.size()) return 2;
    if (yarn) {
        auto model = ModelContract::qwen38_flash_next(); model.qwen_rotary_table = mock_rotary(2);
        SerialSessionStateLayoutRequest sr; sr.context_tokens = o.context_tokens;
        const auto state = build_serial_session_state_layout(model,sr);
        PrefillRequirements owners; owners.state_d1 = state.device_total_bytes + state.host_total_bytes;
        append_yarn_owner_requirements(o,model,state,0,owners);
        if (used != std::set<std::uint32_t>{8192,3840,64,16} ||
            owners.owners.at("merged_routed_csr") != q51_known_rows::prefill_scratch_bytes(3840,512,10)) return 3;
        for (auto rows : o.known_prefix_tail_rows)
            if (!used.contains(rows) || !owners.owners.contains("tail_"+std::to_string(rows)+"_routed_csr")) return 4;
    }
    std::cout << "schedule " << schedule << ": native cursor complete, YaRN " << (yarn ? "owners match" : "refused") << '\n';
}
