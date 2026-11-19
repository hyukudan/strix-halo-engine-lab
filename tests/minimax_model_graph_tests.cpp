#include "../src/minimax/model_graph.hpp"
#include "strix/prefill_row_liveness.hpp"

#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <filesystem>
#include <stdexcept>
#include <string>

namespace {
namespace mm = strix::minimax::graph;
namespace mg = strix::model_graph;
int checks = 0;
void check(bool value, const char* message) {
    ++checks;
    if (!value) throw std::runtime_error(message);
}
template<class F> void rejects(F run) {
    bool refused = false;
    try { run(); } catch (const std::exception&) { refused = true; }
    check(refused, "invalid description/graph/plan accepted");
}
void environment(const char* key, const char* value) {
#ifdef _WIN32
    if (_putenv_s(key, value == nullptr ? "" : value) != 0) throw std::runtime_error("environment update failed");
#else
    if ((value == nullptr ? ::unsetenv(key) : ::setenv(key, value, 1)) != 0) throw std::runtime_error("environment update failed");
#endif
}
struct EnvGuard {
    std::string key, saved;
    bool present;
    explicit EnvGuard(const char* name) : key(name), present(std::getenv(name) != nullptr) {
        if (present) saved = std::getenv(name);
        environment(name, nullptr);
    }
    ~EnvGuard() { environment(key.c_str(), present ? saved.c_str() : nullptr); }
};
} // namespace

int main(int argc, char** argv) {
    try {
        // Host-only probe of the exact helper used by both production apps.
        if (argc == 5 && std::string(argv[1]) == "--inspect") {
            const auto rank = static_cast<std::uint32_t>(std::stoul(argv[3]));
            return mm::inspect_or_check(argv[2], rank, {}, std::string(argv[4]) == "graph", std::string(argv[4]) == "plan") ? 0 : 1;
        }
        if (argc != 2) throw std::runtime_error("expected repository MiniMax model TOML");
        const auto d = strix::minimax::description::load(argv[1]);
        const auto g = mm::build_model_graph(d);
        const auto original = mg::canonical(g);
        check(g.layers.size() == 60 && g.attachments.empty(), "trunk/attachment inventory");
        // Row liveness (strix/prefill_row_liveness.hpp): every trunk layer declares state, so prompt rows stop nowhere early.
        { const auto lp = strix::prefill_liveness::plan_graph(g, {.taps = {}, .every_row = false, .head_row = false, .tail_row_invariant = false});
          check(!lp.cut() && lp.cache_only_last && strix::prefill_liveness::skipped_row_layers(lp, 512) == 0, "row liveness: no layer above the last state owner"); }
        check(g.dimensions.at("dims.hidden") == 6144 && !g.dimensions.contains("dims.vocab_rank_pad"), "model vs physical dimensions");
        check(original == mg::canonical(mm::build_model_graph(d)) && mg::identity(g).size() == 16, "stable canonical identity");
        for (const auto& l : g.layers) {
            check(l.attention == mg::AttentionFamily::msa && l.states.size() == 3, "MSA and index state");
            check(l.norms.size() == 2 && l.norms[0].kind == "gemma_rmsnorm" && l.norms[1].placement == mg::NormPlacement::pre_mlp, "norm semantics");
            check(l.dimensions.at("query_heads") == 64 && l.dimensions.at("index_heads") == 4, "full semantic head inventory");
            if (l.id < 3) check(l.mlps.size() == 1 && l.mlps[0].kind == mg::MlpKind::dense, "dense prefix");
            else check(l.mlps.size() == 2 && l.mlps[0].experts == 128 && l.mlps[0].selected == 4 &&
                       l.mlps[1].kind == mg::MlpKind::shared_expert, "routed and shared experts");
        }
        mm::assert_matches(d, g);
        for (const auto mutate : {
                +[](mg::ModelGraph& value) { value.layers[0].attention = mg::AttentionFamily::dense; },
                +[](mg::ModelGraph& value) { value.layers[3].mlps[0].experts = 64; },
                +[](mg::ModelGraph& value) { value.layers[0].states.pop_back(); },
                +[](mg::ModelGraph& value) { value.layers[0].norms[0].epsilon = "wrong"; },
                +[](mg::ModelGraph& value) { value.layers[0].tensors[0].ownership = "rank0"; }}) {
            auto changed = g; mutate(changed);
            rejects([&] { mm::assert_matches(d, changed); });
        }
        auto wrong = d;
        wrong.fields["name"] = strix::model_description::Value{std::string("unsupported")};
        rejects([&] { (void)mm::build_model_graph(wrong); });
        auto draft = mm::build_model_graph(d, {true, true, 2});
        check(draft.attachments.size() == 1 && draft.attachments[0].kind == mg::AttachmentKind::draft &&
              draft.attachments[0].layers.empty() && draft.dimensions.at("draft.heads") == 2, "configured draft is an opaque attachment, not MTP");
        check(mg::identity(draft) != mg::identity(g), "attachment identity");
        rejects([&] { (void)mm::build_model_graph(d, {false, true, 0}); });
        const auto no_env = [](const char*) -> const char* { return nullptr; };
        for (std::uint32_t rank = 0; rank < 2; ++rank) {
            const auto p = mm::common_decode_plan(d, g, rank, no_env);
            mg::validate(p, g);
            check(p.nodes.size() == 1015 && p.ranks == 2 && p.rank == rank, "complete logical schedule");
            check(p.policies.at("resolved.STRIX_MM_INDEX_CONC") == "1" && !p.engine_local.empty(), "resolver declarations and coverage");
            check(p.policies.at("attention.head_begin") == std::to_string(rank * 32U), "head placement");
            check(p.graph_identity == mg::identity(g) && original == mg::canonical(g), "placement never changes graph");
            check(p.policies.at("head.vocab_end") == (rank == 0 ? "100096" : "200064"), "uneven vocabulary ownership");
            for (std::uint64_t layer = 0; layer < 60; ++layer) {
                std::vector<std::string> operations;
                for (const auto& n : p.nodes) if (n.layer == static_cast<std::int64_t>(layer)) {
                    operations.push_back(n.operation);
                    if (n.operation == "routed_limited_swiglu_intermediate_half")
                        check(n.expert_begin == 0 && n.expert_end == 128, "expert IDs are replicated, not EP shards");
                    check(!n.graph_safe_candidate, "logical nodes do not establish capture safety");
                }
                check(operations.front() == "input_gemma_rmsnorm" && operations.back() == "X2.fixed_rank_order_residual", "layer boundaries");
                const auto x1 = std::find(operations.begin(), operations.end(), "X1.fixed_rank_order_residual");
                const auto norm = std::find(operations.begin(), operations.end(), "post_attention_gemma_rmsnorm");
                check(x1 < norm && norm < operations.end() - 1, "exchange/norm/MLP order");
            }
            unsigned state_count = 0, symbolic = 0;
            for (const auto& b : p.buffers) {
                if (b.lifetime == mg::BufferLifetime::sequence) {
                    ++state_count;
                    check(b.bytes_per_position == (b.state_role == "index_key_cache" ? 256 : 512) && b.live_in && b.live_out, "rank-local BF16 state");
                }
                if (!b.extent.empty()) { ++symbolic; check(b.bytes == 0, "symbolic scratch must not claim byte sizing"); }
            }
            check(state_count == 180 && symbolic == 120, "state/symbolic scratch completeness");
            auto invalid = p; invalid.nodes[0].dependencies = {0};
            rejects([&] { mg::validate(invalid, g); });
            invalid = p; invalid.nodes[0].outputs = {"missing"};
            rejects([&] { mg::validate(invalid, g); });
            invalid = p; invalid.buffers[0].bytes = 0; invalid.buffers[0].extent.clear();
            rejects([&] { mg::validate(invalid, g); });
            check(mg::dump(p).find("\"engine_local\"") != std::string::npos, "plan coverage serialized");
            auto legacy = p; legacy.policies.clear(); legacy.engine_local.clear();
            check(mg::dump(legacy).find("\"policies\"") == std::string::npos &&
                  mg::dump(legacy).find("\"engine_local\"") == std::string::npos, "empty metadata preserves legacy serialization");
        }
        const auto baseline = mm::common_decode_plan(d, g, 0, [](const char* key) -> const char* {
            return std::string(key) == "STRIX_MM_BASELINE" ? "1" : nullptr;
        });
        check(baseline.policies.at("resolved.STRIX_MM_INDEX_CONC") == "0", "baseline resolver policy");
        const auto override_plan = mm::common_decode_plan(d, g, 0, [](const char* key) -> const char* {
            return std::string(key) == "STRIX_MM_INDEX_CONC" ? "0" : nullptr;
        });
        check(override_plan.policies.at("resolved.STRIX_MM_INDEX_CONC") == "0", "explicit resolver override");
        rejects([&] { (void)mm::common_decode_plan(d, g, 2, no_env); });
        EnvGuard gate("STRIX_MINIMAX_MODEL_GRAPH_CHECK"), graph_env("STRIX_PRINT_MODEL_GRAPH"), plan_env("STRIX_PRINT_EXEC_PLAN");
        check(!mm::inspect_or_check("missing-model", 0, {}, false, false), "disabled gate performs no sidecar I/O");
        environment("STRIX_MINIMAX_MODEL_GRAPH_CHECK", "1");
        const auto directory = std::filesystem::path(argv[1]).parent_path().string();
        check(!mm::inspect_or_check(directory, 0, {}, false, false), "check-only continues execution");
        rejects([&] { (void)mm::inspect_or_check("missing-model", 0, {}, false, false); });
        environment("STRIX_PRINT_MODEL_GRAPH", "1");
        check(mm::requested("STRIX_PRINT_MODEL_GRAPH"), "graph environment opt-in");
        environment("STRIX_PRINT_EXEC_PLAN", "0");
        check(!mm::requested("STRIX_PRINT_EXEC_PLAN"), "zero disables print");
        std::printf("minimax model graph: %d checks, 0 failures\n", checks);
        return 0;
    } catch (const std::exception& error) {
        std::fprintf(stderr, "minimax model graph after %d checks: %s\n", checks, error.what());
        return 1;
    }
}
