#include "../src/model_graph/qwen_adapter.hpp"
#include "strix/prefill_row_liveness.hpp"

#include <algorithm>
#include <iostream>
#include <stdexcept>

namespace mg = strix::model_graph;
namespace qd = strix::qwen_description;
namespace {
unsigned checks = 0;
void expect(bool ok, const char* message) {
    ++checks;
    if (!ok) throw std::runtime_error(message);
}
template<class F> void rejects(F action) {
    bool rejected = false;
    try { action(); } catch (const std::invalid_argument&) { rejected = true; }
    expect(rejected, "corrupt fixture accepted");
}
bool contains(const std::vector<std::string>& ids, const std::string& id) {
    return std::find(ids.begin(), ids.end(), id) != ids.end();
}
}  // namespace
int main(int argc, char** argv) {
    try {
        if (argc != 2) throw std::invalid_argument("expected repository model TOML");
        const auto file = strix::model_description::load(argv[1]);
        const auto model = qd::contract_from_description(file);
        const auto graph = qd::model_graph_from_description(file);
        expect(mg::canonical(graph) == mg::canonical(qd::model_graph_from_contract(model)), "file/contract graph differs");
        expect(mg::identity(graph) == mg::identity(qd::model_graph_from_description(file)), "unstable graph identity");
        expect(graph.layers.size() == 48, "trunk layer count");
        // Row liveness (strix/prefill_row_liveness.hpp): every trunk layer declares state, so prompt rows stop nowhere early.
        { const auto lp = strix::prefill_liveness::plan_graph(graph, {.taps = {}, .every_row = false, .head_row = false, .tail_row_invariant = false});
          expect(!lp.cut() && lp.cache_only_last && strix::prefill_liveness::skipped_row_layers(lp, 512) == 0, "row liveness: no layer above the last state owner"); }
        expect(graph.attachments.size() == 2, "PLE/MTP attachment count");
        expect(graph.attachments[0].kind == mg::AttachmentKind::ple &&
            graph.attachments[0].target == "layer.1.before-attention", "PLE insertion");
        expect(graph.attachments[1].kind == mg::AttachmentKind::mtp &&
            graph.attachments[1].layers.size() == model.mtp_full_attention_layers &&
            graph.attachments[1].layers[0].id == 48 &&
            graph.attachments[1].layers[0].block == mg::BlockKind::auxiliary, "MTP sidecar");
        unsigned gdn = 0, qsa = 0;
        for (std::size_t i = 0; i < graph.layers.size(); ++i) {
            const auto& item = graph.layers[i];
            const bool indexed = (i + 1) % 4 == 0;
            expect(item.id == i && item.attention == (indexed ? mg::AttentionFamily::qsa : mg::AttentionFamily::gdn), "hybrid cadence");
            indexed ? ++qsa : ++gdn;
            expect(item.hyper_connection && item.hyper_connection->streams == model.hc_count, "HC streams");
            expect(item.mlps.size() == 2 && item.mlps[0].kind == mg::MlpKind::moe &&
                item.mlps[0].experts == 512 && item.mlps[0].selected == 10 &&
                item.mlps[1].kind == mg::MlpKind::shared_expert, "routed/shared experts");
            expect(item.states.size() == (i == 1 ? 3U : 2U), "persistent layer state inventory");
        }
        expect(gdn == 36 && qsa == 12, "attention totals");
        const auto source = strix::build_serial_decode_execution_plan(model);
        const auto plan = qd::common_decode_plan(source, graph);
        qd::validate_decode_projection(model, source, graph, plan);
        expect(plan.nodes.size() == source.steps.size(), "projection changed node count");
        for (std::size_t i = 0; i < source.steps.size(); ++i) {
            const auto& step = source.steps[i];
            const auto& node = plan.nodes[i];
            expect(node.operation == strix::to_string(step.op) &&
                node.layer == (step.layer ? static_cast<std::int64_t>(*step.layer) : -1), "operation/layer changed");
            expect(std::vector<std::uint64_t>(step.dependencies.begin(), step.dependencies.end()) == node.dependencies,
                "dependencies changed");
            expect(node.graph_safe_candidate == step.graph_safe_candidate && node.dynamic_shape == step.dynamic_shape,
                "capture flags changed");
            // Independent mask/role oracle, including the two different convolution owners.
            for (unsigned bit = 0; bit < 6; ++bit) {
                const auto flag = static_cast<strix::PersistentState>(1U << bit);
                const char* names[] = {"attention_kv", "recurrent_matrix", "convolution_history",
                    "sparse_index", "ngram_history", "position_history"};
                std::string role = names[bit];
                if (bit == 2 && step.op == strix::SerialOp::ple) role = "ple_convolution";
                const auto id = (bit >= 4 ? "sequence" : "layer." + std::to_string(node.layer)) + "." + role;
                expect(contains(node.state_reads, id) == strix::has_state(step.reads_state, flag), "state read mask");
                expect(contains(node.state_writes, id) == strix::has_state(step.writes_state, flag), "state write mask");
            }
        }
        for (const auto& buffer : plan.buffers) {
            expect(buffer.lifetime == mg::BufferLifetime::sequence && buffer.live_in && buffer.live_out &&
                !buffer.extent.empty(), "state lifetime/extent");
            std::uint64_t first = plan.nodes.size(), last = 0;
            for (std::size_t i = 0; i < plan.nodes.size(); ++i) {
                if (contains(plan.nodes[i].state_reads, buffer.id) || contains(plan.nodes[i].state_writes, buffer.id)) {
                    first = std::min(first, static_cast<std::uint64_t>(i));
                    last = i;
                }
            }
            expect(first == buffer.first_use && last == buffer.last_use, "state use interval");
        }
        expect(mg::dump(plan).find("\"graph_safe_candidate\":false") != std::string::npos &&
            mg::dump(plan).find("\"dynamic_shape\":true") != std::string::npos, "capture flags absent from dump");
        const auto ple = static_cast<std::size_t>(std::find_if(source.steps.begin(), source.steps.end(),
            [](const auto& step) { return step.op == strix::SerialOp::ple; }) - source.steps.begin());
        auto broken_graph = graph;
        broken_graph.layers[0].attention = mg::AttentionFamily::qsa;
        rejects([&] { qd::validate_decode_projection(model, source, broken_graph, plan); });
        broken_graph = graph;
        broken_graph.layers[1].states.pop_back();
        rejects([&] { (void)qd::common_decode_plan(source, broken_graph); });
        auto broken = plan;
        broken.nodes.back().dependencies.clear();
        rejects([&] { qd::validate_decode_projection(model, source, graph, broken); });
        broken = plan;
        broken.nodes[ple].state_writes.clear();
        rejects([&] { qd::validate_decode_projection(model, source, graph, broken); });
        broken = plan;
        broken.nodes[ple].graph_safe_candidate = true;
        rejects([&] { qd::validate_decode_projection(model, source, graph, broken); });
        auto broken_source = source;
        broken_source.steps[ple].dynamic_shape = false;
        rejects([&] { qd::validate_decode_projection(model, broken_source, graph, plan); });
        broken_source = source;
        broken_source.steps[0].reads_state = static_cast<strix::PersistentState>(1U << 9U);
        rejects([&] { (void)qd::common_decode_plan(broken_source, graph); });
        broken_source = source;
        broken_source.steps[1].id = 0;
        rejects([&] { (void)qd::common_decode_plan(broken_source, graph); });
        auto altered = model;
        altered.full_attention_interval = 0;
        rejects([&] { (void)qd::model_graph_from_contract(altered); });
        altered = model;
        altered.attention_kv_storage = strix::AttentionKvStorage::int8_token;
        expect(mg::identity(qd::model_graph_from_contract(altered)) != mg::identity(graph), "KV storage omitted from graph");
        altered = model;
        altered.norm_topk_probability = !model.norm_topk_probability;
        expect(mg::identity(qd::model_graph_from_contract(altered)) != mg::identity(graph), "router semantics omitted");
        altered = model;
        altered.full_attention_interval = 6;
        altered.full_attention_layers = 8;
        altered.gdn_layers = 40;
        altered.ple_layer_1based = 6;
        altered.mtp_full_attention_layers = 2;
        const auto alternate_graph = qd::model_graph_from_contract(altered);
        const auto alternate_source = strix::build_serial_decode_execution_plan(altered);
        const auto alternate_plan = qd::common_decode_plan(alternate_source, alternate_graph);
        qd::validate_decode_projection(altered, alternate_source, alternate_graph, alternate_plan);
        expect(alternate_graph.layers[3].attention == mg::AttentionFamily::gdn &&
            alternate_graph.layers[5].attention == mg::AttentionFamily::qsa &&
            alternate_graph.attachments[0].target == "layer.5.before-attention" &&
            alternate_graph.attachments[1].layers.size() == 2, "cadence/PLE/MTP must come from contract");
        std::cout << "Qwen graph/projection: " << checks << " checks passed\n";
        return 0;
    } catch (const std::exception& error) {
        std::cerr << "Qwen graph/projection: " << error.what() << '\n';
        return 1;
    }
}
