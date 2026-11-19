#include "../src/model_graph/deepseek41_adapter.hpp"
#include "strix/prefill_row_liveness.hpp"
#include "strix/deepseek41_model_description.hpp"
#include "strix/model_description.hpp"
#include "strix/serve/json.hpp"

#include <algorithm>
#include <cstdint>
#include <iostream>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace {
#include "deepseek41_real_config.inc"
namespace ds = strix::deepseek41;
namespace mg = strix::model_graph;
void check(bool value, const char* message) {
    if (!value) throw std::runtime_error(message);
}
template<class F> void rejects(F run, const char* message) {
    try { run(); } catch (const std::invalid_argument&) { return; }
    throw std::runtime_error(message);
}
ds::TargetModel fixture() {
    ds::TargetModel model;
    model.config = ds::parse_config_json(real_config_json);
    model.layers = ds::layer_attributes(model.config);
    for (const auto& t : ds::expected_target_tensors(model.config)) {
        strix::ManifestTensor tensor;
        tensor.name = t.name; tensor.dimensions = t.dimensions; tensor.ggml_type = t.ggml_type;
        tensor.role = t.role; tensor.residency = t.residency;
        model.manifest.tensors.push_back(std::move(tensor));
    }
    return model;
}
const mg::PlanBuffer& buffer(const mg::ExecutionPlan& plan, const std::string& id) {
    const auto it = std::find_if(plan.buffers.begin(), plan.buffers.end(), [&](const auto& b) { return b.id == id; });
    if (it == plan.buffers.end()) throw std::runtime_error("missing expected buffer " + id);
    return *it;
}
std::vector<std::string> ids(const std::vector<ds::Buffer>& values) {
    std::vector<std::string> out;
    for (const auto value : values) out.push_back("activation/" + std::string(ds::to_string(value)));
    return out;
}
bool contains(const std::vector<std::string>& values, const std::string& value) {
    return std::find(values.begin(), values.end(), value) != values.end();
}
}  // namespace

int main() try {
    const auto model = fixture();
    const auto description = strix::model_description::parse(ds::describe_model_toml(model.config));
    const auto graph = ds::graph_from_model_description(description, model.config);
    check(mg::canonical(graph) == mg::canonical(ds::model_graph_from_config(model.config)), "description/config graph mismatch");
    check(graph.layers.size() == model.layers.size(), "ordered trunk size");
    // Row liveness (strix/prefill_row_liveness.hpp): every trunk layer declares state, so prompt rows stop nowhere early.
    { const auto lp = strix::prefill_liveness::plan_graph(graph, {.taps = {}, .every_row = false, .head_row = false, .tail_row_invariant = false});
      check(!lp.cut() && lp.cache_only_last && strix::prefill_liveness::skipped_row_layers(lp, 512) == 0, "row liveness: no layer above the last state owner"); }
    check(!graph.attachments.empty() && graph.attachments.front().kind == mg::AttachmentKind::mtp, "MTP attachment");
    check(graph.attachments.front().layers.size() == model.config.num_nextn_predict_layers, "MTP stages");
    bool compressed = false, window_only = false;
    for (std::size_t i = 0; i < graph.layers.size(); ++i) {
        const auto& l = graph.layers[i];
        check(l.id == i && l.hyper_connection && l.hyper_connection->streams == model.config.hc_mult, "ordered HC layers");
        check(l.mlps.size() == 2 && l.mlps[0].experts == model.config.n_routed_experts &&
              l.mlps[1].kind == mg::MlpKind::shared_expert, "routed/shared experts");
        check(std::stod(l.norms.front().epsilon) == model.config.rms_norm_eps, "norm epsilon precision");
        check(l.dimensions.at("compression_ratio") == model.layers[i].compress_ratio, "compression cadence");
        compressed |= l.attention == mg::AttentionFamily::compressed_indexed;
        window_only |= l.attention == mg::AttentionFamily::mla;
    }
    check(compressed && window_only && !graph.states.empty(), "MLA/indexed/history coverage");
    (void)strix::serve::parse_json(mg::dump(graph));
    auto no_mtp = model.config;
    no_mtp.num_nextn_predict_layers = 0;
    no_mtp.compress_ratios.resize(no_mtp.num_hidden_layers);
    check(ds::model_graph_from_config(no_mtp).attachments.empty(), "absent MTP");
    auto missing = description;
    missing.fields.erase("attention");
    rejects([&] { (void)ds::graph_from_model_description(missing, model.config); }, "incomplete description accepted");

    std::size_t projections = 0, external_states = 0;
    const std::vector<std::pair<std::uint32_t, std::uint32_t>> ranges = {{0, 40}, {0, 4}, {2, 8}, {14, 17}, {3, 7}, {24, 26}};
    for (const auto& [first, end] : ranges) for (const bool head : {false, true}) {
        const auto native = ds::build_decode_plan(model, first, end, first == 0, head);
        for (std::uint32_t rank = 0; rank < 2; ++rank) {
            const auto plan = ds::common_decode_plan(native, graph, rank);
            check(plan.nodes.size() == native.steps.size(), "operation count changed");
            for (std::size_t i = 0; i < native.steps.size(); ++i) {
                const auto& a = native.steps[i]; const auto& b = plan.nodes[i];
                check(b.operation == ds::to_string(a.op) && b.layer == a.layer && b.site == a.site &&
                      b.owner == ds::to_string(a.owner) && b.rank == rank && b.arithmetic_identity == a.identity &&
                      b.weights == a.weights && b.inputs == ids(a.inputs) && b.outputs == ids(a.outputs), "lossy plan translation");
                check(i == 0 ? b.dependencies.empty() : b.dependencies == std::vector<std::uint64_t>{i - 1U}, "operation order");
                for (const auto value : a.inputs)
                    check(buffer(plan, "activation/" + std::string(ds::to_string(value))).bytes == native.buffer_elements(value) * 4U,
                          "activation bytes");
                if (a.op == ds::Op::routed_experts)
                    check(b.expert_begin == rank * native.expert_split && b.expert_end == (rank + 1U) * native.expert_split,
                          "TP2 expert ownership");
                if (a.layer >= 0) {
                    const auto& attributes = model.layers[static_cast<std::size_t>(a.layer)];
                    const auto prefix = "layer/" + std::to_string(a.layer) + '/';
                    if (a.op == ds::Op::window_write)
                        check(contains(b.state_reads, prefix + "window") && contains(b.state_writes, prefix + "window"), "ring read/write");
                    if (a.op == ds::Op::compress)
                        check(contains(b.state_writes, prefix + "compressed_kv") && contains(b.state_writes, prefix + "index_keys") &&
                              (!attributes.compressor_gate || contains(b.state_writes, prefix + "compress_carry")), "compressor effects");
                    if (a.op == ds::Op::index) {
                        check(contains(b.state_reads, "layer/" + std::to_string(attributes.kv_source_layer) + "/index_keys") &&
                              contains(b.state_writes, prefix + "selection"), "index source effects");
                        if (attributes.uses_candidates)
                            check(contains(b.state_reads, "layer/" + std::to_string(model.config.candidate_source_layer_id) + "/candidates"),
                                  "candidate source read");
                    }
                    if (a.op == ds::Op::attention) {
                        check(contains(b.state_reads, prefix + "window"), "attention ring read");
                        if (attributes.compress_ratio > 0)
                            check(contains(b.state_reads, "layer/" + std::to_string(attributes.kv_source_layer) + "/compressed_kv") &&
                                  contains(b.state_reads, "layer/" + std::to_string(attributes.index_source_layer) + "/selection"),
                                  "cross-layer attention reads");
                    }
                    if (a.op == ds::Op::engram) check(contains(b.state_reads, "model/token_history"), "engram history read");
                }
            }
            for (const auto& s : native.state) {
                const auto prefix = "layer/" + std::to_string(s.layer) + '/';
                check(buffer(plan, prefix + "window").bytes == s.window_ring_bytes, "window bytes");
                if (s.kv_bytes_per_position > 0)
                    check(buffer(plan, prefix + "compressed_kv").bytes_per_position == s.kv_bytes_per_position &&
                          buffer(plan, prefix + "index_keys").bytes_per_position == s.index_key_bytes_per_position, "compressed state bytes");
                if (s.compress_carry_bytes > 0)
                    check(buffer(plan, prefix + "compress_carry").bytes == s.compress_carry_bytes, "carry bytes");
            }
            for (const auto& b : plan.buffers) if (b.external) {
                ++external_states;
                check(b.live_in && b.layer < first && b.lifetime == mg::BufferLifetime::sequence, "partial state import");
            }
            (void)strix::serve::parse_json(mg::dump(plan));
            ++projections;
        }
    }
    check(external_states > 0, "no cross-range sources exercised");
    const auto native = ds::build_decode_plan(model, 3, 7, false, true);
    const auto plan = ds::common_decode_plan(native, graph, 0);
    const auto bad = [&](auto mutate) {
        auto copy = plan; mutate(copy);
        rejects([&] { mg::validate(copy, graph); }, "corrupt common plan accepted");
    };
    bad([](auto& p) { p.nodes[0].inputs.push_back("missing"); });
    bad([](auto& p) { p.nodes[0].rank = 1; });
    bad([](auto& p) { p.nodes[0].owner = "unknown"; });
    bad([](auto& p) { p.nodes[0].layer = 40; });
    bad([](auto& p) { p.nodes[0].dependencies.push_back(0); });
    bad([](auto& p) { p.nodes[0].arithmetic_identity.clear(); });
    bad([](auto& p) { p.buffers[0].last_use = 0; });
    bad([](auto& p) { p.graph_identity = "wrong"; });
    bad([](auto& p) {
        auto it = std::find_if(p.buffers.begin(), p.buffers.end(), [](const auto& b) { return b.external; });
        it->state_role = "unknown";
    });
    bad([](auto& p) {
        auto it = std::find_if(p.buffers.begin(), p.buffers.end(), [](const auto& b) { return b.external; });
        p.nodes[0].state_writes.push_back(it->id); it->first_use = 0;
    });
    rejects([&] { (void)ds::common_decode_plan(native, graph, 2); }, "rank 2 accepted");
    auto broken_native = native; broken_native.steps.front().layer = 40;
    rejects([&] { (void)ds::common_decode_plan(broken_native, graph, 0); }, "invalid source layer accepted");
    broken_native = native; broken_native.state.pop_back();
    rejects([&] { (void)ds::common_decode_plan(broken_native, graph, 0); }, "missing source state accepted");
    auto other_graph = graph; other_graph.layers[0].dimensions["head_dim"] += 1;
    rejects([&] { (void)ds::common_decode_plan(native, other_graph, 0); }, "wrong graph accepted");
    std::cout << "PASS: DS41 description/graph, " << projections << " TP2/range/head projections, "
              << external_states << " external state views; corruption rejection\n";
    return 0;
} catch (const std::exception& error) {
    std::cerr << "FAIL: " << error.what() << '\n';
    return 1;
}
