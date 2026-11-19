// Host-only description/identity/inventory fixtures. No model payload or device.
#include "../src/glm/load/glm_model_description.hpp"
#include "strix/prefill_row_liveness.hpp"
#include "../src/glm/load/glm_rank_plan_graph.hpp"
#include "strix/serve/json.hpp"

#include <algorithm>
#include <chrono>
#include <cstdio>
#include <filesystem>
#include <fstream>
#include <functional>
#include <iterator>
#include <stdexcept>
#include <string>
#include <vector>

namespace md = strix::model_description;
namespace gm = strix::glm::model;
namespace mg = strix::model_graph;
namespace gl = strix::glm::load;
namespace {
int checks = 0, failures = 0;
#define CHECK(c) do { ++checks; if (!(c)) { ++failures; std::fprintf(stderr, "FAIL line %d: %s\n", __LINE__, #c); } } while (false)
void rejects(const std::function<void()>& action) {
    bool threw = false;
    try { action(); } catch (const std::exception&) { threw = true; }
    CHECK(threw);
}
std::vector<std::string> inventory() {
    // Independent fixture of the tensor families rank_plan consumes. Include
    // logical EXL3 group names and physical index components in separate tests.
    std::vector<std::string> names;
    for (unsigned id = 0; id < 46; ++id) {
        const auto prefix = "model.language_model.layers." + std::to_string(id) + '.';
        names.push_back(prefix + (id != 45 && id % 4 != 3 ? "self_attn.qkv_proj" : "self_attn.kv_a_proj_with_mqa"));
        names.push_back(prefix + (id < 3 ? "mlp.gate_proj" : "mlp.experts.0.gate_proj"));
        if (id >= 3) names.push_back(prefix + "mlp.shared_experts.gate_proj");
        if (id == 45) names.push_back(prefix + "eh_proj");
    }
    return names;
}
void layer_contract(const mg::ModelGraph& graph) {
    CHECK(graph.layers.size() == 45);
    // Row liveness (strix/prefill_row_liveness.hpp): every trunk layer declares state, so prompt rows stop nowhere early.
    { const auto lp = strix::prefill_liveness::plan_graph(graph, {.taps = {}, .every_row = false, .head_row = false, .tail_row_invariant = false});
      CHECK(!lp.cut() && lp.cache_only_last && strix::prefill_liveness::skipped_row_layers(lp, 512) == 0); }
    unsigned kda = 0, indexed = 0;
    for (std::size_t id = 0; id < graph.layers.size(); ++id) {
        const auto& layer = graph.layers[id];
        CHECK(layer.id == id);
        CHECK(layer.block == mg::BlockKind::transformer);
        CHECK(layer.attention == (id % 4 != 3 ? mg::AttentionFamily::kda : mg::AttentionFamily::compressed_indexed));
        if (layer.attention == mg::AttentionFamily::kda) ++kda; else ++indexed;
        CHECK(layer.mlps[0].kind == (id < 3 ? mg::MlpKind::dense : mg::MlpKind::moe));
        CHECK(layer.mlps.size() == (id < 3 ? 1u : 2u));
        if (id >= 3) CHECK(layer.mlps[1].kind == mg::MlpKind::shared_expert);
        CHECK(layer.norms[0].placement == mg::NormPlacement::pre_attention);
        CHECK(layer.norms[1].placement == mg::NormPlacement::pre_mlp);
        CHECK(layer.hyper_connection && layer.hyper_connection->kind == mg::HyperConnectionKind::mhc);
        CHECK(layer.hyper_connection->streams == 4 && layer.hyper_connection->iterations == 20);
        CHECK(layer.states.size() == (id % 4 != 3 ? 2u : 3u));
    }
    CHECK(kda == 34 && indexed == 11);
    CHECK(graph.attachments.size() == 1);
    const auto& mtp = graph.attachments[0];
    CHECK(mtp.kind == mg::AttachmentKind::mtp && mtp.layers.size() == 1);
    CHECK(mtp.layers[0].id == 45);
    CHECK(mtp.layers[0].attention == mg::AttentionFamily::compressed_indexed);
    CHECK(!mtp.layers[0].hyper_connection);
    CHECK(mtp.layers[0].mlps[1].kind == mg::MlpKind::shared_expert);
}
void identity_contract(const md::ModelDescription& description, const mg::ModelGraph& graph) {
    const auto original = mg::identity(graph);
    CHECK(original.size() == 16);
    CHECK(original == mg::identity(gm::build_model_graph(description)));
    CHECK(mg::canonical(graph) == mg::canonical(gm::build_model_graph(description)));
    const std::vector<std::function<void(mg::ModelGraph&)>> changes = {
        [](auto& g) { g.model += "changed"; },
        [](auto& g) { g.semantic_version += "changed"; },
        [](auto& g) { ++g.dimensions["dims.hidden"]; },
        [](auto& g) { g.semantics["arithmetic.rms_eps"] += "changed"; },
        [](auto& g) { std::swap(g.layers[0], g.layers[1]); },
        [](auto& g) { g.layers[0].attention = mg::AttentionFamily::gdn; },
        [](auto& g) { g.layers[0].block = mg::BlockKind::recurrent; },
        [](auto& g) { ++g.layers[0].mlps[0].intermediate; },
        [](auto& g) { g.layers[0].norms[0].placement = mg::NormPlacement::post_attention; },
        [](auto& g) { ++g.layers[0].hyper_connection->streams; },
        [](auto& g) { g.layers[0].tensors[0].format += "changed"; },
        [](auto& g) { g.layers[0].states[0].lifetime += "changed"; },
        [](auto& g) { g.attachments[0].target += "changed"; },
        [](auto& g) { g.attachments[0].layers[0].norms.clear(); },
        [](auto& g) { g.edges[0].kind = mg::EdgeKind::cross_layer; },
        [](auto& g) { g.inputs[0].format += "changed"; },
        [](auto& g) { g.outputs[0].ownership += "changed"; },
        [](auto& g) { g.tensors[0].shape.push_back("changed"); },
    };
    for (const auto& change : changes) { auto altered = graph; change(altered); CHECK(original != mg::identity(altered)); }
    auto escaped = graph;
    escaped.semantics["escape"] = "quote\" slash\\ newline\n tab\t";
    const auto json = strix::serve::parse_json_unique(mg::dump(escaped));
    CHECK(json.find("identity")->as_string() == mg::identity(escaped));
    CHECK(json.find("semantics")->find("escape")->as_string() == escaped.semantics.at("escape"));
    // Exercise future family/attachment vocabulary without adding engine fields.
    for (const auto family : {mg::AttentionFamily::gdn, mg::AttentionFamily::qsa, mg::AttentionFamily::mla, mg::AttentionFamily::msa}) {
        auto future = graph;
        future.layers[0].attention = family;
        future.attachments[0].kind = mg::AttachmentKind::ple;
        CHECK(!mg::dump(future).empty());
        future.attachments[0].kind = mg::AttachmentKind::draft;
        CHECK(!mg::dump(future).empty());
    }
}
void mismatch_contract(const mg::ModelGraph& graph) {
    const auto names = inventory();
    gl::check_rank_plan_graph(graph, names);
    auto physical = names;
    for (auto& name : physical) name += ".trellis";
    std::reverse(physical.begin(), physical.end());
    gl::check_rank_plan_graph(graph, physical);
    CHECK(true);
    auto altered = graph;
    std::swap(altered.layers[0], altered.layers[1]);
    rejects([&] { gl::check_rank_plan_graph(altered, names); });
    altered = graph; altered.layers.pop_back();
    rejects([&] { gl::check_rank_plan_graph(altered, names); });
    altered = graph; altered.layers[3].attention = mg::AttentionFamily::kda;
    rejects([&] { gl::check_rank_plan_graph(altered, names); });
    altered = graph; altered.layers[3].mlps.pop_back();
    rejects([&] { gl::check_rank_plan_graph(altered, names); });
    altered = graph; altered.attachments.clear();
    rejects([&] { gl::check_rank_plan_graph(altered, names); });
    for (const auto& suffix : {"self_attn.qkv_proj", "mlp.shared_experts.gate_proj", "eh_proj"}) {
        auto missing = names;
        missing.erase(std::remove_if(missing.begin(), missing.end(), [&](const auto& name) { return name.ends_with(suffix); }), missing.end());
        rejects([&] { gl::check_rank_plan_graph(graph, missing); });
    }
    auto extra = names; extra.push_back("model.language_model.layers.46.mlp.gate_proj");
    rejects([&] { gl::check_rank_plan_graph(graph, extra); });
    extra = names; extra.push_back("model.language_model.layers.0.self_attn.kv_a_proj_with_mqa");
    rejects([&] { gl::check_rank_plan_graph(graph, extra); });
    extra = names; extra.push_back("model.language_model.layers.bad.weight");
    rejects([&] { gl::check_rank_plan_graph(graph, extra); });
}
void directory_contract(const std::string& fixture, const mg::ModelGraph& graph) {
    const auto root = std::filesystem::temp_directory_path() /
        ("strix-graph-fixture-" + std::to_string(std::chrono::steady_clock::now().time_since_epoch().count()));
    if (!std::filesystem::create_directory(root)) throw std::runtime_error("fixture directory already exists");
    struct Cleanup { std::filesystem::path path; ~Cleanup() { std::error_code ignored; std::filesystem::remove_all(path, ignored); } } cleanup{root};
    std::filesystem::copy_file(fixture, root / "glm-5.3-flash.toml");
    const auto write_index = [&](const std::vector<std::string>& names) {
        std::ofstream file(root / "model.safetensors.index.json");
        file << "{\"weight_map\":{";
        bool comma = false;
        for (const auto& name : names) { if (comma) file << ','; comma = true; file << '"' << name << ".trellis\":\"fixture.safetensors\""; }
        file << "}}";
    };
    write_index(inventory());
    CHECK(gm::print_model_graph(root.string()) == mg::dump(graph));
    auto names = inventory(); names.pop_back(); write_index(names);
    rejects([&] { (void)gm::print_model_graph(root.string()); });
    write_index(inventory());
    { std::ofstream file(root / "config.json"); file << "{\"hidden_size\":1}"; }
    rejects([&] { (void)gm::print_model_graph(root.string()); });
    std::filesystem::remove(root / "config.json");
    std::filesystem::remove(root / "glm-5.3-flash.toml");
    rejects([&] { (void)gm::print_model_graph(root.string()); });
}
}  // namespace
int main(int argc, char** argv) {
    try {
        if (argc != 2) throw std::runtime_error("usage: glm-model-graph-tests MODEL.toml");
        const auto description = md::load(argv[1]);
        const auto graph = gm::build_model_graph(description);
        layer_contract(graph); identity_contract(description, graph); mismatch_contract(graph);
        auto rejected = description; rejected.fields["name"] = md::Value{std::string("unsupported")};
        rejects([&] { (void)gm::build_model_graph(rejected); });
        auto malformed = graph; malformed.layers[1].id = 0;
        rejects([&] { mg::validate(malformed); });
        directory_contract(argv[1], graph);
        std::printf("graph identity: %s\n", mg::identity(graph).c_str());
    } catch (const std::exception& e) { ++failures; std::fprintf(stderr, "exception: %s\n", e.what()); }
    std::printf("glm model graph: %d checks, %d failures\n", checks, failures);
    return failures == 0 ? 0 : 1;
}
