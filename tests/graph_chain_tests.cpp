#include "model_graph/dense_adapter.hpp"
#include "model_graph/qwen_adapter.hpp"
#include "stage_bridge.hpp"
#include "strix/device/profiles.hpp"
#include "strix/graph/launch_plan.hpp"

#ifdef STRIX_D3_HAS_DS41
#include "model_graph/deepseek41_adapter.hpp"
#endif

#include <algorithm>
#include <array>
#include <cinttypes>
#include <cstdio>
#include <stdexcept>
#include <string>
#include <utility>

namespace {

using namespace strix::graph;
using StageIndex = std::uint32_t;
namespace mg = strix::model_graph;

void check(bool ok, const char* message) {
    if (!ok) throw std::runtime_error(message);
}

template<class F>
void rejects(F run, const char* message) {
    try { run(); } catch (const std::invalid_argument&) { return; }
    throw std::runtime_error(message);
}

constexpr StageDecl stage(std::string_view id, Op op, std::string_view out,
                          std::array<std::string_view, 4> in = {}, std::string_view form = {}) {
    StageDecl s{.id = id, .op = op, .form = form, .entry = id, .site = "layer",
                .in = {in[0], in[1], in[2], in[3]}, .out = out};
    if (op == Op::linear || op == Op::head) s.operand = weights(id, 64, 64);
    return s;
}

std::vector<std::string_view> entries(const LaunchPlan& plan) {
    std::vector<std::string_view> result;
    for (const auto& launch : plan.launches) result.push_back(launch.entry);
    return result;
}

std::vector<Form> forms(const LaunchPlan& plan, ChainId id) {
    std::vector<Form> result;
    const auto& chain = plan.chains.at(id);
    for (std::size_t i = 0; i < chain.count; ++i) result.push_back(chain.links[i].form);
    return result;
}

StageGraph layer_graph(std::span<const StageDecl> stages) {
    static constexpr FormatRule formats[] = {{"", Format::f32, {}}};
    return {.layer = stages, .layers = {trait::attn_dense | trait::ffn_dense}, .formats = formats};
}

void test_chains() {
    static_assert(sizeof(Form) == 2);
    for (std::uint16_t id = 1; id <= static_cast<std::uint16_t>(Form::index_pool_append); ++id) {
        const auto form = static_cast<Form>(id);
        check(!form_name(form).empty() && parse_form(form_name(form)) == form, "form ID round trip");
    }
    check(!parse_form("model.private"), "unknown form is unrecognized");
    ChainTable table, other;
    Chain first{};
    first.links[0] = {Form::rope, 1};
    first.links[1] = {Form::store_bf16, 0};
    first.count = 2;
    const auto id = table.intern(first);
    first.links[7] = {Form::softcap, 255};
    check(table.intern(first) == id && table.intern({}) == 0 && table.at(0).count == 0, "intern across unused bytes");
    std::swap(first.links[0], first.links[1]);
    const auto reversed = table.intern(first);
    check(reversed != id && table.identity(reversed) != table.identity(id), "link order is part of identity");
    check(other.identity(other.intern(first)) == table.identity(reversed), "chain identity is independent of table IDs");
    first.links[1].arg = 2;
    check(table.intern(first) != reversed, "arg is part of identity");
    first.count = 9;
    rejects([&] { (void)table.intern(first); }, "oversized chain accepted");
    first.count = 1;
    first.links[0] = {Form::none, 0};
    rejects([&] { (void)table.intern(first); }, "invalid form accepted");
}

constexpr StageDecl dense_stages[] = {
    stage("attn_norm", Op::norm, "xn", {"x"}),
    stage("q", Op::linear, "q", {"xn"}),
    stage("k", Op::linear, "k", {"xn"}),
    stage("v", Op::linear, "v", {"xn"}),
    stage("rope_q", Op::elementwise, "qr", {"q"}, "rope"),
    stage("rope_k", Op::elementwise, "kr", {"k"}, "rope"),
    stage("kv_append", Op::store, "kv", {"kr"}, "kv.append"),
    stage("attend", Op::attend, "ctx", {"qr", "kr", "v", "kv"}),
    stage("o_proj", Op::linear, "o", {"ctx"}),
    stage("attn_residual", Op::combine, "r", {"o", "x"}, "residual.add"),
    stage("ffn_norm", Op::norm, "rn", {"r"}),
    stage("gate", Op::linear, "g", {"rn"}),
    stage("up", Op::linear, "u", {"rn"}),
    stage("silu_mul", Op::elementwise, "gu", {"g", "u"}, "act.silu_mul"),
    stage("down", Op::linear, "d", {"gu"}),
    stage("ffn_residual", Op::combine, "y", {"d", "r"}, "residual.add")
};

void print_layer(const char* kind, const LaunchPlan& plan) {
    std::printf("CORE-SPEC 7.5 %s (host plan, layer 0):\n", kind);
    for (const auto& l : plan.launches) {
        std::printf("  %.*s", static_cast<int>(l.entry.size()), l.entry.data());
        for (std::size_t j = 0; j < l.jobs; ++j) {
            const auto key = l.job_keys[j];
            std::printf(" %.*s[", static_cast<int>(key.size()), key.data());
            for (const auto f : forms(plan, l.job_prologue_chain[j])) {
                const auto name = form_name(f);
                std::printf(" +%.*s", static_cast<int>(name.size()), name.data());
            }
            for (const auto f : forms(plan, l.epilogue_chain[j])) {
                const auto name = form_name(f);
                std::printf(" -%.*s", static_cast<int>(name.size()), name.data());
            }
            std::printf(" ]");
        }
        std::puts("");
    }
}

void dense_layer() {
    strix::model::DenseConfig config;
    config.hidden = 64; config.layers = 1; config.heads = 4; config.kv_heads = 2;
    config.intermediate = 64; config.head_dim = 16; config.vocab = 64; config.context = 128;
    config.rope.dimensions = 16;
    const auto model = mg::dense_decoder_graph(config, "dense-test", "v1");
    const auto graph = mg::to_stage_graph(model, layer_graph(dense_stages));
    const auto today = launch_plan(graph, 0, 1, {.mode = Fusion::today});
    const std::vector<std::string_view> expected{"attn_norm", "q", "k", "v", "rope_q", "rope_k", "kv_append",
        "attend", "o_proj", "attn_residual", "ffn_norm", "gate", "up", "silu_mul", "down", "ffn_residual"};
    check(entries(today) == expected && today.kernel_count() == expected.size(), "dense today launch list");
    const auto identity = plan_identity(today);
    check(identity == 0x52ad491525a9f378ULL, "dense today identity");
    std::printf("plan_identity_dense=%016" PRIx64 "\n", identity);
    constexpr PlanStage bindings[] = {
        {.stage = &dense_stages[13], .paired_tile = true}
    };
    const auto fused = launch_plan(graph, PlanRequest{.stages = bindings});
    check(entries(fused) == std::vector<std::string_view>{"q", "attend", "o_proj", "gate", "down"},
          "dense attention five-launch list");
    check(fused.launches[0].jobs == 3 && fused.launches[3].jobs == 2 &&
          forms(fused, fused.launches[0].prologue_chain) == std::vector{Form::norm_rms} &&
          forms(fused, fused.launches[0].epilogue_chain[1]) == std::vector{Form::rope, Form::kv_append} &&
          forms(fused, fused.launches[3].epilogue_chain[0]) == std::vector{Form::act_silu_mul} &&
          forms(fused, fused.launches.back().epilogue_chain[0]) == std::vector{Form::residual_add}, "dense layer chains");
    print_layer("dense attention", fused);
    const auto state = launch_plan(graph, PlanRequest{.phase = Phase::state_only, .stages = bindings});
    check(entries(state) == std::vector<std::string_view>{"k"} && state.instances.size() == 4 &&
          forms(state, state.launches.front().epilogue_chain[0]) == std::vector{Form::rope, Form::kv_append},
          "state-only plan keeps only transitive state producers");
}

void gdn_layer() {
    constexpr StageDecl stages[] = {
        dense_stages[0],
        stage("front_qkv", Op::linear, "qkv", {"xn"}),
        stage("front_z", Op::linear, "z", {"xn"}),
        stage("conv", Op::store, "conv_pending", {"qkv"}, "conv.tap"),
        stage("recurrence", Op::mix, "core", {"qkv", "z", "conv_pending"}),
        stage("commit", Op::store, "recur_pending", {"core"}, "recur.commit"),
        stage("o_proj", Op::linear, "o", {"core"}),
        dense_stages[9], dense_stages[10], dense_stages[11], dense_stages[12],
        dense_stages[13], dense_stages[14], dense_stages[15]
    };
    auto graph = layer_graph(stages);
    graph.layers[0] = trait::attn_gdn | trait::ffn_dense;
    const PlanStage bindings[] = {{.stage = &stages[11], .paired_tile = true}};
    const auto fused = launch_plan(graph, PlanRequest{.stages = bindings});
    check(entries(fused) == std::vector<std::string_view>{"front_qkv", "recurrence", "o_proj", "gate", "down"} &&
          forms(fused, fused.launches[0].epilogue_chain[0]) == std::vector{Form::conv_tap} &&
          forms(fused, fused.launches[1].epilogue_chain[0]) == std::vector{Form::recur_commit},
          "GDN five-launch list with convolution and recurrence state epilogues");
    print_layer("dense GDN", fused);
}

// The incumbent adapters expose logical calls. They emit no physical kernel-count
// trace. Build a one-call-per-node check table and compare the full source order.
template<class Source, class Name>
std::uint64_t check_adapter(const mg::ExecutionPlan& adapter, const Source& source, Name name) {
    std::vector<StageDecl> declarations;
    std::vector<std::string> ids;
    check(adapter.nodes.size() == source.steps.size(), "adapter call count changed");
    declarations.reserve(adapter.nodes.size());
    ids.reserve(adapter.nodes.size());
    for (std::size_t i = 0; i < adapter.nodes.size(); ++i) {
        const auto& node = adapter.nodes[i];
        const auto [op, layer] = name(source.steps[i]);
        check(node.operation == op && node.layer == layer, "adapter operation or layer changed");
        ids.push_back(std::to_string(node.layer) + "/" + node.site);
        declarations.push_back(stage(ids.back(), Op::elementwise, {}, {}, node.operation));
        declarations.back().entry = node.operation;
        declarations.back().numeric = node.arithmetic_identity;
    }
    const StageGraph graph{.before = declarations};
    const auto plan = launch_plan(graph, 0, 1, {.mode = Fusion::today});
    check(plan.kernel_count() == source.steps.size(), "today call count changed");
    for (std::size_t i = 0; i < source.steps.size(); ++i) {
        const auto& l = plan.launches.at(i);
        check(l.entry == name(source.steps[i]).first && l.kernels == 1 && l.stages == std::vector{static_cast<StageIndex>(i)} &&
              l.prologue.empty() && l.epilogue.empty(), "today call entry or stage changed");
    }
    const auto identity = plan_identity(plan);
    const auto repeated = launch_plan(graph, 0, 1, {.mode = Fusion::today});
    check(identity == plan_identity(repeated), "today identity changed on repeat");
    return identity;
}

void incumbent_checks(const char* qwen_model) {
    namespace qd = strix::qwen_description;
    const auto description = strix::model_description::load(qwen_model);
    const auto source = strix::build_serial_decode_execution_plan(qd::contract_from_description(description));
    const auto adapter = qd::common_decode_plan(source, qd::model_graph_from_description(description));
    const auto qwen = check_adapter(adapter, source, [](const auto& s) {
        return std::pair{strix::to_string(s.op), s.layer ? static_cast<std::int64_t>(*s.layer) : -1};
    });
    check(qwen == 0x434c56d6fae717dbULL, "Qwen today identity");
    std::printf("plan_identity_qwen=%016" PRIx64 "\n", qwen);

#ifdef STRIX_D3_HAS_DS41
    namespace ds = strix::deepseek41;
#include "deepseek41_real_config.inc"
    ds::TargetModel model;
    model.config = ds::parse_config_json(real_config_json);
    model.layers = ds::layer_attributes(model.config);
    for (const auto& expected : ds::expected_target_tensors(model.config)) {
        strix::ManifestTensor tensor;
        tensor.name = expected.name;
        tensor.dimensions = expected.dimensions;
        tensor.ggml_type = expected.ggml_type;
        tensor.role = expected.role;
        tensor.residency = expected.residency;
        model.manifest.tensors.push_back(std::move(tensor));
    }
    const auto native = ds::build_decode_plan(model, 0, model.config.num_hidden_layers);
    const auto graph = ds::model_graph_from_config(model.config);
    for (std::uint32_t rank = 0; rank < native.ranks; ++rank) {
        const auto ds41 = check_adapter(ds::common_decode_plan(native, graph, rank), native, [](const auto& s) {
            return std::pair{ds::to_string(s.op), s.layer};
        });
        check(ds41 == 0x228840a58acdcf42ULL, "DS41 today identity");
        std::printf("plan_identity_ds41_rank_%" PRIu32 "=%016" PRIx64 "\n", rank, ds41);
    }
#endif
}

void wave_pack() {
    strix::device::Profile device{}; device.vendor = "nvidia"; device.units = 4;
    constexpr StageDecl stages[] = {
        stage("a", Op::linear, "a", {"x"}), stage("b", Op::linear, "b", {"y"}),
        stage("c", Op::linear, "c", {"z"}), stage("d", Op::linear, "d", {"w"}),
        stage("consumer", Op::linear, "e", {"a"})
    };
    const PlanStage jobs[] = {
        {.stage = &stages[0], .tiles = 4, .wave_key = "wave"},
        {.stage = &stages[1], .input_transform = {Form::load_round_bf16, 0}, .tiles = 4, .wave_key = "wave"},
        {.stage = &stages[2], .tiles = 2, .wave_key = "wave"},
        {.stage = &stages[3], .tiles = 0, .wave_key = "wave", .split_k_key = "unknown_split"},
        {.stage = &stages[4], .tiles = 1, .wave_key = "wave", .split_k_key = "ordered_split"}
    };
    const auto p = launch_plan(layer_graph(stages), PlanRequest{.device = &device, .stages = jobs});
    check(entries(p) == std::vector<std::string_view>{"wave", "wave", "d"} && p.launches[0].jobs == 2 &&
          p.launches[1].jobs == 2 && p.launches[0].pack == Pack::wave &&
          p.launches[0].prologue_chain == 0 &&
          forms(p, p.launches[0].job_prologue_chain[1]) == std::vector{Form::load_round_bf16} &&
          p.launches[1].job_keys[1] == "consumer", "wave packing, per-job prologue and two-wave limit");
    const auto split = launch_plan(layer_graph(std::span(stages).subspan(4)),
                                   PlanRequest{.device = &device, .stages = jobs});
    check(entries(split) == std::vector<std::string_view>{"ordered_split"}, "isolated tiny grid takes ordered split-K");
    const auto dependent = launch_plan(layer_graph(std::span(stages).first(1)), PlanRequest{});
    check(entries(dependent) == std::vector<std::string_view>{"a"}, "no wave geometry is assumed");
}

void fixed_packs() {
    strix::device::Profile device{}; device.vendor = "nvidia"; device.units = 4;
    constexpr StageDecl sum_stages[] = {
        stage("routed", Op::linear, "merged", {"r"}), stage("shared", Op::linear, "merged", {"s"})
    };
    constexpr std::string_view sum_jobs[] = {"routed", "shared"};
    const PlanPack sum[] = {{Pack::sum, "sum", sum_jobs}};
    const auto ordered = launch_plan(layer_graph(sum_stages), PlanRequest{.packs = sum});
    check(ordered.launches.size() == 1 && ordered.launches[0].pack == Pack::sum &&
          ordered.launches[0].job_keys[0] == "routed" && ordered.launches[0].job_keys[1] == "shared", "fixed sum pack");
    const StageDecl reversed[] = {sum_stages[1], sum_stages[0]};
    check(launch_plan(layer_graph(reversed), PlanRequest{.packs = sum}).launches.size() == 2, "sum order is fixed");
    constexpr std::string_view reverse_jobs[] = {"shared", "routed"};
    const PlanPack reverse_sum[] = {{Pack::sum, "sum", reverse_jobs}};
    check(plan_identity(ordered) != plan_identity(launch_plan(layer_graph(reversed), PlanRequest{.packs = reverse_sum})),
          "sum identity includes job order");

    constexpr StageDecl pipe_stages[] = {
        stage("first", Op::linear, "a", {"x"}), stage("second", Op::linear, "b", {"a"}),
        stage("third", Op::linear, "c", {"b"})
    };
    constexpr std::string_view pipe_jobs[] = {"first", "second", "third"};
    const PlanPack pipe[] = {{Pack::pipe, "pipe", pipe_jobs}};
    const auto fused = launch_plan(layer_graph(pipe_stages), PlanRequest{.packs = pipe});
    check(fused.launches.size() == 1 && fused.launches[0].pack == Pack::pipe && fused.launches[0].jobs == 3, "pipe pack");
    auto illegal = fused;
    illegal.launches[0].pack = Pack::wave;
    rejects([&] { validate_plan(illegal); }, "a wave may not read another job's output");
    const PlanPack fallback[] = {{Pack::pipe, "", pipe_jobs}};
    const auto separate = launch_plan(layer_graph(pipe_stages), PlanRequest{.device = &device, .packs = fallback});
    check(separate.launches.size() == 3 && separate.launches[2].after == 1 && separate.launches[2].dependent,
          "pipe without an instance keeps a dependent launch chain");
}

void plan_rules() {
    constexpr StageDecl stages[] = {
        stage("seed", Op::linear, "row", {"x"}),
        stage("a8", Op::linear, "q", {"row"}),
        stage("fp", Op::linear, "v", {"row"}),
        stage("select", Op::select, "keys", {"q"}),
        stage("attend", Op::attend, "context", {"v", "keys"}),
        stage("published", Op::store, "tx", {"context"}, "xch.publish"),
        stage("rx", Op::copy, "receive", {"rx_wire"}, "xch.join"),
        stage("host", Op::copy, "ready", {"host_wire"}, "wait.host"),
        stage("head", Op::head, "logits", {"context", "receive", "ready"}),
        stage("cap", Op::elementwise, "scores", {"logits"}, "softcap"),
        stage("pick", Op::argmax, "top", {"scores"}),
        stage("sample", Op::argmax, "token", {"top"})
    };
    PlanStage bindings[] = {
        {.stage = &stages[1], .operand_store = Form::store_a8g32, .input_transform = {Form::rotate_hadamard, 1}},
        {.stage = &stages[2], .operand_store = Form::store_f32, .input_transform = {Form::rotate_hadamard, 1}},
        {.stage = &stages[3], .pool_keys = 4096},
        {.stage = &stages[5], .publish_frames = 2},
        {.stage = &stages[8], .head_emit = {Form::emit_topn, 2}},
        {.stage = &stages[11], .sample = true}
    };
    const PlanSidecar sidecar[] = {{"seed", "read_ahead", {}}, {"head", "warm", "rx", 2}};
    const auto graph = layer_graph(stages);
    const auto p = launch_plan(graph, PlanRequest{.rows = 4, .stages = bindings, .sidecars = sidecar});
    check(entries(p) == std::vector<std::string_view>{"seed", "read_ahead", "a8", "fp", "attend", "warm", "head", "pick"},
          "fused rules 8 through 15");
    check(forms(p, p.launches[0].epilogue_chain[0]) ==
          std::vector{Form::rotate_hadamard, Form::store_f32, Form::store_a8g32}, "cross-stage transform then multi-store");
    check(p.launches[2].prologue_chain == 0 && p.launches[4].prologue == std::vector<StageIndex>{3} &&
          forms(p, p.launches[4].epilogue_chain[0]) == std::vector{Form::xch_publish} &&
          p.launches[4].publish_frames[0] == 2, "selection and framed publish");
    check(forms(p, p.launches[6].prologue_chain) == std::vector{Form::xch_join, Form::wait_host} &&
          forms(p, p.launches[6].epilogue_chain[0]) == std::vector{Form::softcap, Form::emit_topn} &&
          p.launches[7].stages == std::vector<StageIndex>{10, 11}, "exchange prologues, head emission and sampled pick");
    check(p.launches[1].sidecar && p.launches[1].wait_for == std::vector<std::uint32_t>{0} &&
          p.launches[5].sidecar && p.launches[5].during == "rx" && p.launches[5].stream == 2 &&
          p.launches[6].rows == 4, "attached and exchange-triggered sidecars preserve rows");

    bindings[1].input_transform = {};
    bindings[2].pool_keys = 4097;
    const auto large = launch_plan(graph, PlanRequest{.stages = bindings});
    check(entries(large) == std::vector<std::string_view>{"seed", "a8", "fp", "select", "attend", "head", "pick"} &&
          forms(large, large.launches[0].epilogue_chain[0]) == std::vector{Form::store_f32, Form::store_a8g32} &&
          forms(large, large.launches[1].job_prologue_chain[0]) == std::vector{Form::rotate_hadamard} &&
          large.launches[2].prologue_chain == 0, "large pool and disagreed transform stay separate");
}

void regimes_and_streams() {
    StageDecl stages[] = {
        stage("dense", Op::attend, "a", {"x"}),
        stage("selected", Op::attend, "a", {"x"}),
        stage("pool_close", Op::store, "pool", {"a"}, "index_pool.append"),
        stage("independent", Op::linear, "b", {"y"}),
        stage("dependent", Op::mix, "c", {"b", "a"})
    };
    stages[0].pos = Pos::prefix_dense;
    stages[1].pos = Pos::prefix_select;
    stages[2].pos = Pos::group_close;
    auto graph = layer_graph(stages);
    graph.select_after = 8;
    graph.group = 4;
    strix::device::Profile device{}; device.units = 4;
    const PlanOverlap overlap[] = {{"selected", "independent", 10, 6}, {"independent", "dependent", 10, 1}};
    const auto p = launch_plan(graph, PlanRequest{.phase = Phase::verify, .rows = 8, .device = &device,
                                                  .slot = 11, .overlap = overlap});
    check(entries(p) == std::vector<std::string_view>{"selected", "independent", "dependent"} &&
          p.launches[0].regime.pos == Pos::group_close && p.launches[0].regime.phase == Phase::verify &&
          p.launches[0].rows == 8 && forms(p, p.launches[0].epilogue_chain[0]) == std::vector{Form::index_pool_append},
          "position guards select one row-cohort variant");
    check(p.launches[1].stream == 1 && p.launches[2].stream == 0 && p.launches[2].after == 0 &&
          p.launches[2].wait_for == std::vector<std::uint32_t>{1}, "measured overlap and cross-stream dependency");
    const PlanOverlap losing[] = {{"selected", "independent", 6, 10}, {"independent", "dependent", 10, 1}};
    const auto serial = launch_plan(graph, PlanRequest{.device = &device, .slot = 11, .overlap = losing});
    check(serial.launches[1].stream == 0 && serial.launches[2].stream == 0, "dependent or losing overlap refused");
    const auto prefix = launch_plan(graph, PlanRequest{.pos = Pos::prefix_dense});
    check(prefix.launches[0].entry == "dense" && prefix.launches[0].epilogue.empty(), "dense prefix has no pool close");
    rejects([&] { (void)launch_plan(graph, PlanRequest{.pos = Pos::prefix_dense, .slot = 11}); },
            "conflicting regime accepted");
}

}  // namespace

int main(int argc, char** argv) try {
    check(argc == 2, "expected Qwen model description path");
    test_chains();
    dense_layer();
    gdn_layer();
    incumbent_checks(argv[1]);
    wave_pack();
    fixed_packs();
    plan_rules();
    regimes_and_streams();
    std::puts("graph chain tests passed");
    return 0;
} catch (const std::exception& error) {
    std::fprintf(stderr, "FAIL: %s\n", error.what());
    return 1;
}
