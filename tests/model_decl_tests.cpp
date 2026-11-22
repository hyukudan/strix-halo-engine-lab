#include "strix/graph/model_decl.hpp"
#include "strix/qwen35/config.hpp"
#include "strix/weight/admission.hpp"
#include "inkling2/inkling2_mini_config.hpp"
#include "inkling2/inkling2_mini_schedule.hpp"
#include "stage_bridge.hpp"

#include <algorithm>
#include <iostream>
#include <stdexcept>

namespace {
using namespace strix::graph;
namespace md = strix::model_description;
namespace mg = strix::model_graph;
namespace mini = strix::inkling2;
namespace weight = strix::weight;
using V = std::string_view;

void check(bool pass, V message) {
    if (!pass) throw std::runtime_error(std::string(message));
}

template<class F>
void refuses(V diagnostic, F run) {
    try {
        run();
    } catch (const std::invalid_argument& error) {
        if (V(error.what()).find(diagnostic) != V::npos) return;
        throw std::runtime_error("Expected " + std::string(diagnostic) + ", got " + error.what());
    }
    throw std::runtime_error("Missing refusal " + std::string(diagnostic));
}

void same_rows(std::span<const StageDecl> actual, std::span<const StageDecl> expected) {
    check(actual.size() == expected.size(), "stage count");
    for (std::size_t i = 0; i < expected.size(); ++i) {
        const auto& a = actual[i];
        const auto& e = expected[i];
        check(a.id == e.id && a.op == e.op && a.form == e.form && a.entry == e.entry && a.site == e.site &&
              std::ranges::equal(a.in, e.in) && a.out == e.out && a.operand.tensor == e.operand.tensor &&
              a.operand.rows == e.operand.rows && a.operand.cols == e.operand.cols &&
              a.operand.items == e.operand.items && a.operand.read == e.operand.read && a.numeric == e.numeric &&
              a.rows == e.rows && a.split == e.split && a.leaves == e.leaves && a.reduce == e.reduce &&
              a.kernels == e.kernels && a.call == e.call && a.when_all == e.when_all && a.when_none == e.when_none &&
              a.pos == e.pos, e.id);
    }
}

// Stage fields from the incumbent table in stage_graph_tests.cpp. Shape and
// layer classification come from the incumbent Qwen 27B configuration.
void qwen(const ModelDecl& decl) {
    const strix::qwen35::Config c;
    using namespace trait;
    const auto gdn_width = c.gdn_v_width();
    const auto width = c.hidden;
    const StageDecl layer[] = {
        {.id="attn_norm", .op=Op::norm, .entry="rmsnorm_fused", .site="attn", .in={"x"}, .out="xn", .reduce=true},
        {.id="qkv", .op=Op::linear, .entry="dense_iu4", .site="attn", .in={"xn"}, .out="qkv",
         .operand=weights("attn_qkv.weight", c.gdn_conv_channels(), width), .when_all=attn_gdn},
        {.id="z", .op=Op::linear, .entry="dense_iu4", .site="attn", .in={"xn"}, .out="z",
         .operand=weights("attn_gate.weight", gdn_width, width), .when_all=attn_gdn},
        {.id="beta", .op=Op::linear, .entry="linear", .site="attn", .in={"xn"}, .out="beta",
         .operand=weights("ssm_beta.weight", c.gdn_value_heads, width), .when_all=attn_gdn},
        {.id="alpha", .op=Op::linear, .entry="linear", .site="attn", .in={"xn"}, .out="alpha",
         .operand=weights("ssm_alpha.weight", c.gdn_value_heads, width), .when_all=attn_gdn},
        {.id="gdn", .op=Op::mix, .entry="gdn_step", .site="attn", .in={"qkv","z","beta","alpha"}, .out="core",
         .rows=Rows::cross_row, .when_all=attn_gdn},
        {.id="gdn_out", .op=Op::linear, .entry="dense_iu4", .site="attn", .in={"core"}, .out="proj",
         .operand=weights("ssm_out.weight", width, gdn_width), .when_all=attn_gdn},
        {.id="q", .op=Op::linear, .entry="dense_iu4", .site="attn", .in={"xn"}, .out="q",
         .operand=weights("attn_q.weight", c.q_gate_width(), width), .when_all=attn_dense},
        {.id="k", .op=Op::linear, .entry="dense_iu4", .site="attn", .in={"xn"}, .out="k",
         .operand=weights("attn_k.weight", c.kv_width(), width), .when_all=attn_dense},
        {.id="v", .op=Op::linear, .entry="dense_iu4", .site="attn", .in={"xn"}, .out="v",
         .operand=weights("attn_v.weight", c.kv_width(), width), .when_all=attn_dense},
        {.id="attend", .op=Op::attend, .entry="dense_decode_flash", .site="attn", .in={"q","k","v"}, .out="ctx",
         .rows=Rows::cross_row, .when_all=attn_dense},
        {.id="o", .op=Op::linear, .entry="dense_iu4", .site="attn", .in={"ctx"}, .out="proj",
         .operand=weights("attn_output.weight", width, c.heads * c.head_dim), .when_all=attn_dense},
        {.id="post_norm", .op=Op::norm, .entry="rmsnorm_fused", .site="ffn", .in={"x","proj"}, .out="xn2", .reduce=true},
        {.id="gate", .op=Op::linear, .entry="dense_iu4", .site="ffn", .in={"xn2"}, .out="g",
         .operand=weights("ffn_gate.weight", c.intermediate, width)},
        {.id="up", .op=Op::linear, .entry="dense_iu4", .site="ffn", .in={"xn2"}, .out="u",
         .operand=weights("ffn_up.weight", c.intermediate, width)},
        {.id="silu_mul", .op=Op::elementwise, .form="silu_mul", .entry="silu_mul_kernel", .site="ffn", .in={"g","u"}, .out="act"},
        {.id="down", .op=Op::linear, .entry="dense_iu4", .site="ffn", .in={"act"}, .out="proj2",
         .operand=weights("ffn_down.weight", width, c.intermediate)}
    };
    const FormatRule formats[] = {{"",Format::iu4,{}}, {"ssm_beta.weight",Format::bf16,{}},
                                  {"ssm_alpha.weight",Format::bf16,{}}};
    mg::ModelGraph legacy;
    legacy.model = "qwen-27b";
    for (std::uint32_t physical = 0; physical < c.layers; ++physical) {
        mg::LayerDesc d;
        d.id = physical;
        d.attention = c.kind(physical) == strix::qwen35::LayerKind::full_attention ?
            mg::AttentionFamily::dense : mg::AttentionFamily::gdn;
        d.mlps = {{mg::MlpKind::dense, c.intermediate, 0, 0}};
        legacy.layers.push_back(d);
        check(decl.schedule.at(physical) == LayerVisit{physical,0,physical}, "Qwen physical schedule");
    }
    const auto expected = mg::to_stage_graph(legacy, {.layer=layer, .formats=formats});
    const auto& actual = decl.stages;
    same_rows(actual.before, expected.before);
    same_rows(actual.layer, expected.layer);
    same_rows(actual.after, expected.after);
    check(actual.layers == expected.layers && actual.select_after == 0 && actual.group == 0, "Qwen graph traits");
    check(actual.formats.size() == std::size(formats), "Qwen format count");
    for (std::size_t i = 0; i < std::size(formats); ++i)
        check(actual.formats[i].tensor == formats[i].tensor && actual.formats[i].format == formats[i].format &&
              actual.formats[i].layers.empty(), "Qwen format rule");
    check(decl.requirements.size() == 496 && decl.states.size() == 2 && decl.drafts.size() == 1, "Qwen v2 tables");
    std::cout << "D1 PASS qwen3.8-27b: 17 layer rows, 64 visits, 3 format rules\n";
}

void inkling(const ModelDecl& decl) {
    // Semantic boundaries of MiniResidentForward::Forward::run_layer and MiniLayers::run.
    constexpr Traits block = 1U << 7, boundary = 1U << 8, capture = 1U << 9, nonfirst = 1U << 10,
        restore = 1U << 12, save = 1U << 13, separate = 1U << 14;
    const auto row = [](V id, Op op, std::initializer_list<V> in, V out, V form = {},
                        Operand operand = {}, Traits when = 0, Traits unless = 0) {
        StageDecl r{.id=id, .op=op, .form=form, .out=out, .operand=operand, .when_all=when, .when_none=unless};
        std::ranges::copy(in, r.in);
        return r;
    };
    const auto h = mini::kHiddenWidth, q = mini::kQueryWidth, f = mini::kExpertWidth, experts = mini::kExperts;
    const StageDecl before[] = {
        row("embed",Op::gather,{"tok"},"embed",{},weights("model.llm.embed.weight",mini::kVocabulary,h)),
        row("embed_norm",Op::norm,{"embed"},"x","rms")};
    StageDecl layer[] = {
        row("save_core_entry",Op::copy,{"x","residual_bank"},"entry",{},{},save),
        row("restore_core_entry",Op::copy,{"entry"},"x",{},{},restore),
        row("capture_boundary",Op::copy,{"x"},"boundary",{},{},boundary),
        row("attn_read",Op::mix,{"x","residual_bank"},"attn_rows","residual_bank",
            weights("model.llm.attn_res.attn_queries.{L-1}",1,h),nonfirst),
        row("query_input",Op::norm,{"x","attn_rows"},"qa","message_rms"),
        row("key_input",Op::norm,{"qa"},"ka","message_rms",{},separate),
        row("value_input",Op::norm,{"qa"},"va","message_rms",{},separate),
        row("qkv",Op::linear,{"qa","ka","va"},"q",{},weights("q_weight",q,h)),
        row("prepare_kv",Op::elementwise,{"k","v"},"kv_rows","conv4.clip.kv_append"),
        row("attend",Op::attend,{"q","rel","gate","sink"},"context_row","prepared_gqa"),
        row("attn_out",Op::linear,{"context_row"},"attention",{},weights("model.llm.layers.{L}.attn.wo_ud.weight",h,q)),
        row("attn_conv",Op::elementwise,{"attention"},"conv_attn","conv4"),
        row("save_block",Op::copy,{"x","conv_attn"},"residual",{},{},block),
        row("add_residual",Op::combine,{"x","conv_attn"},"residual","add",{},0,block),
        row("moe_read",Op::mix,{"residual","residual_bank"},"moe_rows","residual_bank",
            weights("model.llm.attn_res.mlp_queries.{L}",1,h)),
        row("moe_input",Op::norm,{"moe_rows"},"expert_in","message_rms"),
        row("router",Op::linear,{"expert_in"},"router_logits",{},weights("model.llm.layers.{L}.mlp.gate.weight",experts,h)),
        row("route",Op::route,{"router_logits"},"routes","top_k"),
        row("expert_gate_up",Op::linear,{"expert_in","routes"},"expert_act",{},weights("expert_w13",2*f,h,experts,mini::kExpertsPerToken)),
        row("expert_down",Op::linear,{"expert_act","routes"},"expert_rows",{},weights("expert_w2",h,f,experts,mini::kExpertsPerToken)),
        row("expert_merge",Op::combine,{"expert_rows","routes"},"expert_sum","weighted_sum"),
        row("moe_conv",Op::elementwise,{"expert_sum"},"conv_moe","conv4"),
        row("layer_out",Op::combine,{"residual","conv_moe"},"x","add_round_bf16"),
        row("capture_pass",Op::mix,{"boundary","x"},"message_bank","message_bank",{},capture)};
    layer[7].site = "attn";
    layer[7].numeric = "dot.fma.f32";
    layer[7].split = Split::by_heads;
    layer[9].rows = Rows::cross_row;
    layer[17].reduce = true;
    layer[18].split = layer[19].split = Split::by_experts;
    const StageDecl after[] = {
        row("readout",Op::mix,{"x","residual_bank"},"readout","residual_bank"), row("final_norm",Op::norm,{"readout"},"final","rms"),
        row("head",Op::head,{"final"},"logits",{},weights("model.llm.unembed.weight",mini::kVocabulary,h)),
        row("softcap",Op::elementwise,{"logits"},"scored","softcap")};
    const auto& g = decl.stages;
    same_rows(g.before, before);
    same_rows(g.layer, layer);
    same_rows(g.after, after);
    check(g.formats.size() == 7 && g.select_after == 0 && g.group == 0, "Inkling format rules");
    for (std::uint32_t l = 0; l < mini::kPhysicalLayers; ++l) {
        for (const auto& stage : layer) {
            const auto tensor = stage.operand.tensor;
            if (tensor.empty()) continue;
            Format format = Format::bf16;
            if (tensor.starts_with("expert_")) format = Format::none;
            if (tensor.starts_with("model.llm.attn_res.")) format = Format::f32;
            check(format_of(g, tensor, static_cast<std::int32_t>(l)) == format, "Inkling format projection");
        }
    }
    const auto& details = decl.graphs().front().rows;
    const auto expected = mini::mini_schedule();
    check(decl.schedule.size() == expected.size(), "Inkling schedule length");
    for (const auto& e : expected) {
        const LayerVisit visit{e.physical_layer,e.pass,e.logical_layer};
        check(decl.schedule.at(e.logical_layer) == visit, "Inkling visit identity");
        const auto any = [&](V stage) {
            const auto& found = *std::ranges::find(details, stage, &StageFields::id);
            return std::ranges::find(found.active_visits, visit) != found.active_visits.end();
        };
        const auto full = mini::is_full_attention(e.physical_layer);
        check(any("attn_read") == (e.physical_layer != 0) &&
              any("key_input") == (full || e.pass != 0) && any("value_input") == any("key_input") &&
              any("save_block") == mini::starts_residual_block(e) && any("add_residual") != any("save_block") &&
              any("save_core_entry") == (e.physical_layer == mini::kCoreFirst && e.pass == 0) &&
              any("restore_core_entry") == (e.physical_layer == mini::kCoreFirst && e.pass != 0) &&
              any("capture_boundary") == mini::captures_message_boundary(e) &&
              any("capture_pass") == (e.physical_layer + 1 == mini::kCoreLast && e.pass + 1 < mini::kCorePasses),
              "Inkling visit guard");
        check((g.layers.at(e.physical_layer) & (trait::attn_dense | trait::attn_window | trait::ffn_moe)) ==
              (trait::ffn_moe | (full ? trait::attn_dense : trait::attn_window)), "Inkling legacy traits");
    }
    for (const auto& tensor : decl.requirements) {
        if (tensor.name.find(".mlp.experts.") == V::npos) continue;
        const auto layer_index = std::stoul(tensor.name.substr(V("model.llm.layers.").size()));
        check(tensor.format == (layer_index < mini::kMxFp8First ? "nvfp4" : "mxfp8"), "Inkling expert format");
    }
    check(decl.states.size() == 3, "Inkling state table");
    std::cout << "D1 PASS inkling2-mini: 2 before, 24 layer, 4 after, 48 visits\n";
}

constexpr V minimal = R"toml(
schema = "strix-model/v2"
name = "fixture"
[source]
container = "gguf"
architecture = "fixture"
[dims]
layers = 2
hidden = 32
head_dim = 64
[[value]]
id = "x"
scope = "step"
width = "hidden"
)toml";
constexpr V project = R"toml(
[[stage]]
id = "proj"
op = "linear"
form = "matrix"
in = ["x"]
out = "y"
weight = "blk.{L}.matrix"
operand = {rows = "hidden", cols = "hidden"}
[[format]]
format = "f32"
)toml";

md::ModelDescription parse(V tail, V root = minimal) {
    return md::parse(std::string(root) + std::string(tail), md::TableSyntax::root_array_tables);
}

void invalid() {
    const auto bad = [](V tail) { (void)inspect_model_decl(parse(tail)); };
    refuses("expected strix-model/v2", [] { (void)inspect_model_decl(md::parse("schema = \"strix-model/v1\"")); });
    refuses("invalid template path", [&] { bad("\n[graph]\ntemplate = \"../outside\"\n"); });
    refuses("invalid repeat", [&] { bad("\n[schedule]\norder = \"(0..1)*0\"\n"); });
    refuses("omits a physical layer", [&] { bad("\n[schedule]\norder = \"0\"\n"); });
    refuses("unknown key missing", [&] { bad("\n[[stage]]\nid=\"s\"\nop=\"missing\"\nout=\"y\"\n"); });
    refuses("projection without operand", [&] { bad("\n[[stage]]\nid=\"s\"\nop=\"linear\"\nout=\"y\"\n"); });
    refuses("missing entry", [&] { bad("\n[[stage]]\nid=\"s\"\nop=\"custom\"\nout=\"y\"\n"); });
    refuses("unknown insertion anchor", [&] { bad("\n[[stage]]\nid=\"s\"\nafter=\"other\"\nop=\"copy\"\nout=\"y\"\n"); });
    refuses("duplicate stage id", [&] { bad("\n[[stage]]\nid=\"s\"\n[[stage]]\nid=\"s\"\n"); });
    constexpr V copy = "\n[[stage]]\nid=\"s\"\nop=\"copy\"\nout=\"y\"\n";
    const auto bad_copy = [&](V tail) { bad(std::string(copy) + std::string(tail)); };
    refuses("unknown value", [&] { bad_copy("in=[\"z\"]\n"); });
    refuses("unknown name unk", [&] { bad_copy("when=\"false & unk\"\n"); });
    refuses("unknown name unk", [&] { bad_copy("row_when=\"pos < unk\"\n"); });
    refuses("overlapping slices", [&] {
        bad("\n[[stage]]\nid=\"s\"\nop=\"copy\"\nout=[{id=\"a\",cols=\"0:4\"},{id=\"b\",cols=\"2:8\"}]\n");
    });
    refuses("cycle in value references", [&] {
        bad("\n[[stage]]\nid=\"a\"\nop=\"copy\"\nin=[\"b@1\"]\nout=\"a\"\nwhen=\"L == 0\"\n"
            "[[stage]]\nid=\"b\"\nop=\"copy\"\nin=[\"a@0\"]\nout=\"b\"\nwhen=\"L == 1\"\n");
    });
    refuses("replay input not produced in the window", [&] {
        bad_copy("state=\"cache\"\n[[state]]\nid=\"cache\"\nkind=\"conv\"\ndtype=\"f32\"\n"
                 "rollback=\"replay\"\nreplay_inputs=[\"x\"]\n");
    });
    const auto catalogue = inspect_model_decl(parse(std::string(copy) +
        "state=\"cache\"\n[[state]]\nid=\"cache\"\nkind=\"kv_window\"\ndtype=\"bf16\"\n"
        "[[stage]]\nid=\"attn\"\nop=\"attend\"\nform=\"gqa\"\nout=\"z\"\n"
        "[[format]]\ntensor=\"norm\"\ntransform=\"one_plus\"\n"
        "[[image]]\nof=\"norm\"\nformat=\"q8_0\"\nlifetime=\"request\"\n"));
    for (const auto& [kind, key] : {
             std::pair{DeclKeyKind::operation,V("attend:gqa")}, {DeclKeyKind::head_dim,"64"},
             {DeclKeyKind::decoder,"q8_0"}, {DeclKeyKind::state_layout,"kv_window"}, {DeclKeyKind::transform,"one_plus"}})
        refuses(key, [&] { validate(catalogue, [&](const DeclKey& k) { return k.kind != kind || k.key != key; }); });
}

void tensors() {
    auto root = std::string(minimal);
    root.replace(root.find("hidden = 32"), V("hidden = 32").size(), "hidden = \"@fixture.hidden\"");
    const auto descriptor = parse(project, root);
    weight::TensorIndex index;
    refuses("missing metadata", [&] { (void)load_model_decl(descriptor, index); });
    index.metadata.emplace_back("fixture.hidden", "32");
    refuses("missing tensor blk.0.matrix", [&] { (void)load_model_decl(descriptor, index); });
    for (unsigned i = 0; i < 2; ++i) {
        weight::RawTensor tensor;
        tensor.name = "blk." + std::to_string(i) + ".matrix";
        tensor.shape = {{32,32,0,0},2};
        index.tensors.push_back(std::move(tensor));
    }
    const auto good = load_model_decl(descriptor, index);
    check(good.tensors.size() == 2 && good.requirements.at(0).dims.at(0) == 32, "derived tensor map");
    const auto copy = [](ModelDecl declaration) { return declaration; }(load_model_decl(descriptor, index));
    check(copy.identity == good.identity && copy.stages.layer.front().id == "proj" &&
          copy.tensors.at(1).name == "blk.1.matrix", "owned stage and tensor views");
    index.tensors[1].shape.dims[1] = 31;
    refuses("wrong shape", [&] { (void)load_model_decl(descriptor, index); });
    index.tensors[1].shape.dims[1] = 32;
    index.tensors[1].wire_type = 1;
    refuses("wrong format", [&] { (void)load_model_decl(descriptor, index); });
    index.tensors[1].wire_type = 0;
    index.tensors.push_back(index.tensors[1]);
    refuses("duplicate tensor", [&] { (void)load_model_decl(descriptor, index); });
    index.tensors.back().name = "extra";
    refuses("unknown tensor", [&] { (void)load_model_decl(descriptor, index); });
    index.tensors.pop_back();
    index.metadata.front().second = "33";
    index.tensors[0].shape.dims = index.tensors[1].shape.dims = {33,33,0,0};
    check(load_model_decl(descriptor, index).identity != good.identity, "resolved dimension identity");
}

void templates_and_graphs(const ModelDeclOptions& options) {
    auto root = std::string(minimal);
    root.insert(root.find("[[value]]"),
                "q_width=64\nkv_width=32\nattn_width=64\nffn=64\nexperts=4\ntop_k=2\n");
    for (const V name : {"prenorm-gqa-recurrent", "prenorm-gqa-moe", "prenorm-gqa-moe-hc"}) {
        const auto desc = parse("\n[classes]\nfull=[1]\ngdn=[0]\n[graph]\ntemplate=\"decoder/" +
                                std::string(name) + "\"\n", root);
        const auto resolved = inspect_model_decl(desc, options);
        check(resolved.stages.layer.size() == (name.ends_with("-hc") ? 13U : 12U), name);
    }
    const auto multi = parse(R"toml(
[classes]
full = [1]
[[graph]]
id = "text"
kind = "decoder"
splice = {from = "vision", into = "second", marker = 9}
stage = [
    {id = "first", op = "copy", in = ["x"], out = "a", when = "L == 0"},
    {id = "second", op = "copy", in = ["a@L-1"], out = "b", when = "class == full & phase == decode"},
]
[[graph]]
id = "vision"
kind = "encoder"
stage = [{id = "embed", op = "copy", out = "v", table = "before"}]
)toml");
    const auto decl = inspect_model_decl(multi, options);
    check(decl.graphs().size() == 2 && decl.graphs()[1].stages.before[0].id == "embed", "multiple graph tables");
    const auto rows_by_layer = instances(decl.stages, 0);
    check(rows_by_layer.size() == 2 && rows_by_layer[0].layer == 0 && rows_by_layer[1].layer == 1,
          "compiled legacy layer predicate");
    check(inspect_model_decl(md::ModelDescription{multi.fields}, options).identity == decl.identity, "stable identity");
}
} // namespace

int main(int argc, char** argv) try {
    check(argc == 3, "expected template and declaration fixture directories");
    const ModelDeclOptions options{argv[1]};
    const std::filesystem::path fixtures(argv[2]);
    const auto load = [&](V name) {
        return inspect_model_decl(md::load((fixtures / name).string(), md::TableSyntax::root_array_tables), options);
    };
    qwen(load("qwen-27b.toml"));
    inkling(load("inkling2-mini.toml"));
    invalid();
    tensors();
    templates_and_graphs(options);
    std::cout << "D1 PASS schema, catalogue refusals and tensor admission\n";
    return 0;
} catch (const std::exception& error) {
    std::cerr << "D1 FAIL " << error.what() << '\n';
    return 1;
}
