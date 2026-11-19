// Stage graph / launch planner / rows planner host tests (shared part; the Hy4 declaration's check is
// tests/hy4_graph_tests.cpp on the Hy4 branch).
#include "stage_bridge.hpp"
#include "strix/graph/launch_plan.hpp"
#include "strix/graph/rows_plan.hpp"

#include <cmath>
#include <cstdio>
#include <stdexcept>
#include <string>
#include <vector>

namespace {
int failures = 0;
void check(bool ok, const std::string& what) {
    if (!ok) {
        ++failures;
        std::fprintf(stderr, "FAIL: %s\n", what.c_str());
    }
}
template <class F> bool throws(F&& f) {
    try {
        f();
    } catch (const std::invalid_argument&) {
        return true;
    }
    return false;
}

using namespace strix::graph;
namespace t = strix::graph::trait;

// Qwen 27B dense decode layer (src/qwen35/hip/engine.hip decode loop): GDN layers project xn to {qkv, z} (IU4), beta,
// alpha (48 outputs: not an IU4 shape); gated attention layers to {q (+gate), k, v}; the MLP to {gate, up}, down.
// Test fixture (not the family's declaration). Only the projection part is declared in full;
// the recurrent / attention bodies are one stage each.
constexpr StageDecl kQwenLayer[] = {
    {.id = "attn_norm", .op = Op::norm, .entry = "rmsnorm_fused", .site = "attn", .in = {"x"}, .out = "xn", .reduce = true},
    {.id = "qkv", .op = Op::linear, .entry = "dense_iu4", .site = "attn", .in = {"xn"}, .out = "qkv",
     .operand = weights("attn_qkv.weight", 10240, 5120), .when_all = t::attn_gdn},
    {.id = "z", .op = Op::linear, .entry = "dense_iu4", .site = "attn", .in = {"xn"}, .out = "z",
     .operand = weights("attn_gate.weight", 6144, 5120), .when_all = t::attn_gdn},
    {.id = "beta", .op = Op::linear, .entry = "linear", .site = "attn", .in = {"xn"}, .out = "beta",
     .operand = weights("ssm_beta.weight", 48, 5120), .when_all = t::attn_gdn},
    {.id = "alpha", .op = Op::linear, .entry = "linear", .site = "attn", .in = {"xn"}, .out = "alpha",
     .operand = weights("ssm_alpha.weight", 48, 5120), .when_all = t::attn_gdn},
    {.id = "gdn", .op = Op::mix, .entry = "gdn_step", .site = "attn", .in = {"qkv", "z", "beta", "alpha"}, .out = "core",
     .rows = Rows::cross_row, .when_all = t::attn_gdn},
    {.id = "gdn_out", .op = Op::linear, .entry = "dense_iu4", .site = "attn", .in = {"core"}, .out = "proj",
     .operand = weights("ssm_out.weight", 5120, 6144), .when_all = t::attn_gdn},
    {.id = "q", .op = Op::linear, .entry = "dense_iu4", .site = "attn", .in = {"xn"}, .out = "q",
     .operand = weights("attn_q.weight", 12288, 5120), .when_all = t::attn_dense},
    {.id = "k", .op = Op::linear, .entry = "dense_iu4", .site = "attn", .in = {"xn"}, .out = "k",
     .operand = weights("attn_k.weight", 1024, 5120), .when_all = t::attn_dense},
    {.id = "v", .op = Op::linear, .entry = "dense_iu4", .site = "attn", .in = {"xn"}, .out = "v",
     .operand = weights("attn_v.weight", 1024, 5120), .when_all = t::attn_dense},
    {.id = "attend", .op = Op::attend, .entry = "dense_decode_flash", .site = "attn", .in = {"q", "k", "v"}, .out = "ctx",
     .rows = Rows::cross_row, .when_all = t::attn_dense},
    {.id = "o", .op = Op::linear, .entry = "dense_iu4", .site = "attn", .in = {"ctx"}, .out = "proj",
     .operand = weights("attn_output.weight", 5120, 6144), .when_all = t::attn_dense},
    {.id = "post_norm", .op = Op::norm, .entry = "rmsnorm_fused", .site = "ffn", .in = {"x", "proj"}, .out = "xn2", .reduce = true},
    {.id = "gate", .op = Op::linear, .entry = "dense_iu4", .site = "ffn", .in = {"xn2"}, .out = "g",
     .operand = weights("ffn_gate.weight", 17408, 5120)},
    {.id = "up", .op = Op::linear, .entry = "dense_iu4", .site = "ffn", .in = {"xn2"}, .out = "u",
     .operand = weights("ffn_up.weight", 17408, 5120)},
    {.id = "silu_mul", .op = Op::elementwise, .form = "silu_mul", .entry = "silu_mul_kernel", .site = "ffn", .in = {"g", "u"}, .out = "act"},
    {.id = "down", .op = Op::linear, .entry = "dense_iu4", .site = "ffn", .in = {"act"}, .out = "proj2",
     .operand = weights("ffn_down.weight", 5120, 17408)},
};
constexpr FormatRule kQwenFormats[] = {{"", Format::iu4, {}}, {"ssm_beta.weight", Format::bf16, {}}, {"ssm_alpha.weight", Format::bf16, {}}};
constexpr Format kIu4Only[] = {Format::iu4};

std::vector<std::string> groups(const LaunchPlan& p, std::int32_t layer) {
    std::vector<std::string> out;
    for (const auto& l : p.launches) {
        if (l.layer != layer || p.instances[l.stages.front()].decl->op != Op::linear) continue;
        std::string g;
        for (const auto s : l.stages) g += (g.empty() ? "" : "+") + std::string(p.instances[s].decl->id);
        out.push_back(g);
    }
    return out;
}

void qwen_dense() {
    namespace mg = strix::model_graph;
    mg::ModelGraph m;
    m.model = "qwen-27b-fixture";
    for (std::uint64_t l = 0; l < 4; ++l) {
        mg::LayerDesc d;
        d.id = l;
        d.attention = l == 3U ? mg::AttentionFamily::dense : mg::AttentionFamily::gdn;
        d.mlps = {{mg::MlpKind::dense, 17408, 0, 0}};
        m.layers.push_back(d);
    }
    StageGraph decl;
    decl.layer = kQwenLayer;
    decl.formats = kQwenFormats;
    const auto g = mg::to_stage_graph(m, decl);
    check(g.layers.size() == 4U && g.layers[0] == (t::attn_gdn | t::ffn_dense) && g.layers[3] == (t::attn_dense | t::ffn_dense),
          "qwen: to_stage_graph traits");
    const auto today = launch_plan(g, 100U, 1U, {.mode = Fusion::today, .group_formats = kIu4Only});
    const std::vector<std::string> gdn{"qkv+z", "beta", "alpha", "gdn_out", "gate+up", "down"};
    const std::vector<std::string> att{"q+k+v", "o", "gate+up", "down"};
    check(groups(today, 0) == gdn, "qwen today: GDN projection groups {qkv,z} {gate,up}, beta / alpha alone");
    check(groups(today, 3) == att, "qwen today: attention projection groups {q,k,v} {gate,up}");
    // fused: beta and alpha (one input, one format) share a launch; silu_mul (inputs from two group members) is a
    // prologue of down.
    const auto fused = launch_plan(g, 100U, 1U, {.mode = Fusion::fused});
    const std::vector<std::string> gdn_f{"qkv+z", "beta+alpha", "gdn_out", "gate+up", "down"};
    check(groups(fused, 0) == gdn_f, "qwen fused: beta+alpha grouped");
    bool silu_prologue = false;
    for (const auto& l : fused.launches)
        for (const auto s : l.prologue) silu_prologue = silu_prologue || (l.layer == 0 && fused.instances[s].decl->id == "silu_mul");
    check(silu_prologue, "qwen fused: silu_mul is the prologue of down");
    // bytes: IU4 g64 rows, BF16 beta
    const auto inst = instances(g, 100U);
    check(bytes(inst[1]) == 10240ULL * (5120U / 2U + 5120U / 64U * 2U), "qwen: IU4 g64 operand bytes");
    check(bytes(inst[3]) == 48ULL * 5120U * 2U, "qwen: BF16 beta bytes");
    mg::ModelGraph bad = m;
    bad.layers[1].attention = mg::AttentionFamily::kda;
    check(throws([&] { (void)mg::to_stage_graph(bad, decl); }), "qwen: an unmapped attention family throws");
}

void contracts() {
    check(row_bytes(Format::q8_0, 3072) == 3264U && row_bytes(Format::q4_k, 1536) == 864U && row_bytes(Format::q6_k, 1536) == 1260U,
          "row_bytes: Q8_0 / Q4_K / Q6_K");
    check(throws([] { (void)row_bytes(Format::q4_k, 1000); }) && throws([] { (void)row_bytes(Format::none, 32); }),
          "row_bytes: bad width / no format");
    static constexpr std::uint32_t odd[] = {1U, 3U};
    static constexpr TraitList lists[] = {{t::attn_window, odd, false}, {t::attn_indexed, odd, true}};
    const auto w = layer_traits(4, lists);
    check(w == std::vector<Traits>{t::attn_indexed, t::attn_window, t::attn_indexed, t::attn_window}, "layer_traits: list + except");
    static constexpr StageDecl dup[] = {{.id = "a", .op = Op::norm, .out = "x"}, {.id = "a", .op = Op::norm, .out = "y"}};
    StageGraph g;
    g.layer = dup;
    g.layers = {0U};
    check(throws([&] { validate(g); }), "validate: duplicate id");
    static constexpr StageDecl dead[] = {{.id = "a", .op = Op::norm, .out = "x", .when_all = t::ffn_moe}};
    g.layer = dead;
    check(throws([&] { validate(g); }), "validate: stage active on no layer");
    static constexpr StageDecl noop[] = {{.id = "a", .op = Op::linear, .out = "x"}};
    g.layer = noop;
    check(throws([&] { validate(g); }), "validate: linear without operand");
    static constexpr std::uint32_t l1[] = {1U};
    static constexpr FormatRule f[] = {{"", Format::q8_0, {}}, {"w", Format::q4_k, {}}, {"w", Format::q6_k, l1}};
    g.formats = f;
    g.layers = {0U, 0U};
    check(format_of(g, "w", 0) == Format::q4_k && format_of(g, "w", 1) == Format::q6_k && format_of(g, "v", 1) == Format::q8_0,
          "format_of: later rules win, layer lists");
    // fused rule 1 groups by input VALUE: a buffer name rewritten in between is a different input
    static constexpr StageDecl rw[] = {
        {.id = "n1", .op = Op::norm, .in = {"x"}, .out = "xn", .reduce = true},
        {.id = "a", .op = Op::linear, .in = {"xn"}, .out = "ya", .operand = weights("wa", 64, 64)},
        {.id = "n2", .op = Op::norm, .in = {"ya"}, .out = "xn", .reduce = true},
        {.id = "b", .op = Op::linear, .in = {"xn"}, .out = "yb", .operand = weights("wb", 64, 64)},
        {.id = "c", .op = Op::linear, .in = {"xn"}, .out = "yc", .operand = weights("wc", 64, 64)},
    };
    g.layer = rw;
    g.layers = {0U};
    const auto p = launch_plan(g, 0U, 1U, {.mode = Fusion::fused});
    check(p.launches.size() == 2U && p.launches[0].stages.size() == 1U && p.launches[1].stages.size() == 2U &&
              p.launches[1].prologue.size() == 1U,
          "fused: no group across a rewritten input; b + c grouped with n2 as prologue");
    // A stage moved into an earlier launch must not reorder a write of its output name: b (same input as a) would join
    // a's launch and write t twice there, and the prologue n of c would read b's t.
    static constexpr StageDecl ww[] = {
        {.id = "a", .op = Op::linear, .in = {"x"}, .out = "t", .operand = weights("wa", 64, 64)},
        {.id = "n", .op = Op::norm, .in = {"t"}, .out = "u", .reduce = true},
        {.id = "c", .op = Op::linear, .in = {"u"}, .out = "y", .operand = weights("wc", 64, 64)},
        {.id = "b", .op = Op::linear, .in = {"x"}, .out = "t", .operand = weights("wb", 64, 64)},
    };
    g.layer = ww;
    const auto p1 = launch_plan(g, 0U, 1U, {.mode = Fusion::fused});
    check(p1.launches.size() == 3U && p1.launches[0].stages.size() == 1U && p1.launches[1].prologue.size() == 1U &&
              p1.launches[2].stages.size() == 1U && p1.instances[p1.launches[2].stages[0]].decl->id == "b",
          "fused: a second writer of one name opens its own launch");
    // An epilogue folded into an earlier launch must not overwrite a name a later-placed launch still reads.
    static constexpr StageDecl war[] = {
        {.id = "a", .op = Op::linear, .in = {"x"}, .out = "y", .operand = weights("wa", 64, 64)},
        {.id = "b", .op = Op::linear, .in = {"o"}, .out = "z", .operand = weights("wb", 64, 64)},
        {.id = "e", .op = Op::elementwise, .in = {"y"}, .out = "o"},
    };
    g.layer = war;
    const auto p2 = launch_plan(g, 0U, 1U, {.mode = Fusion::fused});
    check(p2.launches.size() == 3U && p2.launches[0].epilogue.empty(), "fused: no epilogue fold over a pending reader of its output");
    // Adjacent norm chains share a launch only when EVERY input comes from before it (n2 reads p, written by n1's launch).
    static constexpr StageDecl nn[] = {
        {.id = "l", .op = Op::linear, .in = {"x"}, .out = "h", .operand = weights("wl", 64, 64)},
        {.id = "n1", .op = Op::norm, .in = {"h"}, .out = "p", .reduce = true},
        {.id = "n2", .op = Op::norm, .in = {"z", "p"}, .out = "r", .reduce = true},
        {.id = "e1", .op = Op::elementwise, .in = {"p"}, .out = "q"},
        {.id = "e2", .op = Op::elementwise, .in = {"r"}, .out = "s"},
    };
    g.layer = nn;
    const auto p3 = launch_plan(g, 0U, 1U, {.mode = Fusion::fused});
    check(p3.launches.size() == 3U && p3.launches[1].stages.size() == 1U && p3.launches[2].stages.size() == 1U,
          "fused: a norm reading the previous norm's output opens its own launch");
    // A reduction whose input is rewritten before its consumer cannot be deferred into that consumer's prologue.
    static constexpr StageDecl cl[] = {
        {.id = "a", .op = Op::linear, .in = {"x"}, .out = "t", .operand = weights("wa", 64, 64)},
        {.id = "n", .op = Op::norm, .in = {"t"}, .out = "u", .reduce = true},
        {.id = "b", .op = Op::linear, .in = {"w"}, .out = "t", .operand = weights("wb", 64, 64)},
        {.id = "c", .op = Op::linear, .in = {"u"}, .out = "y", .operand = weights("wc", 64, 64)},
    };
    g.layer = cl;
    const auto p4 = launch_plan(g, 0U, 1U, {.mode = Fusion::fused});
    bool no_prologue = true;
    for (const auto& l : p4.launches) no_prologue = no_prologue && l.prologue.empty();
    check(p4.launches.size() == 4U && no_prologue, "fused: no prologue across a rewrite of the reduction's input");
}

void rows() {
    check(rows_form(1) == RowsForm::step && rows_form(16) == RowsForm::verify && rows_form(17) == RowsForm::window, "rows_form");
    check(std::fabs(distinct_items(512, 16, 1) - 16.0) < 1e-9 && std::fabs(distinct_items(512, 16, 2) - 31.5) < 1e-9,
          "distinct_items: 16 then 31.5 of 512");
    static constexpr StageDecl s[] = {
        {.id = "e", .op = Op::linear, .in = {"x"}, .out = "y", .operand = weights("exps", 64, 256, 512, 16)},
        {.id = "a", .op = Op::attend, .in = {"y"}, .out = "z", .operand = keys(4, 8, 100), .rows = Rows::cross_row},
    };
    static constexpr FormatRule f[] = {{"", Format::q4_k, {}}};
    StageGraph g;
    g.layer = s;
    g.formats = f;
    g.layers = {0U};
    const auto p1 = rows_plan(g, 0, 1), p2 = rows_plan(g, 0, 2);
    check(std::fabs(p1.bytes - 16.0 * 64 * 144) < 1e-6 && std::fabs(p2.bytes - 31.5 * 64 * 144) < 1e-6, "rows_plan: distinct expert bytes");
    check(p2.macs == 2ULL * (64ULL * 256U * 16U + 4ULL * 8U * 100U), "rows_plan: MACs scale with rows");
    check(p2.stages[1].rule == RowRule::block_causal && p2.stages[0].rule == RowRule::independent, "rows_plan: cross-row rule");
}
}  // namespace

int main() {
    contracts();
    rows();
    qwen_dense();
    if (failures != 0) {
        std::fprintf(stderr, "%d failure(s)\n", failures);
        return 1;
    }
    std::printf("stage graph tests: OK\n");
    return 0;
}
