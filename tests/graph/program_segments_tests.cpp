// Program segments and launch budget: segments re-describe the launch list without changing it.
#include "strix/graph/program_segments.hpp"

#include <cstdio>
#include <cstdlib>
#include <initializer_list>
#include <stdexcept>
#include <string>

using namespace strix::graph;

namespace {

int failures = 0;
void expect(bool ok, const char* what) {
    if (!ok) {
        std::fprintf(stderr, "FAIL %s\n", what);
        ++failures;
    }
}

constexpr StageDecl embed{.id = "embed", .op = Op::gather};
constexpr StageDecl attn_norm{.id = "attn_norm", .op = Op::norm, .reduce = true};
constexpr StageDecl qkv{.id = "qkv", .op = Op::linear};
constexpr StageDecl attend{.id = "attend", .op = Op::attend};
constexpr StageDecl out_proj{.id = "out", .op = Op::linear};
constexpr StageDecl residual{.id = "residual", .op = Op::combine};
constexpr StageDecl mix{.id = "mix", .op = Op::mix, .reduce = true};
constexpr StageDecl ffn_up{.id = "up", .op = Op::linear};
constexpr StageDecl act{.id = "act", .op = Op::elementwise};
constexpr StageDecl ffn_down{.id = "down", .op = Op::linear};
constexpr StageDecl final_norm{.id = "final_norm", .op = Op::norm, .reduce = true};
constexpr StageDecl head{.id = "head", .op = Op::head};
constexpr StageDecl pick{.id = "pick", .op = Op::argmax};

struct Builder {
    LaunchPlan plan;
    std::uint32_t at(const StageDecl& d, std::int32_t layer, Format f = Format::bf16) {
        plan.instances.push_back({&d, layer, f});
        return static_cast<std::uint32_t>(plan.instances.size() - 1);
    }
    void launch(std::int32_t layer, std::initializer_list<const StageDecl*> body,
                std::initializer_list<const StageDecl*> pro = {}, std::initializer_list<const StageDecl*> epi = {},
                std::uint32_t kernels = 1, Format f = Format::bf16) {
        Launch l;
        l.entry = (*body.begin())->id;
        l.layer = layer;
        l.kernels = kernels;
        for (const auto* d : body) l.stages.push_back(at(*d, layer, f));
        for (const auto* d : pro) l.prologue.push_back(at(*d, layer, f));
        for (const auto* d : epi) l.epilogue.push_back(at(*d, layer, f));
        if (!plan.launches.empty()) l.after = static_cast<std::uint32_t>(plan.launches.size() - 1);
        plan.launches.push_back(std::move(l));
    }
    // Attention layer, fused form: norm as qkv prologue, residual as out epilogue; one glue launch (mix).
    void attn_layer(std::int32_t layer, Format f = Format::bf16) {
        launch(layer, {&qkv}, {&attn_norm}, {}, 1, f);
        launch(layer, {&attend});
        launch(layer, {&out_proj}, {}, {&residual}, 1, f);
        launch(layer, {&mix});
    }
    // FFN layer with an unfused tail: standalone norm, up, act, down, residual (two glue launches + norm).
    void ffn_layer(std::int32_t layer) {
        launch(layer, {&attn_norm});
        launch(layer, {&ffn_up}, {}, {&act});
        launch(layer, {&ffn_down});
        launch(layer, {&residual});
    }
    void prelude() { launch(-1, {&embed}); }
    void coda() {
        launch(-1, {&final_norm});
        launch(-1, {&head}, {}, {}, 2);
        launch(-1, {&pick});
    }
};

std::uint32_t covered(const SegmentPlan& s) {
    return s.prelude.launches + s.core.launches * s.core.repeats + s.coda.launches;
}

void periodic_hybrid() {
    Builder b;
    b.prelude();
    for (std::int32_t layer = 0; layer < 8; ++layer) {
        if (layer % 2 == 0) b.attn_layer(layer);
        else b.ffn_layer(layer);
    }
    b.coda();
    const auto s = segment_plan(b.plan);
    expect(covered(s) == b.plan.launches.size(), "hybrid: segments cover every launch");
    expect(s.prelude.launches == 1 && s.coda.launches == 3, "hybrid: prelude 1, coda 3");
    expect(s.core.layers == 2 && s.core.repeats == 4 && s.core.launches == 8, "hybrid: period 2 layers x 4");
    expect(s.core.first == 1 && s.core.first_layer == 0, "hybrid: core starts at layer 0");
    expect(s.launches == 36 && s.kernels == 37, "hybrid: counts per step");
    // glue: per attention layer mix (1); per ffn layer norm + residual (2); coda final_norm (1).
    expect(s.glue == 4 * 1 + 4 * 2 + 1, "hybrid: glue launches per step");
    expect(s.core_glue_per_layer == 2 && s.core_kernels_per_layer == 4, "hybrid: per core layer maxima");
    expect(check_budget(s, {}).ok, "budget: zero is unbounded");
    expect(check_budget(s, {.kernels_per_step = 37, .glue_per_core_layer = 2}).ok, "budget: at the limit passes");
    const auto glue = check_budget(s, {.glue_per_core_layer = 1});
    expect(!glue.ok && glue.reason.find("2 glue launches in one core layer > budget 1") != std::string::npos,
           "budget: glue per core layer refused with reason");
    const auto kernels = check_budget(s, {.kernels_per_step = 36});
    expect(!kernels.ok && kernels.reason.find("37 kernels per step > budget 36") != std::string::npos,
           "budget: kernels per step refused with reason");
    expect(segment_receipt(s) == "segments prelude=1 core=8x4 (period 2 layers from layer 0) coda=3 launches=36 "
                                 "kernels=37 glue=13 glue_per_core_layer=2 kernels_per_core_layer=4",
           "receipt line");
}

void broken_period_goes_to_coda() {
    // Layers 0..4 uniform, layer 5 in another weight format: core 0..4, layer 5 lands in the coda.
    Builder b;
    b.prelude();
    for (std::int32_t layer = 0; layer < 5; ++layer) b.attn_layer(layer);
    b.attn_layer(5, Format::q8_0);
    b.coda();
    const auto s = segment_plan(b.plan);
    expect(s.core.layers == 1 && s.core.repeats == 5 && s.core.launches == 4, "format break: core 5 x 1 layer");
    expect(s.coda.launches == 4 + 3 && covered(s) == b.plan.launches.size(), "format break: layer 5 in coda");
}

void fused_plan_meets_budget_the_unfused_does_not() {
    // The same 4-layer FFN stack, unfused (standalone norm and residual) vs fused (norm prologue, residual epilogue).
    Builder unfused, fused;
    unfused.prelude();
    fused.prelude();
    for (std::int32_t layer = 0; layer < 4; ++layer) {
        unfused.ffn_layer(layer);
        fused.launch(layer, {&ffn_up}, {&attn_norm}, {&act});
        fused.launch(layer, {&ffn_down}, {}, {&residual});
    }
    const LaunchBudget budget{.glue_per_core_layer = 1};
    const auto u = segment_plan(unfused.plan);
    const auto f = segment_plan(fused.plan);
    expect(u.core.repeats == 4 && f.core.repeats == 4, "fusion: both periodic over 4 layers");
    expect(!check_budget(u, budget).ok, "fusion: unfused stack refused");
    expect(check_budget(f, budget).ok && f.glue == 0, "fusion: fused stack has no glue launch");
    expect(f.launches == 9 && u.launches == 17, "fusion: launch counts");
}

void no_repetition_is_all_prelude() {
    Builder b;
    b.prelude();
    b.attn_layer(0);
    b.ffn_layer(1);
    b.coda();
    const auto s = segment_plan(b.plan);
    expect(s.core.repeats == 0 && s.core.launches == 0, "no repetition: empty core");
    expect(s.prelude.launches == b.plan.launches.size() && s.coda.launches == 0, "no repetition: all prelude");
    expect(check_budget(s, {.glue_per_core_layer = 1}).ok, "no repetition: per-layer budget needs a core");
}

void dependency_shape_breaks_period() {
    // Same stages per layer, but layer 2 waits on a different predecessor: not the same body.
    Builder b;
    for (std::int32_t layer = 0; layer < 4; ++layer) b.attn_layer(layer);
    b.plan.launches[9].after = 2;
    const auto s = segment_plan(b.plan);
    expect(s.core.repeats < 4, "edge change: the layer with a different edge leaves the core");
    expect(covered(s) == b.plan.launches.size(), "edge change: still covered");
}

void undeclared_instance_rejected() {
    Builder b;
    b.attn_layer(0);
    b.plan.instances[0].decl = nullptr;
    bool threw = false;
    try {
        (void)segment_plan(b.plan);
    } catch (const std::invalid_argument&) {
        threw = true;
    }
    expect(threw, "undeclared instance rejected");
}

}  // namespace

int main() {
    periodic_hybrid();
    broken_period_goes_to_coda();
    fused_plan_meets_budget_the_unfused_does_not();
    no_repetition_is_all_prelude();
    dependency_shape_breaks_period();
    undeclared_instance_rejected();
    if (failures) return EXIT_FAILURE;
    std::puts("strix-program-segments-tests: PASS");
    return EXIT_SUCCESS;
}
