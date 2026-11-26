#include "exec/cuda_row_epilogue.hpp"
#include "row_epilogue_fixture_cases.hpp"
#include "strix/engine_profile.hpp"
#include "strix/weight/formats/scalar_float.hpp"

#include <algorithm>
#include <array>
#include <bit>
#include <iostream>
#include <stdexcept>

namespace {

using namespace strix::exec;
namespace graph = strix::graph;
namespace device = strix::device;
namespace profile = strix::profile;
namespace lever = strix::lever;
using graph::Form;
using graph::Op;

constexpr KernelKey exact_key{Op::combine, "test.add_round_add", {}, OperandClass::f32_fma, RowsClass::r1};
constexpr KernelKey prefix_key{Op::combine, "test.add_round", {}, OperandClass::f32_fma, RowsClass::r1};
constexpr KernelKey fast_key{Op::combine, "test.add_round_add.fast", {}, OperandClass::f32_fma, RowsClass::r1};
constexpr LauncherRef launcher{"test_bind", "tests/row_epilogue_plan_tests.cpp", "test", 1};
constexpr auto arch = device::ArchMask::sm120;

constexpr std::array steps{
    RowEpilogueStep{{"round-v1", "rne-v1"}, {Form::store_bf16, 0}},
    RowEpilogueStep{{"add-v1", "fp32-add-v1"}, {Form::residual_add, 1}},
};
constexpr RowEpilogueVariant variant{{"head-v1", "fp32-add-v1"}, graph::Format::none, 6, 0, 1, arch, steps,
                                      exact_key, "add-round-add-fixture-v1"};

constexpr std::array declarations{
    graph::StageDecl{.id = "head", .op = Op::combine, .entry = "head-v1", .site = "row",
                     .in = {"x", "first"}, .out = "sum", .operand = {.rows = 6}, .numeric = "fp32-add-v1"},
    graph::StageDecl{.id = "round", .op = Op::elementwise, .form = "store.bf16", .entry = "round-v1", .site = "row",
                     .in = {"sum"}, .out = "rounded", .numeric = "rne-v1"},
    graph::StageDecl{.id = "tail", .op = Op::combine, .form = "residual.add", .entry = "add-v1", .site = "row",
                     .in = {"rounded", "residual"}, .out = "answer", .numeric = "fp32-add-v1"},
    graph::StageDecl{.id = "pick", .op = Op::argmax, .entry = "pick-v1", .site = "output",
                     .in = {"answer"}, .out = "result"},
};
constexpr graph::StageDecl input{.id = "input", .op = Op::gather, .entry = "input-v1", .out = "x"};

constexpr KernelEntry entry(KernelKey key, std::string_view identity, lever::Exactness exactness) {
    return {key, arch, identity, exactness,
        +[](const LaunchSite&, const device::Profile&) { return EntryCost{}; },
        +[](const LaunchSite&, WorkspaceNeed&) {},
        +[](const LaunchSite&, EntryBinder&) { return EntryLaunch{}; }, &launcher};
}

constexpr std::array entries{
    entry(exact_key, "fused-exact-v1", lever::Exactness::bitwise),
    entry(prefix_key, "fused-prefix-v1", lever::Exactness::bitwise),
    entry(fast_key, "fused-fast-v1", lever::Exactness::bit_changing),
};

void require(bool condition, const char* message) {
    if (!condition) throw std::runtime_error(message);
}

RowEpiloguePlan plan(const Catalog& catalog, std::span<const graph::StageDecl> stages,
                     std::span<const RowEpilogueVariant> variants, profile::Name tier = profile::Name::standard,
                     device::ArchMask target = arch, std::uint32_t rows = 1) {
    const graph::StageGraph model{.before = std::span(&input, 1U), .layer = stages, .layers = {0}};
    return plan_row_epilogues(model, catalog, {.rows = rows, .tier = tier, .arch = target, .variants = variants});
}

void check_fallback(const Catalog& catalog, std::span<const graph::StageDecl> stages,
                    std::span<const RowEpilogueVariant> variants, device::ArchMask target = arch,
                    std::uint32_t rows = 1) {
    const graph::StageGraph graph{.before = std::span(&input, 1U), .layer = stages, .layers = {0}};
    const auto original = graph::launch_plan(graph, 0, rows, graph::FusionPolicy{});
    const auto p = plan(catalog, stages, variants, profile::Name::standard, target, rows);
    require(graph::plan_identity(p.graph) == graph::plan_identity(original) &&
            std::ranges::all_of(p.selections, [](const auto& selection) { return selection.entry == nullptr; }),
            "fallback changed the incumbent launch list");
}

void check_graph(const Catalog& catalog) {
    const auto p = plan(catalog, declarations, std::span(&variant, 1U));
    require(p.graph.kernel_count() == 3 && p.selections[1].entry == &entries[0] &&
            p.selections[1].exact_fixture == variant.exact_fixture, "chain or exact fixture is missing");
    const auto& fused = p.graph.launches[1];
    require(fused.entry == entries[0].identity && fused.stages == std::vector<std::uint32_t>{1} &&
            fused.epilogue == std::vector<std::uint32_t>{2, 3} && fused.kernels == 1,
            "head or epilogue order differs from graph");
    const auto& chain = p.graph.chains.at(fused.epilogue_chain[0]);
    require(chain.count == 2 && chain.links[0] == steps[0].link && chain.links[1] == steps[1].link,
            "ordered form or argument slot changed");
    require(p.graph.launches.front().entry == input.entry && p.graph.launches.back().entry == declarations.back().entry,
            "unrelated entries changed");
    std::cout << "PASS B3 graph-derived ordered epilogues\n";
}

void check_rounding(const Catalog& catalog) {
    using strix::weight::scalar_float::bf16_round;
    const auto p = plan(catalog, declarations, std::span(&variant, 1U));
    const auto& chain = p.graph.chains.at(p.graph.launches[1].epilogue_chain[0]);
    namespace data = strix::test::row_epilogue;
    constexpr auto& samples = data::input;
    std::array<float, samples.size()> sum{}, rounded{}, incumbent{}, fused{};
    constexpr float first = data::first_addend;
    constexpr float residual = data::last_addend;
    for (std::size_t i = 0; i < samples.size(); ++i) sum[i] = samples[i] + first;
    std::ranges::transform(sum, rounded.begin(), bf16_round);
    std::ranges::transform(rounded, incumbent.begin(), [](float x) { return x + residual; });
    bool sensitive = false;
    for (std::size_t i = 0; i < samples.size(); ++i) {
        auto value = samples[i] + first;
        for (std::size_t n = 0; n < chain.count; ++n) {
            switch (chain.links[n].form) {
            case Form::store_bf16: value = bf16_round(value); break;
            case Form::residual_add: value += residual; break;
            default: throw std::runtime_error("unknown fixture link");
            }
        }
        fused[i] = value;
        sensitive |= incumbent[i] != bf16_round(sum[i] + residual);
    }
    require(sensitive && std::bit_cast<std::array<std::uint32_t, samples.size()>>(fused) ==
                         std::bit_cast<std::array<std::uint32_t, samples.size()>>(incumbent),
            "chain changed incumbent bits or fixture missed a rounding boundary");
    std::cout << "PASS B3 exact chain float bits\n";
}

void check_declines(const Catalog& catalog) {
    auto changed = declarations;
    changed.back().in[1] = "sum";
    check_fallback(catalog, changed, std::span(&variant, 1U));
    changed = declarations;
    changed[2].site = "other";
    check_fallback(catalog, changed, std::span(&variant, 1U));
    changed = declarations;
    changed[2].reduce = true;
    check_fallback(catalog, changed, std::span(&variant, 1U));
    changed = declarations;
    changed[1].numeric = "other-rounding-v1";
    check_fallback(catalog, changed, std::span(&variant, 1U));
    changed = declarations;
    ++changed[0].operand.rows;
    check_fallback(catalog, changed, std::span(&variant, 1U));
    const std::array reversed{steps[1], steps[0]};
    auto wrong = variant;
    wrong.steps = reversed;
    check_fallback(catalog, declarations, std::span(&wrong, 1U));
    wrong = variant;
    wrong.exact_fixture = {};
    check_fallback(catalog, declarations, std::span(&wrong, 1U));
    check_fallback(Catalog{}, declarations, std::span(&variant, 1U));
    check_fallback(catalog, declarations, std::span(&variant, 1U), device::ArchMask::gfx1151);
    check_fallback(catalog, declarations, std::span(&variant, 1U), arch, 2);

    auto prefix = variant;
    prefix.steps = std::span(steps).first(1);
    prefix.key = prefix_key;
    prefix.exact_fixture = "add-round-fixture-v1";
    const std::array variants{prefix, variant};
    require(plan(catalog, declarations, variants).selections[1].entry == &entries[0], "longest exact match lost");
    changed = declarations;
    changed.back().in[1] = "rounded";
    const auto partial = plan(catalog, changed, variants);
    require(partial.selections[1].entry == &entries[1] && partial.graph.launches[2].entry == "add-v1",
            "rejected extension hid the admitted prefix or its remaining tail");
    changed = declarations;
    changed[1].out = "sum";
    changed[2].in[0] = "sum";
    require(plan(catalog, changed, variants).selections[1].entry == &entries[0], "prior value version was lost");
    std::cout << "PASS B3 fallback and partial-chain legality\n";
}

KernelBinding leaf(std::string_view identity) {
    KernelBinding binding;
    binding.fn = +[](std::span<const std::byte>, const Grid&, void*) {
        throw std::logic_error("planner executed a kernel");
    };
    binding.identity = binding.exact_identity = identity;
    binding.grid = {{1U, 1U, 1U}, {32U, 1U, 1U}, 0U};
    binding.predicted.launches = 1;
    binding.pdl = {PdlWait::before_step_io, PdlOptOut::none};
    return binding;
}

void check_recipe(const Catalog& catalog) {
    auto fast = variant;
    fast.key = fast_key;
    fast.exact_fixture = {};
    const auto fast_only = std::span(&fast, 1U);
    check_fallback(catalog, declarations, fast_only);
    const std::array both{fast, variant};
    require(plan(catalog, declarations, both, profile::Name::fast).selections[1].entry == &entries[0],
            "exact identity lost on an equal-length match");
    for (const bool use_fast : {false, true}) {
        const auto p = use_fast ? plan(catalog, declarations, fast_only, profile::Name::fast) :
                                 plan(catalog, declarations, std::span(&variant, 1U));
        const auto* selected = &entries[use_fast ? 2U : 0U];
        require(p.selections[1].entry == selected, "wrong tier identity");
        CudaProgramRecipe recipe;
        recipe.kernel(leaf(input.entry));
        require(!emit_row_epilogue(recipe, p, 1, leaf("wrong-entry")), "emitter accepted a different body");
        if (!use_fast) {
            auto unknown = leaf(selected->identity);
            unknown.exact_identity.clear();
            require(!emit_row_epilogue(recipe, p, 1, std::move(unknown)), "emitter admitted unproven arithmetic");
        }
        require(emit_row_epilogue(recipe, p, 1, leaf(selected->identity)), "could not emit admitted body");
        recipe.kernel(leaf(declarations.back().entry));
        const auto compiled = recipe.compile(ProgramKey{}, CudaReadback::resident);
        const auto& nodes = compiled.binding.nodes;
        require(nodes.size() == p.graph.launches.size() && compiled.program.launches[1].identity == selected->identity &&
                compiled.program.launches[1].chain_identity.find(selected->identity) != std::string::npos,
                "recipe lost the selected launch or chain identity");
        require(compiled.program.numerical_identity.has_value() == !use_fast, "wrong recipe EXACT declaration");
        require(nodes[1].after == std::vector<CudaDependency>{{0, CudaEdgeKind::programmatic}} &&
                nodes[2].after == std::vector<CudaDependency>{{1, CudaEdgeKind::programmatic}},
                "fused node did not use recipe PDL defaults");
    }
    std::cout << "PASS B3 FAST identity and CUDA PDL recipe\n";
}

}  // namespace

int main() {
    try {
        Catalog catalog;
        catalog.add(entries);
        check_graph(catalog);
        check_rounding(catalog);
        check_declines(catalog);
        check_recipe(catalog);
    } catch (const std::exception& error) {
        std::cerr << error.what() << '\n';
        return 1;
    }
}
