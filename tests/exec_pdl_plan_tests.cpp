#include "cuda_program_recipe.hpp"

#include <iostream>
#include <stdexcept>
#include <string_view>

namespace {

using namespace strix::exec;

void require(bool valid, const char* message) {
    if (!valid) throw std::runtime_error(message);
}

KernelBinding leaf(PdlWait wait, PdlOptOut reason = PdlOptOut::none) {
    KernelBinding binding;
    binding.fn = +[](std::span<const std::byte>, const Grid&, void*) {
        throw std::logic_error("planning executed a kernel thunk");
    };
    binding.identity = binding.exact_identity = "test-cuda-pdl-leaf";
    binding.grid = {{1U, 1U, 1U}, {32U, 1U, 1U}, 0U};
    binding.predicted.launches = 1U;
    binding.pdl = {wait, reason};
    return binding;
}

void expect_edge(const std::vector<CudaNodeBinding>& nodes, std::size_t index, CudaEdgeKind kind) {
    require(nodes.at(index).after == std::vector<CudaDependency>{{index - 1U, kind}},
            "unexpected CUDA PDL predecessor or edge kind");
}

void plans_default_to_pdl() {
    using enum CudaEdgeKind;
    CudaProgramRecipe recipe;
    recipe.kernel(leaf(PdlWait::before_step_io));
    recipe.kernel(leaf(PdlWait::before_step_io));
    recipe.kernel(leaf(PdlWait::absent, PdlOptOut::kernel_without_wait));
    recipe.kernel(leaf(PdlWait::before_step_io));
    recipe.kernel(leaf(PdlWait::before_step_io, PdlOptOut::legacy_schedule));
    unsigned cleared{};
    recipe.clear({&cleared, sizeof(cleared), 0});
    recipe.kernel(leaf(PdlWait::before_step_io));

    const auto compiled = recipe.compile(ProgramKey{}, CudaReadback::executor);
    const auto on = cuda_detail::capture_plan(compiled.program, compiled.binding, true);
    require(on.size() == 7 && on.front().after.empty(), "invalid CUDA plan root or size");
    expect_edge(on, 1, programmatic);
    expect_edge(on, 2, completion);
    expect_edge(on, 3, programmatic);
    expect_edge(on, 4, completion);
    expect_edge(on, 5, completion);
    expect_edge(on, 6, completion);

    const auto off = cuda_detail::capture_plan(compiled.program, compiled.binding, false);
    require(off.size() == on.size() && off.front().after.empty(), "capability fallback changed the CUDA root or size");
    for (std::size_t i = 1; i < off.size(); ++i) expect_edge(off, i, completion);

    CudaProgramRecipe tapped(true);
    tapped.kernel(leaf(PdlWait::before_step_io));
    tapped.kernel(leaf(PdlWait::before_step_io));
    ProgramKey tapped_key{};
    tapped_key.tap = true;
    const auto eager = tapped.compile(tapped_key, CudaReadback::executor);
    expect_edge(eager.binding.nodes, 1, completion);
}

void expect_rejection(auto add_leaf, std::string_view diagnostic) {
    CudaProgramRecipe recipe;
    try {
        add_leaf(recipe);
    } catch (const std::invalid_argument& error) {
        require(std::string_view(error.what()).find(diagnostic) != std::string_view::npos,
                "wrong CUDA PDL rejection diagnostic");
        return;
    }
    throw std::runtime_error("invalid CUDA PDL kernel accepted by the recipe");
}

void rejects_invalid_policy() {
    expect_rejection([](auto& recipe) { recipe.kernel(leaf(PdlWait::absent)); },
                     "requires a grid dependency wait");
    expect_rejection([](auto& recipe) { recipe.kernel(leaf(PdlWait::undeclared)); },
                     "requires a grid dependency wait");
    expect_rejection([](auto& recipe) {
        recipe.kernel(leaf(PdlWait::absent, static_cast<PdlOptOut>(255U)));
    }, "invalid CUDA PDL opt-out reason");
    expect_rejection([](auto& recipe) {
        recipe.kernel(leaf(PdlWait::before_step_io, PdlOptOut::kernel_without_wait));
    }, "opt-out requires a body without");
    expect_rejection([](auto& recipe) {
        recipe.kernel(leaf(PdlWait::absent, PdlOptOut::kernel_without_wait), true);
    }, "conflicts with the kernel wait contract");
}

}  // namespace

int main() {
    try {
        plans_default_to_pdl();
        rejects_invalid_policy();
        std::cout << "PASS exec PDL default plan\n";
    } catch (const std::exception& error) {
        std::cerr << error.what() << '\n';
        return 1;
    }
}
