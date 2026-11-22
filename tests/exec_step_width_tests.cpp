#include "cuda_program_recipe.hpp"

#include <array>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <map>
#include <stdexcept>
#include <string_view>

namespace {

using namespace strix::exec;
using strix::session::StepReplayAction;
using strix::session::StepReplayPolicy;
using cuda_detail::GraphAction;
using cuda_detail::GraphBinding;

constexpr std::array widths{1U, 2U, 4U, 8U, 16U};

void check(bool condition, const char* message) {
    if (!condition) throw std::runtime_error(message);
}

template <class Work>
void refuses(Work&& work, std::string_view reason) {
    try {
        work();
    } catch (const std::invalid_argument& error) {
        check(error.what() == reason, error.what());
        return;
    }
    throw std::runtime_error("unsupported step accepted");
}

struct Args {
    const StepBlock* descriptor;
    std::uint32_t rows;
    std::uint32_t operation;
};

void gather(std::span<const std::byte>, const Grid&, void*) {}
void pick(std::span<const std::byte>, const Grid&, void*) {}

KernelBinding bind(const StepBlock& descriptor, std::uint32_t rows, std::uint32_t operation) {
    KernelBinding kernel;
    kernel.fn = operation == 0 ? gather : pick;
    kernel.identity = operation == 0 ? "test-gather" : "test-pick";
    kernel.exact_identity = kernel.identity;
    kernel.grid = {{rows, 1, 1}, {32, 1, 1}, 0};
    kernel.predicted.launches = 1;
    const Args args{&descriptor, rows, operation};
    const auto bytes = std::as_bytes(std::span(&args, 1));
    kernel.args.assign(bytes.begin(), bytes.end());
    return kernel;
}

[[nodiscard]] CudaRecipeProgram incumbent(const StepBlock& descriptor) {
    CudaProgramRecipe recipe;
    recipe.stage("gather");
    recipe.kernel(bind(descriptor, 1, 0));
    recipe.stage("pick");
    recipe.kernel(bind(descriptor, 1, 1), true);
    return recipe.compile({}, CudaReadback::executor);
}

[[nodiscard]] CudaRecipeProgram lower(const StepBlock& descriptor, ProgramKey key) {
    CudaProgramRecipe recipe;
    recipe.stage("gather");
    recipe.kernel(bind(descriptor, key.rows, 0), false, key.rows);
    recipe.stage("pick");
    recipe.kernel(bind(descriptor, key.rows, 1), true, key.rows);
    return recipe.compile(key, CudaReadback::executor);
}

void check_launches(const CudaRecipeProgram& lowered, const StepBlock& descriptor, std::uint32_t rows) {
    const auto& program = lowered.program;
    check(program.descriptor.rows == rows && program.launches.size() == 2 && program.predicted.launches == 2,
          "step width changed the number of launches");
    const std::vector<CudaNodeBinding> expected{{CudaNodeKind::kernel, {}},
        {CudaNodeKind::kernel, {{0, CudaEdgeKind::programmatic}}}};
    check(cuda_detail::capture_plan(program, lowered.binding, true) == expected, "unexpected width capture plan");

    for (std::uint32_t i = 0; i < program.launches.size(); ++i) {
        const auto& launch = program.launches[i];
        check(launch.fn == (i == 0 ? gather : pick) && launch.entry == (i == 0 ? "test-gather" : "test-pick") &&
            launch.after == (i == 0 ? no_launch : i - 1) && launch.rows == rows && launch.kernels == 1 &&
            launch.grid.blocks == std::array{rows, 1U, 1U} && launch.grid.threads == std::array{32U, 1U, 1U} &&
            launch.grid.shared_bytes == 0 && launch.args_bytes == sizeof(Args), "wrong width launch binding");
        Args args{};
        std::memcpy(&args, program.args.data() + launch.args_offset, sizeof(args));
        check(args.descriptor == &descriptor && args.rows == rows && args.operation == i,
              "wrong width argument block");
    }
}

void check_descriptor(const Program& program, StepBlock& descriptor) {
    const auto rows = program.key.rows;
    descriptor = {};
    descriptor.h.rows = rows;
    descriptor.h.step = 73;
    for (std::uint32_t i = 0; i < rows; ++i) {
        const bool valid = rows == 1 || i + 1 < rows;
        descriptor.row[i] = {100U + i, 4096U + i, 32U - i, kStepNoParent, 1U << i,
                             valid ? std::uint32_t{step_valid} : 0U, 0, 0};
        if (valid) descriptor.h.valid_mask |= 1U << i;
    }
    cuda_detail::validate_x3_input(program, descriptor, 1);
    const auto bytes = std::as_bytes(std::span(&descriptor, 1));
    for (std::uint32_t i = 0; i < rows; ++i) {
        std::array<std::uint32_t, 3> words{};
        std::memcpy(words.data(), bytes.data() + 32U + 32U * i, sizeof(words));
        check(words == std::array{100U + i, 4096U + i, 32U - i}, "position or slot ABI differs");
        check(step_row_valid(descriptor, i) == (rows == 1 || i + 1 < rows), "padding lost row mask");
    }
    auto wrong = descriptor;
    wrong.h.rows = 3;
    refuses([&] { cuda_detail::validate_x3_input(program, wrong, 1); }, "invalid CUDA X3 step header");
}

void test_widths() {
    StepBlock descriptor{};
    const auto base = incumbent(descriptor);
    check_launches(base, descriptor, 1);
    std::array<CudaRecipeProgram, widths.size()> programs;
    std::map<ProgramKey, GraphBinding> graphs;
    for (std::size_t i = 0; i < widths.size(); ++i) {
        ProgramKey key{};
        key.rows = widths[i];
        auto& lowered = programs[i];
        lowered = lower(descriptor, key);
        const auto& program = lowered.program;
        check_launches(lowered, descriptor, key.rows);
        check_descriptor(program, descriptor);
        check(program.numerical_identity && program.numerical_identity == base.program.numerical_identity,
              "width changed the declared arithmetic identity");
        if (key.rows == 1) {
            check(program.identity == base.program.identity && program.args == base.program.args &&
                lowered.binding == base.binding, "R=1 differs from incumbent lowering");
        } else {
            for (std::size_t previous = 0; previous < i; ++previous)
                check(program.identity != programs[previous].program.identity, "width reused physical identity");
        }
        const auto [slot, inserted] = graphs.try_emplace(key);
        check(inserted && slot->second.select(program, 0, lowered.binding) == GraphAction::capture,
              "new width did not get a fresh graph slot");
        slot->second.remember(program, 0, true, lowered.binding);
    }
    for (std::size_t step = 0; step < 1000; ++step) {
        const auto& lowered = programs[step % widths.size()];
        const auto& program = lowered.program;
        const auto epoch = step / widths.size();
        descriptor.h.rows = program.key.rows;
        descriptor.h.valid_mask = 1;
        descriptor.row[0].position = static_cast<std::uint32_t>(step);
        descriptor.row[0].slot = static_cast<std::uint32_t>(step % 17);
        for (std::uint32_t row = 1; row < program.key.rows; ++row) descriptor.row[row].flags = 0;
        cuda_detail::validate_x3_input(program, descriptor, 1);
        auto& graph = graphs.at(program.key);
        check(graph.select(program, epoch, lowered.binding) == GraphAction::replay,
              "width switch or stable layout invalidation recaptured the graph");
        graph.remember(program, epoch, true, lowered.binding);
    }
    const auto& width4 = programs[2];
    Program moved = width4.program;
    moved.args[0] ^= std::byte{1};  // A workspace rebind changes captured argument bytes.
    check(graphs.at(moved.key).select(moved, 200, width4.binding) == GraphAction::rebind,
          "same-width arena change did not request a rebind");
    const auto verify = lower(descriptor, {.phase = Phase::verify, .rows = 4});
    check_launches(verify, descriptor, 4);
    check(graphs.try_emplace(verify.program.key).second, "verify and decode shared a graph slot");
}

void test_policy_and_refusals() {
    StepReplayPolicy policy;
    for (const auto rows : widths) {
        check(policy.select_bound(rows) == StepReplayAction::capture, "new width was treated as the installed graph");
        policy.captured(rows);
        check(policy.select_bound(rows) == StepReplayAction::replay, "installed width did not replay");
    }
    policy.invalidate();
    check(policy.select_bound(16) == StepReplayAction::capture, "invalidation retained the captured width");
    policy.fail();
    check(policy.select_bound(4) == StepReplayAction::eager, "failed graph attempted to capture again");
    policy.reset();
    check(policy.select_bound(4) == StepReplayAction::capture, "reset did not clear failure");
    check(policy.select(true, 4U, true, true) == StepReplayAction::eager, "legacy scalar transport admitted a wide step");

    StepBlock descriptor{};
    CudaProgramRecipe recipe;
    recipe.stage("gather");
    recipe.kernel(bind(descriptor, 4, 0), false, 4);
    recipe.stage("legacy-pick");
    recipe.kernel(bind(descriptor, 1, 1));
    const auto compile = [&](std::uint32_t rows) {
        (void)recipe.compile({.phase = Phase::verify, .rows = rows}, CudaReadback::executor);
    };
    refuses([&] { compile(4); }, "CUDA recipe width unsupported at legacy-pick/1: requested R=4, bound R=1");
    refuses([&] { compile(3); }, "unsupported CUDA step width R=3");

    CudaProgramRecipe transfer;
    transfer.stage("clear");
    transfer.clear({&descriptor, sizeof(std::uint32_t), 0});
    refuses([&] { (void)transfer.compile({.rows = 4}, CudaReadback::executor); },
            "CUDA recipe width unsupported at clear/0: requested R=4, bound R=1");
}

}  // namespace

int main() try {
    test_widths();
    test_policy_and_refusals();
    std::puts("exec_step_widths PASS widths=1,2,4,8,16 R1_nodes=2 mixed_steps=1000");
    return 0;
} catch (const std::exception& error) {
    std::fprintf(stderr, "exec_step_widths FAIL: %s\n", error.what());
    return 1;
}
