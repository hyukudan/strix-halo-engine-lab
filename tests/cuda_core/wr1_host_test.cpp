#include "exec/catalog_cuda_wr1.hpp"
#include "exec/cuda_decode_lowering.hpp"

#include <array>
#include <cstdio>
#include <cstring>
#include <set>
#include <stdexcept>

namespace {

using namespace strix::exec;
namespace decode = strix::exec::cuda_decode;
namespace wr1 = decode::wr1;
namespace mix = strix::core::residual_mix;

void trap(std::span<const std::byte>, const Grid&, void*) { throw std::logic_error("unexpected device launch"); }

void check(bool condition, const char* message) {
    if (!condition) throw std::runtime_error(message);
}

template <class F>
void refuses(F run) {
    try { run(); }
    catch (const std::invalid_argument&) { return; }
    throw std::runtime_error("accepted malformed binding");
}

void test() {
    wr1::Thunks thunks{};
    for (auto& width : thunks) width.fill(trap);
    StepBlock descriptor{};
    alignas(16) std::array<float, 2048> input{}, output{}, second{};
    alignas(16) std::array<float, 4096> scratch{};
    alignas(16) std::array<std::uint16_t, 4096> weights{};
    std::array<strix::head_argmax::Pick2, 16> partial{};
    std::array<RowOut, 16> picks{};
    std::set<std::string> identities;

    for (auto rows : wr1::widths) {
        wr1::Bindings bindings(&descriptor, rows, thunks);
        CudaProgramRecipe recipe;
        // The R1 latency forms are never selected for a wide step (forms off); only their incumbents bind.
        const auto append = [&](const auto& args, bool = false) {
            if constexpr (requires { bindings.bind(args); }) bindings.append(recipe, args);
            else throw std::logic_error("an R1-only form reached a wide recipe");
        };
        const auto fp32 = decode::Bf16DenseGroup{input.data(), 32,
            {strix::core::Bf16RowProjection{weights.data(), 17, output.data()}}, 1, decode::Bf16DenseMode::fp32};
        append(fp32);
        auto dense = fp32;
        dense.projections.front().outputs = 32;
        dense.mode = decode::Bf16DenseMode::mma3;
        append(dense);
        dense.mode = decode::Bf16DenseMode::mma3_round;
        dense.projections.front().round_bf16 = true;
        append(dense);

        const mix::NormalizeArgs bank{input.data(), output.data(), rows, 32, 0.001F};
        append(bank);
        const auto clipped = decode::HeadClipBf16{input.data(), weights.data(), output.data(),
            nullptr, 2, 32, 0, 128, 0.001F, 1.0F, 0.5F};
        append(clipped);
        append(decode::Bf16RelativeProjection16{input.data(), weights.data(), output.data(), 2, 33});
        append(decode::Bf16Softcap{output.data(), 32, 4.0F});
        append(decode::Bf16Round{output.data(), 32});
        append(decode::Bf16SigmoidProduct{input.data(), second.data(), output.data(), 32});
        append(mix::AddArgs{input.data(), second.data(), output.data(), 32});
        append(wr1::Greedy{input.data(), partial.data(), picks.data(), 32, 1});
        const std::array<std::span<const float>, 1> bank_rows{second};
        decode::lower_low_rank_controller(32, 16, 0.001F, std::span(input).first(32), input, bank_rows,
            output, scratch, weights.data(), weights.data(), weights.data(), weights.data(), append, {}, rows);

        ProgramKey key{};
        key.rows = rows;
        const auto lowered = recipe.compile(key, CudaReadback::executor);
        check(lowered.program.launches.size() == 18, "incorrect recipe launch count");
        for (const auto& launch : lowered.program.launches) {
            check(launch.rows == rows && launch.fn == trap && launch.kernels == 1, "incorrect width or thunk");
            const StepBlock* captured{};
            std::memcpy(&captured, lowered.program.args.data() + launch.args_offset, sizeof(captured));
            check(captured == &descriptor, "descriptor address was not captured");
            identities.insert(launch.entry);
        }
        for (std::size_t i = 1; i < lowered.binding.nodes.size(); ++i)
            check(lowered.binding.nodes[i].after.front().kind == CudaEdgeKind::completion, "WR1 edge is not completion");
        check(bindings.bind(clipped).grid.blocks == std::array{2U, rows, 1U}, "clip grid");
        check(bindings.bind(bank).grid.blocks == std::array{rows, 1U, 1U}, "bank grid");
        check(bindings.bind_partial({input.data(), partial.data(), picks.data(), 32, 1}).grid.blocks ==
              std::array{1U, rows, 1U}, "greedy grid");

        key.rows = rows == 1U ? 2U : 1U;
        refuses([&] { (void)recipe.compile(key, CudaReadback::executor); });
        refuses([&] { (void)bindings.bind(mix::NormalizeArgs{input.data(), output.data(), 0, 32, 0.001F}); });
        refuses([&] { (void)bindings.bind(wr1::Greedy{input.data(), partial.data(), nullptr, 32, 1}); });
    }
    check(identities.size() == wr1::widths.size() * static_cast<unsigned>(wr1::Leaf::count), "missing recipe identity");
    check(decode::controller_scratch_floats(32, 16, 1) == 144 &&
          decode::controller_scratch_floats(32, 16, 4) == 832, "controller scratch layout");
    refuses([&] { (void)wr1::Bindings(&descriptor, 3, thunks); });
    refuses([&] { (void)wr1::Bindings(nullptr, 4, thunks); });
}

}  // namespace

int main() try {
    test();
    std::puts("wr1_host PASS widths=1,2,4,8,16 leaves=16");
    return 0;
} catch (const std::exception& error) {
    std::fprintf(stderr, "wr1_host FAIL %s\n", error.what());
    return 1;
}
