#include "executor_cuda.hpp"
#include "strix/device/cuda_enumerator.hpp"
#include "strix/device/runtime.hpp"
#include "strix/exec/profiler.hpp"

#include <algorithm>
#include <array>
#include <cstring>
#include <iostream>
#include <memory>
#include <stdexcept>
#include <type_traits>

namespace {

using namespace strix::exec;
namespace native = strix::device::native;

template<class Args>
void node(Program& program, const char* entry, const Args& args) {
    auto& launch = program.launches.emplace_back();
    launch.entry = entry;
    launch.args_offset = static_cast<std::uint32_t>(program.args.size());
    launch.args_bytes = sizeof(args);
    program.args.resize(program.args.size() + sizeof(args));
    std::memcpy(program.args.data() + launch.args_offset, &args, sizeof(args));
}

template<class Input>
void check_profile(const Input& input, CudaDescriptorLayout layout, const char* name) {
    constexpr std::size_t bytes = 64;
    native::Buffer<unsigned char> device(bytes);
    unsigned char* host = nullptr;
    native::check(cudaMallocHost(&host, bytes), "allocate pinned X6 fixture reply");
    const std::unique_ptr<unsigned char, decltype(&cudaFreeHost)> pinned(host, cudaFreeHost);
    CudaExecutor executor(false, true, layout);
    Program program{};
    program.descriptor.rows = 1;
    node(program, "fixture.fill", CudaMemsetArgs{device.data(), bytes, 0x5a});
    node(program, "fixture.readback", CudaCopyArgs{host, device.data(), bytes});
    CudaProgramBinding binding;
    binding.readback = CudaReadback::bound;
    binding.nodes = {{CudaNodeKind::memset, {}}, {CudaNodeKind::copy_d2h_pinned, {{0}}}};
    const std::array work{LaunchWork{}, LaunchWork{0, bytes}};
    const auto result = profile(executor, program, input, {work, 0, 3}, &binding);
    std::array<unsigned char, bytes> instrumented{};
    std::ranges::copy(std::span(host, bytes), instrumented.begin());
    std::ranges::fill(std::span(host, bytes), static_cast<unsigned char>(0));
    if constexpr (std::is_same_v<Input, StepBlock>) executor.wait(executor.submit_x3(program, input, {&binding}), {});
    else executor.wait(executor.submit(program, input, {&binding}), {});
    const auto stats = executor.graph_stats(program.key);
    if (!std::ranges::equal(std::span(host, bytes), instrumented) ||
        !std::ranges::all_of(instrumented, [](auto value) { return value == 0x5a; }) ||
        result.launches.size() != 2 || result.bytes_per_step != bytes || result.repeats != 3 ||
        !result.cold_wall_ms || !result.wall_ms || *result.wall_ms <= 0 ||
        !result.busy_ms || *result.busy_ms <= 0 || stats.nodes != 2 || stats.captures != 1 ||
        stats.instantiations != 1 || stats.copies != 1 || stats.memsets != 1 || stats.edges != 1)
        throw std::runtime_error("X6 profile did not preserve output or installed graph");
    const auto r1 = profile_receipt(result, {});
    if (!r1.step || r1.step->size() != 3 || r1.step->back().bytes_read != bytes)
        throw std::runtime_error("X6 fixture R1 projection mismatch");
    std::cout << "x6_cuda_fixture: PASS " << name << " nodes=2 captures=1 bytes=64 repeats=3\n";
}

}  // namespace

int main(int argc, char** argv) try {
    strix::device::bind_cuda_main(argc, argv, nullptr);
    check_profile(CudaStepIn{}, CudaDescriptorLayout::legacy, "legacy");
    StepBlock step{};
    step.h.rows = 1;
    check_profile(step, CudaDescriptorLayout::x3, "x3");
    return 0;
} catch (const std::exception& error) {
    std::cerr << "x6_cuda_fixture: FAIL " << error.what() << '\n';
    return 1;
}
