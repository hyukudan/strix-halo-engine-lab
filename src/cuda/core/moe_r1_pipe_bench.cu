#include "cuda/core/moe_r1_pipe_bench.cuh"

#include "cuda/core/moe_r1_pipe.cuh"
#include "exec/executor_cuda.hpp"
#include "strix/device/gate.hpp"
#include "strix/device/runtime.hpp"
#include "strix/fnv.hpp"

#include <algorithm>
#include <array>
#include <cstring>
#include <stdexcept>

namespace strix::exec::bench {
namespace {

namespace r1 = moe_core::microscaled_r1;
namespace pipe = r1::pipe;
namespace native = device::native;
using r1::packed::MatrixFormat;

template <MatrixFormat Format, bool Gate, bool Pipe>
int enqueue(void* binding, void* queue) {
    const auto& args = *static_cast<const r1::Args*>(binding);
    const auto stream = static_cast<cudaStream_t>(queue);
    if constexpr (Gate && Pipe) return pipe::launch_gate_up<Format>(args, stream);
    if constexpr (Gate && !Pipe) return r1::launch_gate_up<Format, pipe::kTop>(args, stream);
    if constexpr (!Gate && Pipe) return pipe::launch_down_combine<Format>(args, stream);
    if constexpr (!Gate && !Pipe) return r1::launch_down_combine<Format, pipe::kTop>(args, stream);
}

template <MatrixFormat Format, bool Gate>
EntryLaunch resolve(void* binding, const LauncherRef& ref, const LaunchSite& site) {
    if (site.backend_args != binding) return {};
    if (ref.mode == "r1.k8.pipe") return {enqueue<Format, Gate, true>, binding};
    if (ref.mode == "r1.k8") return {enqueue<Format, Gate, false>, binding};
    return {};
}

void launch_entry(std::span<const std::byte> bytes, const Grid&, void* queue) {
    EntryLaunch call{};
    std::memcpy(&call, bytes.data(), sizeof(call));
    native::check(static_cast<cudaError_t>(call.enqueue(call.binding, queue)), "enqueue synthetic MoE R1 entry");
}

template <class T>
void upload(native::Buffer<T>& buffer, std::span<const T> host) {
    native::check(cudaMemcpy(buffer.data(), host.data(), host.size_bytes(), cudaMemcpyHostToDevice), "seed synthetic MoE R1 entry");
}

template <MatrixFormat Format, bool Gate>
StepProfile profile_entry(const Options& options, const KernelEntry& entry, const device::Bound& bound) {
    using Operand = r1::packed::PackedOperand<Format>;
    using Expert = moe_core::device_grouped::ExpertWeights;
    using r1::packed::RowOrder;
    // This diagnostic deliberately fixes the Inkling-shaped workload for incumbent/candidate parity.
    constexpr unsigned h = 2304U, n = 1152U, top = pipe::kTop;
    if (options.rows != 1U || options.width != h) throw std::invalid_argument("MoE R1 entry requires --rows 1 --width 2304");
    constexpr unsigned rows = Gate ? 2U * n : h;
    constexpr unsigned width = Gate ? h : n;
    constexpr std::size_t codes_per_expert = std::size_t{rows} * width / (Operand::kNvfp4 ? 2U : 1U);
    constexpr std::size_t scales_per_expert = std::size_t{rows} * width / Operand::kBlockColumns;
    constexpr std::size_t weight_bytes = top * (codes_per_expert + scales_per_expert);
    constexpr std::size_t state_bytes = Gate ? h * sizeof(float) : top * n * sizeof(std::uint16_t);
    constexpr auto allocation = weight_bytes + top * (sizeof(Expert) + sizeof(unsigned) + sizeof(float)) +
        2U * h * sizeof(float) + top * n * sizeof(std::uint16_t);
    if (bound.budget.device_capped() && allocation > bound.budget.device_bytes)
        throw std::runtime_error("synthetic MoE operands exceed the declared device budget");

    native::Buffer<std::uint8_t> codes(top * codes_per_expert), scales(top * scales_per_expert);
    std::vector<std::uint8_t> host_codes(top * codes_per_expert), host_scales(top * scales_per_expert);
    std::uint32_t seed = 9137U;
    for (auto& byte : host_codes) {
        seed = seed * 1664525U + 1013904223U;
        byte = static_cast<std::uint8_t>(seed >> 24U);
        if constexpr (!Operand::kNvfp4) {
            if ((byte & 0x7fU) == 0x7fU) --byte;
        }
    }
    for (auto& byte : host_scales) {
        seed = seed * 1664525U + 1013904223U;
        byte = static_cast<std::uint8_t>(Operand::kNvfp4 ? 32U + (seed >> 27U) : 115U + (seed >> 29U));
    }
    upload(codes, std::span<const std::uint8_t>(host_codes));
    upload(scales, std::span<const std::uint8_t>(host_scales));

    native::Buffer<Expert> matrices(top);
    native::Buffer<unsigned> ids(top);
    native::Buffer<float> weights(top), input(h), output(h);
    native::Buffer<std::uint16_t> activation(top * n);
    std::array<Expert, top> host_matrices{};
    std::array<unsigned, top> host_ids{};
    std::array<float, top> host_weights{};
    for (unsigned expert = 0; expert < top; ++expert) {
        const r1::packed::PackedMatrix matrix{codes.data() + expert * codes_per_expert,
            scales.data() + expert * scales_per_expert, rows, width, 0.375F,
            Gate ? RowOrder::interleaved_halves : RowOrder::contiguous};
        if constexpr (Gate) host_matrices[expert].gate_up = matrix;
        else host_matrices[expert].down = matrix;
        host_ids[expert] = top - 1U - expert;
        host_weights[expert] = static_cast<float>(expert + 1U) / 37.0F;
    }
    upload(matrices, std::span<const Expert>(host_matrices));
    upload(ids, std::span<const unsigned>(host_ids));
    upload(weights, std::span<const float>(host_weights));
    if constexpr (Gate) {
        std::array<float, h> host_input{};
        for (unsigned i = 0; i < h; ++i) host_input[i] = static_cast<float>(static_cast<int>(i % 97U) - 48) / 89.0F;
        upload(input, std::span<const float>(host_input));
    } else {
        std::vector<std::uint16_t> host_activation(top * n);
        for (unsigned i = 0; i < top * n; ++i) host_activation[i] = static_cast<std::uint16_t>(0x3e80U + i % 257U);
        upload(activation, std::span<const std::uint16_t>(host_activation));
    }

    r1::Args args{matrices.data(), ids.data(), weights.data(), nullptr, input.data(), activation.data(), output.data(),
                 top, h, n, 3.0F, 2.0F};
    LaunchSite site;
    site.backend_args = &args;
    site.rows = 1U;
    site.weight_bytes = weight_bytes;
    site.state_bytes = state_bytes;
    site.macs = std::uint64_t{rows} * width * top;
    EntryBinder binder{resolve<Format, Gate>, &args};
    const auto call = entry.bind(site, binder);
    if (!call) throw std::runtime_error("MoE catalogue entry declined synthetic operands");

    Program program{};
    program.key.rows = program.descriptor.rows = 1U;
    program.identity = host_fnv::field(host_fnv::offset64, entry.identity);
    auto& launch = program.launches.emplace_back();
    launch.fn = launch_entry;
    launch.identity = entry.identity;
    launch.entry = entry.key.form;
    launch.capturable = entry.capturable;
    launch.rows = launch.kernels = 1U;
    const bool piped = entry.launcher->mode == "r1.k8.pipe";
    if constexpr (Gate) {
        launch.grid.blocks = {piped ? pipe::gate_blocks(n) : r1::gate_blocks(n, r1::kCudaGeometry.gate_columns), top, 1U};
        launch.grid.threads = {piped ? pipe::kGateWarps * pipe::kWarp : r1::detail::kThreads, 1U, 1U};
    } else {
        launch.grid.blocks = {piped ? pipe::down_blocks(h) : r1::down_blocks(h, r1::kCudaGeometry.down_columns), 1U, 1U};
        launch.grid.threads = {r1::detail::kThreads, 1U, 1U};
    }
    program.stages.push_back({Gate ? "synthetic.moe.gate_up" : "synthetic.moe.down_combine", -1, graph::Format::f32});
    launch.stages.emplace_back();
    launch.args_bytes = sizeof(call);
    const auto bytes = std::as_bytes(std::span(&call, 1));
    program.args.assign(bytes.begin(), bytes.end());

    const LaunchWork work{site.weight_bytes, site.state_bytes, 2U * site.macs, &entry, options.peak_tops};
    CudaExecutor executor(false);
    native::check(cudaMemset(executor.output(), 0, sizeof(CudaStepOut) * kStepMaxRows), "initialize synthetic MoE reply");
    native::check(cudaStreamSynchronize(nullptr), "wait for synthetic MoE reply initialization");
    return profile(executor, program, CudaStepIn{}, {std::span(&work, 1), options.byte_gbps, options.repeats});
}

}  // namespace

const KernelEntry* find_moer_entry(const Options& options) {
    const auto entries = Catalog::builtin_entries(Backend::cuda);
    const auto result = std::ranges::find_if(entries, [&](const KernelEntry& entry) {
        const auto form = entry.key.form;
        return entry.identity == options.entry && (form == "microscaled.packed_gate_up.k8" ||
            form == "microscaled.packed_gate_up.k8.pipe" || form == "microscaled.packed_down_combine.k8" ||
            form == "microscaled.packed_down_combine.k8.pipe");
    });
    return result == entries.end() ? nullptr : &*result;
}

Output run_moer_entry(const Options& options, const KernelEntry& entry, const device::Bound& bound) {
    if ((entry.arch & device::arch_mask(bound.visible.arch)) == device::ArchMask::none)
        throw std::invalid_argument("MoE entry does not support the selected architecture");
    const bool gate = entry.key.op == graph::Op::linear;
    const bool nvfp4 = entry.key.weight.name == "nvidia.nvfp4";
    const auto result = gate ? (nvfp4 ? profile_entry<MatrixFormat::nvfp4, true>(options, entry, bound)
                                     : profile_entry<MatrixFormat::mxfp8, true>(options, entry, bound))
                             : (nvfp4 ? profile_entry<MatrixFormat::nvfp4, false>(options, entry, bound)
                                     : profile_entry<MatrixFormat::mxfp8, false>(options, entry, bound));
    receipt::Receipt receipt;
    receipt.identity.engine = "strix-bench";
    receipt.identity.implementation = "CUDA microscaled MoE R1";
    receipt.identity.kernel = entry.identity;
    receipt.identity.build = options.build;
    receipt.identity.device = bound.visible.name;
    receipt.identity.command = options.command;
    receipt.identity.topology = "single-device";
    receipt.resolution.emplace();
    receipt.resolution->tier = receipt::Tier::base;
    receipt.resolution->description = "synthetic packed MoE R1, repeated weight image";
    auto& device = receipt.devices.emplace().emplace_back();
    device.profile_id = bound.device.profile;
    device.power_state = options.power_state;
    if (options.byte_gbps > 0) device.bytes_per_second = options.byte_gbps * 1e9;
    return {write_step_profile(result, std::move(receipt)), write_step_table(result)};
}

}  // namespace strix::exec::bench
