#include "strix/hip_shared_expert_rows_diagnostic.hpp"
#include "strix/hip_setup_quiescence.hpp"
#include <hip/hip_runtime.h>

#include <algorithm>
#include <array>
#include <bit>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <exception>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

namespace {
using namespace strix::shared_expert_rows_diagnostic;
void check(hipError_t value, const char* operation) {
    if (value != hipSuccess) throw std::runtime_error(std::string(operation) + ": " + hipGetErrorString(value));
}
void require(bool value, const char* message) { if (!value) throw std::runtime_error(message); }
struct Owner {
    strix::HipSetupQuiescenceGuard quiescence;
    hipStream_t stream{};
    std::vector<void*> allocations;
    std::vector<hipGraph_t> graphs;
    std::vector<hipGraphExec_t> executions;
    Owner() { quiescence.run([&] { check(hipStreamCreate(&stream), "shared rows stream"); }); }
    ~Owner() {
        quiescence.recover_or_fail_stop();
        const auto cleanup = [](hipError_t value) { if (value != hipSuccess) std::_Exit(3); };
        for (auto e : executions) cleanup(hipGraphExecDestroy(e));
        for (auto g : graphs) cleanup(hipGraphDestroy(g));
        for (auto p : allocations) cleanup(hipFree(p));
        cleanup(hipStreamDestroy(stream));
    }
    Span allocate(std::uint64_t bytes) {
        void* p = nullptr;
        check(hipMalloc(&p, static_cast<std::size_t>(bytes + 128U)), "shared rows guarded allocation");
        try { allocations.push_back(p); }
        catch (...) { quiescence.recover_or_fail_stop(); (void)hipFree(p); throw; }
        check(hipMemset(p, 0xa5, static_cast<std::size_t>(bytes + 128U)), "shared rows initial poison");
        return {reinterpret_cast<std::uintptr_t>(p) + 64U, bytes};
    }
    void finish() { check(hipStreamSynchronize(stream), "shared rows completion"); quiescence.require_original(); }
};
using Bytes = std::vector<unsigned char>;
void upload(Span span, const Bytes& bytes) {
    require(span.bytes == bytes.size(), "shared rows upload extent");
    check(hipMemcpy(reinterpret_cast<void*>(span.address), bytes.data(), bytes.size(), hipMemcpyHostToDevice), "shared rows upload");
}
Bytes download(Span span) {
    Bytes result(static_cast<std::size_t>(span.bytes));
    check(hipMemcpy(result.data(), reinterpret_cast<const void*>(span.address), result.size(), hipMemcpyDeviceToHost), "shared rows download");
    return result;
}
void guards(Span span) {
    for (auto address : {span.address - 64U, span.address + span.bytes}) {
        std::array<unsigned char, 64> guard{};
        check(hipMemcpy(guard.data(), reinterpret_cast<const void*>(address), guard.size(), hipMemcpyDeviceToHost), "shared rows guard");
        require(std::all_of(guard.begin(), guard.end(), [](auto value) { return value == 0xa5; }), "shared rows canary changed");
    }
}
void u16(Bytes& bytes, std::uint64_t offset, std::uint16_t value) {
    std::memcpy(bytes.data() + static_cast<std::size_t>(offset), &value, sizeof(value));
}
void f32(Bytes& bytes, std::uint64_t offset, float value) {
    std::memcpy(bytes.data() + static_cast<std::size_t>(offset), &value, sizeof(value));
}
void same(const Bytes& actual, const Bytes& expected, const char* stage, Shape shape, unsigned flavor) {
    if (actual == expected) return;
    std::size_t first = 0;
    while (first < actual.size() && actual[first] == expected[first]) ++first;
    std::cerr << "shared rows mismatch stage=" << stage << " R=" << shape.rows << " H=" << shape.hidden
              << " M=" << shape.intermediate << " flavor=" << flavor << " first_byte=" << first << '\n';
    throw std::runtime_error("shared rows bitwise mismatch");
}
void padding(const Bytes& bytes, std::uint64_t stride, std::uint64_t payload, unsigned rows) {
    for (unsigned row = 0; row + 1U < rows; ++row)
        for (auto at = row * stride + payload; at < (row + 1U) * stride; ++at)
            require(bytes[static_cast<std::size_t>(at)] == 0xa5, "shared rows stride padding modified");
}
void fill_q8(Bytes& bytes, unsigned width, unsigned outputs, unsigned flavor, bool down) {
    constexpr std::uint16_t exceptional[]{0, 0x8000U, 1, 0x8001U, 0x7c00U, 0xfc00U, 0x7e11U, 0xfe55U};
    for (unsigned out = 0; out < outputs; ++out) for (unsigned block = 0; block < width / 32U; ++block) {
        const auto at = (static_cast<std::uint64_t>(out) * (width / 32U) + block) * 34U;
        const auto scale = flavor == 4U ? exceptional[(out + block) % 8U] :
                           static_cast<std::uint16_t>(down ? 0x1400U : 0x2400U);
        u16(bytes, at, scale);
        for (unsigned local = 0; local < 32U; ++local) {
            const int code = flavor == 5U ? 0 : flavor == 0U ? 1 :
                             static_cast<int>((out * 19U + block * 13U + local * 7U) % 256U) - 128;
            bytes[static_cast<std::size_t>(at + 2U + local)] =
                std::bit_cast<unsigned char>(static_cast<signed char>(code));
        }
    }
}
void graph_identity(hipGraph_t graph, Shape s, Arm arm) {
    std::size_t count = 0;
    check(hipGraphGetNodes(graph, nullptr, &count), "shared rows graph count");
    require(count == layout(s, arm).launches, "shared rows unexpected captured launch count");
    std::vector<hipGraphNode_t> nodes(count);
    check(hipGraphGetNodes(graph, nodes.data(), &count), "shared rows graph nodes");
    unsigned gate_up = 0, down = 0;
    for (auto node : nodes) {
        hipGraphNodeType kind{};
        check(hipGraphNodeGetType(node, &kind), "shared rows graph node kind");
        require(kind == hipGraphNodeTypeKernel, "shared rows captured nonkernel node");
        hipKernelNodeParams p{};
        check(hipGraphKernelNodeGetParams(node, &p), "shared rows graph params");
        if (p.func == projection_symbol(arm, s.rows, false)) {
            ++gate_up;
            require(p.gridDim.x == (s.intermediate + 7U) / 8U && p.gridDim.y == 1U && p.gridDim.z == 1U &&
                    p.blockDim.x == 256U && p.blockDim.y == 1U && p.blockDim.z == 1U && !p.sharedMemBytes,
                    "shared rows gate/up geometry changed");
        }
        if (p.func == projection_symbol(arm, s.rows, true)) {
            ++down;
            require(p.gridDim.x == (s.hidden + 7U) / 8U && p.gridDim.y == 1U && p.gridDim.z == 1U &&
                    p.blockDim.x == 256U && p.blockDim.y == 1U && p.blockDim.z == 1U && !p.sharedMemBytes,
                    "shared rows down geometry changed");
        }
    }
    const auto copies = arm == Arm::reference_rows ? s.rows : 1U;
    require(gate_up == 2U * copies && down == copies, "shared rows projection identity/count changed");
}
void family(unsigned rows, unsigned hidden, unsigned intermediate, unsigned flavor) {
    const Shape s{hidden, intermediate, rows, 2ULL * hidden + 16U, 2ULL * hidden + 32U, 8ULL * intermediate + 16U};
    const auto l = layout(s, Arm::shared_q8_rows);
    Owner owner;
    const auto input = owner.allocate(l.input_bytes), workspace = owner.allocate(l.workspace_bytes), output = owner.allocate(l.output_bytes);
    const Weights w{owner.allocate(l.gate_up_weight_bytes), owner.allocate(l.gate_up_weight_bytes),
                    owner.allocate(l.down_weight_bytes), owner.allocate(l.scalar_weight_bytes)};
    Bytes inputs(static_cast<std::size_t>(input.bytes), 0xa5), gate(static_cast<std::size_t>(w.gate_q8.bytes)),
          up(static_cast<std::size_t>(w.up_q8.bytes)), down(static_cast<std::size_t>(w.down_q8.bytes)),
          scalar(static_cast<std::size_t>(w.shared_gate_f32.bytes));
    constexpr std::uint16_t small[]{0, 0x8000U, 1, 0x8001U, 0x0080U, 0x8080U};
    constexpr std::uint16_t exceptional[]{0x7f80U, 0xff80U, 0x7fc1U, 0xffd5U, 0x7f81U, 0xff81U};
    for (unsigned row = 0; row < rows; ++row) for (unsigned feature = 0; feature < hidden; ++feature) {
        const auto bits = flavor == 0U ? static_cast<std::uint16_t>(0x3e00U + row * 32U + (feature % 16U)) :
                          flavor == 1U ? small[(feature + row) % 6U] :
                          flavor == 2U ? static_cast<std::uint16_t>((feature + row) & 1U ? 0xbf80U : 0x3f80U) :
                          flavor == 3U ? static_cast<std::uint16_t>((feature + row) & 1U ? 0xff7fU : 0x7f7fU) :
                          flavor == 4U ? exceptional[(feature + row) % 6U] : static_cast<std::uint16_t>(0x3f80U);
        u16(inputs, row * s.input_stride_bytes + 2ULL * feature, bits);
    }
    fill_q8(gate, hidden, intermediate, flavor, false); fill_q8(up, hidden, intermediate, flavor, false);
    fill_q8(down, intermediate, hidden, flavor, true);
    for (unsigned feature = 0; feature < hidden; ++feature)
        f32(scalar, 4ULL * feature, flavor == 5U ? 0.0F : 1.0F / 1024.0F);
    upload(input, inputs); upload(w.gate_q8, gate); upload(w.up_q8, up);
    upload(w.down_q8, down); upload(w.shared_gate_f32, scalar);
    const auto reset = [&] {
        check(hipMemsetAsync(reinterpret_cast<void*>(workspace.address), 0xa5, static_cast<std::size_t>(workspace.bytes), owner.stream), "shared rows workspace poison");
        check(hipMemsetAsync(reinterpret_cast<void*>(output.address), 0xa5, static_cast<std::size_t>(output.bytes), owner.stream), "shared rows output poison");
    };
    reset(); enqueue(s, input, w, workspace, output, Arm::reference_rows, owner.stream); owner.finish();
    const auto expected = download(output), expected_workspace = download(workspace);
    padding(expected, s.output_stride_bytes, 2ULL * hidden, rows);
    padding(expected_workspace, s.workspace_stride_bytes, 8ULL * intermediate, rows);
    if (flavor == 5U) {
        Bytes exact_output(expected.size(), 0xa5), exact_workspace(expected_workspace.size(), 0xa5);
        for (unsigned row = 0; row < rows; ++row) {
            for (unsigned feature = 0; feature < hidden; ++feature) u16(exact_output, row * s.output_stride_bytes + 2ULL * feature, 0);
            for (unsigned feature = 0; feature < 2U * intermediate; ++feature) f32(exact_workspace, row * s.workspace_stride_bytes + 4ULL * feature, 0.0F);
            f32(exact_workspace, row * s.workspace_stride_bytes + 4ULL * intermediate, 0.5F);
        }
        same(expected, exact_output, "independent zero oracle output", s, flavor);
        same(expected_workspace, exact_workspace, "independent zero oracle workspace", s, flavor);
    }
    for (auto arm : {Arm::reference_rows, Arm::shared_q8_rows}) {
        reset(); enqueue(s, input, w, workspace, output, arm, owner.stream); owner.finish();
        same(download(output), expected, "eager output", s, flavor);
        same(download(workspace), expected_workspace, "eager workspace", s, flavor);
        check(hipStreamBeginCapture(owner.stream, hipStreamCaptureModeThreadLocal), "shared rows capture begin");
        enqueue(s, input, w, workspace, output, arm, owner.stream);
        hipGraph_t graph{};
        check(hipStreamEndCapture(owner.stream, &graph), "shared rows capture end");
        owner.graphs.push_back(graph); graph_identity(graph, s, arm);
        hipGraphExec_t execution{};
        check(hipGraphInstantiate(&execution, graph, nullptr, nullptr, 0), "shared rows graph instantiate");
        owner.executions.push_back(execution);
        for (unsigned replay = 0; replay < 3U; ++replay) {
            reset(); check(hipGraphLaunch(execution, owner.stream), "shared rows graph replay"); owner.finish();
            same(download(output), expected, "graph output", s, flavor);
            same(download(workspace), expected_workspace, "graph workspace", s, flavor);
        }
    }
    if (flavor == 0U) {
        auto changed_input = inputs;
        for (unsigned feature = 0; feature < hidden; ++feature) {
            std::uint16_t value{}; std::memcpy(&value, changed_input.data() + 2ULL * feature, 2);
            u16(changed_input, 2ULL * feature, static_cast<std::uint16_t>(value ^ 0x8000U));
        }
        upload(input, changed_input);
        reset(); enqueue(s, input, w, workspace, output, Arm::reference_rows, owner.stream); owner.finish();
        const auto mutant = download(output);
        require(!std::equal(mutant.begin(), mutant.begin() + static_cast<std::ptrdiff_t>(2ULL * hidden), expected.begin()),
                "shared rows input mutation is invisible at BF16 output");
        for (unsigned row = 1; row < rows; ++row)
            require(std::equal(mutant.begin() + static_cast<std::ptrdiff_t>(row * s.output_stride_bytes),
                               mutant.begin() + static_cast<std::ptrdiff_t>(row * s.output_stride_bytes + 2ULL * hidden),
                               expected.begin() + static_cast<std::ptrdiff_t>(row * s.output_stride_bytes)),
                    "single-row input mutation contaminated another row");
        reset(); enqueue(s, input, w, workspace, output, Arm::shared_q8_rows, owner.stream); owner.finish();
        same(download(output), mutant, "row mutation", s, flavor);
        upload(input, inputs);
        auto changed_down = down;
        for (std::size_t at = 1; at < changed_down.size(); at += 34U) changed_down[at] ^= 0x80U;
        upload(w.down_q8, changed_down);
        reset(); enqueue(s, input, w, workspace, output, Arm::reference_rows, owner.stream); owner.finish();
        const auto coefficient_mutant = download(output);
        require(coefficient_mutant != expected, "shared rows Q8 coefficient mutation is invisible");
        reset(); enqueue(s, input, w, workspace, output, Arm::shared_q8_rows, owner.stream); owner.finish();
        same(download(output), coefficient_mutant, "coefficient mutation", s, flavor);
        upload(w.down_q8, down);
    }
    same(download(input), inputs, "readonly input", s, flavor);
    same(download(w.gate_q8), gate, "readonly gate", s, flavor);
    same(download(w.up_q8), up, "readonly up", s, flavor);
    same(download(w.down_q8), down, "readonly down", s, flavor);
    same(download(w.shared_gate_f32), scalar, "readonly scalar", s, flavor);
    for (auto span : {input, workspace, output, w.gate_q8, w.up_q8, w.down_q8, w.shared_gate_f32}) guards(span);
}
void rejection() {
    Owner owner;
    const Shape original{32, 32, 2, 64, 64, 256};
    const auto l = layout(original, Arm::shared_q8_rows);
    const auto input = owner.allocate(l.input_bytes), workspace = owner.allocate(l.workspace_bytes), output = owner.allocate(l.output_bytes);
    const Weights weights{owner.allocate(l.gate_up_weight_bytes), owner.allocate(l.gate_up_weight_bytes),
                          owner.allocate(l.down_weight_bytes), owner.allocate(l.scalar_weight_bytes)};
    const auto before_output = download(output), before_workspace = download(workspace);
    for (unsigned invalid = 0; invalid < 14U; ++invalid) {
        auto s = original; auto in = input; auto w = weights; auto work = workspace; auto out = output;
        auto arm = Arm::shared_q8_rows; auto stream = owner.stream;
        switch (invalid) {
        case 0: s.rows = 3; break;
        case 1: ++s.input_stride_bytes; break;
        case 2: --in.bytes; break;
        case 3: ++in.bytes; break;
        case 4: in.address = 0; break;
        case 5: work.address = in.address; break;
        case 6: out.address = work.address; break;
        case 7: --w.down_q8.bytes; break;
        case 8: ++w.gate_q8.address; break;
        case 9: arm = static_cast<Arm>(42); break;
        case 10: stream = nullptr; break;
        case 11: stream = hipStreamLegacy; break;
        case 12: stream = hipStreamPerThread; break;
        case 13: w.shared_gate_f32.address = 0; break;
        }
        check(hipStreamBeginCapture(owner.stream, hipStreamCaptureModeThreadLocal), "shared rows negative capture begin");
        bool rejected = false; std::exception_ptr unexpected;
        try { enqueue(s, in, w, work, out, arm, stream); }
        catch (const std::invalid_argument&) { rejected = true; }
        catch (...) { unexpected = std::current_exception(); }
        hipGraph_t graph{};
        check(hipStreamEndCapture(owner.stream, &graph), "shared rows negative capture end");
        owner.graphs.push_back(graph);
        if (unexpected) std::rethrow_exception(unexpected);
        std::size_t count = 0;
        check(hipGraphGetNodes(graph, nullptr, &count), "shared rows negative node count");
        require(rejected && count == 0U, "invalid shared rows view enqueued a kernel");
    }
    owner.finish();
    same(download(output), before_output, "negative unchanged output", original, 0);
    same(download(workspace), before_workspace, "negative unchanged workspace", original, 0);
    for (auto span : {input, workspace, output, weights.gate_q8, weights.up_q8, weights.down_q8, weights.shared_gate_f32}) guards(span);
}
} // namespace
int main() try {
    int devices = 0;
    check(hipGetDeviceCount(&devices), "shared rows device count");
    require(devices > 0, "shared rows native fixture requires a HIP device");
    for (unsigned rows : {2U, 4U, 8U}) for (const auto sizes : {std::array{32U, 32U}, std::array{96U, 288U},
                                                           std::array{288U, 96U}, std::array{2560U, 640U}})
        for (unsigned flavor = 0; flavor < 6U; ++flavor) family(rows, sizes[0], sizes[1], flavor);
    rejection();
    std::cout << "shared rows standalone native equivalence fixtures passed; no whole-decoder R>1 claim\n";
    return 0;
} catch (const std::exception& error) { std::cerr << error.what() << '\n'; return 1; }
