#include "strix/moe_batched_activation_diagnostic.hpp"
#include "strix/hip_setup_quiescence.hpp"

#include <hip/hip_runtime.h>

#include <algorithm>
#include <array>
#include <bit>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <iostream>
#include <limits>
#include <stdexcept>
#include <string>
#include <vector>

namespace {
using namespace strix::moe_batched_activation_diagnostic;
void require(bool condition, const char* text) { if (!condition) throw std::runtime_error(text); }
void check(hipError_t error, const char* text) {
    if (error != hipSuccess) throw std::runtime_error(std::string(text) + ": " + hipGetErrorString(error));
}
struct Owner {
    strix::HipSetupQuiescenceGuard guard;
    hipStream_t stream = nullptr;
    std::vector<void*> allocations;
    std::vector<hipGraph_t> graphs;
    std::vector<hipGraphExec_t> executions;
    Owner() {
        require(guard.architecture().starts_with("gfx1151"), "batched activation native fixture requires actual gfx1151");
        try { guard.run([&] { check(hipStreamCreateWithFlags(&stream, hipStreamNonBlocking), "create explicit stream"); }); }
        catch (...) { if (stream && hipStreamDestroy(stream) != hipSuccess) std::_Exit(3); throw; }
    }
    ~Owner() {
        guard.recover_or_fail_stop();
        const auto cleanup = [](hipError_t error) { if (error != hipSuccess) std::_Exit(3); };
        for (const auto execution : executions) cleanup(hipGraphExecDestroy(execution));
        for (const auto graph : graphs) cleanup(hipGraphDestroy(graph));
        for (const auto allocation : allocations) cleanup(hipFree(allocation));
        cleanup(hipStreamDestroy(stream));
    }
    void finish() {
        try { check(hipStreamSynchronize(stream), "complete owned stream"); guard.require_original(); }
        catch (...) { guard.recover_or_fail_stop(); throw; }
    }
    Span allocate(std::uint64_t bytes) {
        void* allocation = nullptr;
        guard.run([&] { check(hipMalloc(&allocation, static_cast<std::size_t>(bytes + 128U)), "allocate guarded activation view"); });
        try { allocations.push_back(allocation); }
        catch (...) { guard.recover_or_fail_stop(); (void)hipFree(allocation); throw; }
        guard.run([&] { check(hipMemsetAsync(allocation, 0xa5, static_cast<std::size_t>(bytes + 128U), stream), "poison allocation guards"); });
        return {reinterpret_cast<std::uintptr_t>(allocation) + 64U, bytes};
    }
    void retain(hipGraph_t graph) {
        try { graphs.push_back(graph); }
        catch (...) { guard.recover_or_fail_stop(); (void)hipGraphDestroy(graph); throw; }
    }
    hipGraphExec_t capture(Shape shape, Bindings bindings, Mode mode) {
        check(hipStreamBeginCapture(stream, hipStreamCaptureModeThreadLocal), "begin activation capture");
        try { enqueue(shape, bindings, mode, stream); }
        catch (...) {
            hipGraph_t abandoned = nullptr;
            if (hipStreamEndCapture(stream, &abandoned) == hipSuccess && abandoned) retain(abandoned);
            guard.recover_or_fail_stop(); throw;
        }
        hipGraph_t graph = nullptr;
        check(hipStreamEndCapture(stream, &graph), "end activation capture"); retain(graph);
        std::size_t count = 0; check(hipGraphGetNodes(graph, nullptr, &count), "count activation graph nodes");
        require(count == (mode == Mode::reference ? shape.slots : 1U), "activation launch batching was not captured");
        std::vector<hipGraphNode_t> nodes(count);
        check(hipGraphGetNodes(graph, nodes.data(), &count), "read activation graph nodes");
        for (const auto node : nodes) {
            hipGraphNodeType type;
            check(hipGraphNodeGetType(node, &type), "read activation node type");
            require(type == hipGraphNodeTypeKernel, "activation seam captured a non-kernel operation");
            hipKernelNodeParams parameters{};
            check(hipGraphKernelNodeGetParams(node, &parameters), "read activation kernel geometry");
            require(parameters.gridDim.x == (shape.intermediate + 255U) / 256U &&
                    parameters.gridDim.y == (mode == Mode::reference ? 1U : shape.slots) &&
                    parameters.gridDim.z == 1U && parameters.blockDim.x == 256U &&
                    parameters.blockDim.y == 1U && parameters.blockDim.z == 1U,
                    "captured activation kernel geometry differs from admitted layout");
        }
        hipGraphExec_t execution = nullptr;
        check(hipGraphInstantiate(&execution, graph, nullptr, nullptr, 0), "instantiate activation graph");
        try { executions.push_back(execution); }
        catch (...) { guard.recover_or_fail_stop(); (void)hipGraphExecDestroy(execution); throw; }
        return execution;
    }
};
template <typename T> void upload(Owner& owner, Span view, const std::vector<T>& values) {
    require(view.bytes == values.size() * sizeof(T), "activation fixture upload extent");
    owner.guard.run([&] { check(hipMemcpyAsync(reinterpret_cast<void*>(view.address), values.data(),
        static_cast<std::size_t>(view.bytes), hipMemcpyHostToDevice, owner.stream), "upload activation values"); });
}
std::vector<unsigned char> read(Owner& owner, Span view) {
    std::vector<unsigned char> result(static_cast<std::size_t>(view.bytes));
    owner.guard.run([&] { check(hipMemcpyAsync(result.data(), reinterpret_cast<const void*>(view.address), result.size(),
        hipMemcpyDeviceToHost, owner.stream), "read activation values"); });
    return result;
}
void guards(Owner& owner, Span view) {
    for (const auto address : {view.address - 64U, view.address + view.bytes}) {
        const auto bytes = read(owner, {address, 64U});
        require(std::all_of(bytes.begin(), bytes.end(), [](auto byte) { return byte == 0xa5; }), "activation view crossed an outer guard");
    }
}
std::vector<float> values(Shape shape, unsigned seed, unsigned flavor) {
    std::vector<float> data(static_cast<std::size_t>(shape.slots) * 2U * shape.intermediate);
    for (std::uint32_t slot = 0; slot < shape.slots; ++slot)
        for (std::uint32_t half = 0; half < 2U; ++half)
            for (std::uint32_t index = 0; index < shape.intermediate; ++index) {
                const auto mixed = (slot * 131U + half * 61U + index * 17U + seed * 29U) % 521U;
                float value = static_cast<float>(static_cast<int>(mixed) - 260) / 64.0F;
                if (flavor == 1U) {
                    constexpr std::array<std::uint32_t, 8> bits{0U, 0x80000000U, 1U, 0x80000001U,
                        0x00800000U, 0x80800000U, 0x3f800000U, 0xbf800000U};
                    value = std::bit_cast<float>(bits[mixed % bits.size()]);
                }
                if (flavor == 2U && half == 0U) value = (mixed & 1U) ? 80.0F : -80.0F;
                if (flavor == 3U && half == 0U) {
                    constexpr std::array<std::uint32_t, 6> bits{0x7fc01234U, 0xffc05678U,
                        0x7f800000U, 0xff800000U, 0U, 0x80000000U};
                    value = std::bit_cast<float>(bits[mixed % bits.size()]);
                }
                data[(static_cast<std::size_t>(slot) * 2U + half) * shape.intermediate + index] = value;
            }
    return data;
}
void compare(Owner& owner, Shape shape, Bindings reference, Bindings candidate,
             const std::vector<float>& before, bool nonfinites, bool sticky) {
    const auto a = read(owner, reference.workspace), b = read(owner, candidate.workspace);
    require(a.size() == b.size(), "activation workspace sizes differ");
    for (std::size_t index = 0; index < before.size(); ++index) {
        std::uint32_t left = 0, right = 0;
        std::memcpy(&left, a.data() + index * sizeof(float), sizeof(left));
        std::memcpy(&right, b.data() + index * sizeof(float), sizeof(right));
        const bool up = index % (2U * shape.intermediate) >= shape.intermediate;
        if (up || sticky) require(left == std::bit_cast<std::uint32_t>(before[index]) && left == right,
                                  "activation changed a read-only up plane or sticky-skipped gate");
        else if (nonfinites && std::isnan(std::bit_cast<float>(left)))
            require(std::isnan(std::bit_cast<float>(right)), "batched raw activation changed incumbent NaN propagation class");
        else require(left == right, "batched activation differs bitwise from original per-slot SiLU");
    }
    require(read(owner, reference.status) == read(owner, candidate.status), "activation changed sticky status bits");
}
void run(Shape shape, unsigned flavor, bool captured, std::uint32_t sticky) {
    Owner owner;
    const auto bytes = static_cast<std::uint64_t>(shape.slots) * 2U * shape.intermediate * sizeof(float);
    const Bindings reference{owner.allocate(bytes), owner.allocate(sizeof(std::uint32_t))};
    const Bindings candidate{owner.allocate(bytes), owner.allocate(sizeof(std::uint32_t))};
    const auto old_graph = captured ? owner.capture(shape, reference, Mode::reference) : nullptr;
    const auto new_graph = captured ? owner.capture(shape, candidate, Mode::batched) : nullptr;
    for (unsigned seed = 0; seed < 3U; ++seed) {
        const auto input = values(shape, seed, flavor);
        upload(owner, reference.workspace, input); upload(owner, candidate.workspace, input);
        upload(owner, reference.status, std::vector<std::uint32_t>{sticky});
        upload(owner, candidate.status, std::vector<std::uint32_t>{sticky});
        const auto status_before = read(owner, reference.status);
        for (unsigned repetition = 0; repetition < 3U; ++repetition) {
            if (captured) {
                check(hipGraphLaunch(old_graph, owner.stream), "replay original per-slot activation");
                check(hipGraphLaunch(new_graph, owner.stream), "replay batched activation");
            } else {
                enqueue(shape, reference, Mode::reference, owner.stream);
                enqueue(shape, candidate, Mode::batched, owner.stream);
            }
            owner.finish();
            compare(owner, shape, reference, candidate, input, flavor == 3U, sticky != 0U);
            require(read(owner, reference.status) == status_before && read(owner, candidate.status) == status_before,
                    "activation kernel wrote caller-owned status");
        }
    }
    for (const auto view : {reference.workspace, reference.status, candidate.workspace, candidate.status}) guards(owner, view);
}
void invalid_admission() {
    Owner owner; const Shape good{640U, 10U};
    const Bindings original{owner.allocate(2U * 640U * 10U * sizeof(float)), owner.allocate(sizeof(std::uint32_t))};
    const auto before = read(owner, original.workspace), status_before = read(owner, original.status);
    for (unsigned kind = 0; kind < 18U; ++kind) {
        auto shape = good; auto bindings = original; auto mode = Mode::batched; auto stream = owner.stream;
        if (kind == 0U) shape.intermediate = 0U;
        if (kind == 1U) shape.intermediate = 641U;
        if (kind == 2U) shape.slots = 0U;
        if (kind == 3U) shape.slots = 11U;
        if (kind == 4U) mode = static_cast<Mode>(99);
        if (kind == 5U) --bindings.workspace.bytes;
        if (kind == 6U) ++bindings.workspace.bytes;
        if (kind == 7U) ++bindings.workspace.address;
        if (kind == 8U) bindings.workspace.address = 0U;
        if (kind == 9U) bindings.status.address = bindings.workspace.address;
        if (kind == 10U) bindings.status.address += 1U;
        if (kind == 11U) bindings.status.bytes = 8U;
        if (kind == 12U) bindings.status.address = 0U;
        if (kind == 13U) bindings.workspace.address = std::numeric_limits<std::uintptr_t>::max() - 3U;
        if (kind == 14U) stream = nullptr;
        if (kind == 15U) stream = hipStreamPerThread;
        if (kind == 16U) stream = hipStreamLegacy;
        if (kind == 17U) bindings.status.address = bindings.workspace.address + 640U * sizeof(float);
        check(hipStreamBeginCapture(owner.stream, hipStreamCaptureModeThreadLocal), "begin rejected activation capture");
        bool rejected = false; hipGraph_t graph = nullptr;
        try { enqueue(shape, bindings, mode, stream); }
        catch (const std::invalid_argument&) { rejected = true; }
        catch (...) { if (hipStreamEndCapture(owner.stream, &graph) == hipSuccess && graph) owner.retain(graph); throw; }
        check(hipStreamEndCapture(owner.stream, &graph), "end rejected activation capture"); owner.retain(graph);
        std::size_t nodes = 0; check(hipGraphGetNodes(graph, nullptr, &nodes), "read rejected graph nodes");
        require(rejected && nodes == 0U, "invalid activation admission submitted work");
        require(read(owner, original.workspace) == before && read(owner, original.status) == status_before,
                "invalid activation admission changed a view");
    }
    guards(owner, original.workspace); guards(owner, original.status);
}
} // namespace
int main() try {
    int devices = 0;
    if (hipGetDeviceCount(&devices) != hipSuccess || !devices) return 77;
    invalid_admission();
    require(kernel_symbol(Mode::reference) != kernel_symbol(Mode::batched), "activation diagnostic did not expose distinct actual kernels");
    for (const bool captured : {false, true}) {
        for (const auto intermediate : {1U, 31U, 32U, 33U, 255U, 256U, 257U, 639U, 640U})
            for (const auto slots : {1U, 3U, 10U}) run({intermediate, slots}, 0U, captured, 0U);
        for (const auto flavor : {1U, 2U, 3U}) run({640U, 10U}, flavor, captured, 0U);
        for (const auto sticky : {1U, 2U, 4U, 8U, 0x80000000U, 0xffffffffU}) run({257U, 10U}, 3U, captured, sticky);
    }
    std::cout << "batched routed activation diagnostic fixtures completed; native model gates remain separate\n";
    return 0;
} catch (const std::exception& error) { std::cerr << error.what() << '\n'; return 1; }
