#include "strix/gdn_recurrent_layout_diagnostic.hpp"
#include "strix/hip_setup_quiescence.hpp"

#include <hip/hip_runtime.h>

#include <algorithm>
#include <array>
#include <bit>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <iostream>
#include <limits>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace {
using namespace strix::gdn_recurrent_layout_diagnostic;
constexpr std::array layouts{Layout::reference, Layout::stride32, Layout::stride33, Layout::transposed};
void check(hipError_t result, const char* operation) {
    if (result != hipSuccess) throw std::runtime_error(std::string(operation) + ": " + hipGetErrorString(result));
}
void require(bool condition, const char* message) { if (!condition) throw std::runtime_error(message); }
struct Owner {
    strix::HipSetupQuiescenceGuard quiescence;
    hipStream_t stream{};
    std::vector<void*> allocations;
    std::vector<hipGraph_t> graphs;
    std::vector<hipGraphExec_t> executions;
    Owner() {
        require(quiescence.architecture().starts_with("gfx1151"), "native diagnostic requires actual gfx1151");
        try {
            quiescence.run([&] { check(hipStreamCreateWithFlags(&stream, hipStreamNonBlocking), "create owned stream"); });
        } catch (...) {
            if (stream && hipStreamDestroy(stream) != hipSuccess) std::_Exit(3);
            throw;
        }
    }
    ~Owner() {
        quiescence.recover_or_fail_stop();
        const auto cleanup = [](hipError_t result) { if (result != hipSuccess) std::_Exit(3); };
        for (auto execution : executions) cleanup(hipGraphExecDestroy(execution));
        for (auto graph : graphs) cleanup(hipGraphDestroy(graph));
        for (auto allocation : allocations) cleanup(hipFree(allocation));
        cleanup(hipStreamDestroy(stream));
    }
    Span allocate(std::uint64_t bytes) {
        void* pointer = nullptr;
        check(hipMalloc(&pointer, static_cast<std::size_t>(bytes + 128U)), "allocate guarded view");
        try { allocations.push_back(pointer); }
        catch (...) { quiescence.recover_or_fail_stop(); (void)hipFree(pointer); throw; }
        check(hipMemsetAsync(pointer, 0xa5, static_cast<std::size_t>(bytes + 128U), stream), "initialize guards");
        return {reinterpret_cast<std::uintptr_t>(pointer) + 64U, bytes};
    }
    void retain(hipGraph_t graph) {
        try { graphs.push_back(graph); }
        catch (...) { quiescence.recover_or_fail_stop(); (void)hipGraphDestroy(graph); throw; }
    }
    void retain(hipGraphExec_t execution) {
        try { executions.push_back(execution); }
        catch (...) { quiescence.recover_or_fail_stop(); (void)hipGraphExecDestroy(execution); throw; }
    }
    void finish() { check(hipStreamSynchronize(stream), "finish owned stream"); quiescence.require_original(); }
};
template <typename T> void upload(Owner& owner, Span view, const std::vector<T>& values) {
    require(view.bytes == values.size() * sizeof(T), "fixture upload extent differs");
    owner.quiescence.run([&] {
        check(hipMemcpyAsync(reinterpret_cast<void*>(view.address), values.data(), static_cast<std::size_t>(view.bytes),
                             hipMemcpyHostToDevice, owner.stream), "upload fixture");
    });
}
std::vector<unsigned char> download(Owner& owner, Span view) {
    std::vector<unsigned char> bytes(static_cast<std::size_t>(view.bytes));
    owner.quiescence.run([&] {
        check(hipMemcpyAsync(bytes.data(), reinterpret_cast<const void*>(view.address), bytes.size(),
                             hipMemcpyDeviceToHost, owner.stream), "download view");
    });
    return bytes;
}
void poison(Owner& owner, Span view, unsigned value) {
    check(hipMemsetAsync(reinterpret_cast<void*>(view.address), static_cast<int>(value), static_cast<std::size_t>(view.bytes),
                        owner.stream), "poison core");
}
void guards(Owner& owner, Span view) {
    if (!view.address) return;
    std::array<unsigned char, 64> bytes{};
    for (const auto address : {view.address - 64U, view.address + view.bytes}) {
        owner.quiescence.run([&] {
            check(hipMemcpyAsync(bytes.data(), reinterpret_cast<const void*>(address), bytes.size(),
                                 hipMemcpyDeviceToHost, owner.stream), "download guard");
        });
        require(std::all_of(bytes.begin(), bytes.end(), [](auto value) { return value == 0xa5; }), "view canary changed");
    }
}
Bindings allocate(Owner& owner, Shape shape, Span qkv, Span beta, Span gate, Span status) {
    return {qkv, beta, gate,
            owner.allocate(static_cast<std::uint64_t>(shape.value_heads) * 128U * 128U * sizeof(float)),
            owner.allocate(static_cast<std::uint64_t>(shape.value_heads) * 128U * sizeof(float)), status};
}
std::vector<float> state_values(Shape shape, unsigned seed, unsigned flavor) {
    std::vector<float> state(static_cast<std::size_t>(shape.value_heads) * 128U * 128U);
    for (unsigned head = 0; head < shape.value_heads; ++head)
        for (unsigned key = 0; key < 128U; ++key) for (unsigned value = 0; value < 128U; ++value) {
            const auto index = (static_cast<std::size_t>(head) * 128U + key) * 128U + value;
            const auto mixed = (head * 131U + key * 17U + value * 11U + (key ^ value) * 7U + seed * 29U) % 521U;
            state[index] = static_cast<float>(static_cast<int>(mixed) - 260) / 65536.0F;
            if (flavor == 1U) {
                constexpr std::uint32_t bits[]{0U, 0x80000000U, 1U, 0x80000001U, 0x00800000U, 0x80800000U};
                state[index] = std::bit_cast<float>(bits[mixed % 6U]);
            }
            if (flavor == 3U) state[index] = (mixed & 1U) ? 0x1.0p100F : -0x1.0p100F;
            if (flavor == 2U && (key == 0U || key == 64U) && value % 32U == 3U)
                state[index] = std::bit_cast<float>(key == 0U ? 0x7fc01234U : 0xffc05678U);
        }
    return state;
}
void inputs(Shape shape, unsigned flavor, std::vector<float>& qkv,
            std::vector<float>& beta, std::vector<float>& gate) {
    qkv.resize((static_cast<std::size_t>(shape.key_heads) * 2U + shape.value_heads) * 128U);
    beta.resize(shape.value_heads); gate.resize(shape.value_heads);
    for (std::size_t i = 0; i < qkv.size(); ++i)
        qkv[i] = static_cast<float>(static_cast<int>((i * 43U + (i / 128U) * 17U) % 137U) - 68) / 4096.0F;
    for (unsigned head = 0; head < shape.value_heads; ++head) {
        beta[head] = static_cast<float>(head % 7U + 1U) / 16.0F;
        gate[head] = -static_cast<float>(head % 3U + 1U) / 8.0F;
    }
    if (flavor == 2U) {
        for (unsigned head = 0; head < shape.key_heads; ++head) {
            const auto q = static_cast<std::size_t>(head) * 128U;
            const auto k = static_cast<std::size_t>(shape.key_heads) * 128U + q;
            qkv[q + 7U] = std::bit_cast<float>(0x7fc04567U);
            qkv[q + 71U] = std::bit_cast<float>(0xffc07654U);
            qkv[k + 11U] = std::numeric_limits<float>::infinity();
            qkv[k + 75U] = -std::numeric_limits<float>::infinity();
        }
    }
}
void equal(Owner& owner, const Bindings& candidate, const Bindings& reference, Shape shape, Layout layout, const char* phase) {
    for (const auto pair : {std::pair{candidate.state, reference.state}, std::pair{candidate.core, reference.core}}) {
        const auto actual = download(owner, pair.first), expected = download(owner, pair.second);
        if (actual != expected) {
            std::size_t first = 0; while (first < actual.size() && actual[first] == expected[first]) ++first;
            std::cerr << "recurrent layout=" << static_cast<int>(layout) << " keys=" << shape.key_heads
                      << " values=" << shape.value_heads << " phase=" << phase << " first_byte=" << first << '\n';
            throw std::runtime_error("full recurrent output/state differs from incumbent bits");
        }
    }
}
hipGraphExec_t capture(Owner& owner, Shape shape, const Bindings& bindings, Layout layout) {
    hipGraph_t graph{};
    check(hipStreamBeginCapture(owner.stream, hipStreamCaptureModeGlobal), "begin recurrence capture");
    try { enqueue(shape, bindings, layout, owner.stream); }
    catch (...) { (void)hipStreamEndCapture(owner.stream, &graph); if (graph) owner.retain(graph); throw; }
    check(hipStreamEndCapture(owner.stream, &graph), "end recurrence capture"); owner.retain(graph);
    std::size_t nodes = 0; check(hipGraphGetNodes(graph, nullptr, &nodes), "count recurrence nodes");
    require(nodes == 1U, "recurrence capture must contain exactly one kernel");
    hipGraphNode_t node{}; check(hipGraphGetNodes(graph, &node, &nodes), "get recurrence node");
    hipGraphNodeType type{}; check(hipGraphNodeGetType(node, &type), "get recurrence node type");
    require(type == hipGraphNodeTypeKernel, "recurrence graph contains a nonkernel node");
    hipKernelNodeParams parameters{}; check(hipGraphKernelNodeGetParams(node, &parameters), "get recurrence node parameters");
    require(parameters.func == kernel_symbol(layout) && parameters.gridDim.x == shape.value_heads * 4U &&
            parameters.gridDim.y == 1U && parameters.gridDim.z == 1U && parameters.blockDim.x == 128U &&
            parameters.blockDim.y == 1U && parameters.blockDim.z == 1U && parameters.sharedMemBytes == 0U,
            "captured recurrence identity or geometry differs");
    hipGraphExec_t execution{}; check(hipGraphInstantiate(&execution, graph, nullptr, nullptr, 0), "instantiate recurrence");
    owner.retain(execution); return execution;
}
void family(Shape shape, unsigned flavor, bool with_status) {
    Owner owner;
    const auto qkv = owner.allocate((2ULL * shape.key_heads + shape.value_heads) * 128U * sizeof(float));
    const auto beta = owner.allocate(static_cast<std::uint64_t>(shape.value_heads) * sizeof(float));
    const auto gate = owner.allocate(static_cast<std::uint64_t>(shape.value_heads) * sizeof(float));
    const auto status = with_status ? owner.allocate(sizeof(std::uint32_t)) : Span{};
    auto reference = allocate(owner, shape, qkv, beta, gate, status);
    auto candidate = allocate(owner, shape, qkv, beta, gate, status);
    std::vector<float> x, b, g; inputs(shape, flavor, x, b, g);
    upload(owner, qkv, x); upload(owner, beta, b); upload(owner, gate, g);
    if (with_status) upload(owner, status, std::vector<std::uint32_t>{0U});
    const auto original_x = download(owner, qkv), original_b = download(owner, beta), original_g = download(owner, gate);
    const auto reference_graph = capture(owner, shape, reference, Layout::reference);
    for (const auto layout : {Layout::stride32, Layout::stride33, Layout::transposed}) {
        require(kernel_symbol(layout) != kernel_symbol(Layout::reference), "candidate aliases incumbent symbol");
        upload(owner, reference.state, state_values(shape, 0U, flavor)); upload(owner, candidate.state, state_values(shape, 0U, flavor));
        const auto before = download(owner, reference.state);
        for (unsigned step = 0; step < 3U; ++step) {
            poison(owner, reference.core, 0x31U + step); poison(owner, candidate.core, 0xd1U + step);
            enqueue(shape, reference, Layout::reference, owner.stream);
            enqueue(shape, candidate, layout, owner.stream); owner.finish();
            equal(owner, candidate, reference, shape, layout, "eager evolving state");
        }
        if (flavor == 0U) require(download(owner, reference.state) != before, "nonzero state fixture did not evolve");
        const auto graph = capture(owner, shape, candidate, layout);
        for (unsigned replay = 0; replay < 3U; ++replay) {
            // A different nonzero state each time exposes retained LDS/stale
            // graph data; the subsequent replay also checks resumed evolution.
            upload(owner, reference.state, state_values(shape, replay + 1U, flavor));
            upload(owner, candidate.state, state_values(shape, replay + 1U, flavor));
            for (unsigned resumed = 0; resumed < 2U; ++resumed) {
                poison(owner, reference.core, 0x41U + replay); poison(owner, candidate.core, 0xe1U + replay);
                check(hipGraphLaunch(reference_graph, owner.stream), "reference graph replay");
                check(hipGraphLaunch(graph, owner.stream), "layout graph replay"); owner.finish();
                equal(owner, candidate, reference, shape, layout, "captured reset/resumed state");
            }
        }
        guards(owner, reference.state); guards(owner, reference.core); guards(owner, candidate.state); guards(owner, candidate.core);
    }
    require(download(owner, qkv) == original_x && download(owner, beta) == original_b && download(owner, gate) == original_g,
            "read-only recurrence inputs changed");
    if (with_status) {
        std::uint32_t bits = 1U;
        owner.quiescence.run([&] {
            check(hipMemcpyAsync(&bits, reinterpret_cast<const void*>(status.address), sizeof(bits),
                                 hipMemcpyDeviceToHost, owner.stream), "status read");
        });
        require(bits == 0U, "raw recurrence unexpectedly published finite validation status");
    }
    for (const auto view : {qkv, beta, gate, status}) guards(owner, view);
}
void sticky_rejection() {
    Owner owner; const Shape shape{2U, 6U, 128U};
    const auto qkv = owner.allocate((2ULL * shape.key_heads + shape.value_heads) * 128U * 4U);
    const auto beta = owner.allocate(shape.value_heads * 4U), gate = owner.allocate(shape.value_heads * 4U);
    const auto status = owner.allocate(4U);
    auto bindings = allocate(owner, shape, qkv, beta, gate, status);
    std::vector<float> x, b, g; inputs(shape, 2U, x, b, g);
    upload(owner, qkv, x); upload(owner, beta, b); upload(owner, gate, g); upload(owner, status, std::vector<std::uint32_t>{8U});
    for (const auto layout : layouts) {
        upload(owner, bindings.state, state_values(shape, 5U, 0U)); poison(owner, bindings.core, 0xd3U);
        const auto state = download(owner, bindings.state), core = download(owner, bindings.core), sticky = download(owner, status);
        const auto graph = capture(owner, shape, bindings, layout);
        for (unsigned repetition = 0; repetition < 3U; ++repetition) {
            enqueue(shape, bindings, layout, owner.stream);
            check(hipGraphLaunch(graph, owner.stream), "sticky reject graph replay"); owner.finish();
            require(download(owner, bindings.state) == state && download(owner, bindings.core) == core && download(owner, status) == sticky,
                    "sticky rejection changed state, output or error bits");
        }
    }
    for (const auto view : {qkv, beta, gate, status, bindings.state, bindings.core}) guards(owner, view);
}
void invalid_preflight() {
    Owner owner; const Shape good{1U, 1U, 128U};
    const auto qkv = owner.allocate(3U * 128U * 4U), beta = owner.allocate(4U), gate = owner.allocate(4U);
    const auto status = owner.allocate(4U);
    const auto original = allocate(owner, good, qkv, beta, gate, status);
    const auto state = download(owner, original.state), core = download(owner, original.core);
    for (const auto valid_layout : layouts) for (unsigned mode = 0; mode < 26U; ++mode) {
        auto shape = good; auto bindings = original; auto layout = valid_layout; auto stream = owner.stream;
        if (mode == 0U) shape.key_heads = 0U;
        if (mode == 1U) shape.key_heads = 2U;
        if (mode == 2U) shape.head_dim = 64U;
        if (mode == 3U) shape.head_dim = 256U;
        if (mode == 4U) shape.value_heads = 65U;
        if (mode == 5U) layout = static_cast<Layout>(99);
        if (mode == 6U) --bindings.qkv.bytes;
        if (mode == 7U) ++bindings.gate.address;
        if (mode == 8U) bindings.core.address = bindings.state.address;
        if (mode == 9U) bindings.state.address = bindings.qkv.address;
        if (mode == 10U) bindings.state.address = std::numeric_limits<std::uintptr_t>::max() - 3U;
        if (mode == 11U) bindings.graph_status.address = 0U;
        if (mode == 12U) bindings.graph_status.bytes = 8U;
        if (mode == 13U) bindings.graph_status.address = bindings.state.address;
        if (mode == 14U) bindings.core.address = 0U;
        if (mode == 15U) stream = nullptr;
        if (mode == 16U) stream = hipStreamPerThread;
        if (mode == 17U) stream = hipStreamLegacy;
        if (mode == 18U) shape.key_heads = 65U;
        if (mode == 19U) shape.value_heads = 0U;
        if (mode == 20U) ++bindings.graph_status.address;
        if (mode == 21U) ++bindings.state.bytes;
        if (mode == 22U) --bindings.core.bytes;
        if (mode == 23U) bindings.beta.bytes = 0U;
        if (mode == 24U) bindings.graph_status.address = bindings.qkv.address;
        if (mode == 25U) bindings.core.address = bindings.gate.address;
        hipGraph_t graph{}; check(hipStreamBeginCapture(owner.stream, hipStreamCaptureModeGlobal), "begin rejection capture");
        bool rejected = false;
        try { enqueue(shape, bindings, layout, stream); }
        catch (const std::invalid_argument&) { rejected = true; }
        catch (...) { (void)hipStreamEndCapture(owner.stream, &graph); if (graph) owner.retain(graph); throw; }
        check(hipStreamEndCapture(owner.stream, &graph), "end rejection capture"); owner.retain(graph);
        std::size_t nodes = 0; check(hipGraphGetNodes(graph, nullptr, &nodes), "get rejection nodes");
        require(rejected && nodes == 0U, "invalid preflight submitted work"); owner.finish();
        require(download(owner, original.state) == state && download(owner, original.core) == core, "invalid preflight mutated views");
    }
    for (const auto view : {qkv, beta, gate, status, original.state, original.core}) guards(owner, view);
}
} // namespace

int main() try {
    invalid_preflight(); sticky_rejection();
    unsigned cases = 0;
    for (const auto shape : {Shape{1U, 1U, 128U}, Shape{2U, 6U, 128U}, Shape{16U, 48U, 128U}})
        for (unsigned flavor = 0; flavor < 4U; ++flavor) for (const bool status : {false, true}) {
            family(shape, flavor, status); ++cases;
        }
    std::cout << "GDN recurrent LDS layout diagnostic PASS cases=" << cases
              << " layouts=3 reference=incumbent full_state_bits=true tiles_per_head=4 timing=false\n";
    return 0;
} catch (const std::exception& error) { std::cerr << error.what() << '\n'; return 1; }
