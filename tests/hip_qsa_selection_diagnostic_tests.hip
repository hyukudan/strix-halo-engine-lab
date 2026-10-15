#include "strix/hip_qsa_selection_diagnostic.hpp"
#include "strix/hip_setup_quiescence.hpp"
#include <hip/hip_runtime.h>
#include <algorithm>
#include <array>
#include <bit>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <functional>
#include <exception>
#include <iostream>
#include <numeric>
#include <stdexcept>
#include <string>
#include <vector>

namespace {
using namespace strix::qsa_selection_diagnostic;
void require(bool ok, const char* text) { if (!ok) throw std::runtime_error(text); }
void check(hipError_t e, const char* text) {
    if (e != hipSuccess) throw std::runtime_error(std::string(text) + ": " + hipGetErrorString(e));
}
struct Owner {
    strix::HipSetupQuiescenceGuard guard;
    hipStream_t stream = nullptr;
    std::vector<void*> allocations;
    std::vector<hipGraph_t> graphs;
    std::vector<hipGraphExec_t> executions;
    Owner() { guard.run([&] { check(hipStreamCreateWithFlags(&stream, hipStreamNonBlocking), "fixture stream"); }); }
    ~Owner() {
        guard.recover_or_fail_stop();
        const auto cleanup = [](hipError_t e) { if (e != hipSuccess) std::_Exit(3); };
        for (auto x : executions) cleanup(hipGraphExecDestroy(x));
        for (auto x : graphs) cleanup(hipGraphDestroy(x));
        for (auto x : allocations) cleanup(hipFree(x));
        if (stream) cleanup(hipStreamDestroy(stream));
    }
    void finish() {
        try { check(hipStreamSynchronize(stream), "fixture stream completion"); guard.require_original(); }
        catch (...) { guard.recover_or_fail_stop(); throw; }
    }
    hipGraphExec_t capture(const std::function<void()>& action) {
        check(hipStreamBeginCapture(stream, hipStreamCaptureModeThreadLocal), "fixture capture begin");
        try { action(); }
        catch (...) {
            hipGraph_t abandoned = nullptr;
            if (hipStreamEndCapture(stream, &abandoned) == hipSuccess && abandoned)
                check(hipGraphDestroy(abandoned), "destroy failed capture");
            throw;
        }
        hipGraph_t graph = nullptr;
        check(hipStreamEndCapture(stream, &graph), "fixture capture end");
        graphs.push_back(graph);
        hipGraphExec_t execution = nullptr;
        check(hipGraphInstantiate(&execution, graph, nullptr, nullptr, 0), "fixture graph instantiate");
        executions.push_back(execution);
        return execution;
    }
};
template<class T> struct Buffer {
    Owner& owner;
    void* base = nullptr;
    View<T> view;
    explicit Buffer(Owner& o, std::uint64_t count) : owner(o) {
        view.count = count;
        if (!count) return;
        const auto bytes = static_cast<std::size_t>(count * sizeof(T) + 128U);
        check(hipMalloc(&base, bytes), "fixture guarded buffer");
        try { owner.allocations.push_back(base); }
        catch (...) { owner.guard.recover_or_fail_stop(); (void)hipFree(base); throw; }
        view.data = reinterpret_cast<T*>(reinterpret_cast<std::uintptr_t>(base) + 64U);
        check(hipMemsetAsync(base, 0xa5, bytes, owner.stream), "fixture poison guarded buffer");
    }
    void upload(const std::vector<T>& values) {
        require(values.size() == view.count, "fixture upload extent");
        if (!values.empty()) owner.guard.run([&] {
            owner.finish();
            check(hipMemcpy(view.data, values.data(), values.size() * sizeof(T), hipMemcpyHostToDevice),
                  "fixture producer input upload");
        });
    }
    std::vector<unsigned char> read_all() const {
        if (!view.count) return {};
        std::vector<unsigned char> bytes(static_cast<std::size_t>(view.count * sizeof(T) + 128U));
        owner.guard.run([&] {
            check(hipMemcpyAsync(bytes.data(), base, bytes.size(), hipMemcpyDeviceToHost, owner.stream), "fixture guarded readback");
            owner.finish();
        });
        return bytes;
    }
    std::vector<T> read() const {
        auto bytes = read_all();
        std::vector<T> values(static_cast<std::size_t>(view.count));
        if (!values.empty()) std::memcpy(values.data(), bytes.data() + 64U, values.size() * sizeof(T));
        return values;
    }
    void guards() const {
        const auto bytes = read_all();
        if (bytes.empty()) return;
        for (std::size_t i = 0; i < 64; ++i)
            require(bytes[i] == 0xa5 && bytes[bytes.size() - 1U - i] == 0xa5, "fixture guard damaged");
    }
};
std::vector<std::uint32_t> oracle(Shape shape, const std::vector<float>& scores) {
    std::vector<std::uint32_t> blocks(shape.blocks);
    std::iota(blocks.begin(), blocks.end(), 0U);
    if (shape.blocks > block_topk) {
        require(scores.size() == shape.blocks, "oracle sparse input size");
        for (const auto score : scores) require(std::isfinite(score), "oracle finite scores");
        std::sort(blocks.begin(), blocks.end(), [&](auto a, auto b) {
            return scores[a] > scores[b] || (scores[a] == scores[b] && a < b);
        });
        blocks.resize(block_topk);
        std::sort(blocks.begin(), blocks.end());
    }
    std::vector<std::uint32_t> tokens;
    for (const auto block : blocks)
        for (std::uint32_t offset = 0; offset < compression; ++offset) tokens.push_back(block * compression + offset);
    for (std::uint32_t i = 0; i < shape.tail; ++i) tokens.push_back(shape.blocks * compression + i);
    return tokens;
}
std::vector<float> pattern(std::uint32_t blocks, unsigned kind) {
    std::vector<float> scores(blocks);
    for (std::uint32_t i = 0; i < blocks; ++i) {
        scores[i] = static_cast<float>((static_cast<std::uint64_t>(i) * 15485863U + 179U) % 65521U) - 32760.0F;
        if (kind == 1) scores[i] = -3.0F;
        if (kind == 2) scores[i] = i % 2U ? -0.0F : 0.0F;
        if (kind == 3) scores[i] = -static_cast<float>(i % 17U) - 0.25F;
        if (kind == 4) scores[i] = std::bit_cast<float>(i % 2U ? 0x00000001U : 0x80000001U);
    }
    return scores;
}
void positive(Shape shape, bool graph_mode) {
    Owner owner;
    Buffer<float> scores(owner, shape.blocks);
    Buffer<std::uint32_t> selected(owner, selected_count(shape));
    Buffer<Item> ping(owner, scratch_items(shape)), pong(owner, scratch_items(shape));
    Buffer<std::uint32_t> status(owner, 1);
    const auto invoke = [&] {
        enqueue(shape, {scores.view.data, scores.view.count}, selected.view, ping.view, pong.view, status.view, owner.stream);
    };
    // Buffers/stream and any queued construction fills outlive the capture.
    owner.finish();
    hipGraphExec_t graph = nullptr;
    if (graph_mode) graph = owner.capture(invoke);
    for (unsigned replay = 0; replay < 5; ++replay) {
        auto input = pattern(shape.blocks, replay);
        scores.upload(input); status.upload({0});
        const auto expected = oracle(shape, input);
        const auto before = scores.read_all();
        if (selected.view.count)
            check(hipMemsetAsync(selected.view.data, 0xcd, static_cast<std::size_t>(selected.view.count * sizeof(std::uint32_t)), owner.stream), "poison output replay");
        if (graph_mode) check(hipGraphLaunch(graph, owner.stream), "fixture graph replay");
        else invoke();
        owner.finish();
        require(status.read()[0] == 0, "finite selection rejected");
        require(selected.read() == expected, "complete chronological token array differs from oracle");
        require(scores.read_all() == before, "score input mutated");
        scores.guards(); selected.guards(); ping.guards(); pong.guards(); status.guards();
    }
}
void blocked_output(Shape shape, bool graph_mode, bool invalid_scores) {
    Owner owner;
    Buffer<float> scores(owner, shape.blocks);
    Buffer<std::uint32_t> selected(owner, selected_count(shape));
    Buffer<Item> ping(owner, scratch_items(shape)), pong(owner, scratch_items(shape));
    Buffer<std::uint32_t> status(owner, 1);
    owner.finish();
    const auto invoke = [&] { enqueue(shape, {scores.view.data, scores.view.count}, selected.view,
                                      ping.view, pong.view, status.view, owner.stream); };
    hipGraphExec_t graph = nullptr;
    if (graph_mode) graph = owner.capture(invoke);
    for (const auto bits : std::array<std::uint32_t,4>{0x7fc01234U,0xffc12345U,0x7f800000U,0xff800000U}) {
        auto input = pattern(shape.blocks, 0);
        if (invalid_scores) input.back() = std::bit_cast<float>(bits);
        scores.upload(input); status.upload({invalid_scores ? 0U : 8U}); owner.finish();
        const auto original = selected.read_all(), original_ping = ping.read_all(), original_pong = pong.read_all();
        const auto readonly = scores.read_all();
        if (graph_mode) check(hipGraphLaunch(graph, owner.stream), "blocked graph replay"); else invoke();
        owner.finish();
        const auto observed = status.read()[0];
        require(invalid_scores ? (observed & nonfinite_score) != 0 : observed == 8U, "sticky/nonfinite status mismatch");
        require(selected.read_all() == original && ping.read_all() == original_ping && pong.read_all() == original_pong,
                "rejected input wrote selected output or scratch");
        require(scores.read_all() == readonly, "rejected scores mutated");
        selected.guards(); ping.guards(); pong.guards(); status.guards();
    }
}
void direct_without_scores() {
    Owner owner;
    // B=512 plus tail must also use direct, matching the incumbent's B<=K branch.
    const Shape shape{512,3};
    Buffer<std::uint32_t> output(owner, selected_count(shape)), status(owner, 1);
    status.upload({0});
    enqueue(shape, {}, output.view, {}, {}, status.view, owner.stream);
    owner.finish(); require(output.read() == oracle(shape, {}), "direct path needs unnecessary scores");
    output.guards(); status.guards();
    const Shape empty{};
    enqueue(empty, {}, {}, {}, {}, status.view, owner.stream);
}
void negative_admission() {
    Owner owner;
    const Shape shape{1025,3};
    Buffer<float> scores(owner, shape.blocks);
    Buffer<std::uint32_t> output(owner, selected_count(shape)), status(owner, 1);
    Buffer<Item> ping(owner, scratch_items(shape)), pong(owner, scratch_items(shape));
    owner.finish();
    const auto reject = [&](const std::function<void()>& action) {
        check(hipStreamBeginCapture(owner.stream, hipStreamCaptureModeThreadLocal), "negative begin capture");
        bool failed = false;
        std::exception_ptr unexpected;
        try { action(); }
        catch (const std::invalid_argument&) { failed = true; }
        catch (...) { unexpected = std::current_exception(); }
        hipGraph_t graph = nullptr;
        check(hipStreamEndCapture(owner.stream, &graph), "negative end capture");
        std::size_t nodes = 99;
        check(hipGraphGetNodes(graph, nullptr, &nodes), "negative node query");
        check(hipGraphDestroy(graph), "negative graph destroy");
        if (unexpected) std::rethrow_exception(unexpected);
        require(failed && nodes == 0, "invalid admission submitted a node");
    };
    const auto run = [&](Shape s, View<const float> input, View<std::uint32_t> out,
                         View<Item> a, View<Item> b, View<std::uint32_t> error, hipStream_t stream) {
        enqueue(s, input, out, a, b, error, stream);
    };
    const View<const float> input{scores.view.data, scores.view.count};
    reject([&] { run({32768,1},input,output.view,ping.view,pong.view,status.view,owner.stream); });
    reject([&] { run({32769,0},input,output.view,ping.view,pong.view,status.view,owner.stream); });
    reject([&] { run({1025,4},input,output.view,ping.view,pong.view,status.view,owner.stream); });
    reject([&] { run(shape,{input.data,input.count-1},output.view,ping.view,pong.view,status.view,owner.stream); });
    reject([&] { run(shape,input,{output.view.data,output.view.count-1},ping.view,pong.view,status.view,owner.stream); });
    reject([&] { run(shape,input,output.view,{ping.view.data,ping.view.count-1},pong.view,status.view,owner.stream); });
    reject([&] { run(shape,input,output.view,ping.view,ping.view,status.view,owner.stream); });
    reject([&] { run(shape,input,{reinterpret_cast<std::uint32_t*>(scores.view.data),output.view.count},ping.view,pong.view,status.view,owner.stream); });
    reject([&] { run(shape,input,output.view,ping.view,pong.view,{output.view.data,1},owner.stream); });
    reject([&] { run(shape,{reinterpret_cast<const float*>(reinterpret_cast<std::uintptr_t>(input.data)+1U),input.count},output.view,ping.view,pong.view,status.view,owner.stream); });
    reject([&] { run(shape,input,output.view,{reinterpret_cast<Item*>(reinterpret_cast<std::uintptr_t>(ping.view.data)+4U),ping.view.count},pong.view,status.view,owner.stream); });
    reject([&] { run(shape,input,output.view,ping.view,pong.view,{},owner.stream); });
    reject([&] { run(shape,input,output.view,ping.view,pong.view,status.view,nullptr); });
}
} // namespace
int main() try {
    int devices = 0;
    if (hipGetDeviceCount(&devices) != hipSuccess || !devices) return 77;
    for (const auto blocks : std::array<std::uint32_t,12>{0,1,511,512,513,1023,1024,1025,2049,4097,8193,32768})
        for (std::uint32_t tail = 0; tail < (blocks == 32768 ? 1U : 4U); ++tail)
            for (const bool graph : {false,true}) positive({blocks,tail}, graph);
    for (const bool graph : {false,true}) {
        blocked_output({2049,3}, graph, false);
        blocked_output({32768,0}, graph, true);
        blocked_output({511,3}, graph, false);
    }
    direct_without_scores(); negative_admission();
    std::cout << "QSA diagnostic source fixtures completed\n";
    return 0;
} catch (const std::exception& error) {
    std::cerr << "QSA diagnostic fixture: " << error.what() << '\n'; return 1;
}
