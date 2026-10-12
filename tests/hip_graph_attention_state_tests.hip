#include "strix/hip_indexed_attention_launcher.hpp"

#include <hip/hip_runtime.h>

#include <array>
#include <bit>
#include <cmath>
#include <cstring>
#include <iostream>
#include <limits>
#include <stdexcept>
#include <string>
#include <vector>

namespace {
void check(hipError_t e, const char* operation) {
    if (e != hipSuccess) throw std::runtime_error(std::string(operation) + ": " + hipGetErrorString(e));
}
void require(bool condition, const char* message) { if (!condition) throw std::runtime_error(message); }
std::uint16_t bf16(float x) {
    auto bits = std::bit_cast<std::uint32_t>(x);
    return static_cast<std::uint16_t>((bits + 0x7fffU + ((bits >> 16U) & 1U)) >> 16U);
}
float unbf16(std::uint16_t x) { return std::bit_cast<float>(static_cast<std::uint32_t>(x) << 16U); }
bool same_bits(const std::vector<float>& a, const std::vector<float>& b) {
    if (a.size() != b.size()) return false;
    for (std::size_t i = 0; i < a.size(); ++i) {
        if (std::bit_cast<std::uint32_t>(a[i]) != std::bit_cast<std::uint32_t>(b[i])) return false;
    }
    return true;
}

struct Fixture {
    static constexpr std::uint32_t heads = 2, dim = 8, stride = 16, kv_width = 8;
    static constexpr std::uint64_t context = 4;
    hipStream_t stream = nullptr;
    hipGraph_t graph = nullptr;
    hipGraphExec_t executable = nullptr;
    std::vector<void*> allocations;
    strix::GraphAttentionStatePlan dynamic, control;

    ~Fixture() {
        if (stream != nullptr && hipStreamQuery(stream) != hipSuccess) return;
        if (executable) static_cast<void>(hipGraphExecDestroy(executable));
        if (graph) static_cast<void>(hipGraphDestroy(graph));
        if (stream) static_cast<void>(hipStreamDestroy(stream));
        for (auto* pointer : allocations) static_cast<void>(hipFree(pointer));
    }
    std::uintptr_t allocate(std::size_t bytes) {
        void* pointer = nullptr;
        check(hipMalloc(&pointer, bytes), "allocate graph attention resource");
        try { allocations.push_back(pointer); } catch (...) { static_cast<void>(hipFree(pointer)); throw; }
        return reinterpret_cast<std::uintptr_t>(pointer);
    }
    template<class T> void put(std::uintptr_t address, const std::vector<T>& values) {
        check(hipMemcpy(reinterpret_cast<void*>(address), values.data(), values.size() * sizeof(T), hipMemcpyHostToDevice), "upload graph attention input");
    }
    template<class T> std::vector<T> get(std::uintptr_t address, std::size_t count) {
        std::vector<T> result(count);
        check(hipMemcpy(result.data(), reinterpret_cast<const void*>(address), count * sizeof(T), hipMemcpyDeviceToHost), "read graph attention resource");
        return result;
    }
    strix::GraphAttentionStatePlan make_plan(bool device) {
        strix::GraphAttentionStateRequest r;
        r.execution_identity = 9; r.query_heads = heads; r.kv_heads = 1; r.head_dim = dim;
        r.query_head_stride = stride; r.rotary_dim = 4; r.epsilon = 3; r.theta = 10000;
        r.context_tokens = context;
        r.position_mode = device ? strix::AttentionPositionMode::device_input : strix::AttentionPositionMode::scalar_control;
        r.arguments = strix::bind_graph_inputs(strix::graph_input_layout(1), {9, allocate(16), 16, 1});
        r.query.address = allocate(heads * stride * sizeof(float)); r.query.bytes = heads * stride * sizeof(float);
        r.query.allocation_generation = 2; r.query.use.generation = 1;
        r.key.address = allocate(kv_width * sizeof(float)); r.key.bytes = kv_width * sizeof(float);
        r.key.allocation_generation = 3; r.key.use.generation = 1;
        r.value.address = allocate(kv_width * sizeof(float)); r.value.bytes = kv_width * sizeof(float);
        r.value.allocation_generation = 4; r.value.use.generation = 1;
        r.query_norm = {0, strix::RuntimeTensorStorage::resident, allocate(dim * sizeof(float)), dim * sizeof(float), 5, 0};
        r.key_norm = r.query_norm;
        put(r.query_norm.address, std::vector<float>(dim, 0));
        r.kv_state.address = allocate((context * kv_width * 2 + 2) * sizeof(std::uint16_t)) + sizeof(std::uint16_t);
        r.kv_state.bytes = context * kv_width * 4; r.kv_state.allocation_generation = 6;
        r.kv_state.state_flag = strix::PersistentState::attention_kv;
        r.status = {9, allocate(4), 4, 7};
        put(r.kv_state.address - sizeof(std::uint16_t), std::vector<std::uint16_t>(context * kv_width * 2 + 2, 0xa55a));
        return strix::prepare_graph_attention_state(r);
    }
    void inputs(const strix::GraphAttentionStatePlan& p, unsigned iteration, bool invalid = false) {
        std::vector<float> q(heads * stride, -777);
        std::vector<float> k(kv_width), v(kv_width);
        for (unsigned h = 0; h < heads; ++h) for (unsigned j = 0; j < dim; ++j) q[h * stride + j] = (j + h + iteration) % 2 == 0 ? 1.0F : -1.0F;
        for (unsigned j = 0; j < kv_width; ++j) { k[j] = (j + iteration) % 2 == 0 ? 1.0F : -1.0F; v[j] = static_cast<float>(j + 1) * -0.125F; }
        if (invalid) v[0] = std::numeric_limits<float>::infinity();
        put(p.request.query.address, q); put(p.request.key.address, k); put(p.request.value.address, v);
    }
    void position(std::uint64_t value) {
        std::vector<std::byte> packet(16);
        std::memcpy(packet.data() + 8, &value, sizeof(value));
        put(dynamic.request.arguments.memory.arena_base, packet);
    }
    std::vector<std::uint16_t> state(const strix::GraphAttentionStatePlan& p) {
        return get<std::uint16_t>(p.request.kv_state.address - sizeof(std::uint16_t), context * kv_width * 2 + 2);
    }
    void replay() {
        check(hipGraphLaunch(executable, stream), "attention graph replay");
        check(hipStreamSynchronize(stream), "attention graph completion");
    }
    void run() {
        check(hipStreamCreateWithFlags(&stream, hipStreamNonBlocking), "create attention stream");
        dynamic = make_plan(true); control = make_plan(false);
        check(hipStreamBeginCapture(stream, hipStreamCaptureModeThreadLocal), "begin attention capture");
        try { strix::launch_hip_graph_attention_state(dynamic, 0, stream); }
        catch (...) { static_cast<void>(hipStreamEndCapture(stream, &graph)); throw; }
        check(hipStreamEndCapture(stream, &graph), "end attention capture");
        check(hipGraphInstantiate(&executable, graph, nullptr, nullptr, 0), "instantiate attention graph");
        unsigned iteration = 0;
        for (const std::uint64_t p : {0ULL, 1ULL, 3ULL}) {
            inputs(dynamic, iteration); inputs(control, iteration); position(p);
            replay();
            strix::launch_hip_graph_attention_state(control, p, stream);
            check(hipStreamSynchronize(stream), "scalar attention control completion");
            require(get<std::uint32_t>(dynamic.request.status.arena_base, 1)[0] == 0, "valid graph status");
            require(state(dynamic) == state(control), "dynamic KV state bits equal scalar-control bits");
            const auto q = get<float>(dynamic.request.query.address, heads * stride);
            require(same_bits(q, get<float>(control.request.query.address, heads * stride)), "dynamic query bits equal scalar control");
            for (unsigned h = 0; h < heads; ++h) for (unsigned j = dim; j < stride; ++j) require(q[h * stride + j] == -777, "query gate padding untouched");
            const auto current = state(dynamic);
            require(current.front() == 0xa55a && current.back() == 0xa55a, "KV guards untouched");
            for (unsigned j = 0; j < kv_width; ++j) {
                float expected = (j + iteration) % 2 == 0 ? 0.5F : -0.5F;
                if (j < 4) {
                    const auto pair = j % 2U;
                    const float angle = static_cast<float>(p) * std::pow(10000.0F, -static_cast<float>(pair) / 2.0F);
                    const float other = (j + 2U + iteration) % 2 == 0 ? 0.5F : -0.5F;
                    expected = j < 2 ? expected * std::cos(angle) - other * std::sin(angle) : expected * std::cos(angle) + other * std::sin(angle);
                }
                const auto actual = unbf16(current[1 + p * kv_width * 2 + j]);
                require(std::isfinite(actual) && std::fabs(actual - expected) < 0.005F, "independent RMS/RoPE state reference");
                require(current[1 + p * kv_width * 2 + kv_width + j] == bf16(static_cast<float>(j + 1) * -0.125F), "independent V state reference");
            }
            ++iteration;
        }
        const auto preserved = state(dynamic);
        for (auto p : {context, std::uint64_t{1} << 32, std::numeric_limits<std::uint64_t>::max()}) {
            inputs(dynamic, 0); position(p); replay();
            require(get<std::uint32_t>(dynamic.request.status.arena_base, 1)[0] == 2, "invalid dynamic position rejected");
            require(state(dynamic) == preserved, "invalid position cannot mutate persistent KV");
        }
        inputs(dynamic, 0, true); position(0); replay();
        require(get<std::uint32_t>(dynamic.request.status.arena_base, 1)[0] == 4, "nonfinite input rejected before state write");
        require(state(dynamic) == preserved, "nonfinite source preserves KV");
        inputs(dynamic, 0);
        auto outlier = get<float>(dynamic.request.query.address, heads * stride);
        outlier[0] = 16.0F;
        put(dynamic.request.query.address, outlier);
        put(dynamic.request.query_norm.address, std::vector<float>(dim, std::numeric_limits<float>::max())); replay();
        require((get<std::uint32_t>(dynamic.request.status.arena_base, 1)[0] & 8U) != 0, "arithmetic overflow prevents state write");
        require(state(dynamic) == preserved, "failed normalization preserves KV");
        // Finite FP32 RoPE can still overflow when rounded to BF16. The K
        // source normalizes exactly to +/-1 with eps3 and two +/-2 entries.
        inputs(dynamic, 0); position(1);
        std::vector<float> large_norm(dim, 0), sparse_key(dim, 0);
        large_norm[0] = unbf16(0x7f00); large_norm[2] = unbf16(0x7f5e);
        sparse_key[0] = 2; sparse_key[2] = -2;
        const float candidate = large_norm[0] * std::cos(1.0F) + large_norm[2] * std::sin(1.0F);
        require(std::isfinite(candidate) && !std::isfinite(unbf16(bf16(candidate))),
            "BF16 rounding overflow fixture must isolate finite FP32 arithmetic");
        put(dynamic.request.query_norm.address, large_norm); put(dynamic.request.key.address, sparse_key); replay();
        require((get<std::uint32_t>(dynamic.request.status.arena_base, 1)[0] & 8U) != 0,
            "finite FP32 RoPE with nonfinite BF16 result rejected");
        require(state(dynamic) == preserved, "BF16 rounding failure cannot commit persistent KV");
        put(dynamic.request.query_norm.address, std::vector<float>(dim, 0)); inputs(dynamic, 0); position(0);
        auto overflow_key = get<float>(dynamic.request.key.address, kv_width);
        overflow_key[0] = std::numeric_limits<float>::max();
        put(dynamic.request.key.address, overflow_key); replay();
        require((get<std::uint32_t>(dynamic.request.status.arena_base, 1)[0] & 8U) != 0,
            "finite input with overflowing RMS reduction cannot silently normalize to zeros");
        require(state(dynamic) == preserved, "RMS reduction overflow preserves KV");
        put(dynamic.request.query_norm.address, std::vector<float>(dim, 0)); inputs(dynamic, 1); position(2); replay();
        require(get<std::uint32_t>(dynamic.request.status.arena_base, 1)[0] == 0, "later valid stage resets status");
        inputs(dynamic, 1); inputs(control, 1);
        auto padded = get<float>(dynamic.request.query.address, heads * stride);
        for (unsigned h = 0; h < heads; ++h) for (unsigned j = dim; j < stride; ++j)
            padded[h * stride + j] = std::numeric_limits<float>::quiet_NaN();
        put(dynamic.request.query.address, padded); put(control.request.query.address, padded);
        replay(); strix::launch_hip_graph_attention_state(control, 2, stream);
        check(hipStreamSynchronize(stream), "padded scalar control completion");
        require(get<std::uint32_t>(dynamic.request.status.arena_base, 1)[0] == 0,
            "non-consumed query gate padding is consistently ignored by this stage");
        require(same_bits(get<float>(dynamic.request.query.address, heads * stride),
            get<float>(control.request.query.address, heads * stride)), "padding retained bitwise in both modes");
        require(state(dynamic) == state(control), "ignored gate padding does not affect KV semantics");
        std::cout << "experimental dynamic position/RoPE/KV fixtures passed; whole-model graph unqualified\n";
    }
};
}
int main() {
    try { Fixture fixture; fixture.run(); return 0; }
    catch (const std::exception& e) { std::cerr << e.what() << '\n'; return 1; }
}
