#include "strix/hip_qsa_gqa_reuse_diagnostic.hpp"
#include "strix/hip_setup_quiescence.hpp"

#include <hip/hip_runtime.h>

#include <algorithm>
#include <array>
#include <bit>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <functional>
#include <iostream>
#include <limits>
#include <stdexcept>
#include <string>
#include <vector>

namespace {
using namespace strix::qsa_gqa_reuse_diagnostic;
constexpr std::array modes{Mode::reference, Mode::heads2, Mode::heads4};
void require(bool condition, const char* text) { if (!condition) throw std::runtime_error(text); }
void check(hipError_t error, const char* text) {
    if (error != hipSuccess) throw std::runtime_error(std::string(text) + ": " + hipGetErrorString(error));
}
template <typename Function> bool rejects(Function&& function) {
    try { function(); } catch (const std::invalid_argument&) { return true; }
    return false;
}
struct Owner {
    strix::HipSetupQuiescenceGuard guard;
    hipStream_t stream = nullptr;
    std::vector<void*> allocations;
    std::vector<hipGraph_t> graphs;
    std::vector<hipGraphExec_t> executions;
    Owner() {
        require(guard.architecture().starts_with("gfx1151"), "QSA GQA native fixture requires actual gfx1151");
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
        guard.run([&] { check(hipMalloc(&allocation, static_cast<std::size_t>(bytes + 128U)), "allocate guarded view"); });
        try { allocations.push_back(allocation); }
        catch (...) { guard.recover_or_fail_stop(); (void)hipFree(allocation); throw; }
        guard.run([&] { check(hipMemsetAsync(allocation, 0xa5, static_cast<std::size_t>(bytes + 128U), stream), "poison guards"); });
        return {reinterpret_cast<std::uintptr_t>(allocation) + 64U, bytes};
    }
    void retain(hipGraph_t graph) {
        try { graphs.push_back(graph); }
        catch (...) { guard.recover_or_fail_stop(); (void)hipGraphDestroy(graph); throw; }
    }
    hipGraphExec_t capture(const std::function<void()>& action, std::size_t expected_nodes = 5U) {
        check(hipStreamBeginCapture(stream, hipStreamCaptureModeThreadLocal), "begin stage capture");
        try { action(); }
        catch (...) {
            hipGraph_t abandoned = nullptr;
            if (hipStreamEndCapture(stream, &abandoned) == hipSuccess && abandoned) retain(abandoned);
            guard.recover_or_fail_stop(); throw;
        }
        hipGraph_t graph = nullptr;
        check(hipStreamEndCapture(stream, &graph), "end stage capture"); retain(graph);
        std::size_t nodes = 0;
        check(hipGraphGetNodes(graph, nullptr, &nodes), "read captured stage node count");
        require(nodes == expected_nodes, "captured attention stage has the wrong kernel node count");
        hipGraphExec_t execution = nullptr;
        check(hipGraphInstantiate(&execution, graph, nullptr, nullptr, 0), "instantiate stage graph");
        try { executions.push_back(execution); }
        catch (...) { guard.recover_or_fail_stop(); (void)hipGraphExecDestroy(execution); throw; }
        return execution;
    }
};
template <typename T> void upload(Owner& owner, Span view, const std::vector<T>& values) {
    require(view.bytes == values.size() * sizeof(T), "upload view extent");
    owner.guard.run([&] { check(hipMemcpyAsync(reinterpret_cast<void*>(view.address), values.data(),
        static_cast<std::size_t>(view.bytes), hipMemcpyHostToDevice, owner.stream), "upload fixture values"); });
}
std::vector<unsigned char> read(Owner& owner, Span view) {
    std::vector<unsigned char> bytes(static_cast<std::size_t>(view.bytes));
    owner.guard.run([&] { check(hipMemcpyAsync(bytes.data(), reinterpret_cast<const void*>(view.address), bytes.size(),
        hipMemcpyDeviceToHost, owner.stream), "read fixture view"); });
    return bytes;
}
void poison(Owner& owner, Span view, int pattern = 0xa5) {
    owner.guard.run([&] { check(hipMemsetAsync(reinterpret_cast<void*>(view.address), pattern,
        static_cast<std::size_t>(view.bytes), owner.stream), "poison attention scratch/output"); });
}
void canaries(Owner& owner, Span view) {
    for (const auto address : {view.address - 64U, view.address + view.bytes}) {
        const auto bytes = read(owner, {address, 64U});
        require(std::all_of(bytes.begin(), bytes.end(), [](auto value) { return value == 0xa5; }), "view outer guard changed");
    }
}
std::uint16_t bf16(float value) {
    const auto bits = std::bit_cast<std::uint32_t>(value);
    return static_cast<std::uint16_t>((bits + 0x7fffU + ((bits >> 16U) & 1U)) >> 16U);
}
float rounded(float value) { return std::bit_cast<float>(static_cast<std::uint32_t>(bf16(value)) << 16U); }
std::vector<float> queries(Shape shape, unsigned flavor) {
    std::vector<float> values(static_cast<std::size_t>(shape.query_heads) * 256U * 2U);
    for (std::uint32_t head = 0; head < shape.query_heads; ++head)
        for (std::uint32_t feature = 0; feature < 512U; ++feature) {
            const auto mixed = (head * 43U + feature * 17U + flavor * 29U) % 257U;
            auto value = rounded(static_cast<float>(static_cast<int>(mixed) - 128) / 128.0F);
            if (flavor == 1U) {
                constexpr std::array<std::uint32_t, 6> bits{0U, 0x80000000U, 0x00010000U, 0x80010000U, 0x00800000U, 0x80800000U};
                value = std::bit_cast<float>(bits[mixed % bits.size()]);
            }
            if (flavor == 2U) value = feature >= 256U ? ((mixed & 1U) ? 100.0F : -100.0F) : value * 20.0F;
            values[static_cast<std::size_t>(head) * 512U + feature] = value;
        }
    return values;
}
std::vector<std::uint16_t> cached_kv(Shape shape, unsigned flavor) {
    const auto row = shape.kv_heads * 256U * 2U;
    std::vector<std::uint16_t> values(static_cast<std::size_t>(shape.context_tokens) * row);
    for (std::uint32_t token = 0; token < shape.context_tokens; ++token)
        for (std::uint32_t feature = 0; feature < row; ++feature) {
            const auto mixed = (token * 97U + feature * 31U + flavor * 53U + (token ^ feature)) % 509U;
            auto value = bf16(static_cast<float>(static_cast<int>(mixed) - 254) / 256.0F);
            if (flavor == 1U) {
                constexpr std::array<std::uint16_t, 6> bits{0U, 0x8000U, 1U, 0x8001U, 0x0080U, 0x8080U};
                value = bits[mixed % bits.size()];
            }
            values[static_cast<std::size_t>(token) * row + feature] = value;
        }
    return values;
}
std::vector<std::uint32_t> selected(Shape shape, std::uint32_t position) {
    std::vector<std::uint32_t> indices(maximum_selected_tokens(shape), std::numeric_limits<std::uint32_t>::max());
    const auto visible = position + 1U, blocks = visible / compression;
    const auto count = std::min(blocks, block_topk);
    std::size_t cursor = 0;
    for (std::uint32_t slot = 0; slot < count; ++slot) {
        const auto block = blocks <= block_topk ? slot : slot * (blocks - 1U) / (block_topk - 1U);
        for (std::uint32_t offset = 0; offset < compression; ++offset) indices[cursor++] = block * compression + offset;
    }
    for (auto token = blocks * compression; token < visible; ++token) indices[cursor++] = token;
    return indices;
}
struct Fixture {
    Owner owner;
    Shape shape;
    Span query, kv, indices, position;
    std::array<Bindings, 3> arms;
    explicit Fixture(Shape requested) : shape(requested) {
        const auto maximum = maximum_selected_tokens(shape);
        query = owner.allocate(static_cast<std::uint64_t>(shape.query_heads) * 512U * sizeof(float));
        kv = owner.allocate(static_cast<std::uint64_t>(shape.context_tokens) * shape.kv_heads * 512U * sizeof(std::uint16_t));
        indices = owner.allocate(static_cast<std::uint64_t>(maximum) * sizeof(std::uint32_t));
        position = owner.allocate(sizeof(std::uint64_t));
        for (auto& arm : arms) arm = {query, kv, indices, position, owner.allocate(sizeof(std::uint32_t)),
            owner.allocate(static_cast<std::uint64_t>(shape.query_heads) * maximum * sizeof(float)),
            owner.allocate(static_cast<std::uint64_t>(shape.query_heads) * 256U * sizeof(float))};
    }
    void initialize(unsigned flavor) { upload(owner, query, queries(shape, flavor)); upload(owner, kv, cached_kv(shape, flavor)); }
    void control(std::uint32_t occupied_position) {
        upload(owner, indices, selected(shape, occupied_position));
        upload(owner, position, std::vector<std::uint64_t>{occupied_position});
    }
    void reset(const Bindings& arm, std::uint32_t status = 0U) {
        poison(owner, arm.scores); poison(owner, arm.output);
        upload(owner, arm.status, std::vector<std::uint32_t>{status});
    }
    std::uint32_t status(const Bindings& arm) {
        const auto bytes = read(owner, arm.status); std::uint32_t result = 0;
        std::memcpy(&result, bytes.data(), sizeof(result)); return result;
    }
    void guards() {
        for (const auto view : {query, kv, indices, position}) canaries(owner, view);
        for (const auto& arm : arms) for (const auto view : {arm.status, arm.scores, arm.output}) canaries(owner, view);
    }
};
void positive(Shape shape, unsigned flavor, bool captured, const std::vector<std::uint32_t>& positions) {
    Fixture fixture(shape); fixture.initialize(flavor);
    const auto query_before = read(fixture.owner, fixture.query), kv_before = read(fixture.owner, fixture.kv);
    std::array<hipGraphExec_t, 3> graphs{};
    std::array<hipGraphExec_t, 3> score_graphs{};
    if (captured) for (std::size_t arm = 0; arm < modes.size(); ++arm) {
        graphs[arm] = fixture.owner.capture([&] { enqueue(shape, fixture.arms[arm], modes[arm], fixture.owner.stream); });
        score_graphs[arm] = fixture.owner.capture([&] {
            enqueue_scores(shape, fixture.arms[arm], modes[arm], fixture.owner.stream);
        }, 2U);
    }
    for (const auto position : positions) {
        fixture.control(position);
        const auto ids_before = read(fixture.owner, fixture.indices), position_before = read(fixture.owner, fixture.position);
        std::vector<unsigned char> raw_reference;
        for (std::size_t arm = 0; arm < modes.size(); ++arm) {
            fixture.reset(fixture.arms[arm]);
            const auto output_before = read(fixture.owner, fixture.arms[arm].output);
            if (captured) check(hipGraphLaunch(score_graphs[arm], fixture.owner.stream), "replay raw QK stage");
            else enqueue_scores(shape, fixture.arms[arm], modes[arm], fixture.owner.stream);
            fixture.owner.finish();
            require(fixture.status(fixture.arms[arm]) == 0U, "finite raw QK fixture reported failure");
            const auto raw_scores = read(fixture.owner, fixture.arms[arm].scores);
            if (arm == 0U) raw_reference = raw_scores;
            else require(raw_scores == raw_reference, "grouped raw QK scores differ bitwise before softmax");
            require(read(fixture.owner, fixture.arms[arm].output) == output_before, "score-only diagnostic wrote V output");
        }
        std::vector<unsigned char> reference_scores, reference_output;
        for (std::size_t arm = 0; arm < modes.size(); ++arm) {
            fixture.reset(fixture.arms[arm]);
            if (captured) check(hipGraphLaunch(graphs[arm], fixture.owner.stream), "replay dynamic attention stage");
            else enqueue(shape, fixture.arms[arm], modes[arm], fixture.owner.stream);
            fixture.owner.finish();
            require(fixture.status(fixture.arms[arm]) == 0U, "finite attention fixture reported failure");
            const auto scores = read(fixture.owner, fixture.arms[arm].scores), output = read(fixture.owner, fixture.arms[arm].output);
            if (arm == 0U) { reference_scores = scores; reference_output = output; }
            else require(scores == reference_scores && output == reference_output,
                         "grouped attention scores/output differ bitwise from literal incumbent kernels");
            for (std::size_t offset = 0; offset < output.size(); offset += sizeof(std::uint32_t)) {
                std::uint32_t bits = 0; std::memcpy(&bits, output.data() + offset, sizeof(bits));
                require((bits & 0xffffU) == 0U && std::isfinite(std::bit_cast<float>(bits)),
                        "attention output stopped representing finite BF16-rounded values in FP32 storage");
            }
        }
        require(read(fixture.owner, fixture.indices) == ids_before && read(fixture.owner, fixture.position) == position_before,
                "attention stage changed selected order or dynamic position");
    }
    require(read(fixture.owner, fixture.query) == query_before && read(fixture.owner, fixture.kv) == kv_before,
            "attention stage changed cached KV or query/gate inputs");
    fixture.guards();
}
void rejection_policy(bool captured) {
    const Shape shape{6U, 2U, 256U, 37U};
    Fixture fixture(shape);
    for (unsigned kind = 0; kind < 8U; ++kind) {
        fixture.initialize(0U); fixture.control(32U);
        std::uint32_t expected = 0U;
        if (kind == 0U) expected = 0x4000U;
        if (kind == 1U) {
            auto q = queries(shape, 0U); q.back() = std::numeric_limits<float>::quiet_NaN();
            upload(fixture.owner, fixture.query, q); expected = nonfinite_source;
        }
        if (kind == 2U) {
            auto kv = cached_kv(shape, 0U); kv[32U * shape.kv_heads * 512U + 513U] = 0x7f80U;
            upload(fixture.owner, fixture.kv, kv); expected = nonfinite_source;
        }
        if (kind == 3U || kind == 4U || kind == 5U) {
            auto ids = selected(shape, 32U);
            if (kind == 3U) ids[0] = 33U;
            if (kind == 4U) ids[2] = ids[1];
            if (kind == 5U) std::swap(ids[1], ids[2]);
            upload(fixture.owner, fixture.indices, ids); expected = invalid_selection;
        }
        if (kind == 6U) {
            upload(fixture.owner, fixture.position, std::vector<std::uint64_t>{shape.context_tokens}); expected = invalid_position;
        }
        if (kind == 7U) {
            auto q = queries(shape, 0U);
            std::fill(q.begin(), q.end(), std::bit_cast<float>(0x7f7f0000U));
            auto kv = cached_kv(shape, 0U); std::fill(kv.begin(), kv.end(), 0x7f7fU);
            upload(fixture.owner, fixture.query, q); upload(fixture.owner, fixture.kv, kv); expected = invalid_arithmetic;
        }
        for (std::size_t arm = 0; arm < modes.size(); ++arm) {
            fixture.reset(fixture.arms[arm], kind == 0U ? expected : 0U);
            const auto scores_before = read(fixture.owner, fixture.arms[arm].scores);
            const auto output_before = read(fixture.owner, fixture.arms[arm].output);
            if (captured) {
                const auto graph = fixture.owner.capture([&] { enqueue(shape, fixture.arms[arm], modes[arm], fixture.owner.stream); });
                check(hipGraphLaunch(graph, fixture.owner.stream), "replay rejected attention stage");
            } else enqueue(shape, fixture.arms[arm], modes[arm], fixture.owner.stream);
            fixture.owner.finish();
            require((fixture.status(fixture.arms[arm]) & expected) == expected, "diagnostic rejection lost its sticky error");
            require(read(fixture.owner, fixture.arms[arm].output) == output_before, "rejected attention published output");
            if (kind != 7U) require(read(fixture.owner, fixture.arms[arm].scores) == scores_before,
                                   "source/control rejection wrote attention scores");
        }
    }
    fixture.guards();
}
void admission() {
    Fixture fixture({24U, 2U, 256U, 33U});
    for (unsigned kind = 0; kind < 22U; ++kind) {
        auto shape = fixture.shape; auto views = fixture.arms.front(); auto mode = Mode::heads4; auto stream = fixture.owner.stream;
        if (kind == 0U) shape.query_heads = 0U;
        if (kind == 1U) shape.kv_heads = 0U;
        if (kind == 2U) shape.query_heads = 65U;
        if (kind == 3U) shape.kv_heads = 5U;
        if (kind == 4U) shape.head_dim = 128U;
        if (kind == 5U) shape.context_tokens = 0U;
        if (kind == 6U) shape.context_tokens = 262145U;
        if (kind == 7U) --views.q_gate.bytes;
        if (kind == 8U) ++views.kv_state.address;
        if (kind == 9U) views.output.address = views.scores.address;
        if (kind == 10U) views.status.address = views.q_gate.address;
        if (kind == 11U) views.position.address += 4U;
        if (kind == 12U) views.status.bytes = 8U;
        if (kind == 13U) --views.selected.bytes;
        if (kind == 14U) ++views.output.bytes;
        if (kind == 15U) views.scores.address = std::numeric_limits<std::uintptr_t>::max() - 3U;
        if (kind == 16U) mode = static_cast<Mode>(99);
        if (kind == 17U) stream = nullptr;
        if (kind == 18U) stream = hipStreamPerThread;
        if (kind == 19U) stream = hipStreamLegacy;
        if (kind == 20U) views.position.address = 0U;
        if (kind == 21U) views.kv_state.bytes = 0U;
        check(hipStreamBeginCapture(fixture.owner.stream, hipStreamCaptureModeThreadLocal), "begin invalid admission capture");
        hipGraph_t graph = nullptr;
        bool rejected = false;
        try { rejected = rejects([&] { enqueue(shape, views, mode, stream); }); }
        catch (...) {
            if (hipStreamEndCapture(fixture.owner.stream, &graph) == hipSuccess && graph) fixture.owner.retain(graph);
            throw;
        }
        check(hipStreamEndCapture(fixture.owner.stream, &graph), "end invalid admission capture"); fixture.owner.retain(graph);
        std::size_t nodes = 0; check(hipGraphGetNodes(graph, nullptr, &nodes), "read invalid admission graph");
        require(rejected && nodes == 0U, "invalid attention admission submitted work");
    }
    const auto base = validate(fixture.shape, fixture.arms.front(), Mode::reference);
    const auto two = validate(fixture.shape, fixture.arms.front(), Mode::heads2);
    const auto four = validate(fixture.shape, fixture.arms.front(), Mode::heads4);
    require(base.score_blocks == four.score_blocks * 4U && base.value_blocks == four.value_blocks * 4U &&
            base.score_blocks == two.score_blocks * 2U && four.score_lds_bytes == 4096U &&
            maximum_selected_tokens({24U, 2U, 256U, 262144U}) == 2051U,
            "QSA GQA declared geometry or capacity-tail accounting changed");
    fixture.guards();
}
} // namespace
int main() try {
    int devices = 0;
    if (hipGetDeviceCount(&devices) != hipSuccess || !devices) return 77;
    admission();
    require(score_kernel_symbol(Mode::reference) != score_kernel_symbol(Mode::heads2) &&
            score_kernel_symbol(Mode::reference) != score_kernel_symbol(Mode::heads4) &&
            value_kernel_symbol(Mode::reference) != value_kernel_symbol(Mode::heads4), "candidate reused reference symbol");
    for (const bool captured : {false, true}) {
        positive({24U, 2U, 256U, 4099U}, 0U, captured, {0U, 3U, 31U, 32U, 2047U, 2048U, 2050U, 4098U, 0U});
        positive({6U, 2U, 256U, 41U}, 2U, captured, {0U, 31U, 40U, 1U});
        positive({14U, 2U, 256U, 260U}, 1U, captured, {0U, 255U, 259U, 2U});
        positive({1U, 1U, 256U, 17U}, 0U, captured, {0U, 16U, 3U});
        rejection_policy(captured);
    }
    std::cout << "QSA GQA diagnostic source fixtures completed; full model gates remain separate\n";
    return 0;
} catch (const std::exception& error) { std::cerr << error.what() << '\n'; return 1; }
