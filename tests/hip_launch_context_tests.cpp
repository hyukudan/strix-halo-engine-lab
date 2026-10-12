#include "strix/hip_launch_context.hpp"

#include <barrier>
#include <cstdint>
#include <iostream>
#include <limits>
#include <stdexcept>
#include <thread>
#include <type_traits>

namespace {
int failures = 0;
void expect(bool condition, const char* message) {
    if (!condition) { std::cerr << "FAIL: " << message << '\n'; ++failures; }
}
bool same(const strix::ExecutionMemoryBinding& a, const strix::ExecutionMemoryBinding& b) {
    return a.execution_identity == b.execution_identity && a.arena_base == b.arena_base &&
        a.arena_bytes == b.arena_bytes && a.allocation_generation == b.allocation_generation;
}
strix::HipLaunchContext fixture() {
    strix::HipLaunchContext context;
    // Metadata-only sentinel: this test never calls a HIP API or dereferences it.
    context.stream = reinterpret_cast<hipStream_t>(std::uintptr_t{0x7000});
    context.graph_capture = true;
    context.arguments = strix::bind_graph_inputs(strix::graph_input_layout(1), {9, 0x1000, 16, 1});
    context.status = {9, 0x2000, 4, 2};
    context.context_tokens = 32;
    context.vocabulary_size = 256;
    context.ngram_input = {9, 0x3000, 28, 3};
    context.ngram_output = {9, 0x4000, 28, 4};
    return context;
}
void rejects(const strix::HipLaunchContext& context, const char* message) {
    bool rejected = false;
    try { const strix::HipLaunchScope scope(context); }
    catch (const std::invalid_argument&) { rejected = true; }
    expect(rejected, message);
    expect(strix::current_hip_launch_context() == nullptr, "failed construction leaves TLS empty");
}
} // namespace

int main() {
    static_assert(!std::is_copy_constructible_v<strix::HipLaunchScope>);
    static_assert(!std::is_move_constructible_v<strix::HipLaunchScope>);
    expect(strix::current_hip_launch_context() == nullptr && strix::hip_execution_stream() == nullptr &&
        strix::hip_graph_position() == nullptr && strix::hip_graph_status() == nullptr,
        "unscoped helper defaults are empty");
    const auto original = fixture();
    {
        auto caller = original;
        const strix::HipLaunchScope scope(caller);
        const auto* owned = strix::current_hip_launch_context();
        expect(owned != &caller && owned->arguments == original.arguments &&
            same(owned->status, original.status) && same(owned->ngram_input, original.ngram_input) &&
            same(owned->ngram_output, original.ngram_output), "scope owns a copy of all bound metadata");
        caller.stream = nullptr; caller.graph_capture = false; caller.arguments = {};
        caller.status = {}; caller.context_tokens = 0; caller.vocabulary_size = 0;
        caller.ngram_input = {}; caller.ngram_output = {};
        expect(owned->stream == original.stream && owned->graph_capture &&
            owned->arguments == original.arguments && same(owned->status, original.status) &&
            owned->context_tokens == original.context_tokens && owned->vocabulary_size == original.vocabulary_size &&
            same(owned->ngram_input, original.ngram_input) && same(owned->ngram_output, original.ngram_output),
            "caller mutation cannot change active scope metadata");
        expect(strix::hip_execution_stream() == original.stream &&
            reinterpret_cast<std::uintptr_t>(strix::hip_graph_position()) == 0x1008 &&
            reinterpret_cast<std::uintptr_t>(strix::hip_graph_status()) == 0x2000,
            "helpers expose the copied device argument/status addresses");
        bool nested_rejected = false;
        try { const strix::HipLaunchScope nested(original); }
        catch (const std::invalid_argument&) { nested_rejected = true; }
        expect(nested_rejected && strix::current_hip_launch_context() == owned,
            "nested rejection preserves the outer scope");

        std::barrier rendezvous(2);
        bool thread_ok = false;
        std::thread worker([&] {
            const bool empty = strix::current_hip_launch_context() == nullptr;
            {
                auto independent = original;
                independent.stream = reinterpret_cast<hipStream_t>(std::uintptr_t{0x8000});
                const strix::HipLaunchScope local(independent);
                thread_ok = empty && strix::hip_execution_stream() == independent.stream;
                rendezvous.arrive_and_wait();
                rendezvous.arrive_and_wait();
            }
            thread_ok = thread_ok && strix::current_hip_launch_context() == nullptr;
        });
        rendezvous.arrive_and_wait();
        expect(strix::current_hip_launch_context() == owned && strix::hip_execution_stream() == original.stream,
            "concurrent thread scope cannot replace parent TLS");
        rendezvous.arrive_and_wait();
        worker.join();
        expect(thread_ok, "other thread starts empty and independently releases its scope");
    }
    expect(strix::current_hip_launch_context() == nullptr, "normal destruction releases TLS");
    try {
        const strix::HipLaunchScope scope(original);
        throw std::runtime_error("controlled unwind");
    } catch (const std::runtime_error&) {}
    expect(strix::current_hip_launch_context() == nullptr, "exception unwind releases TLS");

    for (unsigned fault = 0; fault < 23; ++fault) {
        auto bad = original;
        switch (fault) {
        case 0: bad.stream = nullptr; break;
        case 1: bad.stream = hipStreamPerThread; break;
        case 2: bad.stream = hipStreamLegacy; break;
        case 3: bad.arguments.mode = strix::GraphArgumentMode::baked_scalars; break;
        case 4: ++bad.arguments.device_argument_abi_identity; break;
        case 5: bad.arguments.position_offset = 0; break;
        case 6: bad.arguments.token_offset = 4; break;
        case 7: bad.arguments.memory.execution_identity = 0; break;
        case 8: bad.arguments.memory.allocation_generation = 0; break;
        case 9: bad.arguments.memory.arena_base = 0; break;
        case 10: bad.arguments.memory.arena_base += 4; break;
        case 11: bad.arguments.memory.arena_bytes = 8; break;
        case 12: bad.arguments.memory.arena_base = std::numeric_limits<std::uintptr_t>::max() - 7; break;
        case 13: bad.context_tokens = 0; break;
        case 14: bad.vocabulary_size = 0; break;
        case 15: ++bad.status.execution_identity; break;
        case 16: bad.status.allocation_generation = 0; break;
        case 17: bad.status.arena_base = 0; break;
        case 18: ++bad.status.arena_base; break;
        case 19: bad.status.arena_bytes = 3; break;
        case 20: bad.status.arena_bytes = 8; break;
        case 21: bad.status.arena_base = std::numeric_limits<std::uintptr_t>::max() - 3; break;
        case 22: bad.status.arena_base = 0x100c; break;
        }
        rejects(bad, "malformed stream/ABI/status/limit/range must reject");
    }
    {
        auto adjacent = original; adjacent.status.arena_base = 0x1010;
        const strix::HipLaunchScope scope(adjacent);
        expect(strix::hip_graph_status() != nullptr, "adjacent control ranges do not alias");
    }
    {
        auto eager = original; eager.graph_capture = false; eager.arguments = {}; eager.status = {};
        const strix::HipLaunchScope scope(eager);
        expect(strix::hip_execution_stream() == eager.stream && strix::hip_graph_position() == nullptr &&
            strix::hip_graph_status() == nullptr, "eager owned stream does not expose graph controls");
    }
    expect(strix::current_hip_launch_context() == nullptr, "all scopes finish empty");
    if (!failures) std::cout << "HIP launch scope host-only tests passed\n";
    return failures ? 1 : 0;
}
