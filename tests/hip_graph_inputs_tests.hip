#include "strix/hip_embedding_launcher.hpp"

#include <hip/hip_runtime.h>

#include <array>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

namespace {
void check(hipError_t error, const char* operation) {
    if (error != hipSuccess) throw std::runtime_error(std::string(operation) + ": " + hipGetErrorString(error));
}
void require(bool condition, const char* message) {
    if (!condition) throw std::runtime_error(message);
}
struct Fixture {
    static constexpr std::uint32_t width = 256;
    static constexpr std::uint32_t vocabulary = 3;
    static constexpr std::uint64_t context = 64;
    hipStream_t stream = nullptr;
    hipGraph_t graph = nullptr;
    hipGraphExec_t executable = nullptr;
    void* weights = nullptr;
    void* inputs = nullptr;
    void* output = nullptr;
    void* status = nullptr;
    std::vector<float> host_weights;

    ~Fixture() {
        // A failed device context does not prove graph resources are unused.
        // Keep the entire bounded fixture allocation until process termination.
        if (stream != nullptr && hipStreamQuery(stream) != hipSuccess) return;
        if (executable) static_cast<void>(hipGraphExecDestroy(executable));
        if (graph) static_cast<void>(hipGraphDestroy(graph));
        if (stream) static_cast<void>(hipStreamDestroy(stream));
        if (status) static_cast<void>(hipFree(status));
        if (output) static_cast<void>(hipFree(output));
        if (inputs) static_cast<void>(hipFree(inputs));
        if (weights) static_cast<void>(hipFree(weights));
    }

    void run() {
        check(hipStreamCreateWithFlags(&stream, hipStreamNonBlocking), "create graph stream");
        host_weights.resize(width * vocabulary);
        for (std::uint32_t token = 0; token < vocabulary; ++token) {
            for (std::uint32_t feature = 0; feature < width; ++feature) {
                host_weights[token * width + feature] = static_cast<float>(token * 1000U + feature) * 0.125F;
            }
        }
        check(hipMalloc(&weights, host_weights.size() * sizeof(float)), "allocate weights");
        check(hipMalloc(&inputs, 16), "allocate graph inputs");
        check(hipMalloc(&output, (width + 2U) * sizeof(float)), "allocate guarded output");
        check(hipMalloc(&status, sizeof(std::uint32_t)), "allocate status");
        check(hipMemcpy(weights, host_weights.data(), host_weights.size() * sizeof(float), hipMemcpyHostToDevice), "copy weights");
        const std::vector<float> poison(width + 2U, -777.0F);
        check(hipMemcpy(output, poison.data(), poison.size() * sizeof(float), hipMemcpyHostToDevice), "poison output");
        const auto arguments = strix::bind_graph_inputs(strix::graph_input_layout(1),
            {1, reinterpret_cast<std::uintptr_t>(inputs), 16, 2});
        strix::ExecutionBufferAddress destination;
        destination.address = reinterpret_cast<std::uintptr_t>(output) + sizeof(float);
        destination.bytes = width * sizeof(float);
        destination.allocation_generation = 3;
        destination.use.generation = 1;
        const strix::RuntimeTensorBinding weight_binding{0, strix::RuntimeTensorStorage::resident,
            reinterpret_cast<std::uintptr_t>(weights), host_weights.size() * sizeof(float), 4, 0};
        const strix::ExecutionMemoryBinding status_binding{1, reinterpret_cast<std::uintptr_t>(status), 4, 5};
        const strix::HipEmbeddingLauncherConfig config{width, strix::HipActivationStorage::fp32};
        check(hipStreamBeginCapture(stream, hipStreamCaptureModeThreadLocal), "begin capture");
        try {
            strix::launch_hip_graph_embedding(config, 0, weight_binding, vocabulary, context,
                arguments, destination, status_binding, stream);
        } catch (...) {
            static_cast<void>(hipStreamEndCapture(stream, &graph));
            throw;
        }
        check(hipStreamEndCapture(stream, &graph), "end capture");
        std::size_t nodes = 0;
        check(hipGraphGetNodes(graph, nullptr, &nodes), "graph nodes");
        require(nodes > 0, "capture must contain a device-input kernel");
        check(hipGraphInstantiate(&executable, graph, nullptr, nullptr, 0), "instantiate graph");

        const auto replay = [&](std::uint32_t token, std::uint64_t position, std::uint32_t expected_flags) {
            std::vector<std::byte> packet(16, std::byte{0});
            if (expected_flags == 0) {
                const std::array<std::uint32_t, 1> tokens{token};
                const std::array<std::uint64_t, 1> positions{position};
                packet = strix::pack_graph_inputs(strix::graph_input_layout(1), tokens, positions, vocabulary, context);
            } else {
                // Deliberately bypass host validation to prove the kernel never
                // dereferences an invalid token, and checks position on replay.
                std::memcpy(packet.data(), &token, sizeof(token));
                std::memcpy(packet.data() + 8, &position, sizeof(position));
            }
            check(hipMemcpy(inputs, packet.data(), packet.size(), hipMemcpyHostToDevice), "update stable inputs");
            check(hipGraphLaunch(executable, stream), "graph replay");
            check(hipStreamSynchronize(stream), "graph completion");
            std::vector<float> actual(width + 2U);
            std::uint32_t flags = 99;
            check(hipMemcpy(actual.data(), output, actual.size() * sizeof(float), hipMemcpyDeviceToHost), "read output");
            check(hipMemcpy(&flags, status, sizeof(flags), hipMemcpyDeviceToHost), "read status");
            require(flags == expected_flags, "device-input validity status");
            require(actual.front() == -777.0F && actual.back() == -777.0F, "graph output guards");
            for (std::uint32_t feature = 0; feature < width; ++feature) {
                const auto expected = expected_flags == 0 ? host_weights[token * width + feature] : 0.0F;
                require(actual[feature + 1] == expected, "replay matches independent embedding row");
            }
        };
        replay(0, 0, 0);
        replay(2, 63, 0);
        replay(1, 31, 0);
        replay(vocabulary, 1, 1);
        replay(0, context, 2);
        replay(1, std::uint64_t{1} << 32, 2); // U64 position must not truncate to zero.
        replay(vocabulary, context, 3);
        replay(2, 2, 0); // status clears rather than retaining an old failure.
    }
};
}
int main() {
    try {
        Fixture fixture;
        fixture.run();
        std::cout << "experimental C1 device-input embedding graph fixtures passed; whole-model replay unqualified\n";
        return 0;
    } catch (const std::exception& error) {
        std::cerr << error.what() << '\n';
        return 1;
    }
}
