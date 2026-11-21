#include "strix/device/runtime.hpp"

#include <array>
#include <cstdio>

namespace {

namespace device = strix::device::native;
constexpr unsigned kWords = 32;

__global__ void add(unsigned* words, unsigned scalar) {
    const auto i = threadIdx.x;
    words[i] += scalar + i;
}

__global__ void multiply(unsigned* words, unsigned scalar) {
    const auto i = threadIdx.x;
    words[i] *= 3U * scalar + 1U;
}

__global__ void mix(unsigned* words, unsigned scalar) {
    const auto i = threadIdx.x;
    words[i] ^= scalar << (i % 8U);
}

void enqueue(device::Buffer<unsigned>& words, unsigned scalar, device::Stream stream) {
    device::clear(words.data(), words.span().size_bytes(), stream);
    add<<<1, kWords, 0, stream>>>(words.data(), scalar);
    multiply<<<1, kWords, 0, stream>>>(words.data(), scalar);
    mix<<<1, kWords, 0, stream>>>(words.data(), scalar);
    device::check(cudaGetLastError(), "enqueue graph fixture kernels");
}

}  // namespace

int main() {
    try {
        device::StreamOwner stream;
        device::Buffer<unsigned> eager(kWords), captured(kWords);
        device::StepGraph graph;
        for (unsigned scalar = 1; scalar <= 20; ++scalar) {
            enqueue(eager, scalar, stream.get());
            graph.begin(stream.get());
            try {
                enqueue(captured, scalar, stream.get());
                graph.launch(stream.get());
            } catch (...) {
                graph.abandon(stream.get());
                throw;
            }

            std::array<unsigned, kWords> expected{}, actual{};
            device::copy(expected.data(), eager.data(), sizeof(expected), device::device_to_host, stream.get());
            device::copy(actual.data(), captured.data(), sizeof(actual), device::device_to_host, stream.get());
            device::synchronize(stream.get());
            if (expected != actual) {
                std::fprintf(stderr, "StepGraph differs from eager at scalar %u\n", scalar);
                return 1;
            }
        }
        if (graph.instantiations() != 1U || graph.updates() == 0U) {
            std::fprintf(stderr, "StepGraph recorded %u instantiations and %u updates\n",
                         graph.instantiations(), graph.updates());
            return 1;
        }
        graph.begin(stream.get());
        try {
            enqueue(captured, 99U, stream.get());
        } catch (...) {
            graph.abandon(stream.get());
            throw;
        }
        graph.abandon(stream.get());
        std::array<unsigned, kWords> expected{}, actual{};
        device::copy(expected.data(), eager.data(), sizeof(expected), device::device_to_host, stream.get());
        device::copy(actual.data(), captured.data(), sizeof(actual), device::device_to_host, stream.get());
        device::synchronize(stream.get());
        if (graph.capturing() || expected != actual) return 1;
    } catch (const std::exception& error) {
        std::fprintf(stderr, "%s\n", error.what());
        return 1;
    }
    return 0;
}
