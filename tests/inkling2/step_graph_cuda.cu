#include "strix/device/runtime.hpp"

#include <array>
#include <cstdio>
#include <stdexcept>

namespace {

namespace device = strix::device::native;
constexpr unsigned kWords = 32;

using strix::session::StepDescriptor;

__global__ void add(unsigned* words, unsigned scalar, const StepDescriptor* step) {
    const auto i = threadIdx.x;
    words[i] += (step ? step->position : scalar) + i;
}

__global__ void multiply(unsigned* words, unsigned scalar, const StepDescriptor* step) {
    const auto i = threadIdx.x;
    words[i] *= 3U * (step ? step->token : scalar) + 1U;
}

__global__ void mix(unsigned* words, unsigned scalar, const StepDescriptor* step) {
    const auto i = threadIdx.x;
    words[i] ^= (step ? step->position : scalar) << (i % 8U);
}

void enqueue(device::Buffer<unsigned>& words, unsigned scalar, unsigned token, device::Stream stream,
             const StepDescriptor* step = nullptr) {
    device::clear(words.data(), words.span().size_bytes(), stream);
    add<<<1, kWords, 0, stream>>>(words.data(), scalar, step);
    multiply<<<1, kWords, 0, stream>>>(words.data(), token, step);
    mix<<<1, kWords, 0, stream>>>(words.data(), scalar, step);
    device::check(cudaGetLastError(), "enqueue graph fixture kernels");
}

}  // namespace

int main() {
    try {
        device::StreamOwner stream;
        device::Buffer<unsigned> eager(kWords), captured(kWords);
        device::StepGraph graph;
        unsigned captures = 0U;
        for (unsigned scalar = 1U; scalar <= 128U; ++scalar) {
            enqueue(eager, scalar, 256U - scalar, stream.get());
            if (!graph.try_launch(stream.get(), true, 1U, true, true, {scalar, 256U - scalar}, [&] {
                ++captures;
                enqueue(captured, 0U, 0U, stream.get(), graph.capture_descriptor());
            })) return 1;

            std::array<unsigned, kWords> expected{}, actual{};
            device::copy(expected.data(), eager.data(), sizeof(expected), device::device_to_host, stream.get());
            device::copy(actual.data(), captured.data(), sizeof(actual), device::device_to_host, stream.get());
            device::synchronize(stream.get());
            if (expected != actual) {
                std::fprintf(stderr, "Step graph differs from eager at position %u\n", scalar);
                return 1;
            }
        }
        if (graph.instantiations() != 1U || captures != 1U) return 1;
        if (graph.try_launch(stream.get(), true, 2U, false, true, {}, [&] { ++captures; })) return 1;
        if (graph.try_launch(stream.get(), true, 1U, true, true, {}, [&] {
            ++captures;
            enqueue(captured, 0U, 0U, stream.get(), graph.capture_descriptor());
            throw std::runtime_error("discard test capture");
        })) return 1;
        if (graph.try_launch(stream.get(), true, 1U, true, true, {}, [&] { ++captures; })) return 1;
        std::array<unsigned, kWords> expected{}, actual{};
        device::copy(expected.data(), eager.data(), sizeof(expected), device::device_to_host, stream.get());
        device::copy(actual.data(), captured.data(), sizeof(actual), device::device_to_host, stream.get());
        device::synchronize(stream.get());
        if (captures != 2U || graph.instantiations() != 1U || expected != actual) return 1;
    } catch (const std::exception& error) {
        std::fprintf(stderr, "%s\n", error.what());
        return 1;
    }
    return 0;
}
