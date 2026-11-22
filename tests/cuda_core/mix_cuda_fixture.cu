#include "cuda/core/mix_cuda.cuh"
#include "mix_oracle.hpp"

#include <algorithm>
#include <cstddef>
#include <cstdio>
#include <stdexcept>

namespace {

namespace residual = strix::core::residual_mix;
namespace candidate = strix::cuda::mix;
namespace test = strix::tests::mix;

void check(cudaError_t status) {
    if (status != cudaSuccess) throw std::runtime_error(cudaGetErrorString(status));
}

class ManagedFloats {
public:
    explicit ManagedFloats(std::size_t count) { check(cudaMallocManaged(&data, count * sizeof(float))); }
    ~ManagedFloats() { (void)cudaFree(data); }
    ManagedFloats(const ManagedFloats&) = delete;
    ManagedFloats& operator=(const ManagedFloats&) = delete;
    float* data = nullptr;
};

void same_bits(float actual, float expected, const char* what, unsigned query, std::size_t offset) {
    if (test::bits(actual) == test::bits(expected)) return;
    std::fprintf(stderr, "%s, query %u, offset %zu: %08x != %08x\n",
                 what, query, offset, test::bits(actual), test::bits(expected));
    throw std::runtime_error("byte comparison");
}

void compare_readout(unsigned rows, unsigned width, unsigned banks, unsigned queries, bool normalized,
                     bool equal_logits) {
    const std::size_t image = std::size_t{rows} * width;
    const auto stride = image + 2U;
    constexpr float guard = -1234.25F;
    ManagedFloats bank(banks * image), learned(std::size_t{queries} * width);
    ManagedFloats incumbent(queries * stride), actual(queries * stride);
    std::fill_n(incumbent.data, queries * stride, guard);
    std::fill_n(actual.data, queries * stride, guard);
    for (std::size_t i = 0; i < banks * image; ++i)
        bank.data[i] = static_cast<float>(static_cast<int>((i * 11U + i / width) % 31U) - 15) * 0.137F;
    for (std::size_t i = 0; i < std::size_t{queries} * width; ++i)
        learned.data[i] = equal_logits ? 0.0F :
            static_cast<float>(static_cast<int>((i * 7U + i / width) % 17U) - 8) * 0.073F;

    residual::MixArgs args{.queries = learned.data, .rows = rows, .width = width,
                           .bank_count = banks, .query_count = queries, .epsilon = 1e-6F, .normalized = normalized};
    for (unsigned b = 0; b < banks; ++b) {
        float* source = bank.data + b * image;
        if (normalized) check(residual::launch_normalize({source, source, rows, width, args.epsilon}, nullptr));
        args.banks[b] = source;
    }

    auto old_args = args;
    for (unsigned q = 0; q < queries; ++q) {
        old_args.output[q] = incumbent.data + q * stride + 1U;
        args.output[q] = actual.data + q * stride + 1U;
    }
    check(residual::launch_mix(old_args, nullptr));
    check(candidate::enqueue(args, nullptr));
    check(cudaStreamSynchronize(nullptr));

    for (unsigned q = 0; q < queries; ++q) {
        const float* got = args.output[q];
        const float* reference = old_args.output[q];
        for (std::size_t i = 0; i < image; ++i) same_bits(got[i], reference[i], "incumbent", q, i);
        same_bits(got[-1], guard, "leading canary", q, 0U);
        same_bits(got[image], guard, "trailing canary", q, image);
        same_bits(reference[-1], guard, "incumbent leading canary", q, 0U);
        same_bits(reference[image], guard, "incumbent trailing canary", q, image);
    }

    if (equal_logits) {
        const auto cpu = test::readout_oracle(args);
        for (unsigned q = 0; q < queries; ++q)
            for (std::size_t i = 0; i < image; ++i)
                same_bits(args.output[q][i], cpu[q * image + i], "CPU oracle", q, i);
    }
}

} // namespace

int main() {
    try {
        compare_readout(1U, 2304U, 1U, 1U, false, false);
        compare_readout(1U, 2304U, 5U, 3U, false, false);
        std::puts("PASS K8c CUDA raw residual readout, R1");
        compare_readout(2U, 513U, 8U, 3U, false, false);
        compare_readout(2U, 2304U, 2U, 1U, true, false);
        std::puts("PASS K8c CUDA paired-bank readout, R2");
        compare_readout(3U, 513U, 3U, 3U, false, true);
        compare_readout(33U, 257U, 2U, 1U, true, true);
        std::puts("PASS K8c CUDA serial-bank readout and CPU oracle");
        compare_readout(65536U, 1U, 2U, 1U, true, false);
        std::puts("PASS K8c CUDA bank readout across the grid-stride boundary");
        return 0;
    } catch (const std::exception& error) {
        std::fprintf(stderr, "FAIL K8c CUDA %s\n", error.what());
        return 1;
    }
}
