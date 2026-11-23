// Device fixture of the shared position-indexed BF16 row store (src/cuda/core/step_row_store.cuh), the leaf a
// step program uses to store drafter feature taps. For every step width (1, 2, 4, 8, 16), valid-row masks and
// positions (including a position past the store, which must write nothing):
//   - every valid row's cells at position(row) * pitch + [0, width) hold the BF16 encode of its FP32 input,
//     bitwise equal to the resident tap store (core::row_transforms::StoreBf16Slice over the same row);
//   - no other cell changes (canary over the whole store, including the pitch gap and invalid rows).
// Prints "step-row-store-cuda: PASS <cases>" and returns 0, or names the first mismatch and returns 1.
#include "core/row_transforms.hpp"
#include "cuda/core/step_row_store.cuh"

#include <cuda_runtime.h>

#include <cstdint>
#include <cstdio>
#include <cstring>
#include <random>
#include <stdexcept>
#include <string>
#include <vector>

namespace {

namespace core = strix::cuda::core;
namespace exec = strix::exec;

void check(cudaError_t status, const char* what) {
    if (status != cudaSuccess) throw std::runtime_error(std::string(what) + ": " + cudaGetErrorString(status));
}

template <class T>
struct Device {
    explicit Device(std::size_t count) : size(count) { check(cudaMalloc(&data, count * sizeof(T)), "cudaMalloc"); }
    ~Device() { (void)cudaFree(data); }
    Device(const Device&) = delete;
    Device& operator=(const Device&) = delete;
    void upload(const std::vector<T>& host) { check(cudaMemcpy(data, host.data(), size * sizeof(T), cudaMemcpyHostToDevice), "upload"); }
    std::vector<T> read() const {
        std::vector<T> host(size);
        check(cudaMemcpy(host.data(), data, size * sizeof(T), cudaMemcpyDeviceToHost), "read");
        return host;
    }
    T* data{};
    std::size_t size;
};

template <unsigned Rows>
cudaError_t enqueue(const core::StepRowStoreBf16& a, cudaStream_t stream) {
    return core::enqueue_step_row_store_bf16<Rows>(a, stream);
}

cudaError_t enqueue_width(unsigned rows, const core::StepRowStoreBf16& a) {
    switch (rows) {
    case 1U: return enqueue<1>(a, nullptr);
    case 2U: return enqueue<2>(a, nullptr);
    case 4U: return enqueue<4>(a, nullptr);
    case 8U: return enqueue<8>(a, nullptr);
    case 16U: return enqueue<16>(a, nullptr);
    default: return cudaErrorInvalidValue;
    }
}

}  // namespace

int main() try {
    constexpr unsigned width = 2304U, taps = 4U, pitch = taps * width, positions = 64U;
    constexpr std::uint16_t canary = 0xA5A5U;
    std::mt19937 random(0x7a9cU);
    std::normal_distribution<float> normal(0.0F, 3.0F);
    unsigned cases = 0;
    Device<exec::StepBlock> step(1U);
    Device<float> input(16U * width);
    Device<std::uint16_t> store(std::size_t{positions} * pitch), resident(std::size_t{positions} * pitch);
    for (const unsigned rows : {1U, 2U, 4U, 8U, 16U}) {
        for (const unsigned tap : {0U, 3U}) {
            for (const unsigned first : {0U, 17U, positions - rows, positions - 1U}) {
                for (const unsigned mask_kind : {0U, 1U}) {
                    std::vector<float> host(16U * width);
                    for (auto& value : host) value = normal(random);
                    host[5] = 1.0F / 3.0F;  // ties and specials ride along with the random rows
                    host[6] = -0.0F;
                    host[7] = 65504.0F;
                    input.upload(host);
                    exec::StepBlock block{};
                    block.h.rows = rows;
                    block.h.step = first;
                    for (unsigned r = 0; r < rows; ++r) {
                        const bool valid = mask_kind == 0U || r % 3U != 1U;
                        block.row[r] = {r, first + r, 0U, r == 0U ? exec::kStepNoParent : r - 1U, 0U,
                                        valid ? exec::step_valid : 0U, 0U, 0U};
                        if (valid) block.h.valid_mask |= 1U << r;
                    }
                    for (unsigned r = 0; r < rows; ++r) block.row[r].ancestors = block.h.valid_mask;
                    check(cudaMemcpy(step.data, &block, sizeof(block), cudaMemcpyHostToDevice), "upload step");
                    store.upload(std::vector<std::uint16_t>(store.size, canary));
                    resident.upload(std::vector<std::uint16_t>(resident.size, canary));
                    const core::StepRowStoreBf16 args{step.data, input.data, store.data + std::size_t{tap} * width,
                                                      width, pitch, positions};
                    check(enqueue_width(rows, args), "enqueue step row store");
                    // Resident reference: the per-chunk slice store of the resident forward, one valid row at a time.
                    for (unsigned r = 0; r < rows; ++r) {
                        if ((block.h.valid_mask & (1U << r)) == 0U || first + r >= positions) continue;
                        namespace transforms = strix::core::row_transforms;
                        check(transforms::elements(width, transforms::StoreBf16Slice{
                            input.data + std::size_t{r} * width,
                            resident.data + std::size_t{first + r} * pitch + std::size_t{tap} * width,
                            width, pitch}, nullptr), "resident reference store");
                    }
                    check(cudaDeviceSynchronize(), "synchronize");
                    const auto actual = store.read(), expected = resident.read();
                    if (std::memcmp(actual.data(), expected.data(), actual.size() * sizeof(std::uint16_t)) != 0) {
                        for (std::size_t i = 0; i < actual.size(); ++i) {
                            if (actual[i] == expected[i]) continue;
                            std::fprintf(stderr, "step-row-store-cuda: FAIL rows=%u tap=%u first=%u mask=%u cell=%zu "
                                         "(position %zu, element %zu) got %04x expected %04x\n", rows, tap, first,
                                         mask_kind, i, i / pitch, i % pitch, actual[i], expected[i]);
                            return 1;
                        }
                    }
                    ++cases;
                }
            }
        }
    }
    // Invalid shapes are refused before any launch.
    const core::StepRowStoreBf16 bad{step.data, input.data, store.data, width, width - 1U, positions};
    if (core::enqueue_step_row_store_bf16<1>(bad, nullptr) != cudaErrorInvalidValue) {
        std::fprintf(stderr, "step-row-store-cuda: FAIL pitch below width was admitted\n");
        return 1;
    }
    std::printf("step-row-store-cuda: PASS %u cases (widths 1/2/4/8/16, masks, end-of-store positions)\n", cases);
    return 0;
} catch (const std::exception& error) {
    std::fprintf(stderr, "step-row-store-cuda: FAIL %s\n", error.what());
    return 1;
}
