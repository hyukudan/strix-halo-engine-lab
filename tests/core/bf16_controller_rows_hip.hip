// Gated controller epilogue: enqueue_bf16_controller_rows stores the same bits as the two-step form
// (enqueue_bf16_rows into FP32 forget / gain images, then the pointwise residual_mix controller), and declines decode
// rows so that the caller keeps the two-step form there.
#include "strix/hip/legacy_intrinsics.hpp"
#include "strix/hip/legacy_launch.hpp"
#include "core/fma_bf16_rows.hpp"
#include "core/residual_mix.hpp"

#include <hip/hip_runtime.h>

#include <cstdint>
#include <cstdio>
#include <cstring>
#include <random>
#include <vector>

namespace {

namespace rows_ns = strix::core::fma_rows;
namespace mix = strix::core::residual_mix;
using strix::core::Bf16RowProjection;

bool ok(hipError_t status) { return status == hipSuccess; }

template <class T>
T* upload(const std::vector<T>& host) {
    T* device = nullptr;
    if (!ok(hipMalloc(&device, host.size() * sizeof(T)))) return nullptr;
    if (!ok(hipMemcpy(device, host.data(), host.size() * sizeof(T), hipMemcpyHostToDevice))) return nullptr;
    return device;
}

std::vector<float> values(std::size_t count, std::mt19937& random, float range, bool specials) {
    std::uniform_real_distribution<float> value(-range, range);
    std::vector<float> out(count);
    for (auto& x : out) x = value(random);
    if (specials) {
        const std::uint32_t bits[] = {0x7F800000U, 0xFF800000U, 0x7FC00000U, 0x80000000U, 0x00000001U, 0x7F7FFFFFU};
        for (std::size_t i = 5; i < out.size(); i += 613U) std::memcpy(&out[i], &bits[random() % 6U], sizeof(float));
    }
    return out;
}

std::vector<std::uint16_t> bf16_weights(std::size_t count, std::mt19937& random) {
    std::vector<std::uint16_t> out(count);
    const auto source = values(count, random, 0.5F, false);
    for (std::size_t i = 0; i < count; ++i) {
        std::uint32_t bits;
        std::memcpy(&bits, &source[i], sizeof bits);
        out[i] = static_cast<std::uint16_t>(bits >> 16U);
    }
    return out;
}

// 1 = equal, 0 = differ / error, 2 = declined as expected.
int run(unsigned rows, unsigned K, unsigned width, std::mt19937& random, std::size_t& compared) {
    const std::size_t elements = std::size_t{rows} * width;
    auto* input = upload(values(std::size_t{rows} * K, random, 2.0F, true));
    auto* forget_w = upload(bf16_weights(std::size_t{width} * K, random));
    auto* gain_w = upload(bf16_weights(std::size_t{width} * K, random));
    auto* current = upload(values(elements, random, 3.0F, true));
    auto* message = upload(values(elements, random, 3.0F, true));
    float *forget = nullptr, *gain = nullptr, *two_step = nullptr, *fused = nullptr;
    int result = 0;
    if (input && forget_w && gain_w && current && message && ok(hipMalloc(&forget, elements * sizeof(float))) &&
        ok(hipMalloc(&gain, elements * sizeof(float))) && ok(hipMalloc(&two_step, elements * sizeof(float))) &&
        ok(hipMalloc(&fused, elements * sizeof(float))) && ok(hipMemset(fused, 0x5A, elements * sizeof(float)))) {
        const Bf16RowProjection gates[] = {{forget_w, width, forget}, {gain_w, width, gain}};
        const auto status = rows_ns::enqueue_bf16_controller_rows(input, K, rows, gates, current, message, fused, nullptr);
        if (status == hipErrorNotSupported) {
            result = 2;
        } else if (ok(status) && ok(rows_ns::enqueue_bf16_rows(input, K, rows, gates, width, nullptr)) &&
                   ok(mix::launch_controller({current, message, forget, gain, two_step, elements}, nullptr)) &&
                   ok(hipDeviceSynchronize())) {
            std::vector<float> a(elements), b(elements);
            if (ok(hipMemcpy(a.data(), two_step, elements * sizeof(float), hipMemcpyDeviceToHost)) &&
                ok(hipMemcpy(b.data(), fused, elements * sizeof(float), hipMemcpyDeviceToHost)) &&
                std::memcmp(a.data(), b.data(), elements * sizeof(float)) == 0) {
                result = 1;
                compared += elements;
            }
        }
    }
    for (void* p : {static_cast<void*>(input), static_cast<void*>(forget_w), static_cast<void*>(gain_w),
                    static_cast<void*>(current), static_cast<void*>(message), static_cast<void*>(forget),
                    static_cast<void*>(gain), static_cast<void*>(two_step), static_cast<void*>(fused)})
        if (p) (void)hipFree(p);
    return result;
}

}  // namespace

int main() {
    std::mt19937 random(20261122U);
    std::size_t compared = 0;
    // (rows, K, width): the Inkling controller pair (rank 64, width 2304) with a partial row block, a K16 tail with a
    // partial panel, a small row count above the decode bound.
    const struct { unsigned rows, K, width; } fused_cases[] = {{300U, 64U, 2304U}, {129U, 48U, 200U}, {33U, 32U, 64U}};
    for (const auto& c : fused_cases) {
        if (run(c.rows, c.K, c.width, random, compared) != 1) {
            std::printf("bf16-controller-rows: FAIL rows %u K %u width %u\n", c.rows, c.K, c.width);
            return 1;
        }
    }
    if (run(1U, 64U, 256U, random, compared) != 2) {
        std::printf("bf16-controller-rows: FAIL decode rows were not declined\n");
        return 1;
    }
    std::printf("bf16-controller-rows: %zu outputs bitwise equal (fused epilogue vs projections + controller)\n", compared);
    return 0;
}
