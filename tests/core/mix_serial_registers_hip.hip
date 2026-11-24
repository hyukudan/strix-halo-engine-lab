// Residual bank readout, serial (prefill) schedule: the register form (default) stores the same bits as the loop form
// for raw and normalized banks, 1..8 banks, 1..3 queries, widths up to the register bound and past it (loop form kept).
#include "strix/hip/legacy_intrinsics.hpp"
#include "strix/hip/legacy_launch.hpp"
#pragma clang diagnostic push
#pragma clang diagnostic ignored "-Wunneeded-internal-declaration"
#include "core/residual_mix.hpp"
#pragma clang diagnostic pop

#include <hip/hip_runtime.h>

#include <algorithm>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <random>
#include <vector>

namespace {

namespace mix = strix::core::residual_mix;

bool ok(hipError_t status) { return status == hipSuccess; }

std::vector<float> values(std::size_t count, std::mt19937& random, float range, bool specials) {
    std::uniform_real_distribution<float> value(-range, range);
    std::vector<float> out(count);
    for (auto& x : out) x = value(random);
    if (specials) {
        const std::uint32_t bits[] = {0x7F800000U, 0x7FC00000U, 0x80000000U, 0x00000001U, 0x7F7FFFFFU};
        for (std::size_t i = 11; i < out.size(); i += 50021U) std::memcpy(&out[i], &bits[random() % 5U], sizeof(float));
    }
    return out;
}

// 1 = equal, 0 = differ / error.
int run(unsigned rows, unsigned width, unsigned banks, unsigned queries, bool normalized, std::mt19937& random,
        std::size_t& compared) {
    const std::size_t elements = std::size_t{rows} * width;
    std::vector<float*> device;
    const auto upload = [&](const std::vector<float>& host) -> float* {
        float* p = nullptr;
        if (!ok(hipMalloc(&p, host.size() * sizeof(float)))) return nullptr;
        device.push_back(p);
        return ok(hipMemcpy(p, host.data(), host.size() * sizeof(float), hipMemcpyHostToDevice)) ? p : nullptr;
    };
    const auto zeros = [&](std::size_t count, int byte) -> float* {
        float* p = nullptr;
        if (!ok(hipMalloc(&p, count * sizeof(float)))) return nullptr;
        device.push_back(p);
        return ok(hipMemset(p, byte, count * sizeof(float))) ? p : nullptr;
    };
    mix::MixArgs loop{}, regs{};
    bool ready = true;
    for (unsigned b = 0; b < banks; ++b) {
        loop.banks[b] = regs.banks[b] = upload(values(elements, random, b == banks - 1U ? 4.0F : 1.5F, b == 0U));
        ready = ready && loop.banks[b];
    }
    loop.queries = regs.queries = upload(values(std::size_t{queries} * width, random, 0.3F, false));
    for (unsigned q = 0; q < queries; ++q) {
        loop.output[q] = zeros(elements, 0);
        regs.output[q] = zeros(elements, 0x5A);
        ready = ready && loop.output[q] && regs.output[q];
    }
    loop.rows = regs.rows = rows;
    loop.width = regs.width = width;
    loop.bank_count = regs.bank_count = banks;
    loop.query_count = regs.query_count = queries;
    loop.epsilon = regs.epsilon = 1e-6F;
    loop.normalized = regs.normalized = normalized;
    int result = 0;
    const bool admitted = [&] {
        using namespace mix;
        using namespace mix::detail;
        return fits(loop);
    }();
    if (ready && loop.queries && admitted) {
        const auto blocks = std::min(rows, 65535U);
        if (normalized) {
            mix::detail::mix<true, false, mix::MixArgs, false><<<blocks, mix::detail::kThreads>>>(loop);
            mix::detail::mix<true, false, mix::MixArgs, true><<<blocks, mix::detail::kThreads>>>(regs);
        } else {
            mix::detail::mix<false, false, mix::MixArgs, false><<<blocks, mix::detail::kThreads>>>(loop);
            mix::detail::mix<false, false, mix::MixArgs, true><<<blocks, mix::detail::kThreads>>>(regs);
        }
        bool same = ok(hipGetLastError()) && ok(hipDeviceSynchronize());
        std::vector<float> a(elements), b(elements);
        for (unsigned q = 0; same && q < queries; ++q) {
            same = ok(hipMemcpy(a.data(), loop.output[q], elements * sizeof(float), hipMemcpyDeviceToHost)) &&
                   ok(hipMemcpy(b.data(), regs.output[q], elements * sizeof(float), hipMemcpyDeviceToHost)) &&
                   std::memcmp(a.data(), b.data(), elements * sizeof(float)) == 0;
        }
        if (same) {
            result = 1;
            compared += elements * queries;
        }
    }
    for (float* p : device) (void)hipFree(p);
    return result;
}

}  // namespace

int main() {
    std::mt19937 random(20261124U);
    std::size_t compared = 0;
    unsigned cases = 0;
    // (rows, width, banks, queries, normalized): the Inkling readouts (width 2304, 2..5 banks, 1 or 3 queries; the
    // controller's 2 normalized banks), every bank count, a partial strided lane, the register bound and one past it.
    const struct { unsigned rows, width, banks, queries; bool normalized; } shapes[] = {
        {300U, 2304U, 2U, 1U, false}, {300U, 2304U, 5U, 3U, false}, {129U, 2304U, 2U, 1U, true},
        {64U, 2304U, 8U, 2U, false},  {37U, 1000U, 3U, 1U, false},  {33U, 4096U, 4U, 1U, false},
        {17U, 4224U, 3U, 2U, false},  {9U, 256U, 1U, 1U, false},    {5U, 2304U, 7U, 3U, true},
    };
    for (const auto& s : shapes) {
        ++cases;
        if (run(s.rows, s.width, s.banks, s.queries, s.normalized, random, compared) != 1) {
            std::printf("mix-serial-registers: FAIL rows %u width %u banks %u queries %u normalized %d\n", s.rows,
                        s.width, s.banks, s.queries, s.normalized ? 1 : 0);
            return 1;
        }
    }
    std::printf("mix-serial-registers: EXACT PASS cases=%u values=%zu (register form vs loop form, bitwise)\n", cases,
                compared);
    return 0;
}
