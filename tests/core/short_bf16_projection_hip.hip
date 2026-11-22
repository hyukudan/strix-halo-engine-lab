// ShortBf16Projection: the four-column form (Columns = 4) stores the same bits as the one-output form for every
// output, including non-finite, signed-zero and subnormal operands, and raises the finite-check flag identically.
#include "core/row_transforms.hpp"

#include <hip/hip_runtime.h>

#include <cstdint>
#include <cstdio>
#include <cstring>
#include <random>
#include <vector>

namespace {

namespace prep = strix::core::row_transforms;

bool ok(hipError_t status) { return status == hipSuccess; }

template <unsigned Inputs, bool Check, unsigned Columns>
bool run(const float* input, const std::uint16_t* weights, float* output, unsigned* failure, std::size_t lines,
         unsigned outputs, std::vector<float>& host, unsigned& flag) {
    using Projection = prep::ShortBf16Projection<Inputs, Check, Columns>;
    if (!Projection::fits(input, weights, output, outputs)) return false;
    if (!ok(hipMemset(output, 0xA5, lines * outputs * sizeof(float))) || !ok(hipMemset(failure, 0, sizeof(unsigned))))
        return false;
    if (!ok(prep::elements(Projection::launch_size(lines, outputs), Projection{input, weights, output, outputs, failure},
                           nullptr)))
        return false;
    host.resize(lines * outputs);
    return ok(hipMemcpy(host.data(), output, host.size() * sizeof(float), hipMemcpyDeviceToHost)) &&
           ok(hipMemcpy(&flag, failure, sizeof(unsigned), hipMemcpyDeviceToHost));
}

template <bool Check>
bool compare(const float* input, const std::uint16_t* weights, float* output, unsigned* failure, std::size_t lines,
             unsigned outputs, std::size_t& compared) {
    std::vector<float> narrow, wide;
    unsigned narrow_flag = 2U, wide_flag = 3U;
    if (!run<16U, Check, 1U>(input, weights, output, failure, lines, outputs, narrow, narrow_flag) ||
        !run<16U, Check, 4U>(input, weights, output, failure, lines, outputs, wide, wide_flag))
        return false;
    compared += narrow.size();
    return std::memcmp(narrow.data(), wide.data(), narrow.size() * sizeof(float)) == 0 && narrow_flag == wide_flag;
}

}  // namespace

int main() {
    constexpr unsigned kInputs = 16U;
    std::mt19937 random(20261122U);
    std::size_t compared = 0;
    bool pass = true;
    // (lines, outputs, special-value density): the Inkling relative-logit width plus a narrow and an odd line count.
    const struct { std::size_t lines; unsigned outputs; unsigned special; } cases[] = {
        {37U, 1024U, 0U}, {37U, 1024U, 64U}, {5U, 64U, 8U}, {96U, 4U, 0U}};
    for (const auto& c : cases) {
        std::vector<float> input(c.lines * kInputs);
        std::vector<std::uint16_t> weights(std::size_t{kInputs} * c.outputs);
        std::uniform_real_distribution<float> value(-4.0F, 4.0F);
        for (auto& x : input) x = value(random);
        for (auto& w : weights) {
            const float f = value(random);
            std::uint32_t bits;
            std::memcpy(&bits, &f, sizeof bits);
            w = static_cast<std::uint16_t>(bits >> 16U);
        }
        if (c.special != 0U) {
            const std::uint32_t specials[] = {0x7F800000U, 0xFF800000U, 0x7FC00000U, 0x80000000U, 0x00000001U,
                                              0x00400000U, 0x7F7FFFFFU, 0x80800000U};
            for (std::size_t i = 0; i < input.size(); i += c.special) {
                const std::uint32_t bits = specials[random() % 8U];
                std::memcpy(&input[i], &bits, sizeof bits);
            }
            for (std::size_t i = 3; i < weights.size(); i += c.special)
                weights[i] = static_cast<std::uint16_t>(specials[random() % 8U] >> 16U);
        }
        float* d_input = nullptr;
        float* d_output = nullptr;
        std::uint16_t* d_weights = nullptr;
        unsigned* d_failure = nullptr;
        pass = pass && ok(hipMalloc(&d_input, input.size() * sizeof(float))) &&
               ok(hipMalloc(&d_weights, weights.size() * sizeof(std::uint16_t))) &&
               ok(hipMalloc(&d_output, c.lines * c.outputs * sizeof(float))) && ok(hipMalloc(&d_failure, sizeof(unsigned))) &&
               ok(hipMemcpy(d_input, input.data(), input.size() * sizeof(float), hipMemcpyHostToDevice)) &&
               ok(hipMemcpy(d_weights, weights.data(), weights.size() * sizeof(std::uint16_t), hipMemcpyHostToDevice)) &&
               compare<false>(d_input, d_weights, d_output, d_failure, c.lines, c.outputs, compared) &&
               compare<true>(d_input, d_weights, d_output, d_failure, c.lines, c.outputs, compared);
        (void)hipFree(d_input);
        (void)hipFree(d_weights);
        (void)hipFree(d_output);
        (void)hipFree(d_failure);
        if (!pass) {
            std::printf("short-bf16-projection: FAIL lines %zu outputs %u special %u\n", c.lines, c.outputs, c.special);
            return 1;
        }
    }
    std::printf("short-bf16-projection: %zu outputs bitwise equal (Columns 4 vs 1, both finite-check forms)\n", compared);
    return 0;
}
