#include "strix/hip/legacy_intrinsics.hpp"
#include "strix/hip/legacy_launch.hpp"
#include "moe_core/expert_activation.hpp"

#include <array>
#include <cstdlib>
#include <iostream>
#include <memory>

namespace {

namespace activation = strix::moe_core::expert_activation;
namespace bf16 = strix::weight::scalar_float;
constexpr std::size_t rows = 3, width = 259;
constexpr std::size_t gate_up_row = 2U * width + 13U, output_row = width + 7U;
constexpr float gate_cap = 4.0F, up_cap = 25.0F;
constexpr std::uint16_t sentinel = 0xdeadU;

struct HostMath {
    static float exp(float value) { return std::exp(value); }
    static float tanh(float value) { return std::tanh(value); }
};
constexpr strix::moe::SoftcappedSwiglu<HostMath> oracle{gate_cap, up_cap};

struct Fixture {
    std::array<float, rows * gate_up_row> gate_up{};
    std::array<std::uint16_t, rows * output_row> output{};
};

void require_hip(hipError_t status) {
    if (status == hipSuccess) return;
    std::cerr << hipGetErrorString(status) << '\n';
    std::exit(1);
}

float as_bf16(float value) {
    return bf16::u2f(std::uint32_t{bf16::bf16_encode(value)} << 16U);
}

}  // namespace

int main() {
    constexpr std::array gate{1.00390625F, -8.5F, 0.0F, -0.0F, 33.625F, -90.0F, 3.5F, -0.25F, 15.43F};
    constexpr std::array up{2.0078125F, -90.0F, -3.0F, 2.0F, -0.0F, -1.0F, 30.0F, -3.001F, 1.13F};
    void* storage = nullptr;
    require_hip(hipMallocManaged(&storage, sizeof(Fixture)));
    auto* fixture = std::construct_at(static_cast<Fixture*>(storage));
    fixture->output.fill(sentinel);
    for (std::size_t row = 0; row < rows; ++row) {
        for (std::size_t col = 0; col < width; ++col) {
            fixture->gate_up[row * gate_up_row + col] = gate[(row + col) % gate.size()];
            fixture->gate_up[row * gate_up_row + width + col] = up[(row + col) % up.size()];
        }
    }
    const activation::Args args{fixture->gate_up.data(), fixture->output.data(), rows, width, gate_up_row, output_row};
    hipStream_t stream{};
    require_hip(hipStreamCreate(&stream));
    require_hip(activation::launch(args, gate_cap, up_cap, stream));
    require_hip(hipStreamSynchronize(stream));

    bool passed = true;
    for (std::size_t row = 0; row < rows; ++row) {
        for (std::size_t col = 0; col < output_row; ++col) {
            const auto gv = as_bf16(gate[(row + col) % gate.size()]);
            const auto uv = as_bf16(up[(row + col) % up.size()]);
            const auto expected = col < width ? bf16::bf16_encode(oracle(gv, uv)) : sentinel;
            passed = passed && fixture->output[row * output_row + col] == expected;
        }
    }
    // This row changes if the two preactivation BF16 roundings are omitted.
    passed = passed && fixture->output[0] != bf16::bf16_encode(oracle(gate[0], up[0]));
    if (!passed) std::cerr << "Packed expert activation differs from the BF16 host reference\n";

    auto invalid = args;
    invalid.gate_up_row = 2U * width - 1U;
    const bool rejects = activation::launch(invalid, gate_cap, up_cap, stream) == hipErrorInvalidValue &&
                         activation::launch(args, 0.0F, up_cap, stream) == hipErrorInvalidValue &&
                         activation::launch({}, gate_cap, up_cap, stream) == hipSuccess;
    require_hip(hipStreamDestroy(stream));
    std::destroy_at(fixture);
    require_hip(hipFree(fixture));
    return passed && rejects ? 0 : 1;
}
