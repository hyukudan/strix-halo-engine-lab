#include "cuda/core/chain/launch.cuh"
#include "strix/cuda/amd_math_tables.hpp"

#include <bit>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <exception>
#include <stdexcept>

namespace {

namespace ch = strix::cuda::core::chain;
namespace am = strix::cuda::amdmath;
using ch::Fold;
using ch::Form;
using ch::Math;

using Silu = ch::List<ch::ActivationStep<Form::act_silu_mul, 0, Math::gfx1151_v1>>;
using Gate = ch::List<ch::ActivationStep<Form::gate_sigmoid_mul, 0, Math::gfx1151_v1>>;
using Folded = ch::List<ch::ActivationStep<Form::gate_sigmoid_mul, 0, Math::gfx1151_v1, Fold::descending_upper_first>>;
using Native = ch::List<ch::ActivationStep<Form::gate_sigmoid_mul>>;
static_assert(ch::static_key<Native> == ch::kernel_key(Native::description));
static_assert(ch::static_key<Native> != ch::static_key<Gate>);
static_assert(ch::static_key<Gate> != ch::static_key<Folded>);
static_assert(ch::chain_scope<Gate> == ch::Scope::element);
static_assert(ch::chain_scope<Folded> == ch::Scope::row);

constexpr unsigned kRows = 4;
constexpr unsigned kWidth = 513;
constexpr unsigned kFeatures = 769;
constexpr unsigned kStride = kFeatures + 7;
constexpr unsigned kValues = kRows * kWidth;

struct Results {
    float silu[kValues];
    float element[kValues];
    float folded[kValues];
    float sigmoid[kRows];
};

struct Data {
    float gate[kValues];
    float up[kValues];
    std::uint16_t input[kRows * kStride];
    float weights[kFeatures];
    Results shared;
    Results frozen;
    ch::Gfx1151Silu silu;
    ch::Gfx1151Gate element;
    ch::Gfx1151FoldedGate folded;
};

// Arithmetic frozen from shared_activation_kernel. The same probe tables
// supply expf for both arms. Separate inputs and outputs expose each stage.
__global__ __launch_bounds__(256) void frozen_activation(Data* data, am::Tables tables) {
    __shared__ float scratch[256];
    const unsigned row = blockIdx.y;
    const unsigned index = blockIdx.x * blockDim.x + threadIdx.x;
    const unsigned at = row * kWidth + index;
    if (index < kWidth) {
        const float gate = data->gate[at];
        const float up = data->up[at];
        const float den = __fadd_rn(1.0F, am::amd_expf(tables, -gate));
        data->frozen.silu[at] = __fmul_rn(__fdiv_rn(gate, den), up);
        data->frozen.element[at] = __fmul_rn(up, __fdiv_rn(1.0F, den));
    }
    if (blockIdx.x != 0U) return;
    const auto* input = data->input + row * kStride;
    float sum = 0.0F;
    for (unsigned feature = threadIdx.x; feature < kFeatures; feature += 256U)
        sum = __fmaf_rn(data->weights[feature], __uint_as_float(std::uint32_t{input[feature]} << 16U), sum);
    scratch[threadIdx.x] = sum;
    __syncthreads();
    for (unsigned stride = 128U; stride > 0U; stride >>= 1U) {
        if (threadIdx.x < stride) scratch[threadIdx.x] = __fadd_rn(scratch[threadIdx.x + stride], scratch[threadIdx.x]);
        __syncthreads();
    }
    if (threadIdx.x == 0U)
        data->frozen.sigmoid[row] = __fdiv_rn(1.0F, __fadd_rn(1.0F, am::amd_expf(tables, -scratch[0])));
}

__global__ void frozen_multiply(Data* data) {
    const unsigned i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < kValues) data->frozen.folded[i] = __fmul_rn(data->frozen.silu[i], data->frozen.sigmoid[i / kWidth]);
}

void check(cudaError_t status) {
    if (status != cudaSuccess) throw std::runtime_error(cudaGetErrorString(status));
}

template <std::size_t N>
void compare(const char* name, const float (&actual)[N], const float (&expected)[N]) {
    for (std::size_t i = 0; i < N; ++i) {
        const auto got = std::bit_cast<std::uint32_t>(actual[i]);
        const auto want = std::bit_cast<std::uint32_t>(expected[i]);
        if (got == want) continue;
        std::fprintf(stderr, "%s[%zu]: got %08x, expected %08x\n", name, i,
                     static_cast<unsigned>(got), static_cast<unsigned>(want));
        throw std::runtime_error("bitwise mismatch");
    }
}

template <class Chain>
void launch(const float* input, float* output, const void* binding) {
    ch::ChainArgs args{};
    args.aux[0] = binding;
    check(ch::launch<3>(Chain{}, {input, output, kRows, kWidth}, args, nullptr));
}

void run(const am::Tables& tables) {
    Data* data = nullptr;
    check(cudaMallocManaged(&data, sizeof(Data)));
    for (unsigned i = 0; i < kValues; ++i) {
        data->gate[i] = static_cast<float>(static_cast<int>(i % 161U) - 80) * 0.3125F;
        data->up[i] = static_cast<float>(static_cast<int>(i % 37U) - 18) * 0.125F;
    }
    data->gate[0] = -0.0F;
    data->gate[1] = -100.0F;
    data->gate[2] = 100.0F;
    for (unsigned i = 0; i < kFeatures; ++i)
        data->weights[i] = static_cast<float>(static_cast<int>(i % 17U) - 8) / 47.0F;
    data->weights[0] = 0x1p24F;
    data->weights[1] = 1.0F;
    data->weights[128] = -0x1p24F;
    for (unsigned row = 0; row < kRows; ++row) {
        for (unsigned i = 0; i < kStride; ++i) {
            constexpr std::uint16_t samples[]{0x3f80U, 0xbf80U, 0x3f01U, 0x3e88U, 0x0000U, 0x8000U};
            data->input[row * kStride + i] = row == 0U
                ? ((i == 0U || i == 1U || i == 128U) ? std::uint16_t{0x3f80U} : std::uint16_t{0U})
                : samples[(i / 7U + row) % 6U];
        }
    }
    data->silu = {data->up, tables};
    data->element = {{data->gate, 0.0F, false}, tables};
    data->folded = {data->input, data->weights, kStride, kFeatures, tables, data->shared.sigmoid};

    frozen_activation<<<dim3((kWidth + 255U) / 256U, kRows), 256>>>(data, tables);
    check(cudaGetLastError());
    frozen_multiply<<<(kValues + 255U) / 256U, 256>>>(data);
    check(cudaGetLastError());
    launch<Silu>(data->gate, data->shared.silu, &data->silu);
    launch<Gate>(data->up, data->shared.element, &data->element);
    launch<Folded>(data->shared.silu, data->shared.folded, &data->folded);
    check(cudaDeviceSynchronize());
    if (!(data->frozen.sigmoid[0] > 0.7F && data->frozen.sigmoid[0] < 0.8F))
        throw std::runtime_error("cancellation row did not produce sigmoid(1)");
    compare("SiLU", data->shared.silu, data->frozen.silu);
    compare("element gate", data->shared.element, data->frozen.element);
    compare("folded sigmoid", data->shared.sigmoid, data->frozen.sigmoid);
    compare("folded gate", data->shared.folded, data->frozen.folded);
    check(cudaFree(data));
}

}  // namespace

int main(int argc, char** argv) {
    if (argc != 2) {
        std::fprintf(stderr, "usage: strix-n1-gfx1151-cuda-fixture TABLES_DIR\n");
        return 2;
    }
    try {
        auto tables = am::load_tables(argv[1]);
        run(tables);
        am::free_tables(tables);
        std::puts("N1 gfx1151 CUDA: SiLU, element gate and 256-lane folded gate EXACT");
    } catch (const std::exception& error) {
        std::fprintf(stderr, "N1 gfx1151 CUDA: %s\n", error.what());
        return 1;
    }
}
