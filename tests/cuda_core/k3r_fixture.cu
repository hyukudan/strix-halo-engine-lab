#include "cuda/core/route_project.cuh"

#include <algorithm>
#include <bit>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <random>
#include <stdexcept>

namespace {

namespace core = strix::cuda::core;
using Math = strix::moe::RouteCodyWaiteExpV1;
using strix::moe::RouteScorePlanes;
constexpr unsigned kHidden = 6144, kExperts = 128, kRows = 19;
constexpr unsigned kPairs = kRows * kExperts;
constexpr float kGuard = -12345.0F;

#include "k3r_minimax_route_frozen.inc"

struct Inputs {
    float hidden[kRows * kHidden];
    float weight[kExperts * kHidden];
    float bias[kExperts];
};

struct Outputs {
    float frozen[2U * kPairs + 2U];
    float interleaved[2U * kPairs + 2U];
    float unbiased[kPairs + 2U];
    float biased[kPairs + 2U];
};

void check(cudaError_t status) {
    if (status != cudaSuccess) throw std::runtime_error(cudaGetErrorString(status));
}

void check_bits(float actual, float expected, const char* plane, unsigned index) {
    const auto got = std::bit_cast<std::uint32_t>(actual);
    const auto want = std::bit_cast<std::uint32_t>(expected);
    if (got == want) return;
    std::fprintf(stderr, "%s element %u: got %08x, expected %08x\n", plane, index, got, want);
    throw std::runtime_error("score bytes differ");
}

void fill_inputs(Inputs& in) {
    std::mt19937 rng(0x34552U);
    const auto draw = [&] { return static_cast<float>(static_cast<int>(rng() % 8192U) - 4096); };
    for (float& v : in.hidden) v = draw() / 4096.0F;
    for (float& v : in.weight) v = draw() / 32768.0F;
    for (float& v : in.bias) v = draw() / 8192.0F;
    std::fill_n(in.hidden, 2U * kHidden, 0.0F);
    in.hidden[kHidden] = 1.0F;
    in.weight[0] = 90.0F;
    in.weight[kHidden] = -90.0F;
    in.weight[2U * kHidden] = 87.0F;
    in.weight[3U * kHidden] = -88.0F;
}

template <unsigned Stride>
void launch(const Inputs& in, unsigned rows, RouteScorePlanes<Stride> out) {
    check(core::launch_route_sigmoid_projection<kHidden, kExperts, core::RouteStridedFmaXor32, Math>(
        in.hidden, in.weight, in.bias, rows, out, nullptr));
}

void compare(const Inputs& in, Outputs& out, unsigned rows) {
    std::fill_n(out.frozen, 2U * kPairs + 2U, kGuard);
    std::fill_n(out.interleaved, 2U * kPairs + 2U, kGuard);
    std::fill_n(out.unbiased, kPairs + 2U, kGuard);
    std::fill_n(out.biased, kPairs + 2U, kGuard);
    if (rows != 0U) {
        const auto warps = std::size_t{rows} * kExperts;
        frozen::router_logits_kernel<<<static_cast<unsigned>((warps * 32 + 255) / 256), 256>>>(
            in.hidden, in.weight, in.bias, rows, out.frozen + 1);
        check(cudaGetLastError());
    }
    launch(in, rows, RouteScorePlanes<2>{out.interleaved + 1, out.interleaved + 2});
    launch(in, rows, RouteScorePlanes<1>{out.unbiased + 1, out.biased + 1});
    check(cudaDeviceSynchronize());

    const unsigned pairs = rows * kExperts;
    for (unsigned i = 0; i < pairs; ++i) {
        const float unbiased = out.frozen[2U * i + 1U];
        const float biased = out.frozen[2U * i + 2U];
        check_bits(out.interleaved[2U * i + 1U], unbiased, "interleaved unbiased", i);
        check_bits(out.interleaved[2U * i + 2U], biased, "interleaved biased", i);
        check_bits(out.unbiased[i + 1U], unbiased, "unbiased", i);
        check_bits(out.biased[i + 1U], biased, "biased", i);
    }
    check_bits(out.interleaved[0], kGuard, "interleaved guard", 0U);
    check_bits(out.unbiased[0], kGuard, "unbiased guard", 0U);
    check_bits(out.biased[0], kGuard, "biased guard", 0U);
    for (unsigned i = 2U * pairs + 1U; i < 2U * kPairs + 2U; ++i)
        check_bits(out.interleaved[i], kGuard, "interleaved tail", i);
    for (unsigned i = pairs + 1U; i < kPairs + 2U; ++i) {
        check_bits(out.unbiased[i], kGuard, "unbiased tail", i);
        check_bits(out.biased[i], kGuard, "biased tail", i);
    }
}

}  // namespace

int main() {
    try {
        Inputs* in = nullptr;
        Outputs* out = nullptr;
        check(cudaMallocManaged(&in, sizeof(Inputs)));
        check(cudaMallocManaged(&out, sizeof(Outputs)));
        fill_inputs(*in);
        for (const unsigned rows : {0U, 1U, kRows}) compare(*in, *out, rows);
        check(cudaFree(out));
        check(cudaFree(in));
        std::puts("k3r MiniMax unbiased and biased score bytes PASS rows=1,19 layouts=interleaved,split");
        std::puts("k3r MiniMax output guards and zero rows PASS");
    } catch (const std::exception& error) {
        std::fprintf(stderr, "k3r CUDA FAIL: %s\n", error.what());
        return 1;
    }
}
