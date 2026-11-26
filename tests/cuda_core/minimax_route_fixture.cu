#include "cuda/models/minimax/mm_rtx_kernels.cuh"

#include "cuda/core/route_project.cuh"
#include "cuda/core/route_select.cuh"

#include <algorithm>
#include <bit>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <random>
#include <stdexcept>

namespace {

namespace mm = strix::minimax::rtxpf::cuda;
namespace core = strix::cuda::core;
namespace moe = strix::moe;
using mm::kExperts;
using mm::kHidden;
constexpr unsigned kRows = 19;
constexpr unsigned kPlaneSize = kRows * kExperts;
constexpr unsigned kScores = kPlaneSize * 2U;
constexpr unsigned kPicks = kRows * mm::kTopE;
constexpr float kFloatGuard = -12345.0F;
constexpr std::uint32_t kIdGuard = 0xffffffffU;

#include "k3r_minimax_route_frozen.inc"
#include "mmsel_minimax_frozen.inc"

struct Selection {
    std::uint32_t ids[kPicks + 2U];
    float weights[kPicks + 2U];

    void reset() {
        std::fill_n(ids, kPicks + 2U, kIdGuard);
        std::fill_n(weights, kPicks + 2U, kFloatGuard);
    }
};

struct Buffers {
    float hidden[kRows * kHidden];
    float weights[kExperts * kHidden];
    float bias[kExperts];
    float frozen[kScores + 2U];
    float scores[kScores + 2U];
    float unbiased[kPlaneSize + 2U];
    float biased[kPlaneSize + 2U];
    Selection expected, interleaved, split;
};

void check(cudaError_t status) {
    if (status != cudaSuccess) throw std::runtime_error(cudaGetErrorString(status));
}

template <class T>
void check_bits(T actual, T expected, const char* plane, unsigned index) {
    const auto got = std::bit_cast<std::uint32_t>(actual);
    const auto want = std::bit_cast<std::uint32_t>(expected);
    if (got == want) return;
    std::fprintf(stderr, "%s element %u: got %08x, expected %08x\n", plane, index, got, want);
    throw std::runtime_error("router output bytes differ");
}

template <class T, std::size_t N>
void check_guards(const T (&buffer)[N], unsigned written, T guard, const char* name) {
    check_bits(buffer[0], guard, name, 0U);
    for (unsigned i = written + 1U; i < N; ++i) check_bits(buffer[i], guard, name, i);
}

void fill_inputs(Buffers& b) {
    std::mt19937 rng(0x34552U);
    const auto draw = [&] { return static_cast<float>(static_cast<int>(rng() % 8192U) - 4096); };
    for (float& value : b.hidden) value = draw() / 4096.0F;
    for (float& value : b.weights) value = draw() / 32768.0F;
    for (float& value : b.bias) value = draw() / 8192.0F;
    // Row 0 exercises bias-only selection and ties. Row 1 spans the exp cutoffs.
    std::fill_n(b.hidden, 2U * kHidden, 0.0F);
    b.bias[2] = b.bias[3] = 0.75F;
    b.bias[17] = b.bias[19] = 0.625F;
    b.hidden[kHidden] = 1.0F;
    b.weights[0] = 90.0F;
    b.weights[kHidden] = -90.0F;
    b.weights[2U * kHidden] = 87.0F;
    b.weights[3U * kHidden] = -88.0F;
}

__global__ void frozen_top4(const float* scores, unsigned rows, std::uint32_t* ids, float* weights) {
    const unsigned row = blockIdx.x * blockDim.x + threadIdx.x;
    if (row < rows) frozen::router_top4_row(scores, row, ids, weights);
}

void compare_selection(const Selection& actual, const Selection& expected, unsigned picks) {
    for (unsigned i = 1; i <= picks; ++i) {
        check_bits(actual.ids[i], expected.ids[i], "expert ID", i);
        check_bits(actual.weights[i], expected.weights[i], "expert weight", i);
    }
}

void compare(Buffers& b, unsigned rows) {
    std::fill_n(b.frozen, kScores + 2U, kFloatGuard);
    std::fill_n(b.scores, kScores + 2U, kFloatGuard);
    std::fill_n(b.unbiased, kPlaneSize + 2U, kFloatGuard);
    std::fill_n(b.biased, kPlaneSize + 2U, kFloatGuard);
    b.expected.reset();
    b.interleaved.reset();
    b.split.reset();
    const unsigned pairs = rows * kExperts;
    if (rows != 0U) {
        const unsigned blocks = (pairs + 7U) / 8U;
        frozen::router_logits_kernel<<<blocks, 256>>>(b.hidden, b.weights, b.bias, rows, b.frozen + 1);
        check(cudaGetLastError());
        frozen_top4<<<1, 128>>>(b.frozen + 1, rows, b.expected.ids + 1, b.expected.weights + 1);
        check(cudaGetLastError());
    }
    mm::route(b.hidden, b.weights, b.bias, rows, b.scores + 1, b.interleaved.ids + 1, b.interleaved.weights + 1, nullptr);
    check(cudaGetLastError());
    const moe::RouteScorePlanes<1> split{b.unbiased + 1, b.biased + 1};
    check(core::launch_route_sigmoid_projection<kHidden, kExperts, core::RouteStridedFmaXor32, moe::RouteCodyWaiteExpV1>(
        b.hidden, b.weights, b.bias, rows, split, nullptr));
    check(core::launch_route_score_planes<mm::RouterPolicy>(split, rows, b.split.ids + 1, b.split.weights + 1, nullptr));
    check(cudaGetLastError());
    check(cudaDeviceSynchronize());

    for (unsigned i = 0; i < pairs; ++i) {
        check_bits(b.scores[2U * i + 1U], b.frozen[2U * i + 1U], "interleaved unbiased score", i);
        check_bits(b.scores[2U * i + 2U], b.frozen[2U * i + 2U], "interleaved biased score", i);
        check_bits(b.unbiased[i + 1U], b.frozen[2U * i + 1U], "split unbiased score", i);
        check_bits(b.biased[i + 1U], b.frozen[2U * i + 2U], "split biased score", i);
    }
    const unsigned picks = rows * mm::kTopE;
    compare_selection(b.interleaved, b.expected, picks);
    compare_selection(b.split, b.expected, picks);
    check_guards(b.frozen, 2U * pairs, kFloatGuard, "frozen score guard");
    check_guards(b.scores, 2U * pairs, kFloatGuard, "interleaved score guard");
    check_guards(b.unbiased, pairs, kFloatGuard, "split unbiased score guard");
    check_guards(b.biased, pairs, kFloatGuard, "split biased score guard");
    for (const Selection* selection : {&b.expected, &b.interleaved, &b.split}) {
        check_guards(selection->ids, picks, kIdGuard, "expert ID guard");
        check_guards(selection->weights, picks, kFloatGuard, "expert weight guard");
    }
}

}  // namespace

int main() {
    try {
        Buffers* buffers = nullptr;
        check(cudaMallocManaged(&buffers, sizeof(Buffers)));
        fill_inputs(*buffers);
        for (const unsigned rows : {0U, 1U, kRows}) compare(*buffers, rows);
        // The first eligible NaN stays selected and a later one cannot win.
        buffers->bias[0] = buffers->bias[4] = std::bit_cast<float>(0x7fc00000U);
        compare(*buffers, 1U);
        check(cudaFree(buffers));
        std::puts("PASS MiniMax RTX K3R interleaved score bytes rows=1,19");
        std::puts("PASS MiniMax RTX K3R top-4 ID and weight bytes rows=1,19");
        std::puts("PASS MiniMax RTX K3R output guards and zero rows");
        std::puts("PASS MiniMax RTX MMSEL top-4 ID and weight bytes rows=1,19");
    } catch (const std::exception& error) {
        std::fprintf(stderr, "FAIL MiniMax RTX K3R: %s\n", error.what());
        return 1;
    }
}
