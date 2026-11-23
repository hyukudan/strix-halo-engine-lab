#include "cuda/core/route_select.cuh"
#include "mmsel_cases.hpp"

namespace {

using namespace mmsel;
namespace core = strix::cuda::core;

struct Results {
    Outputs expected, paired, split;
};

void check(cudaError_t status) {
    if (status != cudaSuccess) throw std::runtime_error(cudaGetErrorString(status));
}

__global__ void frozen_router_top4_kernel(const float* scratch, std::size_t rows, std::uint32_t* ids, float* weights) {
    const std::size_t r = static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (r >= rows) return;
    frozen::router_top4_row(scratch, r, ids, weights);
}

void compare_fixture(Inputs& in, Results& out, unsigned rows) {
    out.expected.reset();
    out.paired.reset();
    out.split.reset();
    if (rows != 0U) {
        frozen_router_top4_kernel<<<(rows + 127U) / 128U, 128U>>>(
            in.interleaved, rows, out.expected.ids + 1U, out.expected.weights + 1U);
        check(cudaGetLastError());
    }
    check(core::launch_route_score_planes<Policy>(in.paired(), rows, out.paired.ids + 1U, out.paired.weights + 1U,
                                                  nullptr));
    check(core::launch_route_score_planes<Policy>(in.split(), rows, out.split.ids + 1U, out.split.weights + 1U,
                                                  nullptr));
    check(cudaDeviceSynchronize());
    check_known_rows(out.expected, rows);
    compare(out.expected, out.expected, rows, "frozen guards");
    compare(out.paired, out.expected, rows, "interleaved");
    compare(out.split, out.expected, rows, "split");
}

}  // namespace

int main() {
    try {
        const moe::RouteScorePlanes<2> empty{};
        check(core::launch_route_score_planes<Policy>(empty, 0U, nullptr, nullptr, nullptr));
        if (core::launch_route_score_planes<Policy>(empty, 1U, nullptr, nullptr, nullptr) != cudaErrorInvalidValue)
            throw std::runtime_error("invalid nonempty launch was accepted");
        Inputs* in = nullptr;
        Results* out = nullptr;
        check(cudaMallocManaged(&in, sizeof(Inputs)));
        check(cudaMallocManaged(&out, sizeof(Results)));
        fill_inputs(*in);
        for (const unsigned rows : {0U, 1U, kRows}) compare_fixture(*in, *out, rows);
        check(cudaFree(out));
        check(cudaFree(in));
        std::puts("mmsel CUDA ID and weight bytes PASS rows=1,259 layouts=interleaved,split");
        std::puts("mmsel CUDA ties, NaN/Inf, output guards and zero rows PASS");
    } catch (const std::exception& error) {
        std::fprintf(stderr, "mmsel CUDA FAIL: %s\n", error.what());
        return 1;
    }
}
