// Native per-row fused-boundary qualification: R17 + R7 vs split Tier A, both rank orientations.
#include "glm/xfer/glm_post_exchange.hpp"
#include "glm/kernels/glm_tier_a.hpp"

#include <hip/hip_runtime.h>

#include <cstdio>
#include <cstring>
#include <limits>
#include <stdexcept>
#include <string>
#include <vector>

namespace {
namespace X = strix::glm::xfer;
namespace D = strix::glm::dev;
constexpr unsigned H = 4096, C = 4, M = 24;
void ok(hipError_t e) {
    if (e != hipSuccess) throw std::runtime_error(hipGetErrorString(e));
}
template<class T> struct Buffer {
    T* p = nullptr;
    explicit Buffer(std::size_t n) { ok(hipMalloc(reinterpret_cast<void**>(&p), n * sizeof(T))); }
    ~Buffer() { (void)hipFree(p); }
    Buffer(const Buffer&) = delete;
    Buffer& operator=(const Buffer&) = delete;
};
__global__ void sum_rank_order(const float* local, const float* peer, float* y, unsigned rank, unsigned n) {
    const unsigned i = blockIdx.x * 256 + threadIdx.x;
    if (i < n) y[i] = rank == 0 ? local[i] + peer[i] : peer[i] + local[i];
}
void run(unsigned rows, unsigned rank, unsigned poisoned, unsigned source) {
    const std::size_t nx = static_cast<std::size_t>(rows) * C * H, np = static_cast<std::size_t>(rows) * H;
    std::vector<float> x(nx), mix(static_cast<std::size_t>(rows) * M), local(np), peer(np);
    for (std::size_t i = 0; i < nx; ++i) x[i] = static_cast<float>(static_cast<int>(i % 71) - 35) * 0.031371F;
    for (std::size_t i = 0; i < np; ++i) {
        local[i] = static_cast<float>(static_cast<int>(i % 19) - 9) * 0.062731F;
        peer[i] = static_cast<float>(static_cast<int>(i % 13) - 6) * 0.125117F;
    }
    for (unsigned r = 0; r < rows; ++r) for (unsigned i = 0; i < M; ++i)
        mix[static_cast<std::size_t>(r) * M + i] = static_cast<float>(i % 5 + 1) * 0.125231F;
    if (poisoned < rows) {
        const float inf = std::numeric_limits<float>::infinity();
        if (source == 0) local[static_cast<std::size_t>(poisoned) * H + 17] = inf;
        else if (source == 1) peer[static_cast<std::size_t>(poisoned) * H + 31] = inf;
        else x[static_cast<std::size_t>(poisoned) * C * H + 47] = inf;
    }
    Buffer<float> dx(nx), dm(mix.size()), dl(np), dp(np), dy(np), split(nx), fused(nx), bracket(nx);
    // Guard words and pre-existing bits verify attribution and OR semantics.
    std::vector<unsigned> initial(rows + 2, 0x40U), sf(rows + 2), ff(rows + 2);
    initial.front() = initial.back() = 0xa5a5U;
    Buffer<unsigned> dsf(initial.size()), dff(initial.size());
    ok(hipMemcpy(dx.p, x.data(), nx * sizeof(float), hipMemcpyHostToDevice));
    ok(hipMemcpy(dm.p, mix.data(), mix.size() * sizeof(float), hipMemcpyHostToDevice));
    ok(hipMemcpy(dl.p, local.data(), np * sizeof(float), hipMemcpyHostToDevice));
    ok(hipMemcpy(dp.p, peer.data(), np * sizeof(float), hipMemcpyHostToDevice));
    ok(hipMemcpy(dsf.p, initial.data(), initial.size() * sizeof(unsigned), hipMemcpyHostToDevice));
    sum_rank_order<<<(rows * H + 255) / 256, 256>>>(dl.p, dp.p, dy.p, rank, rows * H);
    ok(hipGetLastError());
    D::hc_expand(dx.p, dm.p, dy.p, split.p, dsf.p + 1, rows, nullptr, 1);
    std::vector<float> expected(nx), actual(nx);
    ok(hipMemcpy(expected.data(), split.p, nx * sizeof(float), hipMemcpyDeviceToHost));
    ok(hipMemcpy(sf.data(), dsf.p, sf.size() * sizeof(unsigned), hipMemcpyDeviceToHost));
    if (sf.front() != initial.front() || sf.back() != initial.back()) throw std::runtime_error("split flag guard");
    for (unsigned r = 0; r < rows; ++r)
        if (sf[r + 1] != (0x40U | (r == poisoned ? 1U : 0U))) throw std::runtime_error("split row attribution");
    Buffer<std::uint64_t> ready(1), gate(1);
    const std::uint64_t sequence = 1;
    ok(hipMemcpy(ready.p, &sequence, sizeof(sequence), hipMemcpyHostToDevice));
    const X::RxView none{};
    for (unsigned wait = 0; wait < 2; ++wait) for (unsigned form = 0; form < 2; ++form) {
        ok(hipMemset(gate.p, 0, sizeof(sequence)));
        const X::RxView view = wait == 0 ? none : X::RxView{dp.p, ready.p, sequence, gate.p};
        const float* remote = wait == 0 ? dp.p : nullptr;
        ok(hipMemcpy(dff.p, initial.data(), initial.size() * sizeof(unsigned), hipMemcpyHostToDevice));
        if (form == 0) X::post_expand(dx.p, dm.p, dl.p, remote, view, rank, fused.p, dff.p + 1, rows, nullptr, 1);
        else {
            X::hc_bracket(dx.p, dm.p, bracket.p, rows, nullptr);
            X::post_expand_bracket(bracket.p, dm.p, dl.p, remote, view, rank, fused.p, dff.p + 1, rows, nullptr, 1);
        }
        ok(hipMemcpy(actual.data(), fused.p, nx * sizeof(float), hipMemcpyDeviceToHost));
        ok(hipMemcpy(ff.data(), dff.p, ff.size() * sizeof(unsigned), hipMemcpyDeviceToHost));
        if (std::memcmp(expected.data(), actual.data(), nx * sizeof(float)) != 0 || sf != ff)
            throw std::runtime_error("fused/split output bits or per-row flags differ");
    }
}
}  // namespace

int main() {
    try {
        for (unsigned rows = 1; rows <= 8; ++rows) for (unsigned rank = 0; rank < 2; ++rank)
            for (unsigned poisoned = 0; poisoned <= rows; ++poisoned) for (unsigned source = 0; source < 3; ++source)
                run(rows, rank, poisoned, source);
        std::puts("post-exchange stride-1: all output bits and row flags match split");
        return 0;
    } catch (const std::exception& e) {
        std::fprintf(stderr, "post-exchange stride-1 FAIL: %s\n", e.what());
        return 1;
    }
}
