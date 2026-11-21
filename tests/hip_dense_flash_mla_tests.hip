// Device self-test of the shared dense flash core's MLA traits (dense_flash_prefill.hpp: row-major V, shared rope key,
// sink), the expanded-MHA prefill form of a sliding MLA layer (Hy4 SwaMhaPolicy shape: D 256 = 192 own + 64 shared rope,
// one head per group, 64 positions per work-group, sliding window, keys from a 16-aligned k0 > 0).
// (1) traits without sink == the classic policy (full K rows, V^T rows) on the same values: BITWISE (same LDS contents);
// (2) traits with sink vs FP64 attention over the operand-rounded values with the sink column: error <= 2^-6 max |v|;
// (3) the classic policy vs FP64 without sink, same bound (the reference itself).
// Run strix-dense-flash-mla-hip-tests explicitly (GPU); host CTest does not run it.
#include "strix/hip/legacy_intrinsics.hpp"
#include "strix/hip/legacy_launch.hpp"
#include "attn_core/dense_flash_prefill.hpp"

#include <hip/hip_runtime.h>

#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <random>
#include <stdexcept>
#include <string>
#include <vector>

namespace {
namespace dense = strix::attn_core::dense;
namespace tile = strix::attn_core::tile;

constexpr unsigned kD = 256U, kR = 64U, kN = kD - kR, kH = 4U;
constexpr float kScale = 0.0625F;

struct MArgs {
    const float* q;
    float* out;
    const std::uint16_t *k, *vt, *kn, *rope, *v;  // classic: k [pos][h][256], vt [h][d][pos]; traits: kn, rope, v
    const float* sinks;
    unsigned* failure;
    unsigned rows, p0, k0, window, vstride;
};
struct Common {
    using Args = MArgs;
    using Shape = dense::Shape<kD, 1U, 2U, 64U>;
    static constexpr float kScaleLog2 = kScale * 1.4426950408889634F;
    static constexpr tile::Operand kOp = tile::Operand::bf16;
    __device__ static bool valid(const Args& a) { return a.rows != 0U && a.p0 >= a.k0; }
    __host__ __device__ static unsigned rows(const Args& a) { return a.rows; }
    __device__ static unsigned position(const Args& a) { return a.p0; }
    __device__ static unsigned window(const Args& a) { return a.window; }
    __device__ static const float* query(const Args& a, unsigned row, unsigned h) { return a.q + (std::size_t(row) * kH + h) * kD; }
    __device__ static float* out(const Args& a, unsigned row, unsigned h) { return a.out + (std::size_t(row) * kH + h) * kD; }
    __device__ static unsigned* failure(const Args& a) { return a.failure; }
};
struct Classic : Common {
    __device__ static const std::uint16_t* key(const Args& a, unsigned p, unsigned h) { return a.k + (std::size_t(p - a.k0) * kH + h) * kD; }
    __device__ static const std::uint16_t* value_t(const Args& a, unsigned h, unsigned d) {
        return a.vt + std::size_t(h * kD + d) * a.vstride - a.k0;  // element pos = V[pos][d], pos >= k0 (16-aligned)
    }
};
struct Traits : Common {
    static constexpr bool kValueRows = true;
    static constexpr unsigned kRope = kR;
    __device__ static const std::uint16_t* key(const Args& a, unsigned p, unsigned h) { return a.kn + (std::size_t(p - a.k0) * kH + h) * kN; }
    __device__ static const std::uint16_t* rope_key(const Args& a, unsigned p) { return a.rope + std::size_t(p - a.k0) * kR; }
    __device__ static const std::uint16_t* value(const Args& a, unsigned p, unsigned h) { return a.v + (std::size_t(p - a.k0) * kH + h) * kD; }
};
struct TraitsSink : Traits {
    __device__ static float sink(const Args& a, unsigned, unsigned h) { return a.sinks[h]; }
};

std::uint16_t bf16(float x) {  // round to nearest even (finite)
    std::uint32_t u;
    std::memcpy(&u, &x, 4);
    u += 0x7fffU + ((u >> 16U) & 1U);
    return static_cast<std::uint16_t>(u >> 16U);
}
float from_bf16(std::uint16_t b) {
    const std::uint32_t u = static_cast<std::uint32_t>(b) << 16U;
    float x;
    std::memcpy(&x, &u, 4);
    return x;
}
void ok(hipError_t e, const char* what) {
    if (e != hipSuccess) throw std::runtime_error(std::string(what) + ": " + hipGetErrorString(e));
}
template <class T> T* up(const std::vector<T>& h) {
    T* d = nullptr;
    ok(hipMalloc(&d, h.size() * sizeof(T)), "malloc");
    ok(hipMemcpy(d, h.data(), h.size() * sizeof(T), hipMemcpyHostToDevice), "upload");
    return d;
}

int fails = 0;
void check(bool c, const char* what) {
    if (!c) {
        std::printf("FAIL %s\n", what);
        ++fails;
    }
}

void run(unsigned rows, unsigned p0, unsigned window) {
    const unsigned k0 = (p0 >= window ? p0 - window + 1U : 0U) & ~15U, keys = p0 + rows - k0 + 16U, vstride = keys + 16U;
    std::mt19937 g(rows * 131U + p0);
    std::uniform_real_distribution<float> u(-1.0F, 1.0F);
    std::vector<float> q(std::size_t(rows) * kH * kD);
    for (auto& x : q) x = from_bf16(bf16(u(g)));
    std::vector<std::uint16_t> kn(std::size_t(keys) * kH * kN), rope(std::size_t(keys) * kR), v(std::size_t(keys) * kH * kD);
    for (auto& x : kn) x = bf16(u(g));
    for (auto& x : rope) x = bf16(u(g));
    for (auto& x : v) x = bf16(u(g));
    std::vector<std::uint16_t> kfull(std::size_t(keys) * kH * kD), vt(std::size_t(kH) * kD * vstride, 0U);
    for (unsigned p = 0; p < keys; ++p)
        for (unsigned h = 0; h < kH; ++h)
            for (unsigned d = 0; d < kD; ++d) {
                kfull[(std::size_t(p) * kH + h) * kD + d] = d < kN ? kn[(std::size_t(p) * kH + h) * kN + d] : rope[std::size_t(p) * kR + d - kN];
                vt[std::size_t(h * kD + d) * vstride + p] = v[(std::size_t(p) * kH + h) * kD + d];
            }
    const std::vector<float> sinks{-2.0F, 0.0F, 1.5F, 4.0F};
    MArgs a{};
    a.q = up(q);
    a.k = up(kfull);
    a.vt = up(vt);
    a.kn = up(kn);
    a.rope = up(rope);
    a.v = up(v);
    a.sinks = up(sinks);
    a.failure = up(std::vector<unsigned>{0U});
    a.rows = rows;
    a.p0 = p0;
    a.k0 = k0;
    a.window = window;
    a.vstride = vstride;
    const std::size_t n = std::size_t(rows) * kH * kD;
    std::vector<float> oc(n), ot(n), os(n);
    float* out = nullptr;
    ok(hipMalloc(&out, n * 4U), "malloc out");
    a.out = out;
    dense::launch<Classic>(a, kH, nullptr);
    ok(hipMemcpy(oc.data(), out, n * 4U, hipMemcpyDeviceToHost), "classic");
    dense::launch<Traits>(a, kH, nullptr);
    ok(hipMemcpy(ot.data(), out, n * 4U, hipMemcpyDeviceToHost), "traits");
    dense::launch<TraitsSink>(a, kH, nullptr);
    ok(hipMemcpy(os.data(), out, n * 4U, hipMemcpyDeviceToHost), "traits sink");
    unsigned failure = 0U;
    ok(hipMemcpy(&failure, a.failure, 4U, hipMemcpyDeviceToHost), "failure");
    check(failure == 0U, "failure word");
    check(std::memcmp(oc.data(), ot.data(), n * 4U) == 0, "traits == classic bitwise");
    double worst_c = 0.0, worst_s = 0.0;
    for (unsigned r = 0; r < rows; ++r)
        for (unsigned h = 0; h < kH; ++h) {
            const unsigned pos = p0 + r, lo = pos + 1U >= window ? pos + 1U - window : 0U;
            std::vector<double> s(pos - lo + 1U);
            double m = -1e300;
            for (unsigned j = lo; j <= pos; ++j) {
                double dot = 0.0;
                for (unsigned d = 0; d < kD; ++d)
                    dot += double(q[(std::size_t(r) * kH + h) * kD + d]) * from_bf16(kfull[(std::size_t(j - k0) * kH + h) * kD + d]);
                s[j - lo] = dot * kScale;
                m = std::fmax(m, s[j - lo]);
            }
            const double ms = std::fmax(m, double(sinks[h]));
            double l = 0.0, ls = std::exp(double(sinks[h]) - ms);
            for (auto& x : s) l += std::exp(x - m), ls += std::exp(x - ms);
            for (unsigned d = 0; d < kD; ++d) {
                double o = 0.0, osk = 0.0;
                for (unsigned j = lo; j <= pos; ++j) {
                    const double vv = from_bf16(v[(std::size_t(j - k0) * kH + h) * kD + d]);
                    o += std::exp(s[j - lo] - m) / l * vv;
                    osk += std::exp(s[j - lo] - ms) / ls * vv;
                }
                const std::size_t at = (std::size_t(r) * kH + h) * kD + d;
                worst_c = std::fmax(worst_c, std::fabs(oc[at] - o));
                worst_s = std::fmax(worst_s, std::fabs(os[at] - osk));
            }
        }
    std::printf("rows %u p0 %u window %u k0 %u: |classic - fp64| %.3g, |traits+sink - fp64| %.3g (bound %.3g)\n", rows, p0, window, k0,
                worst_c, worst_s, 0x1p-6);
    check(worst_c <= 0x1p-6, "classic vs fp64");
    check(worst_s <= 0x1p-6, "traits + sink vs fp64");
    for (const void* p : {static_cast<const void*>(a.q), static_cast<const void*>(a.k), static_cast<const void*>(a.vt),
                          static_cast<const void*>(a.kn), static_cast<const void*>(a.rope), static_cast<const void*>(a.v),
                          static_cast<const void*>(a.sinks), static_cast<const void*>(a.failure), static_cast<const void*>(out)})
        ok(hipFree(const_cast<void*>(p)), "free");
}

}  // namespace

int main() {
    try {
        run(152U, 104U, 48U);  // partial last block, partial 16-key tail, k0 = 48 > 0
        run(256U, 512U, 512U);  // the Hy4 window, first halo: k0 = (512 - 511) & ~15 = 0
        run(128U, 2048U, 512U);  // a later window: k0 = 1536
    } catch (const std::exception& e) {
        std::printf("FAIL exception: %s\n", e.what());
        return 1;
    }
    std::printf("dense flash mla tests: %s (%d failures)\n", fails == 0 ? "PASS" : "FAIL", fails);
    return fails == 0 ? 0 : 1;
}
