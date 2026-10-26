// Exact residual fusions against the standalone GPU operators, including untouched route padding.
#include "strix/hip_ds41_layer.hpp"
#include "strix/hip_ds41_formats.hpp"
#include <hip/hip_runtime.h>
#include <bit>
#include <cstdio>
#include <cstring>
#include <limits>
#include <random>
#include <stdexcept>
#include <vector>

namespace {
using namespace strix::ds41;
void check(hipError_t e) { if (e != hipSuccess) throw std::runtime_error(hipGetErrorString(e)); }
template<class T> struct Dev {
    T* p = nullptr; std::size_t n;
    explicit Dev(std::size_t size) : n(size) { check(hipMalloc(&p, n * sizeof(T))); check(hipMemset(p, 0xA5, n * sizeof(T))); }
    explicit Dev(const std::vector<T>& v) : Dev(v.size()) { put(v); }
    ~Dev() { (void)hipFree(p); }
    void put(const std::vector<T>& v) { check(hipMemcpy(p, v.data(), n * sizeof(T), hipMemcpyHostToDevice)); }
    std::vector<T> get() const { std::vector<T> v(n); check(hipMemcpy(v.data(), p, n * sizeof(T), hipMemcpyDeviceToHost)); return v; }
    Dev(const Dev&) = delete;
};
template<class T> void equal(const std::vector<T>& a, const std::vector<T>& b) {
    if (a.size() != b.size() || std::memcmp(a.data(), b.data(), a.size() * sizeof(T))) throw std::runtime_error("bitwise mismatch");
}
std::mt19937 rng(41);
std::vector<float> random(std::size_t n) {
    std::vector<float> v(n); for (auto& x : v) x = static_cast<float>(static_cast<int>(rng() % 2049U) - 1024) / 512.0F; return v;
}
BuildRoutesView routes(Dev<std::uint32_t>& data, const Dev<std::uint32_t>& ids, unsigned rows, unsigned k, unsigned first, unsigned end) {
    const unsigned n = rows * k;
    BuildRoutesView v;
    v.ids = ids.p; v.rows = rows; v.top_k = k; v.split = 192; v.first = first; v.end = end; v.base = first;
    v.experts = data.p; v.begins = data.p + n; v.rows_out = data.p + 2 * n + 1;
    v.slots = data.p + 3 * n + 1; v.order = data.p + 4 * n + 1;
    v.low_counts = data.p + 5 * n + 1; v.counts = data.p + 5 * n + 1 + rows;
    return v;
}
void router_case(unsigned rows, unsigned kind, unsigned first, unsigned end) {
    constexpr unsigned h = 256, experts = 384, k = 6;
    auto x = random(rows * h), w = random(experts * h), bias = random(experts), image_bias = random(experts);
    if (kind) { for (auto& v : x) v = 0.0F; for (auto& v : bias) v = 0.0F; image_bias = bias; }
    if (kind == 2) { bias[0] = std::numeric_limits<float>::quiet_NaN(); bias[5] = std::numeric_limits<float>::infinity(); image_bias = bias; }
    std::vector<std::uint8_t> image(rows); for (unsigned r = 0; r < rows; ++r) image[r] = static_cast<std::uint8_t>(r % 2);
    Dev<float> dx(x), dw(w), db(bias), di(image_bias), scores0(rows * experts), scores1(rows * experts), weights0(rows*k), weights1(rows*k);
    Dev<std::uint8_t> dim(image);
    Dev<std::uint32_t> ids0(rows*k), ids1(rows*k), data0(5*rows*k + rows + 3), data1(5*rows*k + rows + 3);
    RouterView v;
    v.x = dx.p; v.weight = dw.p; v.bias = db.p; v.bias_image = di.p; v.image_rows = dim.p;
    v.rows = rows; v.hidden = h; v.experts = experts; v.top_k = k; v.route_scale = 2.5F;
    v.scores = scores0.p; v.ids = ids0.p; v.weights = weights0.p;
    enqueue_router(v, nullptr);
    auto a = routes(data0, ids0, rows, k, first, end); enqueue_build_routes(a, nullptr);
    v.scores = scores1.p; v.ids = ids1.p; v.weights = weights1.p;
    auto b = routes(data1, ids1, rows, k, first, end);
    if (!enqueue_router_routes(v, b, nullptr)) throw std::runtime_error("unexpected fallback");
    check(hipDeviceSynchronize());
    equal(scores0.get(), scores1.get()); equal(weights0.get(), weights1.get()); equal(ids0.get(), ids1.get()); equal(data0.get(), data1.get());
    v.fast = 1;
    if (enqueue_router_routes(v, b, nullptr)) throw std::runtime_error("fast router fallback missing");
}
void norm_case(unsigned rows, unsigned n, unsigned kind) {
    auto x = random(rows * n), w = random(n);
    if (kind) for (unsigned i = 0; i < rows*n; ++i) x[i] = std::bit_cast<float>(i % 2 ? 0x80000000U : 0U);
    if (kind == 2) for (unsigned i = 0; i < rows*n; ++i) x[i] = std::bit_cast<float>(i % 2 ? 0x00800001U : 0x80000001U);
    Dev<float> a(x), b(x), dw(w), da(rows*n/32), db(rows*n/32);
    Dev<std::int8_t> ca(rows*n), cb(rows*n);
    Dev<std::int32_t> sa(rows*n/32), sb(rows*n/32);
    enqueue_rms_norm(a.p, dw.p, rows, n, n, 1e-20F, a.p, nullptr);
    enqueue_quantize_a8(a.p, n, rows, ca.p, da.p, sa.p, nullptr);
    NormView v; v.x = b.p; v.x_stride = n; v.weight = dw.p; v.rows = rows; v.n = n;
    v.out = b.p; v.out_stride = n; v.codes = cb.p; v.deltas = db.p; v.sums = sb.p; v.a8_stride = n;
    enqueue_norm(v, nullptr); check(hipDeviceSynchronize());
    equal(a.get(), b.get()); equal(ca.get(), cb.get()); equal(da.get(), db.get()); equal(sa.get(), sb.get());
}
}
int main() try {
    unsigned rc = 0, nc = 0;
    for (unsigned rows = 1; rows <= 8; ++rows) {
        for (unsigned kind = 0; kind < 3; ++kind) {
            for (auto range : {std::pair{0U,384U}, std::pair{192U,384U}, std::pair{384U,385U}}) {
                router_case(rows, kind, range.first, range.second); ++rc;
            }
            for (unsigned n : {512U,1280U,5120U}) { norm_case(rows,n,kind); ++nc; }
        }
    }
    std::printf("R1FUSE PASS router=%u norm=%u (all output bytes, R1..8, ties/NaN bias, image, ranges, zeros/subnormals)\n",rc,nc);
    return 0;
} catch (const std::exception& e) { std::fprintf(stderr,"FAIL %s\n",e.what()); return 1; }
