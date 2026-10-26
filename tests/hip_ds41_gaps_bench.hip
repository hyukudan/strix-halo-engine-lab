// ds41/gaps: exactness hashes and launch-chain timing of the small critical-path ops of one decode layer.
//   norm   : the in-place q norm (q_lora_rank 1280) as DeviceForward enqueues it, plain or with the A8 of q_b
//            (STRIX_DS41_R1FUSE_QNORM_A8) against norm + quantize_a8; STRIX_DS41_NORM_ILP=0/1 picks the kernel.
//   router : router scores + selection + route builder (production) against the fused selection + routes
//            (STRIX_DS41_R1FUSE_ROUTES), BF16 router weights (5120 x 384, top 6), EP range [0, 192).
// Every case prints an FNV-1a of all its output bytes: run once with each flag value and diff the hash lines.
// Times are per chain in a stream (hipEvent), warm, median of 7 batches of 200 chains.
#include "strix/hip_ds41_layer.hpp"
#include "strix/hip_ds41_formats.hpp"
#include <hip/hip_runtime.h>
#include <algorithm>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <functional>
#include <random>
#include <stdexcept>
#include <vector>

namespace {
using namespace strix::ds41;
void check(hipError_t e) { if (e != hipSuccess) throw std::runtime_error(hipGetErrorString(e)); }
template <class T> struct Dev {
    T* p = nullptr; std::size_t n;
    explicit Dev(std::size_t size) : n(size) { check(hipMalloc(&p, n * sizeof(T))); check(hipMemset(p, 0, n * sizeof(T))); }
    explicit Dev(const std::vector<T>& v) : Dev(v.size()) { put(v); }
    ~Dev() { (void)hipFree(p); }
    void put(const std::vector<T>& v) { check(hipMemcpy(p, v.data(), n * sizeof(T), hipMemcpyHostToDevice)); }
    std::vector<T> get() const { std::vector<T> v(n); check(hipMemcpy(v.data(), p, n * sizeof(T), hipMemcpyDeviceToHost)); return v; }
    Dev(const Dev&) = delete;
};
std::uint64_t fnv(std::uint64_t h, const void* p, std::size_t n) {
    const auto* b = static_cast<const unsigned char*>(p);
    for (std::size_t i = 0; i < n; ++i) { h ^= b[i]; h *= 1099511628211ULL; }
    return h;
}
template <class T> std::uint64_t fnv(std::uint64_t h, const Dev<T>& d) { const auto v = d.get(); return fnv(h, v.data(), v.size() * sizeof(T)); }
std::mt19937 rng(41);
std::vector<float> random(std::size_t n, float scale = 1.0F) {
    std::vector<float> v(n);
    for (auto& x : v) x = scale * static_cast<float>(static_cast<int>(rng() % 4097U) - 2048) / 1024.0F;
    return v;
}
double time_chain(hipStream_t s, const std::function<void()>& chain) {
    hipEvent_t a, b;
    check(hipEventCreate(&a)); check(hipEventCreate(&b));
    for (int i = 0; i < 50; ++i) chain();
    std::vector<double> t;
    for (int rep = 0; rep < 7; ++rep) {
        check(hipEventRecord(a, s));
        for (int i = 0; i < 200; ++i) chain();
        check(hipEventRecord(b, s));
        check(hipEventSynchronize(b));
        float ms = 0; check(hipEventElapsedTime(&ms, a, b)); t.push_back(ms * 1000.0 / 200.0);
    }
    std::sort(t.begin(), t.end());
    check(hipEventDestroy(a)); check(hipEventDestroy(b));
    return t[3];
}

void norm_cases(hipStream_t s) {
    for (unsigned n : {1280U, 512U, 5120U})
        for (unsigned rows : {1U, 2U, 4U, 6U, 8U}) {
            const auto x0 = random(std::size_t{rows} * n), w = random(n, 0.25F);
            Dev<float> x(x0), dw(w), deltas(rows * n / 32U);
            Dev<std::int8_t> codes(std::size_t{rows} * n);
            Dev<std::int32_t> sums(rows * n / 32U);
            // exactness: plain in place, then norm + quantize, then norm with the A8 emitted (fresh input each time)
            enqueue_rms_norm(x.p, dw.p, rows, n, n, 1e-20F, x.p, s);
            check(hipStreamSynchronize(s));
            const std::uint64_t h_plain = fnv(14695981039346656037ULL, x);
            x.put(x0);
            NormView v; v.x = x.p; v.x_stride = n; v.weight = dw.p; v.rows = rows; v.n = n; v.out = x.p; v.out_stride = n;
            v.codes = codes.p; v.deltas = deltas.p; v.sums = sums.p; v.a8_stride = n;
            enqueue_norm(v, s);
            check(hipStreamSynchronize(s));
            std::uint64_t h_a8 = fnv(fnv(fnv(fnv(14695981039346656037ULL, x), codes), deltas), sums);
            // timing: in place over the same buffer (values drift but the work does not)
            const double t_plain = time_chain(s, [&] { enqueue_rms_norm(x.p, dw.p, rows, n, n, 1e-20F, x.p, s); });
            const double t_quant = time_chain(s, [&] {
                enqueue_rms_norm(x.p, dw.p, rows, n, n, 1e-20F, x.p, s);
                enqueue_quantize_a8(x.p, n, rows, codes.p, deltas.p, sums.p, s);
            });
            const double t_a8 = time_chain(s, [&] { enqueue_norm(v, s); });
            std::printf("norm n=%u rows=%u hash plain %016llx a8 %016llx | us: norm %.2f, norm+quantize %.2f, norm(A8) %.2f\n", n, rows,
                        static_cast<unsigned long long>(h_plain), static_cast<unsigned long long>(h_a8), t_plain, t_quant, t_a8);
        }
}

void router_cases(hipStream_t s) {
    constexpr unsigned h = 5120, experts = 384, k = 6;
    std::vector<std::uint16_t> wb(std::size_t{experts} * h);
    for (auto& v : wb) { const float f = static_cast<float>(static_cast<int>(rng() % 2049U) - 1024) / 65536.0F; std::uint32_t u; std::memcpy(&u, &f, 4); v = static_cast<std::uint16_t>(u >> 16U); }
    const auto bias = random(experts, 0.01F);
    Dev<std::uint16_t> dw(wb);
    Dev<float> db(bias), dbi(bias);
    for (unsigned rows : {1U, 2U, 4U, 6U, 8U}) {
        const unsigned n = rows * k;
        Dev<float> x(random(std::size_t{rows} * h)), scores(std::size_t{rows} * experts), weights(n);
        Dev<std::uint32_t> ids(n), data(5U * n + rows + 3U), tickets(1);
        RouterView v;
        v.x = x.p; v.weight_bf16 = dw.p; v.bias = db.p; v.bias_image = dbi.p; v.rows = rows; v.hidden = h; v.experts = experts;
        v.top_k = k; v.route_scale = 1.5F; v.scores = scores.p; v.ids = ids.p; v.weights = weights.p;
        BuildRoutesView r;
        r.ids = ids.p; r.rows = rows; r.top_k = k; r.split = 192; r.first = 0; r.end = 192; r.base = 0;
        r.experts = data.p; r.begins = data.p + n; r.rows_out = data.p + 2 * n + 1; r.slots = data.p + 3 * n + 1;
        r.order = data.p + 4 * n + 1; r.low_counts = data.p + 5 * n + 1; r.counts = data.p + 5 * n + 1 + rows;
        enqueue_router(v, s); enqueue_build_routes(r, s); check(hipStreamSynchronize(s));
        const std::uint64_t h_sep = fnv(fnv(fnv(fnv(14695981039346656037ULL, scores), weights), ids), data);
        check(hipMemset(data.p, 0, data.n * 4)); check(hipMemset(ids.p, 0, ids.n * 4));
        if (!enqueue_router_routes(v, r, s)) throw std::runtime_error("fused routes fell back");
        check(hipStreamSynchronize(s));
        const std::uint64_t h_fused = fnv(fnv(fnv(fnv(14695981039346656037ULL, scores), weights), ids), data);
        const double t_sep = time_chain(s, [&] { enqueue_router(v, s); enqueue_build_routes(r, s); });
        const double t_fused = time_chain(s, [&] { (void)enqueue_router_routes(v, r, s); });
        std::printf("router rows=%u hash separate %016llx fused %016llx %s | us: router+select+routes %.2f, router+fused %.2f\n", rows,
                    static_cast<unsigned long long>(h_sep), static_cast<unsigned long long>(h_fused), h_sep == h_fused ? "EQUAL" : "DIFFER",
                    t_sep, t_fused);
    }
}
} // namespace

int main() try {
    hipStream_t s;
    check(hipStreamCreateWithFlags(&s, hipStreamNonBlocking));
    norm_cases(s);
    router_cases(s);
    check(hipStreamDestroy(s));
    std::printf("GAPS BENCH DONE\n");
    return 0;
} catch (const std::exception& e) {
    std::fprintf(stderr, "FAIL %s\n", e.what());
    return 1;
}
