// deepseek41 decode small ops at the model's shapes on synthetic data, 1..8 rows: the router's fast form
// (RouterView::fast, with and without the ticket), the paired F16 projections (enqueue_f16_rows_lanes_pair) and the
// compress kernel with its weights read ahead (set_compress_ahead), each against the current kernels bit for bit (and
// the router against the host reference). STRIX_DS41_SMALL_OPS_BENCH=1 adds cold timings (us per call).
#include "strix/ds41_moe.hpp"
#include "strix/hip_ds41_attention.hpp"
#include "strix/hip_ds41_f16_rows.hpp"
#include "strix/hip_ds41_formats.hpp"
#include "strix/hip_ds41_layer.hpp"

#include <hip/hip_runtime.h>

#include <algorithm>
#include <bit>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <functional>
#include <limits>
#include <random>
#include <stdexcept>
#include <string>
#include <tuple>
#include <utility>
#include <vector>

namespace {

namespace a = strix::ds41::attn;
namespace g = strix::ds41::attn::hip;
namespace ds41 = strix::ds41;

int failures = 0;
void expect(bool ok, const std::string& what) {
    if (!ok) { ++failures; std::fprintf(stderr, "FAIL %s\n", what.c_str()); }
}
void check(hipError_t e, const char* what) {
    if (e != hipSuccess) throw std::runtime_error(std::string(what) + ": " + hipGetErrorString(e));
}
template <class T> struct Dev {
    T* p = nullptr;
    std::size_t n = 0;
    explicit Dev(std::size_t count) : n(count) {
        check(hipMalloc(&p, std::max<std::size_t>(count, 1) * sizeof(T)), "hipMalloc");
        check(hipMemset(p, 0, std::max<std::size_t>(count, 1) * sizeof(T)), "hipMemset");
    }
    explicit Dev(const std::vector<T>& h) : Dev(h.size()) { put(h); }
    ~Dev() { (void)hipFree(p); }
    Dev(const Dev&) = delete;
    Dev& operator=(const Dev&) = delete;
    void put(const std::vector<T>& h) { check(hipMemcpy(p, h.data(), h.size() * sizeof(T), hipMemcpyHostToDevice), "upload"); }
    std::vector<T> get() const {
        std::vector<T> h(n);
        check(hipMemcpy(h.data(), p, n * sizeof(T), hipMemcpyDeviceToHost), "download");
        return h;
    }
};
template <class T> bool same(const std::vector<T>& x, const std::vector<T>& y) {
    return x.size() == y.size() && std::memcmp(x.data(), y.data(), x.size() * sizeof(T)) == 0;
}
std::mt19937 rng(4151);
float uni(float lo, float hi) { return std::uniform_real_distribution<float>(lo, hi)(rng); }
std::vector<float> random_f(std::size_t n, float lo, float hi) {
    std::vector<float> v(n);
    for (auto& x : v) x = uni(lo, hi);
    return v;
}
std::vector<std::uint16_t> random_f16(std::size_t n) {  // |w| in about [2^-9, 2^-5): projection-like magnitudes
    std::vector<std::uint16_t> v(n);
    for (auto& h : v) h = static_cast<std::uint16_t>(0x1800U + (rng() & 0xFFFU) + ((rng() & 1U) << 15U));
    return v;
}

// Cold timing: a 256 MB read before every call evicts the caches (a read leaves no dirty lines to write back during the
// call); median of the passes, in us. Warm: the same call back to back, the mean of 50.
__global__ void flush_kernel(const uint4* p, std::size_t n, unsigned* sink) {
    uint4 acc{0, 0, 0, 0};
    for (std::size_t i = blockIdx.x * 256ULL + threadIdx.x; i < n; i += 256ULL * gridDim.x) {
        const uint4 v = p[i];
        acc.x ^= v.x; acc.y ^= v.y; acc.z ^= v.z; acc.w ^= v.w;
    }
    if ((acc.x & acc.y & acc.z & acc.w) == 0xFFFFFFFFU) *sink = 1U;
}
struct Timer {
    Dev<std::uint8_t> flush{std::size_t{256} << 20U};
    Dev<unsigned> sink{1};
    hipEvent_t start{}, stop{};
    Timer() {
        check(hipEventCreate(&start), "event");
        check(hipEventCreate(&stop), "event");
    }
    ~Timer() {
        (void)hipEventDestroy(start);
        (void)hipEventDestroy(stop);
    }
    double us(const std::function<void()>& call, int passes = 15) {
        std::vector<double> t;
        for (int i = 0; i < passes; ++i) {
            hipLaunchKernelGGL(flush_kernel, dim3(1024), dim3(256), 0, nullptr, reinterpret_cast<const uint4*>(flush.p), flush.n / 16U, sink.p);
            check(hipEventRecord(start, nullptr), "event");
            call();
            check(hipEventRecord(stop, nullptr), "event");
            check(hipEventSynchronize(stop), "event");
            float ms = 0.0F;
            check(hipEventElapsedTime(&ms, start, stop), "event");
            t.push_back(1000.0 * ms);
        }
        std::sort(t.begin(), t.end());
        return t[t.size() / 2U];
    }
    double warm_us(const std::function<void()>& call) {
        call();
        check(hipEventRecord(start, nullptr), "event");
        for (int i = 0; i < 50; ++i) call();
        check(hipEventRecord(stop, nullptr), "event");
        check(hipEventSynchronize(stop), "event");
        float ms = 0.0F;
        check(hipEventElapsedTime(&ms, start, stop), "event");
        return 1000.0 * ms / 50.0;
    }
};
bool bench_on() {
    const char* e = std::getenv("STRIX_DS41_SMALL_OPS_BENCH");
    return e && e[0] == '1';
}

// ---- router ----
constexpr unsigned kExperts = 384, kHidden = 5120, kTopK = 6;
struct RouterCase {
    std::vector<float> weight, bias, bias_image;
    std::vector<std::uint16_t> weight_bf16;
};
// BF16-valued weights; kind 1 adds exact ties (duplicated rows with equal, large biases, so tied experts are chosen),
// kind 2 a NaN bias and signed zeros among the biases.
RouterCase router_case(unsigned kind) {
    RouterCase c;
    c.weight.resize(std::size_t{kExperts} * kHidden);
    std::normal_distribution<float> n(0.0F, 0.02F);
    for (auto& w : c.weight) w = std::bit_cast<float>(std::bit_cast<std::uint32_t>(n(rng)) & 0xFFFF0000U);
    c.bias = random_f(kExperts, -0.05F, 0.05F);
    c.bias_image = random_f(kExperts, -0.05F, 0.05F);
    if (kind == 1U) {
        for (const auto& [dst, src] : std::vector<std::pair<unsigned, unsigned>>{{37U, 5U}, {201U, 200U}, {383U, 64U}, {65U, 64U}, {300U, 7U}}) {
            std::copy_n(c.weight.data() + std::size_t{src} * kHidden, kHidden, c.weight.data() + std::size_t{dst} * kHidden);
            c.bias[src] = c.bias[dst] = 4.0F;
            c.bias_image[src] = c.bias_image[dst] = 4.0F;
        }
    } else if (kind == 2U) {
        c.bias[11] = std::numeric_limits<float>::quiet_NaN();
        c.bias_image[290] = std::numeric_limits<float>::quiet_NaN();
        c.bias[40] = 0.0F;
        c.bias[41] = -0.0F;
    }
    c.weight_bf16.resize(c.weight.size());
    for (std::size_t i = 0; i < c.weight.size(); ++i) c.weight_bf16[i] = static_cast<std::uint16_t>(std::bit_cast<std::uint32_t>(c.weight[i]) >> 16U);
    return c;
}
struct RouterRun {
    std::vector<float> scores, weights;
    std::vector<std::uint32_t> ids;
};

void test_router(Timer& timer, bool bench) {
    for (unsigned kind = 0; kind < 3U; ++kind) {
        const auto c = router_case(kind);
        Dev<float> w(c.weight), bias(c.bias), bias_image(c.bias_image);
        Dev<std::uint16_t> w16(c.weight_bf16);
        Dev<std::uint32_t> tickets(1);
        for (std::uint32_t rows = 1; rows <= 8U; ++rows) {
            const auto x = random_f(std::size_t{rows} * kHidden, -2.0F, 2.0F);
            std::vector<std::uint8_t> image(rows, 0U);
            for (std::uint32_t r = 1; r < rows; r += 3U) image[r] = 1U;
            Dev<float> xd(x), scores(std::size_t{rows} * kExperts), weights(std::size_t{rows} * kTopK);
            Dev<std::uint32_t> ids(std::size_t{rows} * kTopK);
            Dev<std::uint8_t> imaged(image);
            ds41::RouterView v;
            v.x = xd.p; v.weight_bf16 = w16.p; v.bias = bias.p; v.bias_image = bias_image.p; v.image_rows = imaged.p;
            v.rows = rows; v.hidden = kHidden; v.experts = kExperts; v.top_k = kTopK; v.route_scale = 1.5F;
            v.scores = scores.p; v.ids = ids.p; v.weights = weights.p;
            const auto run = [&](std::uint32_t fast, std::uint32_t* t) {
                check(hipMemset(scores.p, 0xFF, scores.n * 4U), "clear");
                check(hipMemset(ids.p, 0xFF, ids.n * 4U), "clear");
                check(hipMemset(weights.p, 0xFF, weights.n * 4U), "clear");
                v.fast = fast;
                v.tickets = t;
                ds41::enqueue_router(v, nullptr);
                check(hipDeviceSynchronize(), "router");
                return RouterRun{scores.get(), weights.get(), ids.get()};
            };
            const auto base = run(0U, nullptr);  // the current default (row waves)
            if (kind != 2U) {
                const auto ref = ds41::reference_router(x, rows, kHidden, c.weight, kExperts, c.bias, c.bias_image, image, kTopK, 1.5F);
                expect(same(base.scores, ref.scores) && base.ids == ref.ids && same(base.weights, ref.weights),
                       "router default == host reference, case " + std::to_string(kind) + " rows " + std::to_string(rows));
            }
            for (const std::uint32_t fast : {1U, 4U, 8U, 16U})
                for (const bool fused : {false, true}) {
                    const auto got = run(fast, fused ? tickets.p : nullptr);
                    expect(same(got.scores, base.scores) && got.ids == base.ids && same(got.weights, base.weights),
                           "router fast " + std::to_string(fast) + (fused ? " fused" : " two launches") + " == default, case " +
                               std::to_string(kind) + " rows " + std::to_string(rows));
                }
            expect(tickets.get()[0] == 0U, "router ticket left zero");
            if (bench && kind == 0U) {
                const auto call = [&] { ds41::enqueue_router(v, nullptr); };
                const std::pair<std::uint32_t, bool> forms[] = {{0U, false}, {1U, false}, {1U, true}, {16U, true}};
                double cold[4], warm[4];
                for (unsigned f = 0; f < 4U; ++f) {
                    v.fast = forms[f].first;
                    v.tickets = forms[f].second ? tickets.p : nullptr;
                    cold[f] = timer.us(call);
                    warm[f] = timer.warm_us(call);
                }
                std::printf("bench router R%u cold/warm us: default %.1f/%.1f | fast1 two launches %.1f/%.1f | fast1 fused %.1f/%.1f | wide16 fused %.1f/%.1f\n",
                            rows, cold[0], warm[0], cold[1], warm[1], cold[2], warm[2], cold[3], warm[3]);
            }
        }
    }
    std::printf("router: fast forms (4/8/16 waves, fused or not) == default == host reference, rows 1..8, ties and NaN cases\n");
}

// ---- paired F16 projections: compress (kv, gate on x) and index (q_b on qr, proj on x) ----
void test_pair(Timer& timer, bool bench) {
    constexpr unsigned H = kHidden, Q = 1280, D = 512, IQ = a::kIdxHeads * a::kIdxDim, IW = a::kIdxHeads;
    Dev<std::uint16_t> wkv(random_f16(std::size_t{D} * H)), wgate(random_f16(std::size_t{D} * H));
    Dev<std::uint16_t> wqb(random_f16(std::size_t{IQ} * Q)), wproj(random_f16(std::size_t{IW} * H));
    for (std::uint32_t rows = 1; rows <= 8U; ++rows) {
        Dev<float> x(random_f(std::size_t{rows} * H, -2.0F, 2.0F)), qr(random_f(std::size_t{rows} * Q, -2.0F, 2.0F));
        Dev<float> o1(std::size_t{rows} * D), o2(std::size_t{rows} * D), o3(std::size_t{rows} * IQ), o4(std::size_t{rows} * IW);
        const auto view = [&](const std::uint16_t* w, unsigned width, unsigned count, const float* in, float* out) {
            ds41::F16RowsView v;
            v.weight = w; v.width = width; v.first = 0; v.count = count; v.x = in; v.rows = rows; v.x_stride = width;
            v.out_stride = count; v.out = out;
            return v;
        };
        const auto kv = view(wkv.p, H, D, x.p, o1.p), gate = view(wgate.p, H, D, x.p, o2.p);
        const auto qb = view(wqb.p, Q, IQ, qr.p, o3.p), proj = view(wproj.p, H, IW, x.p, o4.p);
        const auto outputs = [&] { return std::vector<std::vector<float>>{o1.get(), o2.get(), o3.get(), o4.get()}; };
        const auto clear = [&] {
            for (auto* d : {&o1, &o2, &o3, &o4}) check(hipMemset(d->p, 0xFF, d->n * 4U), "clear");
        };
        clear();
        ds41::enqueue_f16_rows(kv, nullptr); ds41::enqueue_f16_rows(gate, nullptr);
        ds41::enqueue_f16_rows(qb, nullptr); ds41::enqueue_f16_rows(proj, nullptr);
        check(hipDeviceSynchronize(), "f16 rows");
        const auto base = outputs();
        clear();
        expect(ds41::f16_rows_pair_supported(kv, gate) && ds41::f16_rows_pair_supported(qb, proj), "pair views supported");
        ds41::enqueue_f16_rows_lanes_pair(kv, gate, nullptr);
        ds41::enqueue_f16_rows_lanes_pair(qb, proj, nullptr);
        check(hipDeviceSynchronize(), "f16 pair");
        const auto got = outputs();
        for (std::size_t i = 0; i < base.size(); ++i)
            expect(same(got[i], base[i]), "F16 pair output " + std::to_string(i) + " == separate launches, rows " + std::to_string(rows));
        clear();
        ds41::enqueue_f16_rows_lanes_pair(proj, qb, nullptr);  // the other order
        check(hipDeviceSynchronize(), "f16 pair");
        expect(same(o3.get(), base[2]) && same(o4.get(), base[3]), "F16 pair (proj, q_b) == separate launches, rows " + std::to_string(rows));
        if (bench) {
            const auto c_old = [&] { ds41::enqueue_f16_rows(kv, nullptr); ds41::enqueue_f16_rows(gate, nullptr); };
            const auto c_new = [&] { ds41::enqueue_f16_rows_lanes_pair(kv, gate, nullptr); };
            const auto i_old = [&] { ds41::enqueue_f16_rows(qb, nullptr); ds41::enqueue_f16_rows(proj, nullptr); };
            const auto i_new = [&] { ds41::enqueue_f16_rows_lanes_pair(qb, proj, nullptr); };
            std::printf("bench f16 R%u cold/warm us: compress kv+gate %.1f/%.1f -> pair %.1f/%.1f | index q_b+proj %.1f/%.1f -> pair %.1f/%.1f\n",
                        rows, timer.us(c_old), timer.warm_us(c_old), timer.us(c_new), timer.warm_us(c_new), timer.us(i_old),
                        timer.warm_us(i_old), timer.us(i_new), timer.warm_us(i_new));
        }
    }
    std::printf("F16 pair: compress kv+gate and index q_b+proj in one launch == separate launches, rows 1..8\n");
}

// ---- compress kernel: weights read ahead ----
void test_compress(Timer& timer, bool bench) {
    const auto norm = random_f(512, 0.5F, 1.5F), knorm = random_f(128, 0.5F, 1.5F);
    Dev<float> dnorm(norm), dknorm(knorm);
    Dev<std::uint16_t> dwk(random_f16(128 * 512));
    for (const unsigned ratio : {1U, 2U})
        for (std::uint32_t rows = 1; rows <= 8U; ++rows)
            for (const std::uint64_t first : {std::uint64_t{1000}, std::uint64_t{1001}}) {
                std::vector<std::uint64_t> pos(rows);
                std::vector<a::RopeRow> rope(rows);
                for (std::uint32_t r = 0; r < rows; ++r) {
                    pos[r] = first + r;
                    rope[r] = a::rope_row(a::RopeKind::compressed, pos[r] / ratio * ratio);
                }
                const std::size_t entries = (first + rows) / ratio + 1U;
                Dev<float> kv(random_f(std::size_t{rows} * 512, -3.0F, 3.0F)), sc(random_f(std::size_t{rows} * 512, -4.0F, 4.0F));
                Dev<float> state_in(random_f(1024, -3.0F, 3.0F)), state_out(1024);
                Dev<std::uint32_t> valid_in(std::vector<std::uint32_t>{1U}), valid_out(1);
                Dev<std::uint64_t> dpos(pos);
                Dev<a::RopeRow> drope(rope);
                Dev<std::uint8_t> kvc(entries * a::kKvEntryBytes), idxc(entries * a::kIdxEntryBytes);
                const g::CompressView v{ratio, rows, kv.p, sc.p, dnorm.p, dwk.p, dknorm.p, drope.p, dpos.p, state_in.p, valid_in.p,
                                        state_out.p, valid_out.p, kvc.p, idxc.p};
                const auto run = [&](int mode) {
                    check(hipMemset(kvc.p, 0, kvc.n), "clear");
                    check(hipMemset(idxc.p, 0, idxc.n), "clear");
                    check(hipMemset(state_out.p, 0xFF, 4096), "clear");
                    g::set_compress_ahead(mode);
                    g::enqueue_compress(v, nullptr);
                    check(hipDeviceSynchronize(), "compress");
                    return std::tuple{kvc.get(), idxc.get(), state_out.get(), valid_out.get()};
                };
                const auto base = run(0), got = run(1);
                const bool equal = same(std::get<0>(base), std::get<0>(got)) && same(std::get<1>(base), std::get<1>(got)) &&
                                   same(std::get<2>(base), std::get<2>(got)) && same(std::get<3>(base), std::get<3>(got));
                expect(equal, "compress ahead == current, ratio " + std::to_string(ratio) + " rows " + std::to_string(rows) +
                                        " first " + std::to_string(first));
                if (bench && first == 1001U && (rows == 1U || rows == 2U || rows == 6U || rows == 8U)) {
                    const auto call = [&] { g::enqueue_compress(v, nullptr); };
                    g::set_compress_ahead(0);
                    const double t0 = timer.us(call), w0 = timer.warm_us(call);
                    g::set_compress_ahead(1);
                    const double t1 = timer.us(call), w1 = timer.warm_us(call);
                    std::printf("bench compress ratio %u R%u cold/warm us: kernel %.1f/%.1f -> ahead %.1f/%.1f\n", ratio, rows, t0, w0, t1, w1);
                }
            }
    g::set_compress_ahead(-1);
    std::printf("compress: ahead kernel == current kernel (caches, state), ratio 1 and 2, rows 1..8\n");
}

}  // namespace

int main() try {
    Timer timer;
    const bool bench = bench_on();
    test_router(timer, bench);
    test_pair(timer, bench);
    test_compress(timer, bench);
    if (failures == 0) std::printf("deepseek41 small ops tests passed\n");
    return failures == 0 ? 0 : 1;
} catch (const std::exception& e) {
    std::fprintf(stderr, "deepseek41 small ops tests: %s\n", e.what());
    return 2;
}
