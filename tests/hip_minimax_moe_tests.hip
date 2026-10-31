// Device test of the fused post-exchange kernel: residual, norm and router top-4 equal the host reference bit for bit,
// rows are independent (the same row gives the same bits in calls of 1 and 8 rows), and ties pick the smaller id.
#include "strix/hip_minimax_moe.hpp"

#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <limits>
#include <random>
#include <vector>

namespace moe = strix::minimax::moe;

namespace {

int g_fail = 0;
void check(bool ok, const char* what) {
    if (!ok) {
        ++g_fail;
        std::fprintf(stderr, "FAIL: %s\n", what);
    }
}
void hip_ok(hipError_t e, const char* what) {
    if (e != hipSuccess) {
        std::fprintf(stderr, "HIP error %s: %s\n", what, hipGetErrorString(e));
        std::exit(2);
    }
}
template <class T> T* dev(const std::vector<T>& h) {
    T* d = nullptr;
    hip_ok(hipMalloc(&d, h.size() * sizeof(T)), "malloc");
    hip_ok(hipMemcpy(d, h.data(), h.size() * sizeof(T), hipMemcpyHostToDevice), "h2d");
    return d;
}
template <class T> std::vector<T> host(const T* d, std::size_t n) {
    std::vector<T> h(n);
    hip_ok(hipMemcpy(h.data(), d, n * sizeof(T), hipMemcpyDeviceToHost), "d2h");
    return h;
}

struct Out {
    std::vector<float> h, y, w;
    std::vector<std::uint32_t> ids;
};

Out run_device(unsigned rows, const std::vector<float>& h0, const std::vector<float>& p0, const std::vector<float>& p1,
               const std::vector<float>& nw, const std::vector<float>& rw, const std::vector<float>& bias, bool route,
               unsigned batch_rows = 0, unsigned ilp = 0, bool timing = false) {
    float *dh = dev(h0), *dp0 = dev(p0), *dp1 = dev(p1), *dnw = dev(nw), *drw = dev(rw), *db = dev(bias);
    float* dy = dev(std::vector<float>(h0.size(), 0.0F));
    constexpr unsigned guard = 8 * moe::kTopE;
    float* dw = dev(std::vector<float>(rows * moe::kTopE + guard, -99.25F));
    auto* di = dev(std::vector<std::uint32_t>(rows * moe::kTopE + guard, 0xdeadbeefU));
    float* scratch = batch_rows == 32 ? dev(std::vector<float>(moe::hip::router_tiled_scratch_floats(rows), -77.5F)) : nullptr;
    const auto launch = [&] {
        moe::hip::PostExchangeView view{rows, dh, dp0, dp1, dnw, dy, route && batch_rows == 0 ? drw : nullptr, db, di, dw};
        view.ilp = ilp;
        moe::hip::enqueue_post_exchange(view, nullptr);
        if (route && batch_rows == 32)
            moe::hip::enqueue_router_tiled(rows, dy, drw, db, di, dw, scratch, nullptr);
        else if (route && batch_rows != 0)
            moe::hip::enqueue_router_batch(rows, dy, drw, db, di, dw, batch_rows, nullptr);
    };
    launch();
    hip_ok(hipDeviceSynchronize(), "sync");
    Out o{host(dh, h0.size()), host(dy, h0.size()), host(dw, rows * moe::kTopE), host(di, rows * moe::kTopE)};
    const auto tail_w = host(dw + rows * moe::kTopE, guard);
    const auto tail_i = host(di + rows * moe::kTopE, guard);
    for (unsigned k = 0; k < guard; ++k)
        check(tail_w[k] == -99.25F && tail_i[k] == 0xdeadbeefU, "router tail guard");
    if (timing) {
        hipEvent_t begin = nullptr, end = nullptr;
        hip_ok(hipEventCreate(&begin), "event create");
        hip_ok(hipEventCreate(&end), "event create");
        std::vector<float> ms;
        for (unsigned it = 0; it < 20; ++it) {
            hip_ok(hipMemcpy(dh, h0.data(), h0.size() * sizeof(float), hipMemcpyHostToDevice), "reset h");
            hip_ok(hipEventRecord(begin, nullptr), "event begin");
            launch();
            hip_ok(hipEventRecord(end, nullptr), "event end");
            hip_ok(hipEventSynchronize(end), "event sync");
            float elapsed = 0.0F;
            hip_ok(hipEventElapsedTime(&elapsed, begin, end), "elapsed");
            ms.push_back(elapsed);
        }
        std::sort(ms.begin(), ms.end());
        std::printf("post+router rows=%u batch=%u median20=%.3f ms\n", rows, batch_rows, (ms[9] + ms[10]) * 0.5F);
        hip_ok(hipEventDestroy(begin), "event destroy");
        hip_ok(hipEventDestroy(end), "event destroy");
    }
    for (void* p : {static_cast<void*>(dh), static_cast<void*>(dp0), static_cast<void*>(dp1), static_cast<void*>(dnw),
                    static_cast<void*>(drw), static_cast<void*>(db), static_cast<void*>(dy), static_cast<void*>(dw),
                    static_cast<void*>(di), static_cast<void*>(scratch)})
        if (p != nullptr) (void)hipFree(p);
    return o;
}

void router_batch_tests() {
    std::mt19937 rng(29);
    std::uniform_real_distribution<float> random(-1.0F, 1.0F);
    for (unsigned rows : {1U, 7U, 8U, 9U, 15U, 16U, 17U, 31U, 32U, 33U, 100U, 1024U, 4096U}) {
        const std::size_t size = std::size_t{rows} * moe::kHidden;
        std::vector<float> h(size), p0(size), p1(size), nw(moe::kHidden), rw(moe::kExperts * moe::kHidden), bias(moe::kExperts);
        for (unsigned kind = 0; kind < 5; ++kind) {
            for (auto& x : h) x = random(rng);
            for (auto& x : p0) x = random(rng);
            for (auto& x : p1) x = random(rng);
            for (auto& x : nw) x = random(rng) * 0.3F;
            for (auto& x : rw) x = random(rng) * 0.02F;
            for (auto& x : bias) x = random(rng);
            if (kind == 1) {  // all choices equal: ids must ascend
                std::fill(rw.begin(), rw.end(), 0.0F);
                std::fill(bias.begin(), bias.end(), 0.0F);
            } else if (kind >= 2) {
                const float magnitude = kind == 2 ? 0.0F : kind == 3 ? std::numeric_limits<float>::denorm_min() : 1.0e12F;
                for (std::size_t j = 0; j < size; ++j) {
                    h[j] = (j & 1U) != 0 ? -magnitude : magnitude;
                    p0[j] = (j & 2U) != 0 ? -magnitude : magnitude;
                    p1[j] = (j & 4U) != 0 ? -magnitude : magnitude;
                }
                if (kind == 3)
                    for (auto& x : rw) x = std::numeric_limits<float>::denorm_min();
                if (kind == 4)
                    for (auto& x : rw) x *= 1.0e20F;
            }
            // Exercise every Ilp form on a masked tail; the full size sweep uses the default form.
            const unsigned modes = rows == 17 ? 3U : 1U;
            for (unsigned ilp = 0; ilp < modes; ++ilp) {
                const bool timing = kind == 0 && ilp == 0 && (rows == 1024 || rows == 4096);
                const Out old = run_device(rows, h, p0, p1, nw, rw, bias, true, 0, ilp, timing);
                for (unsigned batch : {4U, 8U, 32U}) {
                    const Out split = run_device(rows, h, p0, p1, nw, rw, bias, true, batch, ilp, timing);
                    const bool equal = std::memcmp(old.h.data(), split.h.data(), size * sizeof(float)) == 0 &&
                        std::memcmp(old.y.data(), split.y.data(), size * sizeof(float)) == 0 &&
                        std::memcmp(old.ids.data(), split.ids.data(), old.ids.size() * sizeof(std::uint32_t)) == 0 &&
                        std::memcmp(old.w.data(), split.w.data(), old.w.size() * sizeof(float)) == 0;
                    if (!equal) std::fprintf(stderr, "router batch mismatch rows=%u R=%u kind=%u ilp=%u\n", rows, batch, kind, ilp);
                    check(equal, "fused vs split: h/y/ids/weights bytes");
                    if (kind == 1)
                        for (unsigned r = 0; r < rows; ++r)
                            for (unsigned k = 0; k < moe::kTopE; ++k)
                                check(split.ids[r * moe::kTopE + k] == k, "all-choice tie id ascending");
                }
            }
        }
    }
}

}  // namespace

int main() {
    std::mt19937 rng(7);
    std::normal_distribution<float> n(0.0F, 1.0F);
    const unsigned R = 8, D = moe::kHidden;
    std::vector<float> h0(R * D), p0(R * D), p1(R * D), nw(D), rw(moe::kExperts * D), bias(moe::kExperts);
    for (auto& x : h0) x = n(rng) * 4.0F;
    for (auto& x : p0) x = n(rng);
    for (auto& x : p1) x = n(rng);
    for (auto& x : nw) x = n(rng) * 0.3F;
    for (auto& x : rw) x = n(rng) * 0.02F;
    for (unsigned e = 0; e < moe::kExperts; ++e) bias[e] = 16.4F + 0.001F * static_cast<float>(e % 7);
    // Exact ties: experts 10 and 20 share their router row and bias with expert 5 (same logit, same choice).
    for (unsigned e : {10U, 20U}) {
        std::memcpy(&rw[e * D], &rw[5 * D], D * sizeof(float));
        bias[e] = bias[5];
    }
    // Rows 3..5 are steered so that expert 5 (and its twins) wins: a tie decided by the smaller id.
    for (unsigned r = 3; r < 6; ++r)
        for (unsigned j = 0; j < D; ++j) h0[r * D + j] += 3.0F * rw[5 * D + j] * 50.0F;
    for (bool route : {false, true}) {
        const Out d = run_device(R, h0, p0, p1, nw, rw, bias, route);
        std::vector<float> h = h0, y(R * D);
        bool ok_h = true, ok_y = true, ok_r = true;
        for (unsigned r = 0; r < R; ++r) {
            moe::post_exchange_row(&h[r * D], &p0[r * D], &p1[r * D], nw.data(), &y[r * D]);
            if (!route) continue;
            const auto rr = moe::route_row(&y[r * D], rw.data(), bias.data());
            for (unsigned k = 0; k < moe::kTopE; ++k) {
                ok_r &= d.ids[r * moe::kTopE + k] == rr.ids[k];
                ok_r &= std::memcmp(&d.w[r * moe::kTopE + k], &rr.weights[k], 4) == 0;
            }
            if (r == 3) std::printf("row 3 experts %u %u %u %u\n", rr.ids[0], rr.ids[1], rr.ids[2], rr.ids[3]);
        }
        ok_h = std::memcmp(h.data(), d.h.data(), h.size() * 4) == 0;
        ok_y = std::memcmp(y.data(), d.y.data(), y.size() * 4) == 0;
        check(ok_h, "residual bits");
        check(ok_y, "norm bits");
        if (route) check(ok_r, "router ids and weights bits");
    }
    // Batch invariance: row 6 alone equals row 6 of the 8-row call.
    {
        const Out all = run_device(R, h0, p0, p1, nw, rw, bias, true);
        std::vector<float> h1(h0.begin() + 6 * D, h0.begin() + 7 * D), q0(p0.begin() + 6 * D, p0.begin() + 7 * D),
            q1(p1.begin() + 6 * D, p1.begin() + 7 * D);
        const Out one = run_device(1, h1, q0, q1, nw, rw, bias, true);
        check(std::memcmp(one.y.data(), all.y.data() + 6 * D, D * 4) == 0 &&
                  std::memcmp(one.w.data(), all.w.data() + 6 * moe::kTopE, moe::kTopE * 4) == 0 &&
                  std::memcmp(one.ids.data(), all.ids.data() + 6 * moe::kTopE, moe::kTopE * 4) == 0,
              "batch invariance");
    }
    router_batch_tests();
    if (g_fail != 0) {
        std::fprintf(stderr, "%d failure(s)\n", g_fail);
        return 1;
    }
    std::printf("strix-hip-minimax-moe-tests: all passed\n");
    return 0;
}
