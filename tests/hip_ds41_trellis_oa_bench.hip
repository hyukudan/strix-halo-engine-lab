// Trellis o_a adapter micro-benchmark (not a ctest): the wo_a groups of one TP rank (4 x 4096 -> 1024, ours pack) as the
// engine runs them: path A = today's per-group enqueue_trellis_linear calls (the attention hook), B = TrellisGroupRunner
// mode 1 (one stage + one scatter kernel, no staging for a single row), C = mode 2 (B with the groups' library calls on
// concurrent streams), U6 = today's c-u6 o_a kernel (enqueue_q8_0_a8 on random U6 rows of the same shape, + its A8).
// Checks: B and C bitwise == A for every row at R = 1..8, 9, 13 on several (layer, rank) sets; every row == the same
// row alone (R1). Timing: cold (rotating over every layer and both ranks' groups, ~840 MB) and warm, median us per call.
// "trace" mode: one call per (path, R) separated by a marker kernel and a device sync, for rocprofv3 launch counts.
// usage: strix-hip-ds41-trellis-oa-bench PACK_DIR [layers 40] [iterations 60] [trace]
#include "strix/ds41_trellis.hpp"
#include "strix/ds41_trellis_group.hpp"
#include "strix/hip_ds41_formats.hpp"
#include "strix/hip_ds41_layer.hpp"

#include <hip/hip_runtime.h>

#include <algorithm>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <functional>
#include <random>
#include <string>
#include <vector>

using namespace strix::ds41;

namespace {

void check(hipError_t e, const char* what) {
    if (e != hipSuccess) { std::fprintf(stderr, "%s: %s\n", what, hipGetErrorString(e)); std::exit(2); }
}
template <class T> T* dev(std::size_t n) { T* p = nullptr; check(hipMalloc(&p, std::max<std::size_t>(n, 1) * sizeof(T)), "hipMalloc"); return p; }

__global__ void marker_kernel(int id, int* sink) { if (threadIdx.x == 0 && id < 0) *sink = id; }

hipStream_t g_s = nullptr;

double time_us(int iterations, const std::function<void(int)>& launch) {
    hipEvent_t a, b;
    check(hipEventCreate(&a), "event");
    check(hipEventCreate(&b), "event");
    for (int i = 0; i < 3; ++i) launch(i);
    std::vector<float> t;
    for (int i = 0; i < iterations; ++i) {
        check(hipEventRecord(a, g_s), "record");
        launch(i + 3);
        check(hipEventRecord(b, g_s), "record");
        check(hipEventSynchronize(b), "sync");
        float ms = 0;
        check(hipEventElapsedTime(&ms, a, b), "elapsed");
        t.push_back(ms);
    }
    (void)hipEventDestroy(a);
    (void)hipEventDestroy(b);
    std::sort(t.begin(), t.end());
    return 1000.0 * t[t.size() / 2];
}
// one "layer" of N calls back to back (as 40 layers in a token): mean us per call over `calls` calls
double stream_us(int calls, const std::function<void(int)>& launch) {
    hipEvent_t a, b;
    check(hipEventCreate(&a), "event");
    check(hipEventCreate(&b), "event");
    for (int i = 0; i < 4; ++i) launch(i);
    check(hipEventRecord(a, g_s), "record");
    for (int i = 0; i < calls; ++i) launch(i);
    check(hipEventRecord(b, g_s), "record");
    check(hipEventSynchronize(b), "sync");
    float ms = 0;
    check(hipEventElapsedTime(&ms, a, b), "elapsed");
    (void)hipEventDestroy(a);
    (void)hipEventDestroy(b);
    return 1000.0 * ms / calls;
}

std::vector<float> down(const float* p, std::size_t n) {
    std::vector<float> h(n);
    check(hipStreamSynchronize(g_s), "sync");
    check(hipMemcpy(h.data(), p, n * 4U, hipMemcpyDeviceToHost), "download");
    return h;
}

}  // namespace

int main(int argc, char** argv) {
    if (argc < 2) { std::fprintf(stderr, "usage: %s PACK_DIR [layers] [iterations] [trace]\n", argv[0]); return 1; }
    const std::string dir = argv[1];
    const int layers = argc > 2 ? std::atoi(argv[2]) : 40;
    const int iterations = argc > 3 ? std::atoi(argv[3]) : 60;
    const bool trace = argc > 4 && std::string(argv[4]) == "trace";
    check(hipSetDevice(0), "device");
    check(hipStreamCreateWithFlags(&g_s, hipStreamNonBlocking), "stream");
    TrellisPack pack(dir);
    TrellisContext ctx;
    // sets[l * 2 + r] = layer l's groups [4 r, 4 r + 4)
    std::vector<std::vector<const TrellisSite*>> sets;
    std::uint64_t set_bytes = 0;
    for (int l = 0; l < layers; ++l)
        for (std::uint32_t r = 0; r < 2U; ++r) {
            std::vector<const TrellisSite*> s;
            for (std::uint32_t g = 4U * r; g < 4U * r + 4U; ++g) {
                const auto name = trellis_pack_name("blk." + std::to_string(l) + ".attn_output_a.weight", static_cast<int>(g));
                s.push_back(&ctx.prepare(pack, *name));
            }
            if (sets.empty())
                for (const auto* t : s) set_bytes += t->device_bytes;
            sets.push_back(std::move(s));
        }
    ctx.reserve(g_s, 8);
    TrellisGroupRunner run1(sets[0], 1, g_s), run2(sets[0], 2, g_s);
    std::printf("{\"library\":\"%s\",\"sets\":%zu,\"bytes_per_set\":%llu,\"run1\":\"%s\",\"run2\":\"%s\"}\n", TrellisContext::library_version(),
                sets.size(), (unsigned long long)set_bytes, run1.describe().c_str(), run2.describe().c_str());
    constexpr std::uint32_t RMAX = 16, G = 4, I = 4096, N = 1024, XW = G * I, YW = G * N;
    std::vector<float> hx(std::size_t{RMAX} * XW);
    std::mt19937 rng(20261021);
    std::normal_distribution<float> nd(0.0F, 1.0F);
    for (auto& v : hx) v = nd(rng);
    float* x = dev<float>(hx.size());
    check(hipMemcpy(x, hx.data(), hx.size() * 4U, hipMemcpyHostToDevice), "upload");
    float* ya = dev<float>(std::size_t{RMAX} * YW);
    float* yb = dev<float>(std::size_t{RMAX} * YW);
    float* yc = dev<float>(std::size_t{RMAX} * YW);
    float* y1 = dev<float>(YW);
    int* sink = dev<int>(1);

    using Path = std::function<void(const std::vector<const TrellisSite*>&, float*, std::uint32_t)>;
    const Path path_a = [&](const std::vector<const TrellisSite*>& s, float* y, std::uint32_t R) {
        for (std::size_t g = 0; g < s.size(); ++g) enqueue_trellis_linear(*s[g], x + g * I, y + g * N, R, XW, YW, false, g_s);
    };
    const Path path_b = [&](const std::vector<const TrellisSite*>& s, float* y, std::uint32_t R) { run1.enqueue(s, x, XW, y, YW, R, g_s); };
    const Path path_c = [&](const std::vector<const TrellisSite*>& s, float* y, std::uint32_t R) { run2.enqueue(s, x, XW, y, YW, R, g_s); };

    if (trace) {
        int id = 0;
        for (const std::uint32_t R : {1U, 6U}) {
            for (int p = 0; p < 3; ++p) {
                check(hipDeviceSynchronize(), "sync");
                marker_kernel<<<1, 64, 0, g_s>>>(id, sink);
                const Path& f = p == 0 ? path_a : p == 1 ? path_b : path_c;
                f(sets[0], ya, R);
                check(hipDeviceSynchronize(), "sync");
                std::printf("{\"trace_marker\":%d,\"path\":\"%c\",\"rows\":%u}\n", id++, "ABC"[p], R);
            }
        }
        marker_kernel<<<1, 64, 0, g_s>>>(id, sink);
        check(hipDeviceSynchronize(), "sync");
        std::printf("{\"trace_marker\":%d,\"end\":1}\n", id);
        return 0;
    }

    // exactness: B, C == A bitwise; every row == the row alone (A at R1)
    std::uint64_t compared = 0, differ_b = 0, differ_c = 0, differ_r1 = 0;
    for (const std::size_t si : {std::size_t{0}, std::size_t{1}, sets.size() / 2U + 1U, sets.size() - 1U}) {
        const auto& s = sets[si];
        std::vector<std::vector<float>> alone(RMAX);
        for (std::uint32_t r = 0; r < RMAX; ++r) {
            check(hipMemsetAsync(y1, 0xff, YW * 4U, g_s), "clear");
            for (std::size_t g = 0; g < G; ++g) enqueue_trellis_linear(*s[g], x + std::size_t{r} * XW + g * I, y1 + g * N, 1, XW, YW, false, g_s);
            alone[r] = down(y1, YW);
        }
        for (const std::uint32_t R : {1U, 2U, 3U, 4U, 5U, 6U, 7U, 8U, 9U, 13U, 16U}) {
            check(hipMemsetAsync(ya, 0xff, std::size_t{R} * YW * 4U, g_s), "clear");
            check(hipMemsetAsync(yb, 0xfe, std::size_t{R} * YW * 4U, g_s), "clear");
            check(hipMemsetAsync(yc, 0xfd, std::size_t{R} * YW * 4U, g_s), "clear");
            path_a(s, ya, R);
            path_b(s, yb, R);
            path_c(s, yc, R);
            const auto a = down(ya, std::size_t{R} * YW), b = down(yb, std::size_t{R} * YW), c = down(yc, std::size_t{R} * YW);
            for (std::uint32_t r = 0; r < R; ++r) {
                ++compared;
                if (std::memcmp(a.data() + std::size_t{r} * YW, b.data() + std::size_t{r} * YW, YW * 4U) != 0) {
                    std::size_t k = 0;
                    while (std::memcmp(a.data() + std::size_t{r} * YW + k, b.data() + std::size_t{r} * YW + k, 4U) == 0) ++k;
                    std::size_t nd = 0;
                    for (std::size_t j = 0; j < YW; ++j)
                        nd += std::memcmp(a.data() + std::size_t{r} * YW + j, b.data() + std::size_t{r} * YW + j, 4U) != 0 ? 1U : 0U;
                    std::printf("{\"differ\":\"B\",\"set\":%zu,\"R\":%u,\"row\":%u,\"col\":%zu,\"values\":%zu,\"a\":%a,\"b\":%a}\n", si, R, r, k,
                                nd, static_cast<double>(a[std::size_t{r} * YW + k]), static_cast<double>(b[std::size_t{r} * YW + k]));
                }
                differ_b += std::memcmp(a.data() + std::size_t{r} * YW, b.data() + std::size_t{r} * YW, YW * 4U) != 0 ? 1U : 0U;
                differ_c += std::memcmp(a.data() + std::size_t{r} * YW, c.data() + std::size_t{r} * YW, YW * 4U) != 0 ? 1U : 0U;
                differ_r1 += std::memcmp(a.data() + std::size_t{r} * YW, alone[r].data(), YW * 4U) != 0 ? 1U : 0U;
            }
        }
    }
    std::printf("{\"check\":\"rows\",\"compared\":%llu,\"B_differ_A\":%llu,\"C_differ_A\":%llu,\"A_differ_alone\":%llu}\n",
                (unsigned long long)compared, (unsigned long long)differ_b, (unsigned long long)differ_c, (unsigned long long)differ_r1);
    std::fflush(stdout);

    // timing: cold (rotate every set, ~840 MB) per call; stream = 80 calls back to back (mean); warm = set 0 repeated
    const int nsets = static_cast<int>(sets.size());
    for (const std::uint32_t R : {1U, 2U, 4U, 5U, 6U, 7U, 8U}) {
        for (int p = 0; p < 3; ++p) {
            const auto& f = p == 0 ? path_a : p == 1 ? path_b : path_c;
            const double cold = time_us(iterations, [&](int i) { f(sets[static_cast<std::size_t>(i % nsets)], ya, R); });
            const double strm = stream_us(nsets, [&](int i) { f(sets[static_cast<std::size_t>(i % nsets)], ya, R); });
            const double warm = time_us(iterations, [&](int) { f(sets[0], ya, R); });
            std::printf("{\"path\":\"%c\",\"rows\":%u,\"cold_us\":%.1f,\"stream_us\":%.1f,\"warm_us\":%.1f,\"cold_gbs\":%.1f,\"stream_gbs\":%.1f}\n",
                        "ABC"[p], R, cold, strm, warm, static_cast<double>(set_bytes) / (cold * 1e3), static_cast<double>(set_bytes) / (strm * 1e3));
            std::fflush(stdout);
        }
    }

    // today's c-u6 o_a (4 groups x 1024 rows x 4096, U6 26-byte blocks, random), + its A8 of the input
    {
        const std::size_t bytes = std::size_t{4096} * (4096U / 32U * 26U);
        const std::size_t copies = std::max<std::size_t>(4, (840ULL << 20) / bytes + 1);
        std::vector<std::uint8_t> host(bytes);
        std::uniform_int_distribution<int> byte(0, 255);
        for (auto& v : host) v = static_cast<std::uint8_t>(byte(rng));
        for (std::size_t blk = 0; blk < bytes / 26U; ++blk) {
            const auto d = __builtin_bit_cast(std::uint16_t, static_cast<_Float16>(0.01F + 0.0001F * static_cast<float>(blk % 100U)));
            host[blk * 26U] = static_cast<std::uint8_t>(d & 0xffU);
            host[blk * 26U + 1U] = static_cast<std::uint8_t>(d >> 8U);
        }
        auto* w = dev<std::uint8_t>(bytes * copies);
        for (std::size_t c = 0; c < copies; ++c) check(hipMemcpy(w + c * bytes, host.data(), bytes, hipMemcpyHostToDevice), "upload");
        auto* qc = dev<std::int8_t>(std::size_t{8} * XW);
        auto* qd = dev<float>(std::size_t{8} * XW / 32U);
        auto* qs = dev<std::int32_t>(std::size_t{8} * XW / 32U);
        for (const std::uint32_t R : {1U, 2U, 4U, 5U, 6U, 7U, 8U}) {
            const auto call = [&](int i) {
                enqueue_quantize_a8(x, XW, R, qc, qd, qs, g_s);
                Q8A8View v;
                v.weight = w + (static_cast<std::size_t>(i) % copies) * bytes;
                v.width = 4096; v.count = 4096; v.per_group = 1024; v.first = 0;
                v.codes = qc; v.deltas = qd; v.rows = R; v.x_stride = XW; v.out_stride = YW; v.out = ya; v.packed = 6;
                enqueue_q8_0_a8(v, g_s);
            };
            const double cold = time_us(iterations, call);
            const double strm = stream_us(static_cast<int>(copies), call);
            std::printf("{\"path\":\"U6\",\"rows\":%u,\"cold_us\":%.1f,\"stream_us\":%.1f,\"cold_gbs\":%.1f,\"stream_gbs\":%.1f}\n", R, cold, strm,
                        static_cast<double>(bytes) / (cold * 1e3), static_cast<double>(bytes) / (strm * 1e3));
            std::fflush(stdout);
        }
        (void)hipFree(w); (void)hipFree(qc); (void)hipFree(qd); (void)hipFree(qs);
    }
    return 0;
}
