// R6 reads micro-benchmark (not a ctest): attn_q_b and attn_output_a of one tensor-parallel rank at the decode
// window rows 1..8, Q8_0 and U6 (the c-u6 overlay) weights, cold (rotating copies over >= 512 MB, as one layer
// after another in the engine). Kernels: the site kernel as the engine launches it (enqueue_q8_0_a8), the expert
// window kernel (routes product, forms 0 and 2) and the multi-row form of hip_ds41_rn_wide.hpp in each registered
// shape. Every result is compared bitwise with the one-row site kernel run row by row (the R1 identity).
// usage: strix-hip-ds41-rn-wide-bench [iterations] [sites: all or a list like q_b,o_a] [format filter: q8|u6|all]

#include "strix/ds41_formats.hpp"
#include "strix/hip_ds41_formats.hpp"
#include "strix/hip_ds41_layer.hpp"
#include "strix/hip_ds41_rn_wide.hpp"

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

namespace {

using namespace strix::ds41;

void check(hipError_t e, const char* what) {
    if (e != hipSuccess) { std::fprintf(stderr, "%s: %s\n", what, hipGetErrorString(e)); std::exit(2); }
}
template <class T> T* dev(std::size_t n) { T* p = nullptr; check(hipMalloc(&p, std::max<std::size_t>(n, 1) * sizeof(T)), "hipMalloc"); return p; }
template <class T> T* dev(const std::vector<T>& h) { T* p = dev<T>(h.size()); check(hipMemcpy(p, h.data(), h.size() * sizeof(T), hipMemcpyHostToDevice), "upload"); return p; }
template <class T> std::vector<T> down(const T* p, std::size_t n) {
    std::vector<T> h(n);
    check(hipDeviceSynchronize(), "sync");
    check(hipMemcpy(h.data(), p, n * sizeof(T), hipMemcpyDeviceToHost), "download");
    return h;
}
std::uint16_t to_f16(float x) { return __builtin_bit_cast(std::uint16_t, static_cast<_Float16>(x)); }

// Median kernel time (ms) of launch(i) over iterations (each i a different weight copy), one event pair per call.
double time_ms(int iterations, const std::function<void(int)>& launch) {
    hipEvent_t a, b;
    check(hipEventCreate(&a), "event");
    check(hipEventCreate(&b), "event");
    for (int i = 0; i < 3; ++i) launch(i);
    std::vector<float> t;
    for (int i = 0; i < iterations; ++i) {
        check(hipEventRecord(a, nullptr), "record");
        launch(i + 3);
        check(hipEventRecord(b, nullptr), "record");
        check(hipEventSynchronize(b), "sync");
        float ms = 0;
        check(hipEventElapsedTime(&ms, a, b), "elapsed");
        t.push_back(ms);
    }
    (void)hipEventDestroy(a);
    (void)hipEventDestroy(b);
    std::sort(t.begin(), t.end());
    return t[t.size() / 2];
}

struct Site { const char* name; std::uint32_t width, count, per_group; };
struct Format { const char* name; std::uint32_t packed, block_bytes; WeightFormat format; };

void bench(const Site& s, const Format& f, int iterations) {
    std::mt19937 rng(20261020);
    const std::uint32_t groups = s.count / s.per_group, gw = s.width / 32U;
    const std::size_t row_bytes = std::size_t{gw} * f.block_bytes, bytes = row_bytes * s.count;
    const std::size_t copies = std::max<std::size_t>(4, (640ULL << 20) / bytes + 1);
    std::vector<std::uint8_t> host(bytes);
    std::uniform_int_distribution<int> byte(0, 255);
    std::uniform_real_distribution<float> sc(0.001F, 0.05F);
    for (auto& b : host) b = static_cast<std::uint8_t>(byte(rng));
    for (std::size_t blk = 0; blk < bytes / f.block_bytes; ++blk) {
        const auto d = to_f16(sc(rng));
        host[blk * f.block_bytes] = static_cast<std::uint8_t>(d & 0xffU);
        host[blk * f.block_bytes + 1] = static_cast<std::uint8_t>(d >> 8U);
        if (f.packed == 0U)  // Q8_0 codes in -127..127
            for (std::size_t k = 2; k < 34; ++k)
                if (host[blk * 34 + k] == 0x80U) host[blk * 34 + k] = 0x81U;
    }
    auto* w = dev<std::uint8_t>(bytes * copies);
    for (std::size_t c = 0; c < copies; ++c) check(hipMemcpy(w + c * bytes, host.data(), bytes, hipMemcpyHostToDevice), "upload");
    // L2 / MALL flush buffer for the warm-vs-cold check is implicit: consecutive calls read different copies.
    const std::uint32_t x_stride = groups * s.width;
    for (const std::uint32_t rows : {1U, 2U, 3U, 4U, 5U, 6U, 7U, 8U}) {
        std::vector<std::int8_t> codes(std::size_t{rows} * x_stride);
        std::vector<float> deltas(std::size_t{rows} * x_stride / 32U);
        std::uniform_int_distribution<int> code(-127, 127);
        std::uniform_real_distribution<float> dl(0.001F, 0.1F);
        for (auto& c : codes) c = static_cast<std::int8_t>(code(rng));
        for (auto& d : deltas) d = dl(rng);
        std::vector<float> xf(std::size_t{rows} * x_stride);
        for (auto& x : xf) x = dl(rng) * static_cast<float>(code(rng));
        auto* dc = dev(codes);
        auto* dd = dev(deltas);
        auto* dx = dev(xf);
        auto* qc = dev<std::int8_t>(xf.size());
        auto* qd = dev<float>(xf.size() / 32U);
        auto* qs = dev<std::int32_t>(xf.size() / 32U);
        const std::size_t outs = std::size_t{rows} * s.count;
        auto* out = dev<float>(outs);
        auto base_view = [&](int i) {
            Q8A8View v;
            v.weight = w + (static_cast<std::size_t>(i) % copies) * bytes;
            v.width = s.width; v.count = s.count; v.per_group = s.per_group; v.first = 0;
            v.codes = dc; v.deltas = dd; v.rows = rows; v.x_stride = x_stride; v.out_stride = s.count; v.out = out;
            v.packed = f.packed;
            return v;
        };
        // Reference: the one-row site kernel, row by row.
        for (std::uint32_t r = 0; r < rows; ++r) {
            Q8A8View v = base_view(0);
            v.rows = 1;
            v.codes = dc + std::size_t{r} * x_stride;
            v.deltas = dd + std::size_t{r} * x_stride / 32U;
            v.out = out + std::size_t{r} * s.count;
            enqueue_q8_0_a8(v, nullptr);
        }
        const auto ref = down(out, outs);
        const auto report = [&](const char* kernel, const std::string& shape, const std::function<void(int)>& launch) {
            check(hipMemset(out, 0xff, outs * 4), "clear");
            launch(0);
            const auto got = down(out, outs);
            const bool exact = std::memcmp(got.data(), ref.data(), outs * 4) == 0;
            const double ms = time_ms(iterations, launch);
            const double warm = time_ms(std::max(5, iterations / 4), [&](int) { launch(0); });
            std::printf("{\"site\":\"%s\",\"format\":\"%s\",\"rows\":%u,\"kernel\":\"%s\",\"shape\":\"%s\",\"bytes\":%zu,\"ms\":%.4f,"
                        "\"gbs\":%.1f,\"warm_gbs\":%.1f,\"exact\":%s}\n",
                        s.name, f.name, rows, kernel, shape.c_str(), bytes, ms, static_cast<double>(bytes) / (ms * 1e-3) / 1e9,
                        static_cast<double>(bytes) / (warm * 1e-3) / 1e9, exact ? "true" : "false");
            std::fflush(stdout);
        };
        report("site", "engine", [&](int i) { enqueue_q8_0_a8(base_view(i), nullptr); });
        // The expert window kernel over the same weights: one "expert" per group, entry r of group g reads
        // activation row r * groups + g and writes output row r * groups + g (out[r * count + g * per_group + o]).
        {
            std::vector<std::uint32_t> experts(groups), begins(groups + 1), inputs, outputs;
            for (std::uint32_t g = 0; g < groups; ++g) {
                experts[g] = g;
                begins[g] = static_cast<std::uint32_t>(inputs.size());
                for (std::uint32_t r = 0; r < rows; ++r) { inputs.push_back(r * groups + g); outputs.push_back(r * groups + g); }
            }
            begins[groups] = static_cast<std::uint32_t>(inputs.size());
            auto* de = dev(experts);
            auto* db = dev(begins);
            auto* di = dev(inputs);
            auto* dout = dev(outputs);
            const bool fits = std::size_t{rows <= 1 ? 1U : rows <= 2 ? 2U : rows <= 4 ? 4U : 8U} * s.width * 9U / 8U <= 64U * 1024U;
            for (const int form : {0, 2}) {
                if (!fits) break;
                moe_kernel_tuning().shared = form;
                report("expert-window", "form" + std::to_string(form), [&](int i) {
                    RoutesProductView v;
                    v.weights = w + (static_cast<std::size_t>(i) % copies) * bytes;
                    v.expert_stride_bytes = std::uint64_t{s.per_group} * row_bytes;
                    v.row_bytes = row_bytes; v.width = s.width; v.outputs = s.per_group;
                    v.routes.experts = de; v.routes.begins = db; v.routes.inputs = di; v.routes.outputs = dout;
                    v.routes.expert_count = groups; v.routes.max_rows = rows;
                    v.codes = dc; v.deltas = dd; v.out = out;
                    enqueue_routes_product(f.format, v, nullptr);
                });
            }
            moe_kernel_tuning().shared = -1;
            (void)hipFree(de); (void)hipFree(db); (void)hipFree(di); (void)hipFree(dout);
        }
        const RnWideShape shapes[] = {{16, 1, true, false}, {16, 2, true, false}, {8, 1, false, false}, {8, 2, false, false},
                                      {16, 1, false, false}, {4, 4, false, true}, {8, 4, false, true}, {8, 2, false, true},
                                      {4, 8, false, true},
                                      // forms 2 (flat, wave sync), 3 (segments of 8 lanes), 4 (segments of 4 lanes)
                                      {4, 4, false, false, 2}, {8, 4, false, false, 2}, {8, 2, false, false, 2}, {4, 8, false, false, 2},
                                      {8, 4, false, false, 3}, {4, 4, false, false, 3}, {2, 4, false, false, 3}, {4, 8, false, false, 3},
                                      {8, 8, false, false, 3}, {8, 4, true, false, 3}, {4, 4, true, false, 3}, {4, 8, true, false, 3},
                                      {4, 8, false, false, 4}, {4, 8, true, false, 4}};
        for (const auto& sh : shapes) {
            Q8A8View probe = base_view(0);
            if (!rn_wide_supported(probe, sh)) continue;
            const std::string name = std::to_string(sh.outputs_per_wave) + "," + std::to_string(sh.waves) + "," + (sh.lds ? "1" : "0") +
                                     (sh.flat ? ",flat" : "") +
                                     (sh.form ? ",form" + std::to_string(sh.form) : "");
            report("rn", name, [&](int i) { enqueue_q8_0_a8_rn(base_view(i), sh, nullptr); });
        }
        if (rows >= 2) {
            const RnWideShape def = rn_wide_default_shape(base_view(0));
            if (rn_wide_supported(base_view(0), def))
                report("rn-default", std::to_string(def.outputs_per_wave) + "," + std::to_string(def.waves) + (def.flat ? ",flat" : ""),
                       [&](int i) { enqueue_q8_0_a8_rn(base_view(i), def, nullptr); });
        }
        // The A8 quantization of the site input (a separate launch inside the engine's op time).
        {
            const double ms = time_ms(iterations, [&](int) { enqueue_quantize_a8(dx, x_stride, rows, qc, qd, qs, nullptr); });
            std::printf("{\"site\":\"%s\",\"format\":\"%s\",\"rows\":%u,\"kernel\":\"quantize_a8\",\"ms\":%.4f}\n", s.name, f.name, rows, ms);
        }
        (void)hipFree(dc); (void)hipFree(dd); (void)hipFree(dx); (void)hipFree(qc); (void)hipFree(qd); (void)hipFree(qs); (void)hipFree(out);
    }
    (void)hipFree(w);
}

} // namespace

int main(int argc, char** argv) {
    const int iterations = argc > 1 ? std::atoi(argv[1]) : 40;
    const std::string site_filter = argc > 2 ? argv[2] : "all", format_filter = argc > 3 ? argv[3] : "all";
    // One rank of the tensor-parallel attention: q_b heads [32r, 32r + 32) (16,384 outputs of 1,280), o_a groups
    // [4r, 4r + 4) (4 x 1,024 outputs of 4,096 inputs each).
    // q_a and kv (5,120 -> 1,280 / 512, whole on each rank) for reference only (another site owns them).
    const Site sites[] = {{"q_b", 1280, 16384, 16384}, {"o_a", 4096, 4096, 1024}, {"q_a", 5120, 1280, 1280}, {"kv", 5120, 512, 512}};
    const Format formats[] = {{"q8", 0, 34, WeightFormat::q8_0}, {"u6", 6, 26, WeightFormat::u6}};
    for (const auto& f : formats) {
        if (format_filter != "all" && format_filter != f.name) continue;
        for (const auto& s : sites) {
            if (site_filter != "all" && site_filter.find(s.name) == std::string::npos) continue;
            bench(s, f, iterations);
        }
    }
    return 0;
}
