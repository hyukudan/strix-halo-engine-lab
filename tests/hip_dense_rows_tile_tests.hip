// Device checks of the rows tile of launch_hip_dense_rows (strix/dense_rows_tile.hpp), synthetic data, no model:
//  1. identity: rows form (one launch) == one single-row launch per row (row_loop = true), BITWISE, on
//     F32 512 x 3072 fp32 input rows 2048 (model-B router), F32 16 x 3072 fp32 rows 37 (indexer proj, ragged rows),
//     BF16 1003 x 3000 bf16 input rows 13 with strides (ragged outputs, width not a multiple of 256),
//     Q8_0 300 x 2048 bf16 rows 9, F16 64 x 768 fp32 rows 2; outputs outside the stored range stay untouched.
//  2. timing (printed, not asserted): router geometry, rows 2048: per-row loop vs rows tile, median of 5.
//  3. grouped rows (HipDenseRowsView::groups) == one plain launch per group, BITWISE, wave and block form and group_loop:
//     BF16 expert slices 1536 x 1536 with device ids (shared / per-group input), BF16 absorb heads (strided in / out),
//     Q8_0 block form; a slice id outside the weights leaves NaN in that group only. Timing printed (expert gate, K_b).
// Exit 77 when no HIP device is present.
#include "strix/hip_lm_head_launcher.hpp"

#include <hip/hip_runtime.h>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <stdexcept>
#include <string>
#include <vector>

namespace {

int failures = 0;
void check(bool ok, const char* what) {
    std::printf("%s %s\n", ok ? "ok  " : "FAIL", what);
    if (!ok) ++failures;
}
void hip(hipError_t e, const char* what) {
    if (e != hipSuccess) throw std::runtime_error(std::string(what) + ": " + hipGetErrorString(e));
}
std::uint32_t lcg(std::uint32_t& s) {
    s = s * 1664525U + 1013904223U;
    return s;
}
float random_float(std::uint32_t& s) {
    const auto u = lcg(s);
    const float m = static_cast<float>(u >> 8U) / 16777216.0F - 0.5F;
    return (u & 7U) == 0U ? m * 16.0F : m;
}
std::uint16_t bf16_bits(float f) {
    std::uint32_t b = 0;
    std::memcpy(&b, &f, 4);
    return static_cast<std::uint16_t>(b >> 16U);
}
std::uint16_t f16_bits(std::uint32_t& s) {  // finite half, exponent 9 .. 16
    const auto u = lcg(s);
    return static_cast<std::uint16_t>(((u >> 31U) << 15U) | ((9U + (u >> 4U) % 8U) << 10U) | ((u >> 8U) & 0x3ffU));
}

template <typename T>
T* upload(const std::vector<T>& h) {
    T* d = nullptr;
    hip(hipMalloc(reinterpret_cast<void**>(&d), h.size() * sizeof(T)), "malloc");
    hip(hipMemcpy(d, h.data(), h.size() * sizeof(T), hipMemcpyHostToDevice), "upload");
    return d;
}
std::vector<float> download(const float* d, std::size_t n) {
    std::vector<float> h(n);
    hip(hipMemcpy(h.data(), d, n * sizeof(float), hipMemcpyDeviceToHost), "download");
    return h;
}

struct Case {
    std::uint32_t ggml_type, outputs, width, rows, x_stride, out_stride;
    bool bf16_input;
    const char* what;
};

// Weight bytes for [outputs][width] of the given type, random.
std::vector<std::uint8_t> weights(const Case& c, std::uint32_t& s) {
    std::vector<std::uint8_t> w;
    const std::size_t n = std::size_t{c.outputs} * c.width;
    if (c.ggml_type == 0U) {
        w.resize(n * 4U);
        for (std::size_t i = 0; i < n; ++i) {
            const float f = random_float(s);
            std::memcpy(&w[i * 4U], &f, 4);
        }
    } else if (c.ggml_type == 1U || c.ggml_type == 30U) {
        w.resize(n * 2U);
        for (std::size_t i = 0; i < n; ++i) {
            const std::uint16_t b = c.ggml_type == 1U ? f16_bits(s) : bf16_bits(random_float(s));
            std::memcpy(&w[i * 2U], &b, 2);
        }
    } else {  // Q8_0: {f16 d, int8 q[32]} per 32 values
        const std::size_t blocks = n / 32U;
        w.resize(blocks * 34U);
        for (std::size_t b = 0; b < blocks; ++b) {
            const std::uint16_t d = f16_bits(s);
            std::memcpy(&w[b * 34U], &d, 2);
            for (std::size_t q = 0; q < 32U; ++q) w[b * 34U + 2U + q] = static_cast<std::uint8_t>(lcg(s) >> 24U);
        }
    }
    return w;
}

void identity(const Case& c, bool time_it) {
    std::uint32_t s = 0x9e3779b9U ^ c.outputs ^ (c.rows << 12U);
    const auto w = weights(c, s);
    std::vector<float> xf(std::size_t{c.rows} * c.x_stride);
    for (auto& f : xf) f = random_float(s);
    std::vector<std::uint16_t> xb(xf.size());
    for (std::size_t i = 0; i < xf.size(); ++i) xb[i] = bf16_bits(xf[i]);
    auto* dw = upload(w);
    void* dx = c.bf16_input ? static_cast<void*>(upload(xb)) : static_cast<void*>(upload(xf));
    const std::size_t on = std::size_t{c.rows} * c.out_stride;
    const std::vector<float> sentinel(on, -12345.0F);
    auto* da = upload(sentinel);
    auto* db = upload(sentinel);
    strix::HipDenseRowsView v;
    v.x = dx;
    v.input = c.bf16_input ? strix::HipActivationStorage::bf16 : strix::HipActivationStorage::fp32;
    v.weights = dw;
    v.ggml_type = c.ggml_type;
    v.width = c.width;
    v.outputs = c.outputs;
    v.rows = c.rows;
    v.x_stride = c.x_stride;
    v.out_stride = c.out_stride;
    v.out = da;
    v.row_loop = true;
    strix::launch_hip_dense_rows(v, nullptr);
    v.out = db;
    v.row_loop = false;
    strix::launch_hip_dense_rows(v, nullptr);
    hip(hipDeviceSynchronize(), "sync");
    const auto a = download(da, on), b = download(db, on);
    bool same = std::memcmp(a.data(), b.data(), on * sizeof(float)) == 0;
    bool written = true;
    for (std::uint32_t r = 0; r < c.rows; ++r)
        for (std::uint32_t o = 0; o < c.out_stride; ++o) {
            const bool stored = std::memcmp(&b[std::size_t{r} * c.out_stride + o], &sentinel[0], 4) != 0;
            if (stored != (o < c.outputs)) written = false;
        }
    check(same && written, c.what);
    if (time_it) {
        hipEvent_t e0, e1;
        hip(hipEventCreate(&e0), "event");
        hip(hipEventCreate(&e1), "event");
        float ms[2][5] = {};
        for (int rep = 0; rep < 5; ++rep)
            for (int form = 0; form < 2; ++form) {
                v.row_loop = form == 0;
                hip(hipEventRecord(e0, nullptr), "record");
                strix::launch_hip_dense_rows(v, nullptr);
                hip(hipEventRecord(e1, nullptr), "record");
                hip(hipEventSynchronize(e1), "sync");
                hip(hipEventElapsedTime(&ms[form][rep], e0, e1), "elapsed");
            }
        std::sort(ms[0], ms[0] + 5);
        std::sort(ms[1], ms[1] + 5);
        std::printf("TIMING router F32 %u x %u rows %u: per-row loop %.3f ms, rows tile %.3f ms (median of 5)\n",
                    c.outputs, c.width, c.rows, static_cast<double>(ms[0][2]), static_cast<double>(ms[1][2]));
        hip(hipEventDestroy(e0), "event");
        hip(hipEventDestroy(e1), "event");
    }
    hip(hipFree(dw), "free");
    hip(hipFree(dx), "free");
    hip(hipFree(da), "free");
    hip(hipFree(db), "free");
}

// Grouped rows (HipDenseRowsView::groups): G products [outputs][width] in one launch, group g on slice ids[g] (device ids)
// or g, its input at x + g * gx, its outputs at out + g * go. Reference = one plain single-row launch per group on the
// host-offset pointers. Grouped (wave and block form) and group_loop must equal it BITWISE; a slice id outside the
// weights must leave NaN in that group only.
struct Grouped {
    std::uint32_t ggml_type, outputs, width, slices, groups, gx, go;
    bool ids, bf16_input, bad_id;
    const char* what;
};

void grouped(const Grouped& c, bool time_it) {
    std::uint32_t s = 0x2545f491U ^ c.outputs ^ (c.groups << 16U);
    const Case wc{c.ggml_type, c.outputs * c.slices, c.width, 1U, c.width, c.outputs, false, c.what};
    const auto w = weights(wc, s);
    const std::uint64_t slice_bytes = w.size() / c.slices;
    const std::uint32_t inputs = c.gx ? c.groups : 1U;
    std::vector<float> xf(std::size_t{inputs - 1U} * c.gx + c.width);
    for (auto& f : xf) f = random_float(s);
    std::vector<std::uint16_t> xb(xf.size());
    for (std::size_t i = 0; i < xf.size(); ++i) xb[i] = bf16_bits(xf[i]);
    std::vector<std::uint32_t> ids(c.groups);
    for (std::uint32_t g = 0; g < c.groups; ++g) ids[g] = c.ids ? (g * 7U + 3U) % c.slices : g;  // distinct when slices % 7 != 0
    const std::uint32_t bad = c.bad_id ? c.groups / 2U : c.groups;
    if (c.bad_id) ids[bad] = c.slices;
    auto* dw = upload(w);
    auto* dids = upload(ids);
    void* dx = c.bf16_input ? static_cast<void*>(upload(xb)) : static_cast<void*>(upload(xf));
    const std::size_t elem = c.bf16_input ? 2U : 4U;
    const std::size_t on = std::size_t{c.groups} * c.go;
    const std::vector<float> sentinel(on, -12345.0F);
    float* outs[4] = {upload(sentinel), upload(sentinel), upload(sentinel), upload(sentinel)};
    strix::HipDenseRowsView v;
    v.input = c.bf16_input ? strix::HipActivationStorage::bf16 : strix::HipActivationStorage::fp32;
    v.ggml_type = c.ggml_type;
    v.width = c.width;
    v.outputs = c.outputs;
    auto reference = [&] {  // one plain launch per group
        for (std::uint32_t g = 0; g < c.groups; ++g) {
            if (g == bad) continue;
            strix::HipDenseRowsView p = v;
            p.x = static_cast<const std::uint8_t*>(dx) + std::size_t{g} * c.gx * elem;
            p.weights = dw + ids[g] * slice_bytes;
            p.out = outs[0] + std::size_t{g} * c.go;
            strix::launch_hip_dense_rows(p, nullptr);
        }
    };
    auto group_launch = [&](float* out, bool loop, bool block) {
        strix::HipDenseRowsView p = v;
        p.x = dx;
        p.weights = dw;
        p.out = out;
        p.groups = c.groups;
        p.group_ids = c.ids || c.bad_id ? dids : nullptr;
        p.group_slices = c.slices;
        p.group_weight_bytes = slice_bytes;
        p.group_x_stride = c.gx;
        p.group_out_stride = c.go;
        p.group_loop = loop;
        p.block_rows = block;
        strix::launch_hip_dense_rows(p, nullptr);
    };
    reference();
    group_launch(outs[1], false, false);
    group_launch(outs[2], true, false);
    group_launch(outs[3], false, true);
    hip(hipDeviceSynchronize(), "sync");
    const auto ref = download(outs[0], on);
    bool same = true, nan_ok = true, written = true;
    for (int k = 1; k < 4; ++k) {
        const auto got = download(outs[k], on);
        for (std::uint32_t g = 0; g < c.groups; ++g)
            for (std::uint32_t o = 0; o < c.go; ++o) {
                const std::size_t i = std::size_t{g} * c.go + o;
                if (o >= c.outputs) {
                    if (std::memcmp(&got[i], &sentinel[0], 4) != 0) written = false;
                } else if (g == bad) {
                    if (!std::isnan(got[i])) nan_ok = false;
                } else if (std::memcmp(&got[i], &ref[i], 4) != 0) {
                    same = false;
                }
            }
    }
    check(same && nan_ok && written, c.what);
    if (time_it) {
        hipEvent_t e0, e1;
        hip(hipEventCreate(&e0), "event");
        hip(hipEventCreate(&e1), "event");
        float ms[2][5] = {};
        for (int rep = 0; rep < 5; ++rep)
            for (int form = 0; form < 2; ++form) {
                hip(hipEventRecord(e0, nullptr), "record");
                if (form == 0) reference();
                else group_launch(outs[1], false, false);
                hip(hipEventRecord(e1, nullptr), "record");
                hip(hipEventSynchronize(e1), "sync");
                hip(hipEventElapsedTime(&ms[form][rep], e0, e1), "elapsed");
            }
        std::sort(ms[0], ms[0] + 5);
        std::sort(ms[1], ms[1] + 5);
        const double mb = static_cast<double>(slice_bytes) * c.groups / 1e6;
        std::printf("TIMING grouped %s: %u x %u x %u (%.1f MB): per-group launches %.3f ms, grouped %.3f ms (%.0f GB/s), median of 5\n",
                    c.ggml_type == 30U ? "BF16" : "type", c.groups, c.outputs, c.width, mb, static_cast<double>(ms[0][2]),
                    static_cast<double>(ms[1][2]), mb / static_cast<double>(ms[1][2]));
        hip(hipEventDestroy(e0), "event");
        hip(hipEventDestroy(e1), "event");
    }
    hip(hipFree(dw), "free");
    hip(hipFree(dids), "free");
    hip(hipFree(dx), "free");
    for (auto* o : outs) hip(hipFree(o), "free");
}

}  // namespace

int main() {
    int devices = 0;
    if (hipGetDeviceCount(&devices) != hipSuccess || devices == 0) {
        std::printf("no HIP device: skipped\n");
        return 77;
    }
    try {
        identity({0U, 512U, 3072U, 2048U, 3072U, 512U, false,
                  "rows tile == per-row, bitwise: F32 512 x 3072 fp32 rows 2048 (router)"}, true);
        identity({0U, 16U, 3072U, 37U, 3072U, 16U, false,
                  "rows tile == per-row, bitwise: F32 16 x 3072 fp32 rows 37 (indexer proj)"}, false);
        identity({30U, 1003U, 3000U, 13U, 3072U, 1010U, true,
                  "rows tile == per-row, bitwise: BF16 1003 x 3000 bf16 rows 13, strides"}, false);
        identity({8U, 300U, 2048U, 9U, 2048U, 300U, true,
                  "rows tile == per-row, bitwise: Q8_0 300 x 2048 bf16 rows 9"}, false);
        identity({1U, 64U, 768U, 2U, 768U, 64U, false,
                  "rows tile == per-row, bitwise: F16 64 x 768 fp32 rows 2"}, false);
        grouped({30U, 1536U, 1536U, 64U, 16U, 0U, 1536U, true, false, false,
                 "grouped == per-group, bitwise: BF16 16 of 64 slices 1536 x 1536, device ids, shared fp32 input (expert gate)"}, true);
        grouped({30U, 1536U, 1536U, 64U, 16U, 1536U, 1536U, true, false, false,
                 "grouped == per-group, bitwise: BF16 16 slices 1536 x 1536, input per group (expert down)"}, false);
        grouped({30U, 512U, 128U, 32U, 32U, 192U, 576U, false, false, false,
                 "grouped == per-group, bitwise: BF16 32 heads 512 x 128, strided in / out (absorb K_b)"}, true);
        grouped({30U, 256U, 512U, 32U, 32U, 512U, 256U, false, false, false,
                 "grouped == per-group, bitwise: BF16 32 heads 256 x 512 (absorb V_b)"}, false);
        grouped({8U, 64U, 256U, 9U, 4U, 256U, 70U, true, true, false,
                 "grouped == per-group, bitwise: Q8_0 4 of 9 slices 64 x 256, bf16 input (block form)"}, false);
        grouped({30U, 96U, 64U, 8U, 6U, 0U, 96U, true, false, true,
                 "grouped: slice id outside the weights -> NaN in that group only, others bitwise"}, false);
    } catch (const std::exception& e) {
        std::printf("FAIL exception: %s\n", e.what());
        return 1;
    }
    std::printf("%s\n", failures == 0 ? "PASS" : "FAIL");
    return failures == 0 ? 0 : 1;
}
