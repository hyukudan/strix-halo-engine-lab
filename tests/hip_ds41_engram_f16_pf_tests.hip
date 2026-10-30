// Native exactness gate and microbench for the Engram F16 prefill projection (enqueue_engram_f16_rows).
// No arguments: representative full 6144 -> 25600 shapes across rows 1..8192, plus stride/fallback guards.
// --bench [--rows N] [--iters N] [--copies N]: gate first, then old / pf event medians.
// Without --iters, use one timed launch per arm/round at rows >=1024; five balanced rounds.
#include "strix/hip_ds41_layer.hpp"

#include <hip/hip_runtime.h>

#include <algorithm>
#include <bit>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <random>
#include <stdexcept>
#include <string>
#include <vector>

namespace {
using namespace strix::ds41;
constexpr unsigned width = 6144, count = 25600;
constexpr std::size_t weight_elements = std::size_t{width} * count;

void check(hipError_t e, const char* what) {
    if (e != hipSuccess) throw std::runtime_error(std::string(what) + ": " + hipGetErrorString(e));
}
void select(bool on) {
    if (on ? unsetenv("STRIX_DS41_ENGRAM_F16_PF") : setenv("STRIX_DS41_ENGRAM_F16_PF", "0", 1))
        throw std::runtime_error("set Engram switch");
}
template <class T> struct Device {
    T* p = nullptr;
    explicit Device(std::size_t n) { check(hipMalloc(reinterpret_cast<void**>(&p), n * sizeof(T)), "allocate"); }
    ~Device() { if (p) (void)hipFree(p); }
    Device(const Device&) = delete;
    Device& operator=(const Device&) = delete;
};
struct Events {
    hipEvent_t a{}, b{};
    Events() { check(hipEventCreate(&a), "event a"); check(hipEventCreate(&b), "event b"); }
    ~Events() { (void)hipEventDestroy(a); (void)hipEventDestroy(b); }
};

bool equal(const float* a, const float* b, std::size_t n) {
    constexpr std::size_t chunk = 262144;
    std::vector<std::uint32_t> ha(chunk), hb(chunk);
    for (std::size_t i = 0; i < n; i += chunk) {
        const std::size_t k = std::min(chunk, n - i);
        check(hipMemcpy(ha.data(), a + i, k * sizeof(float), hipMemcpyDeviceToHost), "read old");
        check(hipMemcpy(hb.data(), b + i, k * sizeof(float), hipMemcpyDeviceToHost), "read new");
        if (std::memcmp(ha.data(), hb.data(), k * sizeof(float)) != 0) {
            for (std::size_t j = 0; j < k; ++j)
                if (ha[j] != hb[j]) {
                    std::fprintf(stderr, "first mismatch element %zu: %08x vs %08x\n", i + j, ha[j], hb[j]);
                    break;
                }
            return false;
        }
    }
    return true;
}

void enqueue_arm(unsigned arm, const F16RowsView& v) {
    // main() clears the old generic controls. Benchmark explicit kernels, not the switchable wrapper.
    if (arm == 0U) enqueue_f16_rows(v, nullptr);
    else enqueue_engram_f16_pf(v, nullptr);
}

void verify(F16RowsView v, float* other, unsigned copies) {
    const std::size_t n = std::size_t{v.rows} * v.out_stride;
    for (unsigned c = 0; c < copies; ++c) {
        F16RowsView a = v, b = v;
        a.weight += c * weight_elements;
        b.weight = a.weight;
        b.out = other;
        // Different poison patterns detect missing stores; padding must retain the identical sentinel below.
        check(hipMemset(a.out, 0xa5, n * sizeof(float)), "poison old");
        check(hipMemset(b.out, 0x5a, n * sizeof(float)), "poison new");
        if (v.out_stride > v.count) {
            // Identical guard values let the full-buffer comparison include padding.
            check(hipMemset(a.out, 0xa5, n * sizeof(float)), "guard old");
            check(hipMemset(b.out, 0xa5, n * sizeof(float)), "guard new");
        }
        select(false);
        enqueue_engram_f16_rows(a, nullptr);
        select(true);
        enqueue_engram_f16_rows(b, nullptr);
        check(hipDeviceSynchronize(), "bitwise sync");
        if (!equal(a.out, b.out, n)) throw std::runtime_error("old/default mismatch");
        if (v.out_stride > v.count) {
            std::vector<std::uint32_t> row(v.out_stride);
            for (unsigned r = 0; r < v.rows; ++r) {
                check(hipMemcpy(row.data(), b.out + std::size_t{r} * v.out_stride, row.size() * sizeof(float), hipMemcpyDeviceToHost), "guards");
                for (unsigned o = v.count; o < v.out_stride; ++o)
                    if (row[o] != 0xa5a5a5a5U) throw std::runtime_error("output padding overwritten");
            }
        }
    }
    std::printf("6144 -> %u rows %u strides %u/%u: BITWISE EQUAL (%s, %u copies)\n", v.count, v.rows, v.x_stride, v.out_stride,
                engram_f16_pf_supported(v) ? "pf" : "fallback", copies);
    std::fflush(stdout);
}

unsigned bench_iters(unsigned rows) {
    if (rows <= 128U) return 6U;
    if (rows <= 512U) return 2U;
    return 1U;
}

void bench(F16RowsView v, unsigned iters, unsigned copies) {
    if (!engram_f16_pf_supported(v)) throw std::invalid_argument("bench requires rows 32..8192 and a full aligned view");
    std::vector<float> samples[2];
    Events ev;
    // Alternating order, reversed every round; all arms use the same copies (including with iters == 1).
    for (unsigned round = 0; round < 5; ++round)
        for (unsigned j = 0; j < 2; ++j) {
            const unsigned arm = (round + j) % 2U;
            auto warm = v;
            warm.weight += ((round + copies - 1U) % copies) * weight_elements;
            enqueue_arm(arm, warm);
            check(hipDeviceSynchronize(), "warmup");
            check(hipEventRecord(ev.a), "start");
            for (unsigned i = 0; i < iters; ++i) {
                auto t = v;
                t.weight += ((i + round) % copies) * weight_elements;
                enqueue_arm(arm, t);
            }
            check(hipEventRecord(ev.b), "stop");
            check(hipEventSynchronize(ev.b), "timing sync");
            float ms = 0.0F;
            check(hipEventElapsedTime(&ms, ev.a, ev.b), "elapsed");
            samples[arm].push_back(ms / static_cast<float>(iters));
        }
    for (auto& s : samples) std::sort(s.begin(), s.end());
    std::printf("BENCH rows %u old %.6f ms pf %.6f ms pf/old %.4f copies %u iters %u (median of 5)\n", v.rows, samples[0][2],
                samples[1][2], samples[1][2] / samples[0][2], copies, iters);
    std::fflush(stdout);
}

void rejected(const F16RowsView& v) {
    if (engram_f16_pf_supported(v)) throw std::runtime_error("unsupported view admitted");
    // Guard the explicit entry as well as its predicate; F8 never reaches the tile.
    bool threw = false;
    try {
        enqueue_engram_f16_pf(v, nullptr);
    } catch (const std::invalid_argument&) { threw = true; }
    if (!threw) throw std::runtime_error("unsupported explicit launch accepted");
}
}  // namespace

int main(int argc, char** argv) try {
    bool timing = false;
    unsigned only_rows = 0, iters = 0, copies = 1;  // zero internally means automatic per-shape iteration count
    for (int i = 1; i < argc; ++i) {
        const std::string arg = argv[i];
        if (arg == "--bench") { timing = true; continue; }
        if ((arg != "--rows" && arg != "--iters" && arg != "--copies") || i + 1 == argc)
            throw std::invalid_argument("usage: [--bench] [--rows 1..8192] [--iters 1..10000] [--copies 1..8]");
        const auto n = std::stoul(argv[++i]);
        if (n == 0 || n > (arg == "--rows" ? 8192U : arg == "--copies" ? 8U : 10000U))
            throw std::invalid_argument("argument out of range");
        if (arg == "--rows") only_rows = static_cast<unsigned>(n);
        else if (arg == "--copies") copies = static_cast<unsigned>(n);
        else iters = static_cast<unsigned>(n);
    }
    int devices = 0;
    const auto device_status = hipGetDeviceCount(&devices);
    if (device_status == hipErrorNoDevice || (device_status == hipSuccess && devices == 0)) return 77;
    check(device_status, "device count");
    // Fix the old arm to the production dispatch, independent of the invoking shell's tuning overrides.
    if (unsetenv("STRIX_DS41_F16_TILED") || unsetenv("STRIX_DS41_F16_TILE")) throw std::runtime_error("clear old tile controls");
    if (timing && only_rows != 0U && only_rows < 32U) throw std::invalid_argument("bench rows must be >=32");
    std::vector<unsigned> shapes = only_rows ? std::vector<unsigned>{only_rows} : timing ?
        std::vector<unsigned>{32, 128, 512, 1024, 2048, 4096, 8192} :
        std::vector<unsigned>{1, 2, 4, 8, 9, 16, 17, 31, 32, 33, 47, 48, 49, 63, 64, 65, 95, 96, 97, 127, 128, 129,
                              511, 512, 513, 1023, 1024, 1025, 2048, 4096, 8192};
    const unsigned max_rows = std::max(1024U, *std::max_element(shapes.begin(), shapes.end()));
    Device<std::uint16_t> w(weight_elements * copies);
    Device<float> x(std::size_t{max_rows} * (width + 4U) + 4U), a(std::size_t{max_rows} * (count + 4U)),
                  b(std::size_t{max_rows} * (count + 4U));
    std::mt19937 rng(41030);
    std::vector<std::uint16_t> hw(weight_elements);
    for (unsigned c = 0; c < copies; ++c) {
        for (auto& h : hw) h = static_cast<std::uint16_t>((rng() % 0x7c00U) | ((rng() & 1U) << 15U));
        check(hipMemcpy(w.p + c * weight_elements, hw.data(), hw.size() * sizeof(hw[0]), hipMemcpyHostToDevice), "weights");
    }
    std::vector<float> hx(std::size_t{max_rows} * (width + 4U) + 4U);
    // Engram rows are BF16-rounded FP32. Include both zeros and a broad finite exponent/mantissa range.
    for (auto& f : hx) {
        const auto exponent = (static_cast<std::uint32_t>(rng()) % 25U + 115U) << 23U;
        f = std::bit_cast<float>((static_cast<std::uint32_t>(rng()) & 0x807f0000U) | exponent);
    }
    for (std::size_t i = 0; i < hx.size(); i += 37U) hx[i] = i % 3U == 0U ? 0.0F : (i % 3U == 1U ? -0.0F : 0x1p-100F);
    check(hipMemcpy(x.p, hx.data(), hx.size() * sizeof(float), hipMemcpyHostToDevice), "inputs");
    F16RowsView v;
    v.weight = w.p; v.width = width; v.count = count; v.x = x.p; v.x_stride = width; v.out_stride = count; v.out = a.p;
    // Pin admission independently of byte equality: falling back everywhere would pass equality too.
    for (const unsigned rows : {1U, 16U, 31U, 32U, 33U, 48U, 64U, 1024U, 8192U, 8193U}) {
        v.rows = rows;
        if (engram_f16_pf_supported(v) != (rows >= 32U && rows <= 8192U)) throw std::runtime_error("row admission changed");
        const bool admitted = rows >= 32U && rows <= 8192U && rows != 64U && rows != 8193U;  // 64: 96 padded rows vs 64
        if (engram_f16_pf_admitted(v) != admitted) throw std::runtime_error("default admission changed");
    }
    for (const unsigned rows : shapes) {
        v.rows = rows;
        verify(v, b.p, copies);
        if (timing) bench(v, iters ? iters : bench_iters(rows), copies);
    }
    // Padded strides and guards: a row predicate cannot mask a bad view guard.
    v.rows = 65; v.x_stride = width + 4U; v.out_stride = count + 4U;
    verify(v, b.p, 1);
    for (const unsigned rows : {48U, 1024U}) {
        v.rows = rows;
        if (!engram_f16_pf_supported(v)) throw std::runtime_error("padded view rejected");
        verify(v, b.p, 1);
        v.x = x.p + 1;
        rejected(v);
        verify(v, b.p, 1);
        v.x = x.p; v.weight = w.p + 1;
        rejected(v);
        v.weight = w.p; v.first = 1; v.count = count - 1U;
        rejected(v);
        verify(v, b.p, 1);
        v.first = 0; v.count = count; v.weight_f8 = reinterpret_cast<const std::uint8_t*>(w.p);
        rejected(v);  // deliberately no F8 execution: w contains F16, not packed F8 data
        v.weight_f8 = nullptr;
        v.x_stride = width + 1U;
        rejected(v);
        v.x_stride = width - 4U;
        rejected(v);
        v.x_stride = width + 4U; v.out_stride = count - 1U;
        rejected(v);
        v.out_stride = count + 4U;
    }
    std::printf("ALL BITWISE EQUAL\n");
    return 0;
} catch (const std::exception& e) {
    std::fprintf(stderr, "error: %s\n", e.what());
    return 1;
}
