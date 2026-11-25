// No-model fixture of the resident dense routes per weight storage (bf16, q8_0, q4_g64, q6_g64).
// Checks (through the resident entry points core::enqueue_resident_q8_rows / enqueue_resident_q8_projections):
//   - every storage against the host reference decode (finite, within 1e-4 relative for the FP32 sweep, the FP16
//     operand bound 2^-10 + K 2^-23 for low-bit), rows past the count untouched:
//     bf16 / q8_0 at the sweep buckets 1..16, low-bit at 1..129 rows (both routes, every row tile count, two row blocks);
//   - low-bit bits: a row's output is bit-identical at every row count, in a grouped launch, in a list mixed with q8_0
//     (both orders), and on the forced small-M schedule and the forced tile at the same rows (one arithmetic);
// then weight-read bandwidth per row count (each timed pass walks >= 512 MiB of distinct 16-byte aligned copies, dram=1;
// tiny shapes are capped at 4096 copies, dram=0), forced small / tile timings around the switch point, and with the
// default shapes q | k | v (3072 + 1024 + 1024 outputs, K 2304) as one grouped launch.
// Lever rows-select (mode `rows-select N K ...`): its launches against the incumbent route they select, bit for bit,
// alone and in a list, then interleaved timings of the f16 group-scale form, the lever and the FP32 sweep at 1..8 rows.
// Usage: strix-hip-dense-lowbit-bench [N K ...], N > 0, K a positive multiple of 128
//        (default: 3072 2304  1024 2304  2304 3072  384 2304  1000 2304  17 256; the last two have output tails)
// Output: "check ... PASS|FAIL" lines, "bench fmt=F N=.. K=.. route=entry|small|tile rows=R bytes=.. dram=.. us=.. GBps=.."
// lines, then "lowbit bench checks PASS|FAIL".
#include "strix/hip/legacy_intrinsics.hpp"
#include "strix/hip/legacy_launch.hpp"
#include "core/resident_q8_rows.hpp"
#include "core/f16_tile_gemm_small.hpp"
#include "core/iu4_onfly.hpp"
#include "inkling2/inkling2_mini_dense_hip.hpp"

#include <algorithm>
#include <array>
#include <span>
#include <cmath>
#include <cstdlib>
#include <cstring>
#include <iostream>
#include <random>
#include <string>
#include <vector>

namespace {

namespace wei = strix::weight;
namespace core = strix::core;
namespace tg = strix::core::f16_tile_gemm;
namespace wd = strix::core::weight_decode;
constexpr auto f16_group_scale = core::LowBitArithmetic::f16_group_scale;

void ok(hipError_t e, const char* what) {
    if (e != hipSuccess) {
        std::cerr << what << ": " << hipGetErrorString(e) << '\n';
        std::exit(2);
    }
}

std::uint16_t to_bf16(float f) {
    std::uint32_t x;
    std::memcpy(&x, &f, 4);
    return static_cast<std::uint16_t>((x + 0x7fffU + ((x >> 16U) & 1U)) >> 16U);
}

struct Packed {
    wd::Format format;
    wei::FormatId storage;
    std::size_t row_bytes;
    std::vector<std::uint8_t> bytes;
    std::vector<float> dequant;  // [N][K] reference weights
};

Packed pack(wd::Format format, const std::vector<std::uint16_t>& bits, unsigned n, unsigned k) {
    Packed p{format, wei::kBf16WeightStorage, wd::row_bytes(format, k), {}, std::vector<float>(std::size_t(n) * k)};
    p.bytes.resize(p.row_bytes * n);
    std::vector<float> w(bits.size());
    for (std::size_t i = 0; i < w.size(); ++i) w[i] = core::onfly::bf16_bits_to_f32(bits[i]);
    switch (format) {
    case wd::Format::bf16:
        std::memcpy(p.bytes.data(), bits.data(), p.bytes.size());
        p.dequant = w;
        break;
    case wd::Format::q8_0: {
        p.storage = wei::kQ8WeightStorage;
        p.bytes = core::onfly::bf16_to_q8_0_host(bits.data(), n, k);
        for (std::size_t i = 0; i < w.size(); ++i) {
            const auto* blk = p.bytes.data() + (i / k) * p.row_bytes + (i % k) / 32U * 34U;
            p.dequant[i] = core::onfly::f16_bits_to_f32(wei::Q8Storage::scale_bits(blk)) *
                           float(wei::Q8Storage::code(blk, (i % k) % 32U));
        }
        break;
    }
    case wd::Format::q4_g64:
    case wd::Format::q6_g64: {
        const bool q4 = format == wd::Format::q4_g64;
        p.storage = q4 ? wei::kQ4G64WeightStorage : wei::kQ6G64WeightStorage;
        const unsigned block = q4 ? wei::Q4G64Storage::bytes : wei::Q6G64Storage::bytes;
        for (std::size_t b = 0; b < w.size() / 128U; ++b) {
            auto* out = p.bytes.data() + b * block;
            const bool encoded = q4 ? wei::lowbit::encode_block<wei::Q4G64Storage>(w.data() + b * 128U, out)
                                    : wei::lowbit::encode_block<wei::Q6G64Storage>(w.data() + b * 128U, out);
            if (!encoded) std::exit(3);
            for (unsigned j = 0; j < 128U; ++j)
                p.dequant[b * 128U + j] = q4 ? wei::lowbit::decode<wei::Q4G64Storage>(out, j)
                                             : wei::lowbit::decode<wei::Q6G64Storage>(out, j);
        }
        break;
    }
    }
    return p;
}

const char* name(wd::Format f) {
    return f == wd::Format::bf16 ? "bf16" : f == wd::Format::q8_0 ? "q8_0" : f == wd::Format::q4_g64 ? "q4_g64" : "q6_g64";
}
bool lowbit(wd::Format f) { return f == wd::Format::q4_g64 || f == wd::Format::q6_g64; }
std::size_t aligned(std::size_t bytes) { return (bytes + 15U) / 16U * 16U; }

// `copies` distinct 16-byte aligned copies of `bytes` (the resident row contract), enough that a pass reads DRAM.
struct Copies {
    std::uint8_t* w{};
    std::size_t stride{}, copies{}, dram_copies{};
};
Copies upload(const std::uint8_t* bytes, std::size_t size) {
    Copies d;
    d.stride = aligned(size);
    d.dram_copies = (std::size_t(512) << 20U) / d.stride + 1U;
    d.copies = std::max<std::size_t>(8U, std::min<std::size_t>(d.dram_copies, 4096U));
    ok(hipMalloc(&d.w, d.copies * d.stride), "malloc w");
    ok(hipMemcpy(d.w, bytes, size, hipMemcpyHostToDevice), "copy w");
    for (std::size_t done = 1; done < d.copies;) {
        const std::size_t step = std::min(done, d.copies - done);
        ok(hipMemcpy(d.w + done * d.stride, d.w, step * d.stride, hipMemcpyDeviceToDevice), "replicate w");
        done += step;
    }
    return d;
}

wei::ResidentMatrixView view(const Packed& p, const std::uint8_t* w, unsigned n, unsigned k) {
    return {{w, p.bytes.size()}, p.storage, n, k, p.row_bytes};
}

enum class Route { entry, small, tile, rt1, rt2, fp32_sweep };
const char* name(Route r) {
    constexpr const char* names[] = {"entry", "small", "tile", "small_rt1", "small_rt2", "fp32-sweep"};
    return names[static_cast<int>(r)];
}

hipError_t launch(const Packed& p, const std::uint8_t* w, unsigned n, unsigned k, const float* x, float* y,
                  unsigned rows, Route route = Route::entry) {
    if (p.format == wd::Format::bf16) {
        const core::dense_linear::View dense{w, k, n, p.row_bytes};
        if (rows == 1U) core::detail::launch_q8_tight_rows<1, wd::Format::bf16>(dense, x, y, rows, nullptr);
        else if (rows <= 8U) core::detail::launch_q8_tight_rows<8, wd::Format::bf16>(dense, x, y, rows, nullptr);
        else core::detail::launch_q8_tight_rows<16, wd::Format::bf16>(dense, x, y, rows, nullptr);
        return hipGetLastError();
    }
    if (route == Route::fp32_sweep)
        return core::enqueue_resident_q8_rows(view(p, w, n, k), n, x, rows, y, nullptr);
    if (route == Route::entry)
        return lowbit(p.format)
            ? core::enqueue_resident_lowbit_rows(view(p, w, n, k), n, x, rows, y, nullptr, f16_group_scale)
            : core::enqueue_resident_q8_rows(view(p, w, n, k), n, x, rows, y, nullptr);
    tg::Q8ProjectionGroup<1> group{};
    group.append({w, n, y});
    const bool q4 = p.format == wd::Format::q4_g64;
    if (route == Route::small) {  // the small schedule with this device's block plan
        const bool wide = core::lowbit_wide_block(rows, group.first_panel[group.count], core::detail::current_lowbit_row_plan());
        return q4 ? tg::small_m::gemm_lowbit<wei::Q4G64Storage>(x, rows, k, group, nullptr, wide)
                  : tg::small_m::gemm_lowbit<wei::Q6G64Storage>(x, rows, k, group, nullptr, wide);
    }
    if (route != Route::tile) {  // one row-tile instance of the small schedule, as many row blocks as the rows need
        const auto forced = [&](auto storage) {
            using S = decltype(storage);
            if (route == Route::rt1) tg::small_m::launch<S, 1U>(x, rows, k, group, nullptr);
            else tg::small_m::launch<S, 2U>(x, rows, k, group, nullptr);
            return hipGetLastError();
        };
        return q4 ? forced(wei::Q4G64Storage{}) : forced(wei::Q6G64Storage{});
    }
    return q4 ? tg::gemm_lowbit_projections<wei::Q4G64Storage>(x, rows, k, group, nullptr)
              : tg::gemm_lowbit_projections<wei::Q6G64Storage>(x, rows, k, group, nullptr);
}

// Mean microseconds per launch of `run(copy)` over three passes of all copies (after one warm pass).
template <class Run>
double time_us(std::size_t copies, const Run& run) {
    hipEvent_t a, b;
    ok(hipEventCreate(&a), "event");
    ok(hipEventCreate(&b), "event");
    for (std::size_t c = 0; c < copies; ++c) ok(run(c), "warm");
    constexpr int passes = 3;
    ok(hipEventRecord(a, nullptr), "record");
    for (int it = 0; it < passes; ++it)
        for (std::size_t c = 0; c < copies; ++c) ok(run(c), "run");
    ok(hipEventRecord(b, nullptr), "record");
    ok(hipEventSynchronize(b), "sync");
    float ms = 0.0F;
    ok(hipEventElapsedTime(&ms, a, b), "elapsed");
    ok(hipEventDestroy(a), "event");
    ok(hipEventDestroy(b), "event");
    return 1000.0 * ms / (double(passes) * double(copies));
}

std::vector<float> fetch(const float* dy, std::size_t floats) {
    std::vector<float> y(floats);
    ok(hipDeviceSynchronize(), "sync");
    ok(hipMemcpy(y.data(), dy, floats * 4U, hipMemcpyDeviceToHost), "copy y");
    return y;
}

// Rows [0, rows) of y (row pitch n) bit-identical to the same rows of base: number of differing floats.
std::size_t bit_diff(const std::vector<float>& y, const std::vector<float>& base, unsigned rows, unsigned n) {
    std::size_t diff = 0;
    for (std::size_t i = 0; i < std::size_t(rows) * n; ++i) diff += std::memcmp(&y[i], &base[i], 4) != 0 ? 1U : 0U;
    return diff;
}

// Every byte 0xFF (a NaN pattern no launch produces here): a destination a launch must write, or must leave alone.
void poison(float* dy, std::size_t floats) { ok(hipMemset(dy, 0xFF, floats * 4U), "poison"); }

// Floats of y in [from, y.size()) that no longer hold the poison pattern.
std::size_t overwritten(const std::vector<float>& y, std::size_t from) {
    std::size_t count = 0;
    for (std::size_t i = from; i < y.size(); ++i) {
        std::uint32_t bits;
        std::memcpy(&bits, &y[i], 4);
        count += bits != 0xFFFFFFFFU ? 1U : 0U;
    }
    return count;
}

int failures = 0;
void report(bool pass, const std::string& line) {
    failures += pass ? 0 : 1;
    std::cout << "check " << line << (pass ? " PASS" : " FAIL") << '\n';
}

// ---- Lever "lowbit.rows-select" (LowBitArithmetic::f16_rows_select, a FAST / TURBO candidate) ----
// strix-hip-dense-lowbit-bench rows-select N K ...: per shape and storage, the lever's launches equal bit for bit the
// incumbent they select (the FP32 sweep inside a measured cell of this device, the f16 group-scale form elsewhere),
// alone and in a two-projection list; how many outputs the sweep and the FP16 form share; then interleaved timings.
int rows_select_ticket(const std::vector<unsigned>& shapes) {
    constexpr auto select = core::LowBitArithmetic::f16_rows_select;
    constexpr auto sweep = core::LowBitArithmetic::fp32_sweep;
    const std::string arch(core::detail::current_device_arch());
    {
        // The f16 form's incumbent at these rows is the small-M route of the device plan (unbound: the tile everywhere).
        int dev = 0;
        ok(hipGetDevice(&dev), "device");
        report(strix::device::bind_lowbit_rows(dev, {64U, 64U}), "rows-select: bind the declared plan small_rows=64 wide_panels=64");
    }
    std::cout << "rows-select arch=" << arch << " identity=" << core::kLowBitRowsSelectIdentity.name << '@'
              << core::kLowBitRowsSelectIdentity.revision << '\n';
    const std::vector<unsigned> check_rows{1U, 2U, 3U, 4U, 8U, 16U, 17U}, bench_rows{1U, 2U, 4U, 8U};
    for (std::size_t s = 0; s + 1 < shapes.size(); s += 2) {
        const unsigned n = shapes[s], k = shapes[s + 1];
        std::mt19937 rng(11U + n + k);
        std::normal_distribution<float> normal(0.0F, 0.02F);
        std::vector<std::uint16_t> bits(std::size_t(n) * k);
        for (auto& b : bits) b = to_bf16(normal(rng));
        std::vector<float> x(std::size_t(32U) * k);
        for (auto& v : x) v = normal(rng) * 50.0F;
        const std::size_t y_floats = std::size_t(32U) * n;
        float *dx = nullptr, *dy = nullptr, *dy2 = nullptr, *dz = nullptr;
        ok(hipMalloc(&dx, x.size() * 4U), "malloc x");
        ok(hipMalloc(&dy, y_floats * 4U), "malloc y");
        ok(hipMalloc(&dy2, y_floats * 4U), "malloc y2");
        ok(hipMalloc(&dz, y_floats * 4U), "malloc z");
        ok(hipMemcpy(dx, x.data(), x.size() * 4U, hipMemcpyHostToDevice), "copy x");
        for (const auto format : {wd::Format::q4_g64, wd::Format::q6_g64}) {
            const auto p = pack(format, bits, n, k);
            const auto dev = upload(p.bytes.data(), p.bytes.size());
            const std::uint8_t* w = dev.w;
            const auto v = view(p, w, n, k);
            const std::string tag = std::string("fmt=") + name(format) + " N=" + std::to_string(n) + " K=" + std::to_string(k);
            const auto entry = [&](float* y, unsigned rows, core::LowBitArithmetic a) {
                ok(core::enqueue_resident_lowbit_rows(v, n, dx, rows, y, nullptr, a), "entry");
                return fetch(y, std::size_t(rows) * n);
            };
            for (const unsigned rows : check_rows) {
                const bool cell = core::lowbit_sweep_cell(arch, format == wd::Format::q4_g64 ? 4U : 6U, n, k, rows);
                const auto want = entry(dy, rows, cell ? sweep : f16_group_scale);
                poison(dy2, y_floats);
                const auto got = entry(dy2, rows, select);
                const auto diff = bit_diff(got, want, rows, n);
                // The same matrix twice in one list: each projection gets the bits of its selected route.
                const core::ResidentRowProjection list[2]{{v, n, dy2}, {v, n, dz}};
                poison(dy2, y_floats);
                poison(dz, y_floats);
                const auto status = core::enqueue_resident_q8_projections<2>(dx, k, rows, list, nullptr, select);
                ok(status, "list");
                const auto diff_list = bit_diff(fetch(dy2, std::size_t(rows) * n), want, rows, n) +
                                       bit_diff(fetch(dz, std::size_t(rows) * n), want, rows, n);
                report(diff == 0U && diff_list == 0U, tag + " rows=" + std::to_string(rows) + " selects=" +
                                                         (cell ? "fp32-sweep" : "f16-group") + " bits_vs_selected=" +
                                                         std::to_string(diff) + " list_bits_vs_selected=" + std::to_string(diff_list));
            }
            for (const unsigned rows : {1U, 2U}) {
                const auto a = entry(dy, rows, sweep);
                const auto b = entry(dy2, rows, f16_group_scale);
                std::cout << "info " << tag << " rows=" << rows << " fp32-sweep_vs_f16-group differing=" << bit_diff(a, b, rows, n)
                          << " of " << std::size_t(rows) * n << '\n';
            }
            const auto bench = [&](const char* arm, core::LowBitArithmetic a, unsigned rows, int round) {
                const double us = time_us(dev.copies, [&](std::size_t c) {
                    return core::enqueue_resident_lowbit_rows(view(p, dev.w + c * dev.stride, n, k), n, dx, rows, dy, nullptr, a);
                });
                std::cout << "bench rows-select " << tag << " arm=" << arm << " rows=" << rows << " round=" << round
                          << " us=" << us << " Gwps=" << double(n) * k / (us * 1e3) << '\n';
            };
            for (int round = 0; round < 2; ++round)
                for (const unsigned rows : bench_rows) {
                    bench("f16-group", f16_group_scale, rows, round);
                    bench("rows-select", select, rows, round);
                    bench("fp32-sweep", sweep, rows, round);
                }
            ok(hipFree(dev.w), "free w");
        }
        ok(hipFree(dx), "free x");
        ok(hipFree(dy), "free y");
        ok(hipFree(dy2), "free y2");
        ok(hipFree(dz), "free z");
    }
    std::cout << "lowbit rows-select checks " << (failures == 0 ? "PASS" : "FAIL") << '\n';
    return failures == 0 ? 0 : 1;
}

}  // namespace

int main(int argc, char** argv) {
    if (argc > 1 && std::strcmp(argv[1], "rows-select") == 0) {
        std::vector<unsigned> select_shapes;
        for (int i = 2; i < argc; ++i) select_shapes.push_back(static_cast<unsigned>(std::strtoul(argv[i], nullptr, 10)));
        bool valid = !select_shapes.empty() && select_shapes.size() % 2U == 0U;
        for (std::size_t i = 0; valid && i < select_shapes.size(); i += 2)
            valid = select_shapes[i] != 0U && select_shapes[i + 1] != 0U && select_shapes[i + 1] % 128U == 0U;
        if (!valid) {
            std::cerr << "usage: strix-hip-dense-lowbit-bench rows-select N K [N K ...], K a positive multiple of 128\n";
            return 2;
        }
        return rows_select_ticket(select_shapes);
    }
    // Default shapes: the decode projections, then two output tails (N not a multiple of 64 outputs).
    std::vector<unsigned> shapes{3072, 2304, 1024, 2304, 2304, 3072, 384, 2304, 1000, 2304, 17, 256};
    const bool default_shapes = argc <= 1;
    if (!default_shapes) {
        shapes.clear();
        for (int i = 1; i < argc; ++i) shapes.push_back(static_cast<unsigned>(std::strtoul(argv[i], nullptr, 10)));
    }
    bool shapes_ok = !shapes.empty() && shapes.size() % 2U == 0U;
    for (std::size_t s = 0; shapes_ok && s < shapes.size(); s += 2)
        shapes_ok = shapes[s] != 0U && shapes[s + 1] != 0U && shapes[s + 1] % 128U == 0U;
    if (!shapes_ok) {
        std::cerr << "usage: strix-hip-dense-lowbit-bench [N K ...] with N > 0 and K a positive multiple of 128\n";
        return 2;
    }
    const wd::Format formats[] = {wd::Format::bf16, wd::Format::q8_0, wd::Format::q4_g64, wd::Format::q6_g64};
    // bf16 / q8_0: every bucket of the tight-rows sweep plus the partial buckets 3 and 13. Low-bit: 1..3 row tiles of the
    // small schedule (full and partial, one and several row blocks) and the tile (65, 129).
    const std::vector<unsigned> sweep_rows{1U, 2U, 3U, 4U, 8U, 13U, 16U};
    const std::vector<unsigned> lowbit_rows{1U, 2U, 3U, 8U, 13U, 16U, 17U, 31U, 32U, 33U, 40U, 48U, 49U, 64U, 65U, 129U};
    const std::vector<unsigned> sweep_bench{1U, 8U, 16U}, lowbit_bench{1U, 8U, 16U, 32U, 40U, 48U};
    const std::vector<unsigned> switch_bench{48U, 64U, 96U, 128U, 192U, 512U};
    constexpr unsigned max_check = 129U, max_rows = 512U;
    constexpr std::uint32_t sentinel = 0xffffffffU;  // a NaN: no correct output keeps or produces it
    {
        // The route plan the launches read is the one bound to this device (a loader binds its declaration's
        // `lowbit_rows`); unbound = one route. The fixture binds the measured gfx1151 / 40-CU switch points it checks.
        int dev = 0;
        hipDeviceProp_t prop{};
        ok(hipGetDevice(&dev), "device");
        ok(hipGetDeviceProperties(&prop, dev), "device properties");
        const auto unbound = core::detail::current_lowbit_row_plan();
        report(unbound.small_rows == 0U && unbound.wide_panels == 0U, "an unbound device runs one route");
        report(strix::device::bind_lowbit_rows(dev, {64U, 64U}), "bind the declared plan");
        const auto plan = core::detail::current_lowbit_row_plan();
        const std::string arch(prop.gcnArchName);
        std::cout << "plan arch=" << arch << " multiprocessors=" << prop.multiProcessorCount
                  << " bound small_rows=" << plan.small_rows << " wide_panels=" << plan.wide_panels << '\n';
        report(plan.small_rows == 64U && plan.wide_panels == 64U, "launches read the bound plan");
    }
    for (std::size_t s = 0; s + 1 < shapes.size(); s += 2) {
        const unsigned n = shapes[s], k = shapes[s + 1];
        std::mt19937 rng(11U + n + k);
        std::normal_distribution<float> normal(0.0F, 0.02F);
        std::vector<std::uint16_t> bits(std::size_t(n) * k);
        for (auto& b : bits) b = to_bf16(normal(rng));
        std::vector<float> x(std::size_t(max_rows) * k);
        for (auto& v : x) v = normal(rng) * 50.0F;
        const std::size_t y_floats = std::size_t(max_rows) * n;
        float *dx = nullptr, *dy = nullptr, *dy2 = nullptr;
        ok(hipMalloc(&dx, x.size() * 4U), "malloc x");
        ok(hipMalloc(&dy, y_floats * 4U), "malloc y");
        ok(hipMalloc(&dy2, y_floats * 4U), "malloc y2");
        ok(hipMemcpy(dx, x.data(), x.size() * 4U, hipMemcpyHostToDevice), "copy x");
        const auto q8 = pack(wd::Format::q8_0, bits, n, k);
        std::uint8_t* dq8 = nullptr;  // one copy of the q8_0 pack for the mixed-list checks
        ok(hipMalloc(&dq8, q8.bytes.size()), "malloc q8");
        ok(hipMemcpy(dq8, q8.bytes.data(), q8.bytes.size(), hipMemcpyHostToDevice), "copy q8");
        for (const auto format : formats) {
            const bool lb = lowbit(format);
            const auto p = format == wd::Format::q8_0 ? q8 : pack(format, bits, n, k);
            const auto dev = upload(p.bytes.data(), p.bytes.size());
            const std::uint8_t* w = dev.w + (dev.copies - 1U) * dev.stride;  // a nonzero 16-byte offset
            const unsigned ref_rows = lb ? max_check : 16U;
            std::vector<double> ref(std::size_t(ref_rows) * n), mag(std::size_t(ref_rows) * n);
            for (unsigned r = 0; r < ref_rows; ++r)
                for (unsigned o = 0; o < n; ++o) {
                    double acc = 0.0, m = 0.0;
                    for (unsigned kk = 0; kk < k; ++kk) {
                        const double t = double(p.dequant[std::size_t(o) * k + kk]) * x[std::size_t(r) * k + kk];
                        acc += t;
                        m += std::fabs(t);
                    }
                    ref[std::size_t(r) * n + o] = acc;
                    mag[std::size_t(r) * n + o] = m;
                }
            // bf16 / q8_0 sweep: FP32 products. Low-bit routes: the activation rounded to FP16 (2^-11 relative), exact code
            // products, FP32 WMMA / FMA chains over K: |error| <= (2^-10 + K 2^-23) * sum |a w| leaves a factor-2 margin.
            const double tolerance = lb ? std::ldexp(1.0, -10) + double(k) * std::ldexp(1.0, -23) : 1e-4;
            const std::string tag = std::string("fmt=") + name(format) + " N=" + std::to_string(n) + " K=" + std::to_string(k);
            // Low-bit bit reference: the entry at 129 rows (the tile route).
            std::vector<float> base;
            if (lb) {
                ok(launch(p, w, n, k, dx, dy, max_check), "launch base");
                base = fetch(dy, y_floats);
            }
            for (const unsigned rows : lb ? lowbit_rows : sweep_rows) {
                ok(hipMemset(dy, 0xff, y_floats * 4U), "sentinel y");
                ok(launch(p, w, n, k, dx, dy, rows), "launch check");
                const auto y = fetch(dy, y_floats);
                double worst = 0.0;
                std::size_t bad = 0, touched = 0;
                for (unsigned r = 0; r <= max_check; ++r)
                    for (unsigned o = 0; o < n; ++o) {
                        const float v = y[std::size_t(r) * n + o];
                        if (r >= rows) {
                            std::uint32_t b;
                            std::memcpy(&b, &v, 4);
                            touched += b != sentinel ? 1U : 0U;
                            continue;
                        }
                        const double e = std::fabs(double(v) - ref[std::size_t(r) * n + o]) /
                                         (mag[std::size_t(r) * n + o] + 1e-30);
                        if (!std::isfinite(v) || !(e < tolerance)) ++bad;
                        else worst = std::max(worst, e);
                    }
                const std::size_t diff = lb ? bit_diff(y, base, rows, n) : 0U;
                report(bad == 0U && touched == 0U && diff == 0U,
                       tag + " route=entry rows=" + std::to_string(rows) + " rel_err=" + std::to_string(worst) +
                           " bad=" + std::to_string(bad) + " past_rows_written=" + std::to_string(touched) +
                           " bits_vs_129=" + std::to_string(diff));
            }
            if (lb) {
                // The incumbent FP32 sweep remains available and still refuses row 17.
                for (const unsigned rows : {1U, 2U, 3U, 4U, 8U, 13U, 16U, 17U}) {
                    poison(dy, y_floats);
                    const auto status = launch(p, w, n, k, dx, dy, rows, Route::fp32_sweep);
                    const auto y = fetch(dy, y_floats);
                    std::size_t bad = 0U;
                    if (rows <= 16U)
                        for (std::size_t at = 0; at < std::size_t{rows} * n; ++at)
                            if (!std::isfinite(y[at]) ||
                                std::fabs(double(y[at]) - ref[at]) >= 1e-4 * (mag[at] + 1e-30))
                                ++bad;
                    const bool valid = rows <= 16U;
                    const auto past = overwritten(y, valid ? std::size_t{rows} * n : 0U);
                    report(status == (valid ? hipSuccess : hipErrorInvalidValue) && bad == 0U && past == 0U,
                           tag + " route=fp32-sweep rows=" + std::to_string(rows) + " bad=" + std::to_string(bad) +
                               " past_rows_written=" + std::to_string(past));
                }
                // Forced routes at the same row count give the same bits across every row block boundary.
                for (const unsigned rows : {16U, 40U, 48U, 129U})
                    for (const Route route : {Route::small, Route::tile, Route::rt1, Route::rt2}) {
                        ok(launch(p, w, n, k, dx, dy, rows, route), "launch forced");
                        const auto diff = bit_diff(fetch(dy, y_floats), base, rows, n);
                        report(diff == 0U, tag + " route=" + name(route) + " rows=" + std::to_string(rows) +
                                               " bits_vs_129=" + std::to_string(diff));
                    }
                // Grouped (the same matrix twice, one launch) and mixed with q8_0 in both orders: low-bit bits unchanged,
                // q8_0 equal to its own single launch, every order accepted.
                const auto lowbit_view = view(p, w, n, k), q8_view = view(q8, dq8, n, k);
                for (const unsigned rows : {1U, 16U, 32U, 40U}) {
                    const std::array<core::ResidentRowProjection, 2> pair{{{lowbit_view, n, dy}, {lowbit_view, n, dy2}}};
                    poison(dy, y_floats);
                    poison(dy2, y_floats);
                    const auto status = core::enqueue_resident_q8_projections<2>(dx, k, rows, pair, nullptr, f16_group_scale);
                    const auto y1 = fetch(dy, y_floats), y2 = fetch(dy2, y_floats);
                    const auto d1 = bit_diff(y1, base, rows, n), d2 = bit_diff(y2, base, rows, n);
                    const auto past = overwritten(y1, std::size_t(rows) * n) + overwritten(y2, std::size_t(rows) * n);
                    report(status == hipSuccess && d1 == 0U && d2 == 0U && past == 0U,
                           tag + " grouped rows=" + std::to_string(rows) + " bits_vs_129=" + std::to_string(d1 + d2) +
                               " past_rows_written=" + std::to_string(past));
                }
                {
                    // Grouped with a different first matrix whose outputs end in a tail panel (N = 1000), above 64 rows
                    // (tile route): its null columns stage next to the second matrix's live panel; both lists equal
                    // their single launches.
                    constexpr unsigned tail_n = 1000U;
                    std::vector<std::uint16_t> tail_bits(std::size_t(tail_n) * k);
                    for (auto& b : tail_bits) b = to_bf16(normal(rng));
                    const auto tail = pack(format, tail_bits, tail_n, k);
                    std::uint8_t* dt = nullptr;
                    float* dy3 = nullptr;
                    const std::size_t tail_floats = std::size_t(max_rows) * tail_n;
                    ok(hipMalloc(&dt, tail.bytes.size()), "malloc tail");
                    ok(hipMemcpy(dt, tail.bytes.data(), tail.bytes.size(), hipMemcpyHostToDevice), "copy tail");
                    ok(hipMalloc(&dy3, tail_floats * 4U), "malloc y3");
                    const auto tail_view = view(tail, dt, tail_n, k);
                    for (const unsigned rows : {65U, 129U}) {
                        ok(core::enqueue_resident_lowbit_rows(tail_view, tail_n, dx, rows, dy3, nullptr, f16_group_scale), "tail alone");
                        const auto tail_alone = fetch(dy3, tail_floats);
                        const std::array<core::ResidentRowProjection, 2> list{
                            {{tail_view, tail_n, dy3}, {lowbit_view, n, dy}}};
                        poison(dy3, tail_floats);  // a skipped projection keeps poison: both lists must be written
                        poison(dy, y_floats);
                        const auto status = core::enqueue_resident_q8_projections<2>(dx, k, rows, list, nullptr, f16_group_scale);
                        const auto y3 = fetch(dy3, tail_floats), y1 = fetch(dy, y_floats);
                        const auto d1 = bit_diff(y3, tail_alone, rows, tail_n);
                        const auto d2 = bit_diff(y1, base, rows, n);
                        const auto past = overwritten(y3, std::size_t(rows) * tail_n) + overwritten(y1, std::size_t(rows) * n);
                        report(status == hipSuccess && d1 == 0U && d2 == 0U && past == 0U,
                               tag + " grouped tail-first rows=" + std::to_string(rows) + " tail_bits=" +
                                   std::to_string(d1) + " second_bits=" + std::to_string(d2) +
                                   " past_rows_written=" + std::to_string(past));
                    }
                }
                for (const unsigned rows : {2U, 40U}) {
                    ok(core::enqueue_resident_q8_rows(q8_view, n, dx, rows, dy2, nullptr), "q8 alone");
                    const auto q8_alone = fetch(dy2, y_floats);
                    for (const bool lowbit_first : {true, false}) {
                        const core::ResidentRowProjection a{lowbit_view, n, dy}, b{q8_view, n, dy2};
                        const std::array<core::ResidentRowProjection, 2> list{
                            {lowbit_first ? a : b, lowbit_first ? b : a}};
                        poison(dy, y_floats);
                        poison(dy2, y_floats);
                        const auto status = core::enqueue_resident_q8_projections<2>(dx, k, rows, list, nullptr, f16_group_scale);
                        const auto d1 = bit_diff(fetch(dy, y_floats), base, rows, n);
                        const auto d2 = bit_diff(fetch(dy2, y_floats), q8_alone, rows, n);
                        report(status == hipSuccess && d1 == 0U && d2 == 0U,
                               tag + " mixed=" + (lowbit_first ? "lowbit,q8_0" : "q8_0,lowbit") + " rows=" +
                                   std::to_string(rows) + " lowbit_bits=" + std::to_string(d1) +
                                   " q8_bits=" + std::to_string(d2));
                    }
                }
                // The grouped contract is one matrix row per output for every storage, checked before anything is
                // enqueued: a q8_0 entry whose matrix has one row more than its outputs (a live prefix) is refused in
                // either position, at the sweep and at the tile, and neither destination is written.
                if (n > 1U) {
                    for (const unsigned rows : {2U, 40U})
                        for (const bool bad_first : {true, false}) {
                            const core::ResidentRowProjection good{lowbit_view, n, dy}, bad{q8_view, n - 1U, dy2};
                            const std::array<core::ResidentRowProjection, 2> list{
                                {bad_first ? bad : good, bad_first ? good : bad}};
                            poison(dy, y_floats);
                            poison(dy2, y_floats);
                            const auto status = core::enqueue_resident_q8_projections<2>(dx, k, rows, list, nullptr, f16_group_scale);
                            const auto written =
                                overwritten(fetch(dy, y_floats), 0U) + overwritten(fetch(dy2, y_floats), 0U);
                            report(status == core::row_transforms::invalid && written == 0U,
                                   tag + " grouped q8_0 live-prefix refused order=" + (bad_first ? "bad,good" : "good,bad") +
                                       " rows=" + std::to_string(rows) + " written=" + std::to_string(written));
                        }
                }
                // Overlapping destinations are refused for the whole list before anything is enqueued, in both
                // descriptor orders: two projections into one destination (low-bit + low-bit of one launch group, and
                // low-bit + q8_0 across storage runs) and an aligned partial overlap (second output = first + 4).
                for (const unsigned rows : {1U, 40U})
                    for (const bool swap : {false, true})
                        for (const unsigned shift : {0U, 4U}) {
                            for (const bool cross : {false, true}) {
                                const core::ResidentRowProjection first{lowbit_view, n, dy},
                                    second{cross ? q8_view : lowbit_view, n, dy + shift};
                                const std::array<core::ResidentRowProjection, 2> list{
                                    {swap ? second : first, swap ? first : second}};
                                poison(dy, y_floats);
                                const auto status = core::enqueue_resident_q8_projections<2>(dx, k, rows, list, nullptr, f16_group_scale);
                                const auto written = overwritten(fetch(dy, y_floats), 0U);
                                report(status == core::row_transforms::invalid && written == 0U,
                                       tag + " grouped overlap refused " + (cross ? "lowbit+q8_0" : "lowbit+lowbit") +
                                           " shift=" + std::to_string(shift) + " order=" + (swap ? "b,a" : "a,b") +
                                           " rows=" + std::to_string(rows) + " written=" + std::to_string(written));
                            }
                        }
                // An output that aliases weights another projection of the list reads is refused before anything
                // is enqueued, in both descriptor orders: A writes from the first byte of a writable copy of B's
                // packed weights (one launch group, and across storage runs with a q8_0 B). Neither B's weights nor
                // B's destination change.
                for (const unsigned rows : {1U, 40U})
                    for (const bool swap : {false, true})
                        for (const bool cross : {false, true}) {
                            const auto& read_view = cross ? q8_view : lowbit_view;
                            const std::size_t weight_bytes = read_view.bytes.size();
                            const std::size_t alias_bytes = std::max(weight_bytes, std::size_t(rows) * n * 4U);
                            std::uint8_t* alias = nullptr;
                            ok(hipMalloc(&alias, alias_bytes), "malloc alias");
                            ok(hipMemset(alias, 0x5a, alias_bytes), "fill alias");
                            ok(hipMemcpy(alias, read_view.bytes.data(), weight_bytes, hipMemcpyDeviceToDevice), "copy alias");
                            auto b_view = read_view;
                            b_view.bytes = {alias, weight_bytes};
                            const core::ResidentRowProjection a{lowbit_view, n, reinterpret_cast<float*>(alias)},
                                b{b_view, n, dy};
                            const std::array<core::ResidentRowProjection, 2> list{{swap ? b : a, swap ? a : b}};
                            poison(dy, y_floats);
                            const auto status = core::enqueue_resident_q8_projections<2>(dx, k, rows, list, nullptr, f16_group_scale);
                            std::vector<std::uint8_t> before(weight_bytes), after(weight_bytes);
                            ok(hipMemcpy(before.data(), read_view.bytes.data(), weight_bytes, hipMemcpyDeviceToHost), "fetch w");
                            ok(hipMemcpy(after.data(), alias, weight_bytes, hipMemcpyDeviceToHost), "fetch alias");
                            const auto written = overwritten(fetch(dy, y_floats), 0U) + (before == after ? 0U : 1U);
                            report(status == core::row_transforms::invalid && written == 0U,
                                   tag + " grouped output-on-weights refused " + (cross ? "lowbit+q8_0" : "lowbit+lowbit") +
                                       " order=" + (swap ? "b,a" : "a,b") + " rows=" + std::to_string(rows) +
                                       " written=" + std::to_string(written));
                            ok(hipFree(alias), "free alias");
                        }
                // A mixed BF16 / compact list is admitted whole before its first launch: a valid BF16 projection
                // followed by a compact view with a wrong row pitch is refused and no destination is written (the BF16
                // launcher here writes its destination, so a launch before the refusal would show).
                {
                    std::uint16_t* bf16_weights = nullptr;
                    ok(hipMalloc(&bf16_weights, std::size_t(n) * k * 2U), "malloc bf16");
                    auto bad_pitch = lowbit_view;
                    bad_pitch.row_bytes += 16U;
                    for (const unsigned rows : {1U, 40U})
                        for (const bool malformed : {true, false}) {
                            const std::array<core::ResidentRowProjection, 2> list{
                                {{static_cast<const std::uint16_t*>(bf16_weights), n, dy},
                                 {malformed ? bad_pitch : lowbit_view, n, dy2}}};
                            poison(dy, y_floats);
                            poison(dy2, y_floats);
                            unsigned launched = 0U;
                            const auto status = core::dispatch_resident_rows<2>(
                                dx, k, rows, std::span<const core::ResidentRowProjection>(list), hipSuccess,
                                core::row_transforms::invalid,
                                [&](std::span<const core::Bf16RowProjection> run) {
                                    ++launched;
                                    for (const auto& r : run)
                                        if (const auto e = hipMemset(r.output, 0, std::size_t(rows) * r.outputs * 4U);
                                            e != hipSuccess)
                                            return e;
                                    return hipSuccess;
                                },
                                [&](std::span<const core::ResidentRowProjection> run) {
                                    ++launched;
                                    return core::enqueue_resident_q8_projections<2>(dx, k, rows, run, nullptr, f16_group_scale);
                                },
                                [&](std::span<const core::Bf16RowProjection> run) {
                                    return strix::inkling2::mini_hip_dense_rows_admitted(dx, k, rows, run);
                                },
                                [&](std::span<const core::ResidentRowProjection> run) {
                                    return core::resident_q8_projections_admitted<2>(dx, k, rows, run, f16_group_scale);
                                });
                            ok(hipDeviceSynchronize(), "sync mixed");
                            const auto written = overwritten(fetch(dy, y_floats), 0U) + overwritten(fetch(dy2, y_floats), 0U);
                            if (malformed)
                                report(status == core::row_transforms::invalid && written == 0U && launched == 0U,
                                       tag + " mixed bf16,bad-pitch refused rows=" + std::to_string(rows) +
                                           " written=" + std::to_string(written) + " launched=" + std::to_string(launched));
                            else
                                report(status == hipSuccess && launched == 2U,
                                       tag + " mixed bf16,lowbit accepted rows=" + std::to_string(rows) +
                                           " launched=" + std::to_string(launched));
                        }
                    ok(hipFree(bf16_weights), "free bf16");
                }
            }
            const auto bench = [&](unsigned rows, Route route) {
                const double us = time_us(dev.copies, [&](std::size_t c) {
                    return launch(p, dev.w + c * dev.stride, n, k, dx, dy, rows, route);
                });
                std::cout << "bench " << tag << " route=" << name(route) << " rows=" << rows
                          << " bytes=" << p.bytes.size() << " dram=" << (dev.copies >= dev.dram_copies ? 1 : 0)
                          << " us=" << us << " GBps=" << double(p.bytes.size()) / (us * 1e3) << '\n';
            };
            for (const unsigned rows : lb ? lowbit_bench : sweep_bench) bench(rows, Route::entry);
            if (lb) {
                for (const unsigned rows : {1U, 8U, 16U}) bench(rows, Route::fp32_sweep);
                for (const unsigned rows : switch_bench)
                    for (const Route route : {Route::small, Route::tile}) bench(rows, route);
                for (const unsigned rows : {24U, 32U, 40U, 64U})
                    for (const Route route : {Route::rt1, Route::rt2}) bench(rows, route);
            }
            ok(hipFree(dev.w), "free w");
        }
        ok(hipFree(dq8), "free q8");
        ok(hipFree(dx), "free x");
        ok(hipFree(dy), "free y");
        ok(hipFree(dy2), "free y2");
    }
    {
        // Geometry passes byte-range checks, but Inkling's BF16 adapter refuses outputs above its vocabulary.
        constexpr unsigned k = 128U, bad_outputs = strix::inkling2::kVocabulary + 1U;
        const auto q4 = pack(wd::Format::q4_g64, std::vector<std::uint16_t>(k), 1U, k);
        std::uint8_t* dw = nullptr;
        std::uint16_t* db = nullptr;
        float *dx = nullptr, *dy = nullptr, *dy_bf16 = nullptr;
        ok(hipMalloc(&dw, q4.bytes.size()), "malloc q4");
        ok(hipMalloc(&db, std::size_t{bad_outputs} * k * sizeof(*db)), "malloc BF16");
        ok(hipMalloc(&dx, k * sizeof(*dx)), "malloc input");
        ok(hipMalloc(&dy, sizeof(*dy)), "malloc compact output");
        ok(hipMalloc(&dy_bf16, bad_outputs * sizeof(*dy_bf16)), "malloc BF16 output");
        ok(hipMemcpy(dw, q4.bytes.data(), q4.bytes.size(), hipMemcpyHostToDevice), "copy q4");
        ok(hipMemset(dx, 0, k * sizeof(*dx)), "init input");
        const core::ResidentRowProjection compact{view(q4, dw, 1U, k), 1U, dy};
        const core::ResidentRowProjection bf16{static_cast<const std::uint16_t*>(db), bad_outputs, dy_bf16};
        for (const auto& list : {std::array{compact, bf16}, std::array{bf16, compact}}) {
            poison(dy, 1U);
            poison(dy_bf16, bad_outputs);
            const auto status = core::dispatch_resident_rows<2>(dx, k, 1U, list, hipSuccess, hipErrorInvalidValue,
                [&](auto run) { return strix::inkling2::enqueue_mini_hip_dense_rows(dx, k, 1U, run, nullptr); },
                [&](auto run) { return core::enqueue_resident_q8_projections<2>(dx, k, 1U, run, nullptr, f16_group_scale); },
                [&](auto run) { return strix::inkling2::mini_hip_dense_rows_admitted(dx, k, 1U, run); },
                [&](auto run) { return core::resident_q8_projections_admitted<2>(dx, k, 1U, run, f16_group_scale); });
            const auto written = overwritten(fetch(dy, 1U), 0U) + overwritten(fetch(dy_bf16, bad_outputs), 0U);
            report(status == hipErrorInvalidValue && written == 0U,
                   "mixed malformed BF16 refused with destinations untouched, written=" + std::to_string(written));
        }
        ok(hipFree(dw), "free q4");
        ok(hipFree(db), "free BF16");
        ok(hipFree(dx), "free input");
        ok(hipFree(dy), "free compact output");
        ok(hipFree(dy_bf16), "free BF16 output");
    }
    if (default_shapes) {
        // q | k | v of one input as ONE grouped launch through the entry (3072 + 1024 + 1024 outputs, K 2304).
        constexpr unsigned k = 2304U, nq = 3072U, nk = 1024U;
        std::mt19937 rng(7U);
        std::normal_distribution<float> normal(0.0F, 0.02F);
        std::vector<std::uint16_t> qbits(std::size_t(nq) * k), kbits(std::size_t(nk) * k);
        for (auto& b : qbits) b = to_bf16(normal(rng));
        for (auto& b : kbits) b = to_bf16(normal(rng));
        std::vector<float> x(std::size_t(48U) * k);
        for (auto& v : x) v = normal(rng) * 50.0F;
        float *dx = nullptr, *dy = nullptr;
        ok(hipMalloc(&dx, x.size() * 4U), "malloc x");
        ok(hipMalloc(&dy, std::size_t(48U) * (nq + 2U * nk) * 4U), "malloc y");
        ok(hipMemcpy(dx, x.data(), x.size() * 4U, hipMemcpyHostToDevice), "copy x");
        for (const auto format : {wd::Format::q4_g64, wd::Format::q6_g64}) {
            const auto pq = pack(format, qbits, nq, k), pk = pack(format, kbits, nk, k);
            const std::size_t at_k = aligned(pq.bytes.size()), at_v = at_k + aligned(pk.bytes.size());
            std::vector<std::uint8_t> trio(at_v + pk.bytes.size());
            std::memcpy(trio.data(), pq.bytes.data(), pq.bytes.size());
            std::memcpy(trio.data() + at_k, pk.bytes.data(), pk.bytes.size());
            std::memcpy(trio.data() + at_v, pk.bytes.data(), pk.bytes.size());
            const auto dev = upload(trio.data(), trio.size());
            const std::size_t bytes = pq.bytes.size() + 2U * pk.bytes.size();
            for (const unsigned rows : {1U, 8U, 16U, 32U, 40U, 48U}) {
                const double us = time_us(dev.copies, [&](std::size_t c) {
                    const auto* w = dev.w + c * dev.stride;
                    const std::array<core::ResidentRowProjection, 3> list{
                        {{view(pq, w, nq, k), nq, dy},
                         {view(pk, w + at_k, nk, k), nk, dy + std::size_t(rows) * nq},
                         {view(pk, w + at_v, nk, k), nk, dy + std::size_t(rows) * (nq + nk)}}};
                    return core::enqueue_resident_q8_projections<3>(dx, k, rows, list, nullptr, f16_group_scale);
                });
                std::cout << "bench fmt=" << name(format) << " qkv N=" << nq + 2U * nk << " K=" << k
                          << " route=entry rows=" << rows << " bytes=" << bytes
                          << " dram=" << (dev.copies >= dev.dram_copies ? 1 : 0) << " us=" << us
                          << " GBps=" << double(bytes) / (us * 1e3) << '\n';
            }
            ok(hipFree(dev.w), "free w");
        }
        ok(hipFree(dx), "free x");
        ok(hipFree(dy), "free y");
    }
    std::cout << (failures ? "lowbit bench FAIL" : "lowbit bench checks PASS") << '\n';
    return failures ? 1 : 0;
}
