// No-model fixture of the resident dense rows family per weight storage (bf16, q8_0, q4_g64, q6_g64):
// correctness of every storage against the host reference decode, then weight-read bandwidth on decode-sized
// projections. Each timed pass walks enough distinct weight copies (>= 512 MiB, dram=1; tiny shapes are capped at 4096
// copies, dram=0) that every launch reads DRAM. Copies start 16-byte aligned; bytes= is the logical payload.
// Usage: strix-hip-dense-lowbit-bench [N K ...], N > 0, K a positive multiple of 128
//        (default: 3072 2304  1024 2304  2304 3072  384 2304  1000 2304  17 256; the last two have output tails)
// Output: per storage one "check" line per row count (every bucket and two partial buckets; finite outputs within
// 1e-4 relative of the host reference, rows past the count untouched), a refusal check at 17 rows for the low-bit
// storages, then "bench fmt=F N=.. K=.. rows=R bytes=.. dram=.. us=.. GBps=.." lines.
#include "strix/hip/legacy_intrinsics.hpp"
#include "strix/hip/legacy_launch.hpp"
#include "core/resident_q8_rows.hpp"
#include "core/iu4_onfly.hpp"

#include <algorithm>
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
namespace wd = strix::core::weight_decode;

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

hipError_t launch(const Packed& p, const std::uint8_t* w, unsigned n, unsigned k, const float* x, float* y,
                  unsigned rows) {
    if (p.format == wd::Format::bf16) {
        const core::dense_linear::View view{w, k, n, p.row_bytes};
        if (rows == 1U) core::detail::launch_q8_tight_rows<1, wd::Format::bf16>(view, x, y, rows, nullptr);
        else if (rows <= 8U) core::detail::launch_q8_tight_rows<8, wd::Format::bf16>(view, x, y, rows, nullptr);
        else core::detail::launch_q8_tight_rows<16, wd::Format::bf16>(view, x, y, rows, nullptr);
        return hipGetLastError();
    }
    const wei::ResidentMatrixView view{{w, p.bytes.size()}, p.storage, n, k, p.row_bytes};
    return core::enqueue_resident_q8_rows(view, n, x, rows, y, nullptr);
}

bool lowbit(wd::Format f) { return f == wd::Format::q4_g64 || f == wd::Format::q6_g64; }

bool all_sentinel(const std::vector<float>& y, std::uint32_t sentinel) {
    return std::all_of(y.begin(), y.end(), [sentinel](float v) {
        std::uint32_t b;
        std::memcpy(&b, &v, 4);
        return b == sentinel;
    });
}

}  // namespace

int main(int argc, char** argv) {
    // Default shapes: the decode projections, then two output tails (N not a multiple of the outputs per block).
    std::vector<unsigned> shapes{3072, 2304, 1024, 2304, 2304, 3072, 384, 2304, 1000, 2304, 17, 256};
    if (argc > 1) {
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
    // Every row bucket of the tight-rows sweep (1, 2, 4, 8, 16) plus the partial buckets 3 and 13.
    const unsigned check_rows[] = {1U, 2U, 3U, 4U, 8U, 13U, 16U};
    const unsigned row_set[] = {1U, 8U, 16U};
    constexpr unsigned max_rows = 16U, refused_rows = max_rows + 1U;
    constexpr std::uint32_t sentinel = 0xffffffffU;  // a NaN: no correct output keeps or produces it
    int failures = 0;
    for (std::size_t s = 0; s + 1 < shapes.size(); s += 2) {
        const unsigned n = shapes[s], k = shapes[s + 1];
        std::mt19937 rng(11U + n + k);
        std::normal_distribution<float> normal(0.0F, 0.02F);
        std::vector<std::uint16_t> bits(std::size_t(n) * k);
        for (auto& b : bits) b = to_bf16(normal(rng));
        std::vector<float> x(std::size_t(refused_rows) * k);
        for (auto& v : x) v = normal(rng) * 50.0F;
        const std::size_t y_floats = std::size_t(refused_rows) * n;
        float *dx = nullptr, *dy = nullptr;
        ok(hipMalloc(&dx, x.size() * 4U), "malloc x");
        ok(hipMalloc(&dy, y_floats * 4U), "malloc y");
        ok(hipMemcpy(dx, x.data(), x.size() * 4U, hipMemcpyHostToDevice), "copy x");
        std::vector<float> y(y_floats);
        for (const auto format : formats) {
            const auto p = pack(format, bits, n, k);
            // Distinct copies so that a pass never re-reads a cached matrix. Each copy starts 16-byte aligned (the
            // resident row contract); the logical payload is p.bytes.size().
            const std::size_t stride = (p.bytes.size() + 15U) / 16U * 16U;
            const std::size_t dram_copies = (std::size_t(512) << 20U) / stride + 1U;
            const std::size_t copies = std::max<std::size_t>(8U, std::min<std::size_t>(dram_copies, 4096U));
            std::uint8_t* dw = nullptr;
            ok(hipMalloc(&dw, copies * stride), "malloc w");
            ok(hipMemcpy(dw, p.bytes.data(), p.bytes.size(), hipMemcpyHostToDevice), "copy w");
            for (std::size_t done = 1; done < copies;) {
                const std::size_t step = std::min(done, copies - done);
                ok(hipMemcpy(dw + done * stride, dw, step * stride, hipMemcpyDeviceToDevice), "replicate w");
                done += step;
            }
            // Host reference of the stored weights for all 16 rows (output row r depends on input row r only).
            std::vector<double> ref(std::size_t(max_rows) * n), mag(std::size_t(max_rows) * n);
            for (unsigned r = 0; r < max_rows; ++r)
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
            // Correctness on the last copy (a nonzero 16-byte offset): rows < R finite and within 1e-4 relative of
            // the reference, rows >= R keep the sentinel.
            for (const unsigned rows : check_rows) {
                ok(hipMemset(dy, 0xff, y_floats * 4U), "sentinel y");
                ok(launch(p, dw + (copies - 1U) * stride, n, k, dx, dy, rows), "launch check");
                ok(hipMemcpy(y.data(), dy, y_floats * 4U, hipMemcpyDeviceToHost), "copy y");
                double worst = 0.0;
                std::size_t bad = 0, touched = 0;
                for (unsigned r = 0; r < refused_rows; ++r)
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
                        if (!std::isfinite(v) || !(e < 1e-4)) ++bad;
                        else worst = std::max(worst, e);
                    }
                const bool pass = bad == 0U && touched == 0U;
                failures += pass ? 0 : 1;
                std::cout << "check fmt=" << name(format) << " N=" << n << " K=" << k << " rows=" << rows
                          << " rel_err=" << worst << " bad=" << bad << " past_rows_written=" << touched
                          << (pass ? " PASS" : " FAIL") << '\n';
            }
            if (lowbit(format)) {
                // Low-bit rows have no consumer above 16 rows: the request is refused and nothing is written.
                ok(hipMemset(dy, 0xff, y_floats * 4U), "sentinel y");
                const hipError_t refused = launch(p, dw, n, k, dx, dy, refused_rows);
                ok(hipDeviceSynchronize(), "sync refusal");
                ok(hipMemcpy(y.data(), dy, y_floats * 4U, hipMemcpyDeviceToHost), "copy y");
                const bool untouched = all_sentinel(y, sentinel);
                const bool pass = refused != hipSuccess && untouched;
                failures += pass ? 0 : 1;
                std::cout << "check fmt=" << name(format) << " N=" << n << " K=" << k << " rows=" << refused_rows
                          << " refused=" << (refused != hipSuccess ? 1 : 0) << " untouched=" << (untouched ? 1 : 0)
                          << (pass ? " PASS" : " FAIL") << '\n';
            }
            for (const unsigned rows : row_set) {
                hipEvent_t a, b;
                ok(hipEventCreate(&a), "event");
                ok(hipEventCreate(&b), "event");
                for (std::size_t c = 0; c < copies; ++c) ok(launch(p, dw + c * stride, n, k, dx, dy, rows), "warm");
                constexpr int passes = 3;
                ok(hipEventRecord(a, nullptr), "record");
                for (int it = 0; it < passes; ++it)
                    for (std::size_t c = 0; c < copies; ++c) ok(launch(p, dw + c * stride, n, k, dx, dy, rows), "run");
                ok(hipEventRecord(b, nullptr), "record");
                ok(hipEventSynchronize(b), "sync");
                float ms = 0.0F;
                ok(hipEventElapsedTime(&ms, a, b), "elapsed");
                const double launches = double(passes) * double(copies);
                const double us = 1000.0 * ms / launches;
                std::cout << "bench fmt=" << name(format) << " N=" << n << " K=" << k << " rows=" << rows
                          << " bytes=" << p.bytes.size() << " dram=" << (copies >= dram_copies ? 1 : 0)
                          << " us=" << us << " GBps=" << double(p.bytes.size()) / (us * 1e3) << '\n';
                ok(hipEventDestroy(a), "event");
                ok(hipEventDestroy(b), "event");
            }
            ok(hipFree(dw), "free w");
        }
        ok(hipFree(dx), "free x");
        ok(hipFree(dy), "free y");
    }
    std::cout << (failures ? "lowbit bench FAIL" : "lowbit bench checks PASS") << '\n';
    return failures ? 1 : 0;
}
