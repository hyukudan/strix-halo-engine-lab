// Measurement-only probe of the shared q8a8 dense policy (src/core/f16_dense_q8a8.hpp on the tall core) at chosen shapes
// and address layouts: per arm, ms per launch and the FP16 WMMA rate (2 x MACs / time, padded K counted). It separates the
// causes of a slow launch that the per-launch work cannot explain:
//   address layout   : the same products with production row strides, a compact head-major layout, or padded strides
//                      (only addresses change; every arm computes the same chains on the same kind of operands);
//   batched vs tall  : the same MACs as one product with all heads stacked as rows (one weight, same K and outputs);
//   control shapes   : wide-K launches of the same core (stride 2^k vs padded).
// Operand values: a fixed finite pattern, or (rand) fixed-seed full-mantissa values; the output bits are hashed per arm.
// Build: hipcc -std=c++20 -O3 --offload-arch=gfx1151 -Iinclude -Isrc tools/hip_q8a8_shape_probe.hip -o q8a8_shape_probe
// Run:   q8a8_shape_probe [seconds per arm, default 0.4] [rows, default 2048] [layout (default) | dense] [const | rand]
//        dense = one product per arm at prefill projection widths; rand = full-mantissa operands, fixed seed
// Output: one line per arm: "arm <name> ms <per launch> tf <rate> launches <n> bits <hash of the output>"; nothing names
// a model.
#include "core/f16_dense_q8a8.hpp"

#include <hip/hip_runtime.h>

#include <algorithm>
#include <chrono>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <iterator>
#include <span>
#include <string>
#include <vector>

#define CK(x)                                                                                   \
    do {                                                                                        \
        const hipError_t e_ = (x);                                                              \
        if (e_ != hipSuccess) {                                                                 \
            std::fprintf(stderr, "HIP error %s at line %d: %s\n", #x, __LINE__, hipGetErrorString(e_)); \
            std::exit(2);                                                                       \
        }                                                                                       \
    } while (0)

namespace q8 = strix::core::f16_dense_tall::q8a8;

namespace {
struct Buffers {
    std::uint8_t* weight = nullptr;
    std::int8_t* codes = nullptr;
    float* deltas = nullptr;
    float* out = nullptr;
    std::size_t weight_bytes = 0, codes_bytes = 0, delta_floats = 0, out_floats = 0;
};

struct Shape {
    const char* name;
    std::uint32_t rows, in, count, x_stride, out_stride;
    std::uint32_t batch, batch_in, batch_out;  // batch <= 1: one product
};

double now_s() { return std::chrono::duration<double>(std::chrono::steady_clock::now().time_since_epoch()).count(); }

// Highest byte / element any arm touches (checked before the launch, so an arm never reads past its buffers).
std::size_t codes_extent(const Shape& s) {
    const std::size_t b = s.batch > 1U ? s.batch : 1U;
    return std::size_t{s.rows - 1U} * s.x_stride + std::size_t{b - 1U} * s.batch_in + s.in;
}
std::size_t delta_extent(const Shape& s) { return codes_extent(s) / 32U + 1U; }
std::size_t out_extent(const Shape& s) {
    const std::size_t b = s.batch > 1U ? s.batch : 1U;
    return std::size_t{s.rows - 1U} * s.out_stride + std::size_t{b - 1U} * s.batch_out + s.count;
}
std::size_t weight_extent(const Shape& s) {
    const std::size_t b = s.batch > 1U ? s.batch : 1U;
    return b * s.count * (s.in / 32U) * 34U;
}

void run(const Shape& s, const Buffers& buf, double secs) {
    if (codes_extent(s) > buf.codes_bytes || delta_extent(s) > buf.delta_floats || out_extent(s) > buf.out_floats ||
        weight_extent(s) > buf.weight_bytes) {
        std::printf("arm %s SKIP buffers too small\n", s.name);
        return;
    }
    q8::View v{};
    v.weight = buf.weight;
    v.codes = buf.codes;
    v.deltas = buf.deltas;
    v.out = buf.out;
    v.row_bytes = (s.in / 32U) * 34U;
    v.in = s.in;
    v.count = s.count;
    v.rows = s.rows;
    v.x_stride = s.x_stride;
    v.out_stride = s.out_stride;
    v.batch = s.batch;
    v.batch_in = s.batch_in;
    v.batch_out = s.batch_out;
    v.batch_weight_bytes = std::uint64_t{s.count} * v.row_bytes;
    if (!q8::supported(v)) {
        std::printf("arm %s SKIP unsupported\n", s.name);
        return;
    }
    const double macs = double(s.rows) * s.in * s.count * (s.batch > 1U ? s.batch : 1U);
    hipEvent_t e0, e1;
    CK(hipEventCreate(&e0));
    CK(hipEventCreate(&e1));
    for (int i = 0; i < 2; ++i) q8::enqueue(v, nullptr);  // warm (code object, caches, clocks)
    CK(hipGetLastError());
    CK(hipEventRecord(e0));
    q8::enqueue(v, nullptr);
    CK(hipEventRecord(e1));
    CK(hipEventSynchronize(e1));
    float one = 0.0F;
    CK(hipEventElapsedTime(&one, e0, e1));
    const int n = one > 0.0F ? static_cast<int>(secs * 1000.0 / one) + 1 : 1;
    const double t0 = now_s();
    CK(hipEventRecord(e0));
    for (int i = 0; i < n; ++i) q8::enqueue(v, nullptr);
    CK(hipEventRecord(e1));
    CK(hipEventSynchronize(e1));
    CK(hipGetLastError());
    float ms = 0.0F;
    CK(hipEventElapsedTime(&ms, e0, e1));
    const double per = static_cast<double>(ms) / static_cast<double>(n);
    // Output bits (FNV-1a over the arm's output extent after the timed launches; accumulate is off, so every launch
    // writes the same values): equal lines across two builds = the same bits for this shape and data.
    std::vector<std::uint32_t> host(out_extent(s));
    CK(hipMemcpy(host.data(), buf.out, host.size() * 4U, hipMemcpyDeviceToHost));
    std::uint64_t h = 1469598103934665603ULL;
    for (const std::uint32_t w : host) h = (h ^ w) * 1099511628211ULL;
    std::printf("arm %s ms %.3f tf %.2f launches %d bits %016llx window %.3f-%.3f\n", s.name, per,
                2.0 * macs / (per * 1e-3) / 1e12, n, static_cast<unsigned long long>(h), t0, now_s());
    std::fflush(stdout);
    CK(hipEventDestroy(e0));
    CK(hipEventDestroy(e1));
}
}  // namespace

int main(int argc, char** argv) {
    const double secs = argc > 1 ? std::atof(argv[1]) : 0.4;
    const std::uint32_t R = argc > 2 ? static_cast<std::uint32_t>(std::atoi(argv[2])) : 2048U;
    const std::string set = argc > 3 ? argv[3] : "layout", data = argc > 4 ? argv[4] : "const";
    if (R == 0U || R > 4096U || secs <= 0.0 || secs > 5.0 || (set != "layout" && set != "dense") ||
        (data != "const" && data != "rand")) {
        std::fprintf(stderr, "usage: q8a8_shape_probe [seconds 0-5] [rows 1-4096] [layout|dense] [const|rand]\n");
        return 1;
    }
    // Head-batched products: 64 heads, K_b form (256 padded K -> 512 outputs, row = 576 floats per head) and V_b form
    // (512 K -> 256 outputs); production strides, compact head-major, padded strides; then the stacked-rows control and
    // wide-K controls (codes stride 16384 = 2^14 vs 16512).
    const std::uint32_t H = 64U;
    const Shape shapes[] = {
        {"kb_prod", R, 256U, 512U, H * 256U, H * 576U, H, 256U, 576U},
        {"kb_compact", R, 256U, 512U, 256U, 512U, H, R * 256U, R * 512U},
        {"kb_pad", R, 256U, 512U, H * 256U + 128U, H * 576U + 32U, H, 256U, 576U},
        {"vb_prod", R, 512U, 256U, H * 512U, H * 256U, H, 512U, 256U},
        {"vb_compact", R, 512U, 256U, 512U, 256U, H, R * 512U, R * 256U},
        {"vb_pad", R, 512U, 256U, H * 512U + 128U, H * 256U + 32U, H, 512U, 256U},
        {"vb_rows", R * H, 512U, 256U, 512U, 256U, 1U, 0U, 0U},
        {"kb_rows", R * H, 256U, 512U, 256U, 512U, 1U, 0U, 0U},
        {"wide_o_2p14", R, 16384U, 3072U, 16384U, 3072U, 1U, 0U, 0U},
        {"wide_o_pad", R, 16384U, 3072U, 16384U + 128U, 3072U, 1U, 0U, 0U},
        {"wide_gate", R, 3072U, 16384U, 3072U, 16384U, 1U, 0U, 0U},
    };
    // Dense set: one product per arm at the widths of a 3072-wide MLA model with 1024 / 512 low-rank projections,
    // 64 x 256 gated heads and 1536-wide routed latents (operand pitch of strix/a8_row_pitch.hpp for 4096-byte
    // multiples), plus the head-batched K_b / V_b products at their padded layout.
    const Shape dense[] = {
        {"d_3072_16384", R, 3072U, 16384U, 3072U, 16384U, 1U, 0U, 0U},
        {"d_3072_1024", R, 3072U, 1024U, 3072U, 1024U, 1U, 0U, 0U},
        {"d_1024_16384", R, 1024U, 16384U, 1024U, 16384U, 1U, 0U, 0U},
        {"d_3072_576", R, 3072U, 576U, 3072U, 576U, 1U, 0U, 0U},
        {"d_16384_3072", R, 16384U, 3072U, 16384U + 128U, 3072U, 1U, 0U, 0U},
        {"d_13824_3072", R, 13824U, 3072U, 13824U, 3072U, 1U, 0U, 0U},
        {"d_3072_1536", R, 3072U, 1536U, 3072U, 1536U, 1U, 0U, 0U},
        {"d_1536_3072", R, 1536U, 3072U, 1536U, 3072U, 1U, 0U, 0U},
        {"kb_pad", R, 256U, 512U, H * 256U + 128U, H * 576U + 32U, H, 256U, 576U},
        {"vb_pad", R, 512U, 256U, H * 512U + 128U, H * 256U + 32U, H, 512U, 256U},
    };
    const Shape* first = set == "dense" ? dense : shapes;
    const std::size_t arms = set == "dense" ? std::size(dense) : std::size(shapes);
    Buffers buf;
    for (const Shape& s : std::span<const Shape>(first, arms)) {
        buf.codes_bytes = std::max(buf.codes_bytes, codes_extent(s));
        buf.delta_floats = std::max(buf.delta_floats, delta_extent(s));
        buf.out_floats = std::max(buf.out_floats, out_extent(s));
        buf.weight_bytes = std::max(buf.weight_bytes, weight_extent(s));
    }
    CK(hipMalloc(&buf.weight, buf.weight_bytes));
    CK(hipMalloc(&buf.codes, buf.codes_bytes));
    CK(hipMalloc(&buf.deltas, buf.delta_floats * 4U));
    CK(hipMalloc(&buf.out, buf.out_floats * 4U));
    // 0x11 bytes: Q8_0 scale 0x1111 (finite FP16), codes 17; 0x3a bytes: deltas 0x3a3a3a3a (finite FP32).
    CK(hipMemset(buf.weight, 0x11, buf.weight_bytes));
    CK(hipMemset(buf.codes, 0x11, buf.codes_bytes));
    CK(hipMemset(buf.deltas, 0x3a, buf.delta_floats * 4U));
    CK(hipMemset(buf.out, 0, buf.out_floats * 4U));
    if (data == "rand") {  // full-mantissa operands (the sustained clock depends on the operand bits): fixed seed
        std::uint64_t st = 0x9e3779b97f4a7c15ULL;
        const auto next = [&st] {
            st ^= st << 13U, st ^= st >> 7U, st ^= st << 17U;
            return st;
        };
        std::vector<std::uint8_t> w(buf.weight_bytes);
        for (std::size_t b = 0; b + 34U <= w.size(); b += 34U) {  // Q8_0 blocks: FP16 scale in [2^-9, 2^-8), 32 codes
            const auto sc = static_cast<std::uint16_t>(0x1800U | (next() & 0x3ffU));
            w[b] = static_cast<std::uint8_t>(sc & 0xffU), w[b + 1U] = static_cast<std::uint8_t>(sc >> 8U);
            for (unsigned i = 2; i < 34U; ++i) w[b + i] = static_cast<std::uint8_t>(next() % 255U + 129U);  // -127..127
        }
        std::vector<std::int8_t> c(buf.codes_bytes);
        for (auto& x : c) x = static_cast<std::int8_t>(static_cast<int>(next() % 255U) - 127);
        std::vector<float> d(buf.delta_floats);
        for (auto& x : d) x = 1.0F / 127.0F * (0.5F + static_cast<float>(next() % 1024U) / 1024.0F);
        CK(hipMemcpy(buf.weight, w.data(), w.size(), hipMemcpyHostToDevice));
        CK(hipMemcpy(buf.codes, c.data(), c.size(), hipMemcpyHostToDevice));
        CK(hipMemcpy(buf.deltas, d.data(), d.size() * 4U, hipMemcpyHostToDevice));
    }
    CK(hipDeviceSynchronize());
    std::printf("probe q8a8 rows %u seconds %.2f set %s data %s bytes weight %zu codes %zu deltas %zu out %zu\n", R, secs,
                set.c_str(), data.c_str(), buf.weight_bytes, buf.codes_bytes, buf.delta_floats * 4U, buf.out_floats * 4U);
    for (const Shape& s : std::span<const Shape>(first, arms)) run(s, buf, secs);
    CK(hipFree(buf.weight));
    CK(hipFree(buf.codes));
    CK(hipFree(buf.deltas));
    CK(hipFree(buf.out));
    std::printf("probe done\n");
    return 0;
}
