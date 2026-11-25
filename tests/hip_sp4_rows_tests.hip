// strix.sp4t resident rows on HIP (core/resident_q8_rows.hpp -> small_m::gemm_sp4), no model:
//   check: device rows == the host dequantised reference within the kernel's FP32 accumulation bound, every row of a
//          multi-row launch bitwise equal to its one-row launch, the softcap head on SP4 == softcap of the rows;
//   --bench: B1 (1 / 2 / 8 / 16 rows) of the Inkling 2 Mini head shape 200058 x 2304, SP4 against q4_g64 on the same
//            f16 group-scale route, interleaved (GPU).
#include "strix/hip/legacy_intrinsics.hpp"
#include "strix/hip/legacy_launch.hpp"
#include "core/resident_q8_rows.hpp"
#include "strix/weight/host_lowbit_gptq.hpp"

#include <algorithm>
#include <bit>
#include <chrono>
#include <cmath>
#include <cstring>
#include <iostream>
#include <random>
#include <stdexcept>
#include <string>
#include <string_view>
#include <vector>

namespace {
namespace core = strix::core;
namespace weight = strix::weight;
namespace sp = strix::weight::sp;
using strix::hip_check;
using S = sp::SP4Storage;

template <class T>
struct Buffer {
    T* data{};
    explicit Buffer(std::size_t size) { hip_check(hipMalloc(&data, size * sizeof(T)), "allocate"); }
    ~Buffer() { static_cast<void>(hipFree(data)); }
    Buffer(const Buffer&) = delete;
    Buffer& operator=(const Buffer&) = delete;
    void upload(std::span<const T> values) const {
        hip_check(hipMemcpy(data, values.data(), values.size_bytes(), hipMemcpyHostToDevice), "upload");
    }
    void download(std::span<T> values) const {
        hip_check(hipMemcpy(values.data(), data, values.size_bytes(), hipMemcpyDeviceToHost), "download");
    }
};

struct Sp4Matrix {
    std::vector<std::uint8_t> bytes;
    sp::Header header;
    std::vector<float> weights;  // the host dequantised reference (sp::decode)
};

Sp4Matrix make_sp4(std::size_t n, std::size_t k, std::mt19937& rng, float sigma) {
    std::normal_distribution<float> dist(0.0F, sigma);
    std::vector<float> w(n * k);
    for (auto& v : w) v = dist(rng);
    Sp4Matrix m;
    m.header = sp::make_header<S>(w.data(), n, k, sp::default_table<4>());
    m.bytes.resize(n * S::row_bytes(k));
    m.weights.resize(n * k);
    weight::lowbit::gptq::Options o;
    o.dequant = m.weights.data();
    if (!weight::lowbit::gptq::encode_rows_sp<S>(m.header, w.data(), n, k, m.bytes.data(), o))
        throw std::runtime_error("SP4 encode");
    return m;
}

weight::ResidentMatrixView view(const std::uint8_t* device, std::size_t bytes, weight::FormatId storage, std::size_t n,
                                std::size_t k, std::uint64_t row_bytes, std::int32_t outer) {
    weight::ResidentMatrixView v{{device, bytes}, storage, n, k, row_bytes};
    v.outer_exponent = outer;
    return v;
}

float fp16_round(float x) { return static_cast<float>(static_cast<_Float16>(x)); }

void check_rows(std::mt19937& rng) {
    // 1000 outputs (a partial last panel), K 2304 (the head's width), weights of the head's scale.
    constexpr std::size_t n = 1000, k = 2304, max_rows = 33;
    const auto m = make_sp4(n, k, rng, 0.02F);
    Buffer<std::uint8_t> w(m.bytes.size());
    w.upload(m.bytes);
    std::uniform_real_distribution<float> act(-3.0F, 3.0F);
    std::vector<float> x(max_rows * k);
    for (auto& v : x) v = act(rng);
    Buffer<float> dx(x.size()), dy(max_rows * n), dy1(n);
    dx.upload(x);
    const auto v = view(w.data, m.bytes.size(), weight::kSP4TWeightStorage, n, k, S::row_bytes(k), m.header.base);
    // One-row launches: the reference rows of the bitwise row-count check.
    std::vector<float> one(max_rows * n), got(max_rows * n), row(n);
    for (std::size_t r = 0; r < max_rows; ++r) {
        hip_check(core::enqueue_resident_q8_rows(v, n, dx.data + r * k, 1U, dy1.data, nullptr,
                                                 core::LowBitArithmetic::f16_group_scale), "sp4 one row");
        hip_check(hipDeviceSynchronize(), "sync");
        dy1.download(row);
        std::copy(row.begin(), row.end(), one.begin() + static_cast<std::ptrdiff_t>(r * n));
    }
    double worst = 0.0;
    for (std::size_t r = 0; r < max_rows; ++r)
        for (std::size_t c = 0; c < n; ++c) {
            double ref = 0.0, mag = 0.0;
            for (std::size_t i = 0; i < k; ++i) {
                const double t = static_cast<double>(fp16_round(x[r * k + i])) * static_cast<double>(m.weights[c * k + i]);
                ref += t;
                mag += std::fabs(t);
            }
            // FP32 accumulation: 32-term WMMA chains, 72 group fmas, x 2^base (exact): |err| <= ~104 u sum|t|.
            const double err = std::fabs(static_cast<double>(one[r * n + c]) - ref);
            const double bound = 0x1p-16 * mag + 1e-30;
            worst = std::max(worst, err / bound);
            if (!(err <= bound))
                throw std::runtime_error("SP4 row " + std::to_string(r) + " col " + std::to_string(c) + " err " +
                                         std::to_string(err) + " > bound " + std::to_string(bound));
        }
    for (const unsigned rows : {2U, 8U, 16U, 17U, 33U}) {
        hip_check(core::enqueue_resident_q8_rows(v, n, dx.data, rows, dy.data, nullptr,
                                                 core::LowBitArithmetic::f16_group_scale), "sp4 rows");
        hip_check(hipDeviceSynchronize(), "sync");
        dy.download(std::span(got).first(std::size_t{rows} * n));
        if (std::memcmp(got.data(), one.data(), std::size_t{rows} * n * sizeof(float)) != 0)
            throw std::runtime_error("SP4 rows " + std::to_string(rows) + " differ from the one-row launches");
    }
    // The fp32 sweep has no SP4 form: refused, nothing enqueued.
    if (core::enqueue_resident_q8_rows(v, n, dx.data, 1U, dy.data, nullptr, core::LowBitArithmetic::fp32_sweep) ==
        hipSuccess)
        throw std::runtime_error("SP4 accepted on the fp32 sweep");
    // Softcap head on SP4 == the shared softcap of the SP4 rows.
    constexpr float cap = 30.0F;
    hip_check(core::enqueue_resident_q8_softcap_head(v, n, dx.data, 1U, std::span(dy.data, n), cap, nullptr,
                                                     core::LowBitArithmetic::f16_group_scale), "sp4 head");
    hip_check(hipDeviceSynchronize(), "sync");
    dy.download(std::span(got).first(n));
    hip_check(hipMemcpy(dy1.data, one.data(), n * sizeof(float), hipMemcpyHostToDevice), "upload rows");
    hip_check(core::enqueue_bf16_head_softcap(1U, n, dy1.data, cap, nullptr), "softcap");
    hip_check(hipDeviceSynchronize(), "sync");
    dy1.download(row);
    if (std::memcmp(got.data(), row.data(), n * sizeof(float)) != 0)
        throw std::runtime_error("SP4 softcap head differs from softcap(rows)");
    std::cout << "sp4 rows check PASS (n " << n << ", k " << k << ", base " << int{m.header.base}
              << ", worst err/bound " << worst << ", rows 1..33 bitwise per row)\n";
}

void bench(std::mt19937& rng) {
    // Same bytes per weight (68 per 128); the time does not depend on the values: random codes, sane scales.
    constexpr std::size_t n = 200058, k = 2304;
    std::vector<std::uint8_t> sp4(n * S::row_bytes(k)), q4(n * weight::Q4G64Storage::row_bytes(k));
    for (auto& b : sp4) b = static_cast<std::uint8_t>(rng());
    for (auto& b : q4) b = static_cast<std::uint8_t>(rng());
    for (std::size_t blk = 0; blk < sp4.size() / S::bytes; ++blk)
        for (unsigned g = 0; g < 4U; ++g) sp4[blk * S::bytes + S::scale_offset + g] = static_cast<std::uint8_t>(64U + (rng() & 63U));
    for (std::size_t blk = 0; blk < q4.size() / weight::Q4G64Storage::bytes; ++blk)
        for (unsigned g = 0; g < 2U; ++g) {  // FP16 scale 2^-10 per group
            q4[blk * weight::Q4G64Storage::bytes + weight::Q4G64Storage::scale_offset + 2U * g] = 0x00U;
            q4[blk * weight::Q4G64Storage::bytes + weight::Q4G64Storage::scale_offset + 2U * g + 1U] = 0x14U;
        }
    Buffer<std::uint8_t> ws(sp4.size()), wq(q4.size());
    ws.upload(sp4);
    wq.upload(q4);
    std::vector<float> x(16 * k, 0.5F);
    Buffer<float> dx(x.size()), dy(16 * n);
    dx.upload(x);
    const auto vs = view(ws.data, sp4.size(), weight::kSP4TWeightStorage, n, k, S::row_bytes(k), -40);
    const auto vq = view(wq.data, q4.size(), weight::kQ4G64WeightStorage, n, k, weight::Q4G64Storage::row_bytes(k), 0);
    hipEvent_t a, b;
    hip_check(hipEventCreate(&a), "event");
    hip_check(hipEventCreate(&b), "event");
    const auto time = [&](const weight::ResidentMatrixView& v, unsigned rows) {
        constexpr int reps = 50;
        for (int i = 0; i < 5; ++i)
            hip_check(core::enqueue_resident_q8_rows(v, n, dx.data, rows, dy.data, nullptr,
                                                     core::LowBitArithmetic::f16_group_scale), "warm");
        hip_check(hipEventRecord(a, nullptr), "record");
        for (int i = 0; i < reps; ++i)
            hip_check(core::enqueue_resident_q8_rows(v, n, dx.data, rows, dy.data, nullptr,
                                                     core::LowBitArithmetic::f16_group_scale), "run");
        hip_check(hipEventRecord(b, nullptr), "record");
        hip_check(hipEventSynchronize(b), "sync");
        float ms = 0.0F;
        hip_check(hipEventElapsedTime(&ms, a, b), "elapsed");
        return ms / reps;
    };
    for (const unsigned rows : {1U, 2U, 8U, 16U}) {
        float s = 0.0F, q = 0.0F;
        for (int round = 0; round < 3; ++round) {  // interleaved: q4, sp4, q4, sp4 ...
            q += time(vq, rows);
            s += time(vs, rows);
        }
        s /= 3.0F;
        q /= 3.0F;
        std::cout << "sp4 bench rows " << rows << " q4_g64 " << q << " ms sp4t " << s << " ms (" << n * k * 17 / 32
                  << " B each) sp4/q4 " << s / q << '\n';
    }
}

}  // namespace

int main(int argc, char** argv) {
    try {
        std::mt19937 rng(20261125U);
        check_rows(rng);
        if (argc == 2 && std::string_view(argv[1]) == "--bench") bench(rng);
    } catch (const std::exception& error) {
        std::cerr << "FAIL sp4 rows: " << error.what() << '\n';
        return 1;
    }
    return 0;
}
