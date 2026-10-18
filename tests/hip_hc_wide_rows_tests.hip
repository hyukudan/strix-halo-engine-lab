// Prefill wide HC kernels: every projection/output kernel form against the reference forms
// (projection kernel 0, output kernel 0) on the same inputs, bitwise over every written byte
// (low values, injection, output) plus guard bytes, for BF16 and FP32 outputs, several row counts
// and input flavors. STRIX_HC_WIDE_BENCH=1 adds per-kernel timings at 2048 rows.
#include "strix/hip_hc_wide_rows_tests.hpp"
#include <hip/hip_runtime.h>
#include <algorithm>
#include <array>
#include <bit>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <iostream>
#include <random>
#include <stdexcept>
#include <string>
#include <vector>

namespace {
void check(hipError_t e, const char* text) {
    if (e != hipSuccess) throw std::runtime_error(std::string(text) + ": " + hipGetErrorString(e));
}
struct Buffer {
    void* p = nullptr;
    std::size_t n = 0;
    explicit Buffer(std::size_t bytes) : n(bytes) { check(hipMalloc(&p, bytes), "allocate"); }
    ~Buffer() { if (p) static_cast<void>(hipFree(p)); }
    Buffer(const Buffer&) = delete;
    template <class T> void upload(const std::vector<T>& v) { check(hipMemcpy(p, v.data(), v.size() * sizeof(T), hipMemcpyHostToDevice), "upload"); }
    std::vector<unsigned char> download() const {
        std::vector<unsigned char> v(n);
        check(hipMemcpy(v.data(), p, n, hipMemcpyDeviceToHost), "download");
        return v;
    }
};
constexpr std::uint32_t kHidden = 2560U, kStreams = 4U, kRank = 320U, kWidth = kHidden * kStreams;
// Row strides with padding (the engine's rows are one stride apart, not packed).
constexpr std::uint64_t kNormalizedStride = (kWidth + kRank + 64U) * 4U, kInjectionStride = 64U;

std::vector<std::uint8_t> q8_weights(std::uint32_t rows, std::uint32_t width, unsigned flavor, std::mt19937& rng) {
    std::vector<std::uint8_t> w(static_cast<std::size_t>(rows) * (width / 32U) * 34U);
    std::uniform_int_distribution<int> code(-128, 127), mant(0, 1023), ex(9, 13);
    constexpr std::array<std::uint16_t, 8> special{0x0000, 0x8000, 0x0001, 0x8001, 0x03ff, 0x7bff, 0xfbff, 0x3c00};
    for (std::size_t b = 0; b < w.size() / 34U; ++b) {
        std::uint16_t scale = static_cast<std::uint16_t>((static_cast<unsigned>(ex(rng)) << 10U) | static_cast<unsigned>(mant(rng)) | (rng() & 1U ? 0x8000U : 0U));
        if (flavor == 1U && b % 7U == 3U) scale = special[(b / 7U) % special.size()];
        w[b * 34U] = static_cast<std::uint8_t>(scale & 255U);
        w[b * 34U + 1U] = static_cast<std::uint8_t>(scale >> 8U);
        for (unsigned k = 0; k < 32U; ++k) w[b * 34U + 2U + k] = static_cast<std::uint8_t>(code(rng));
    }
    return w;
}

struct Fixture {
    std::uint32_t rows;
    bool bf16;
    std::uint64_t output_stride;
    Buffer normalized, injection, output, down, inject, up;
    std::vector<unsigned char> normalized_host;
    Fixture(std::uint32_t r, bool bf, unsigned flavor, std::uint32_t seed)
        : rows(r), bf16(bf), output_stride((kHidden + 32U) * (bf ? 2U : 4U)),
          normalized(static_cast<std::size_t>(r) * kNormalizedStride), injection(static_cast<std::size_t>(r) * kInjectionStride),
          output(static_cast<std::size_t>(r) * output_stride), down(static_cast<std::size_t>(kRank) * (kWidth / 32U) * 34U),
          inject(static_cast<std::size_t>(kStreams) * kWidth * 4U), up(static_cast<std::size_t>(kStreams) * kHidden * (kRank / 32U) * 34U) {
        std::mt19937 rng(seed);
        std::normal_distribution<float> normal(0.0F, 1.0F);
        std::vector<float> x(static_cast<std::size_t>(r) * kNormalizedStride / 4U, 0.0F);
        constexpr std::array<std::uint32_t, 8> bits{0U, 0x80000000U, 1U, 0x80000001U, 0x00800000U, 0x80800000U, 0x3f800000U, 0x42f00000U};
        for (std::uint32_t row = 0; row < r; ++row)
            for (std::uint32_t i = 0; i < kWidth; ++i) {
                float v = normal(rng);
                if (flavor == 1U && (i * 7U + row) % 97U == 5U) v = std::bit_cast<float>(bits[(i + row) % bits.size()]);
                if (flavor == 2U) v *= (i % 640U < 4U) ? 300.0F : 1.0F;  // outlier channels
                x[static_cast<std::size_t>(row) * kNormalizedStride / 4U + i] = v;
            }
        normalized.upload(x);
        normalized_host = normalized.download();
        down.upload(q8_weights(kRank, kWidth, flavor, rng));
        up.upload(q8_weights(kStreams * kHidden, kRank, flavor, rng));
        std::vector<float> w(static_cast<std::size_t>(kStreams) * kWidth);
        for (auto& v : w) v = normal(rng) * 1e-3F;
        inject.upload(w);
    }
    strix::testing::WideHyperRowsTestViews views() const {
        strix::testing::WideHyperRowsTestViews v;
        v.normalized = static_cast<float*>(normalized.p); v.normalized_stride = kNormalizedStride;
        v.injection = static_cast<float*>(injection.p); v.injection_stride = kInjectionStride;
        v.output = output.p; v.output_stride = output_stride;
        v.rows = rows; v.down = down.p; v.inject = inject.p; v.up = up.p;
        return v;
    }
    // Poisons everything the kernels may write (lows, injection, output, padding) and restores the inputs.
    void reset(unsigned char poison) {
        normalized.upload(normalized_host);
        std::vector<unsigned char> n = normalized_host;
        for (std::uint32_t row = 0; row < rows; ++row)
            std::memset(n.data() + row * kNormalizedStride + kWidth * 4U, poison, kNormalizedStride - kWidth * 4U);
        normalized.upload(n);
        check(hipMemset(injection.p, poison, injection.n), "poison injection");
        check(hipMemset(output.p, poison, output.n), "poison output");
    }
    std::vector<unsigned char> written() const {
        auto a = normalized.download(), b = injection.download(), c = output.download();
        a.insert(a.end(), b.begin(), b.end());
        a.insert(a.end(), c.begin(), c.end());
        return a;
    }
};

strix::HipHyperMixLauncherConfig config(bool bf16) {
    strix::HipHyperMixLauncherConfig c;
    c.hidden_size = kHidden; c.hc_count = kStreams; c.lowrank = kRank; c.rms_epsilon = 1e-6F;
    c.parallel = true; c.splitk_projection = true; c.lane_chain_output = true;
    c.activation = bf16 ? strix::HipActivationStorage::bf16 : strix::HipActivationStorage::fp32;
    return c;
}

void compare(std::uint32_t rows, bool bf16, unsigned flavor, unsigned projection, unsigned output, hipStream_t stream) {
    Fixture f(rows, bf16, flavor, 1234U + rows + flavor * 17U + (bf16 ? 1U : 0U));
    const auto c = config(bf16);
    f.reset(0xa5);
    strix::testing::enqueue_wide_hyper_rows_test(c, f.views(), 0U, 0U, stream);
    check(hipStreamSynchronize(stream), "reference");
    const auto expected = f.written();
    f.reset(0xa5);
    strix::testing::enqueue_wide_hyper_rows_test(c, f.views(), projection, output, stream);
    check(hipStreamSynchronize(stream), "candidate");
    const auto actual = f.written();
    std::size_t differ = 0, first = 0;
    for (std::size_t i = 0; i < expected.size(); ++i)
        if (expected[i] != actual[i]) { if (!differ) first = i; ++differ; }
    std::cout << "rows " << rows << (bf16 ? " bf16" : " fp32") << " flavor " << flavor << " projection " << projection << " output "
              << output << ": " << expected.size() << " bytes, differ " << differ << '\n';
    if (differ) throw std::runtime_error("wide HC kernels differ from the reference forms at byte " + std::to_string(first));
}

// Fast profile kernels (2/2): not bitwise; error of the low values, injection and output against the
// reference forms relative to the RMS of the reference values, and every padding byte untouched.
float to_float(const unsigned char* p, bool bf16) {
    if (!bf16) { float v; std::memcpy(&v, p, 4); return v; }
    std::uint16_t b; std::memcpy(&b, p, 2);
    return std::bit_cast<float>(static_cast<std::uint32_t>(b) << 16U);
}
void compare_fast(std::uint32_t rows, bool bf16, unsigned flavor, hipStream_t stream, unsigned output = 2U) {
    Fixture f(rows, bf16, flavor, 1234U + rows + flavor * 17U + (bf16 ? 1U : 0U));
    const auto c = config(bf16);
    f.reset(0xa5);
    strix::testing::enqueue_wide_hyper_rows_test(c, f.views(), 0U, 0U, stream);
    check(hipStreamSynchronize(stream), "reference");
    const auto en = f.normalized.download(), ei = f.injection.download(), eo = f.output.download();
    f.reset(0xa5);
    strix::testing::enqueue_wide_hyper_rows_test(c, f.views(), 2U, output, stream);
    check(hipStreamSynchronize(stream), "candidate");
    const auto an = f.normalized.download(), ai = f.injection.download(), ao = f.output.download();
    struct Err { double sq = 0, ref = 0, max = 0; std::size_t n = 0, nonfinite = 0; };
    const auto add = [](Err& e, float ref, float got) {
        if (!std::isfinite(got) || !std::isfinite(ref)) { ++e.nonfinite; return; }
        const double d = static_cast<double>(got) - ref;
        e.sq += d * d; e.ref += static_cast<double>(ref) * ref; e.max = std::max(e.max, std::abs(d)); ++e.n;
    };
    Err low, inj, out;
    std::size_t pad = 0;
    for (std::uint32_t r = 0; r < rows; ++r) {
        const std::size_t base = static_cast<std::size_t>(r) * kNormalizedStride;
        for (std::uint32_t k = 0; k < kRank; ++k)
            add(low, to_float(&en[base + (kWidth + k) * 4U], false), to_float(&an[base + (kWidth + k) * 4U], false));
        for (std::size_t b = base + (kWidth + kRank) * 4U; b < base + kNormalizedStride; ++b) pad += an[b] != en[b];
        for (std::uint32_t s = 0; s < kStreams; ++s)
            add(inj, to_float(&ei[r * kInjectionStride + s * 4U], false), to_float(&ai[r * kInjectionStride + s * 4U], false));
        for (std::size_t b = r * kInjectionStride + kStreams * 4U; b < (r + 1U) * kInjectionStride; ++b) pad += ai[b] != ei[b];
        const std::size_t e = bf16 ? 2U : 4U;
        for (std::uint32_t h = 0; h < kHidden; ++h)
            add(out, to_float(&eo[r * f.output_stride + h * e], bf16), to_float(&ao[r * f.output_stride + h * e], bf16));
        for (std::size_t b = r * f.output_stride + kHidden * e; b < (r + 1U) * f.output_stride; ++b) pad += ao[b] != eo[b];
    }
    const auto rel = [](const Err& e) { return e.ref > 0 ? std::sqrt(e.sq / e.ref) : std::sqrt(e.sq); };
    const auto rms = [](const Err& e) { return e.n ? std::sqrt(e.ref / static_cast<double>(e.n)) : 0.0; };
    std::cout << "fast rows " << rows << (bf16 ? " bf16" : " fp32") << " flavor " << flavor << " output " << output << ": low rel " << rel(low) << " max "
              << low.max / rms(low) << " | injection rel " << rel(inj) << " max " << inj.max / rms(inj) << " | output rel " << rel(out)
              << " max " << out.max / rms(out) << " | nonfinite " << low.nonfinite + inj.nonfinite + out.nonfinite << " padding " << pad << '\n';
    if (pad || low.nonfinite + inj.nonfinite + out.nonfinite || rel(low) > 2e-2 || rel(inj) > 2e-2 || rel(out) > 2e-2)
        throw std::runtime_error("fast HC kernels outside the error bound or touched padding");
}

void bench(hipStream_t stream) {
    Fixture f(2048U, true, 0U, 99U);
    const auto c = config(true);
    hipEvent_t a, b;
    check(hipEventCreate(&a), "event"); check(hipEventCreate(&b), "event");
    const auto time = [&](unsigned projection, unsigned output) {
        std::vector<float> samples;
        for (unsigned it = 0; it < 25U; ++it) {
            check(hipEventRecord(a, stream), "record");
            strix::testing::enqueue_wide_hyper_rows_test(c, f.views(), projection, output, stream);
            check(hipEventRecord(b, stream), "record");
            check(hipEventSynchronize(b), "sync");
            float ms = 0; check(hipEventElapsedTime(&ms, a, b), "elapsed");
            if (it >= 5U) samples.push_back(ms * 1000.0F);
        }
        std::sort(samples.begin(), samples.end());
        return samples[samples.size() / 2U];
    };
    const double flop = 2.0 * 2048.0 * 10240.0 * 324.0, flop_out = 2.0 * 2048.0 * 10240.0 * 320.0;
    for (unsigned projection = 0; projection <= 1U; ++projection) {
        const float us = time(projection, 255U);
        std::cout << "bench rows 2048 projection " << projection << ": " << us << " us (" << flop / us / 1e6 << " TFLOPS)\n";
    }
    {
        const float us = time(2U, 1U);
        std::cout << "bench rows 2048 projection 2 + output 1 (WMMA + lane chain, with pack): " << us << " us\n";
    }
    {
        const float us = time(2U, 255U);
        std::cout << "bench rows 2048 projection 2 (WMMA, with pack): " << us << " us (" << flop / us / 1e6 << " TFLOPS)\n";
    }
    for (unsigned output = 0; output <= 2U; ++output) {
        // The wide2 output kernel reads the transposed low values the wide2 projection writes, the WMMA output kernel
        // the FP16 low values of the WMMA projection.
        strix::testing::enqueue_wide_hyper_rows_test(c, f.views(), output == 2U ? 2U : 1U, 255U, stream);
        const float us = time(255U, output);
        std::cout << "bench rows 2048 output " << output << ": " << us << " us (" << flop_out / us / 1e6 << " TFLOPS)\n";
    }
}
}  // namespace

int main() try {
    hipStream_t stream = nullptr;
    check(hipStreamCreateWithFlags(&stream, hipStreamNonBlocking), "stream");
    if (!std::getenv("STRIX_HC_WIDE_BENCH_ONLY")) {
        for (const bool bf16 : {true, false})
            for (const std::uint32_t rows : {8U, 40U, 264U})
                for (unsigned flavor = 0; flavor < 3U; ++flavor) {
                    compare(rows, bf16, flavor, 1U, 0U, stream);
                    compare(rows, bf16, flavor, 1U, 1U, stream);
                }
        compare(2048U, true, 0U, 1U, 1U, stream);
        std::cout << "wide HC kernels == reference forms: ok\n";
    }
    if (!std::getenv("STRIX_HC_WIDE_BENCH_ONLY")) {
        // Flavor 1 has FP16-overflowing effective weights (scale 65504); the fast kernels run on flavors 0 and 2.
        for (const bool bf16 : {true, false})
            for (const std::uint32_t rows : {8U, 40U, 264U, 2048U})
                for (const unsigned flavor : {0U, 2U})
                    for (const unsigned output : {1U, 2U}) compare_fast(rows, bf16, flavor, stream, output);
        std::cout << "fast HC kernels within the error bound: ok\n";
    }
    if (std::getenv("STRIX_HC_WIDE_BENCH") || std::getenv("STRIX_HC_WIDE_BENCH_ONLY")) bench(stream);
    check(hipStreamDestroy(stream), "stream");
    return 0;
} catch (const std::exception& e) {
    std::cerr << "wide HC fixture failed: " << e.what() << '\n';
    return 1;
}
