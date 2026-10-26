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
        constexpr std::array<std::uint16_t, 6> nonfinite{0x7c00, 0xfc00, 0x7e00, 0xfe01, 0x7d55, 0x0000};
        if (flavor == 3U && b % 11U == 4U) scale = nonfinite[(b / 11U) % nonfinite.size()];
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
                constexpr std::array<std::uint32_t, 4> nonfinite{0x7f800000U, 0xff800000U, 0x7fc00001U, 0xffa00000U};
                if (flavor == 3U && (i * 13U + row) % 4099U == 17U) v = std::bit_cast<float>(nonfinite[(i + row) % nonfinite.size()]);
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
        check(hipDeviceSynchronize(), "poison drain");  // the test stream does not wait for the null stream
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

void compare(std::uint32_t rows, bool bf16, unsigned flavor, unsigned projection, unsigned output, hipStream_t stream,
    unsigned ref_projection = 0U, unsigned ref_output = 0U) {
    Fixture f(rows, bf16, flavor, 1234U + rows + flavor * 17U + (bf16 ? 1U : 0U));
    const auto c = config(bf16);
    f.reset(0xa5);
    strix::testing::enqueue_wide_hyper_rows_test(c, f.views(), ref_projection, ref_output, stream);
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

// FP16-input projection (the fast-profile norm's FP16 rows): low values and output bitwise equal to the FP32-input
// form (same RNE operands), injection within the FP16-input error; also run at 8192 rows (largest tiled block).
void compare_half(std::uint32_t rows, bool bf16, unsigned flavor, hipStream_t stream) {
    Fixture f(rows, bf16, flavor, 4321U + rows + flavor);
    const auto c = config(bf16);
    constexpr std::uint32_t pitch = kWidth + 64U;  // the engine's padded FP16 row pitch
    std::vector<_Float16> half(static_cast<std::size_t>(rows) * pitch);
    for (std::uint32_t r = 0; r < rows; ++r)
        for (std::uint32_t i = 0; i < kWidth; ++i) {
            float v;
            std::memcpy(&v, &f.normalized_host[static_cast<std::size_t>(r) * kNormalizedStride + i * 4U], 4);
            half[static_cast<std::size_t>(r) * pitch + i] = static_cast<_Float16>(v);
        }
    Buffer h(half.size() * 2U);
    h.upload(half);
    f.reset(0xa5);
    strix::testing::enqueue_wide_hyper_rows_test(c, f.views(), 2U, 1U, stream);
    check(hipStreamSynchronize(stream), "fp32-input form");
    const auto en = f.normalized.download(), ei = f.injection.download(), eo = f.output.download();
    f.reset(0xa5);
    auto v = f.views();
    v.normalized16 = h.p;
    v.normalized16_pitch = pitch;
    strix::testing::enqueue_wide_hyper_rows_test(c, v, 2U, 1U, stream);
    check(hipStreamSynchronize(stream), "fp16-input form");
    const auto an = f.normalized.download(), ai = f.injection.download(), ao = f.output.download();
    std::size_t differ = 0;
    for (std::size_t i = 0; i < en.size(); ++i) differ += en[i] != an[i];
    for (std::size_t i = 0; i < eo.size(); ++i) differ += eo[i] != ao[i];
    double sq = 0, ref = 0;
    for (std::uint32_t r = 0; r < rows; ++r)
        for (std::uint32_t k = 0; k < kStreams; ++k) {
            float a, b;
            std::memcpy(&a, &ei[r * kInjectionStride + k * 4U], 4); std::memcpy(&b, &ai[r * kInjectionStride + k * 4U], 4);
            sq += (static_cast<double>(b) - a) * (static_cast<double>(b) - a); ref += static_cast<double>(a) * a;
        }
    const double rel = std::sqrt(sq / ref);
    std::cout << "half rows " << rows << (bf16 ? " bf16" : " fp32") << " flavor " << flavor << ": low/output bytes differ " << differ
              << ", injection rel vs fp32-input " << rel << '\n';
    if (differ || !(rel < 1e-4)) throw std::runtime_error("FP16-input HC projection differs from the FP32-input form");
    // FP16-only rows (STRIX_HCW_NORM16=2): the FP32 normalized values are NaN, so any read of them would show; low values and
    // injection equal the FP16-input run, the output differs only by the FP16 rounding of the mix's normalized values.
    {
        std::vector<unsigned char> poisoned = f.normalized_host;
        const float nan = std::bit_cast<float>(0x7fc00000U);
        for (std::uint32_t r = 0; r < rows; ++r)
            for (std::uint32_t i = 0; i < kWidth; ++i) std::memcpy(&poisoned[static_cast<std::size_t>(r) * kNormalizedStride + i * 4U], &nan, 4);
        f.normalized.upload(poisoned);
        check(hipMemset(f.injection.p, 0xa5, f.injection.n), "poison injection");
        check(hipMemset(f.output.p, 0xa5, f.output.n), "poison output");
        check(hipDeviceSynchronize(), "poison drain");
        v.normalized16_only = true;
        strix::testing::enqueue_wide_hyper_rows_test(c, v, 2U, 1U, stream);
        check(hipStreamSynchronize(stream), "fp16-only form");
        const auto bn = f.normalized.download(), bi = f.injection.download(), bo = f.output.download();
        std::size_t low_differ = 0, injection_differ = 0, nonfinite = 0;
        for (std::uint32_t r = 0; r < rows; ++r)
            for (std::size_t b = static_cast<std::size_t>(r) * kNormalizedStride + kWidth * 4U; b < static_cast<std::size_t>(r) * kNormalizedStride + (kWidth + kRank) * 4U; ++b)
                low_differ += bn[b] != an[b];
        for (std::size_t b = 0; b < bi.size(); ++b) injection_differ += bi[b] != ai[b];
        double osq = 0, oref = 0;
        const std::size_t e = bf16 ? 2U : 4U;
        for (std::uint32_t r = 0; r < rows; ++r)
            for (std::uint32_t k = 0; k < kHidden; ++k) {
                const float x = to_float(&ao[r * f.output_stride + k * e], bf16), y = to_float(&bo[r * f.output_stride + k * e], bf16);
                if (!std::isfinite(y)) { ++nonfinite; continue; }
                osq += (static_cast<double>(y) - x) * (static_cast<double>(y) - x); oref += static_cast<double>(x) * x;
            }
        const double orel = std::sqrt(osq / oref);
        std::cout << "  fp16-only rows: low bytes differ " << low_differ << ", injection bytes differ " << injection_differ
                  << ", nonfinite outputs " << nonfinite << ", output rel vs fp16-input " << orel << '\n';
        if (low_differ || injection_differ || nonfinite || !(orel < 2e-3))
            throw std::runtime_error("FP16-only HC rows differ from the FP16-input form beyond the mix rounding");
    }
}

// STRIX_QPF2K_HC folded inject (+ combine) + norm against the window kernel (tile by tile), every written byte: streams
// (in place and separate), normalized rows, combined rows; BF16 inputs with edge values (zeros, denormals, large, ties).
void compare_inject_norm(std::uint32_t rows, bool combine, bool in_place, unsigned flavor, hipStream_t stream) {
    std::mt19937 rng(777U + rows + (combine ? 1U : 0U) + (in_place ? 2U : 0U) + flavor * 5U);
    std::normal_distribution<float> normal(0.0F, 1.0F);
    const auto bf16 = [](float v) {
        std::uint32_t b;
        std::memcpy(&b, &v, 4);
        return static_cast<std::uint16_t>((b + 0x7fffU + ((b >> 16U) & 1U)) >> 16U);
    };
    constexpr std::array<std::uint16_t, 8> edge{0x0000, 0x8000, 0x0001, 0x8001, 0x0080, 0x7f7f, 0x3f80, 0xbf80};
    const auto value = [&](std::size_t i) {
        if (flavor == 1U && i % 61U == 7U) return edge[(i / 61U) % edge.size()];
        constexpr std::array<std::uint16_t, 5> nonfinite{0x7f80, 0xff80, 0x7fc0, 0x7f81, 0xffc1};
        if (flavor == 3U && i % 4093U == 11U) return nonfinite[(i / 4093U) % nonfinite.size()];
        // RNE boundaries: values whose FP32 inject result lands near a BF16 tie.
        if (flavor == 1U && i % 61U == 9U) return static_cast<std::uint16_t>(0x3f80U + (i % 3U));
        float v = normal(rng);
        if (flavor == 2U && i % 640U < 3U) v *= 500.0F;
        return bf16(v);
    };
    constexpr std::uint64_t hs = (kWidth + 64U) * 2U, ds = (kHidden + 32U) * 2U, gs = 32U, ns = (kWidth + kRank + 64U) * 4U;
    std::vector<std::uint16_t> hyper(rows * hs / 2U), hidden(rows * ds / 2U), routed(rows * ds / 2U);
    for (std::size_t i = 0; i < hyper.size(); ++i) hyper[i] = value(i);
    for (std::size_t i = 0; i < hidden.size(); ++i) { hidden[i] = value(i + 3U); routed[i] = value(i + 11U); }
    if (flavor == 4U) {
        // Exact BF16 halfway cases of both parities: gate 1, hyper in {+-1, +-(1 + 2^-7)} (even / odd BF16 mantissa),
        // hidden = +-2^-8 = half an ULP at 1 (combine: shared +-1 / +-(1 + 2^-7), routed +-2^-8 gives the same ties in the
        // combined row; the inject then adds a combined value to the hyper value).
        constexpr std::array<std::uint16_t, 4> base{0x3f80, 0x3f81, 0xbf80, 0xbf81};
        constexpr std::array<std::uint16_t, 2> half{0x3b80, 0xbb80};
        for (std::size_t i = 0; i < hyper.size(); ++i) hyper[i] = base[(i * 7U) % 4U];
        for (std::size_t i = 0; i < hidden.size(); ++i) {
            hidden[i] = combine ? base[(i * 5U) % 4U] : half[(i / 3U) % 2U];
            routed[i] = half[(i / 5U) % 2U];
        }
    }
    std::vector<float> gates(rows * gs / 4U), norm(kWidth);
    for (auto& g : gates) g = flavor == 4U ? 1.0F : 2.0F / (1.0F + std::exp(-normal(rng)));
    for (auto& w : norm) w = 1.0F + 0.1F * normal(rng);
    if (flavor == 1U) { gates[0] = 0.0F; gates[1] = -0.0F; norm[5] = 0.0F; }
    if (flavor == 3U) {
        // Non-finite gates and norm weights (BF16 Inf / NaN inputs come from value()).
        gates[2] = std::bit_cast<float>(0x7f800000U); gates[5] = std::bit_cast<float>(0x7fc00003U); gates[9] = std::bit_cast<float>(0xff800000U);
        norm[7] = std::bit_cast<float>(0x7fc00000U); norm[300] = std::bit_cast<float>(0xff800000U);
    }
    Buffer bh(hyper.size() * 2U), bs(hyper.size() * 2U), bd(hidden.size() * 2U), br(routed.size() * 2U), bg(gates.size() * 4U),
        bc(hidden.size() * 2U), bn(static_cast<std::size_t>(rows) * ns), bw(norm.size() * 4U);
    bg.upload(gates);
    bw.upload(norm);
    const auto run = [&](bool reference) {
        bh.upload(hyper);
        bd.upload(hidden);
        br.upload(routed);
        check(hipMemset(bs.p, 0xa5, bs.n), "poison");
        check(hipMemset(bc.p, 0xa5, bc.n), "poison");
        check(hipMemset(bn.p, 0xa5, bn.n), "poison");
        // The test stream is non-blocking: it does not wait for null-stream memsets, so drain them first.
        check(hipDeviceSynchronize(), "poison drain");
        strix::testing::Qpf2kInjectNormTestViews v;
        v.hyper = bh.p; v.hyper_stride = hs;
        v.streams = in_place ? bh.p : bs.p; v.streams_stride = hs;
        // In place: the streams replace their input and the combined row its shared input (the arena's layout).
        v.hidden = bd.p; v.hidden_stride = ds;
        if (combine) { v.routed = br.p; v.routed_stride = ds; v.combined = in_place ? bd.p : bc.p; v.combined_stride = ds; }
        v.gates = static_cast<const float*>(bg.p); v.gates_stride = gs;
        v.normalized = static_cast<float*>(bn.p); v.normalized_stride = ns;
        v.norm = static_cast<const float*>(bw.p);
        v.rows = rows;
        if (reference) {
            // The production prefill kernels in launch order: combine, inject, HC norm.
            const void* hidden_in = bd.p;
            if (combine) {
                strix::testing::enqueue_elementwise_prefill_rows_test(true, bd.p, ds, br.p, ds, nullptr, 0U, v.combined, ds, rows, kHidden, kStreams, stream);
                hidden_in = v.combined;
            }
            strix::testing::enqueue_elementwise_prefill_rows_test(false, bh.p, hs, hidden_in, ds, static_cast<const float*>(bg.p), gs, v.streams, hs, rows,
                kHidden, kStreams, stream);
        }
        strix::testing::enqueue_qpf2k_inject_norm_test(config(true), v, reference, stream);
        check(hipStreamSynchronize(stream), "inject norm");
        auto out = bh.download();
        for (const auto& b : {bs.download(), bd.download(), bc.download(), bn.download()}) out.insert(out.end(), b.begin(), b.end());
        return out;
    };
    const auto expected = run(true), actual = run(false);
    std::size_t differ = 0, first = 0;
    for (std::size_t i = 0; i < expected.size(); ++i)
        if (expected[i] != actual[i]) { if (!differ) first = i; ++differ; }
    std::cout << "inject norm rows " << rows << (combine ? " combine" : "") << (in_place ? " in place" : "") << " flavor " << flavor << ": "
              << expected.size() << " bytes, differ " << differ << '\n';
    if (differ) throw std::runtime_error("qpf2k inject norm differs from the window kernel at byte " + std::to_string(first));
}

void bench(hipStream_t stream) {
    // STRIX_HC_WIDE_BENCH_ROWS: bench row count (default 2048; the engine's tiled prefill blocks are 8192 rows).
    const std::uint32_t n = [] { const char* v = std::getenv("STRIX_HC_WIDE_BENCH_ROWS"); return v ? static_cast<std::uint32_t>(std::atoi(v)) : 2048U; }();
    Fixture f(n, true, 0U, 99U);
    const auto c = config(true);
    hipEvent_t a, b;
    check(hipEventCreate(&a), "event"); check(hipEventCreate(&b), "event");
    // FP16 rows for the FP16-input projection (zeros: timing only).
    Buffer half(static_cast<std::size_t>(n) * (kWidth + 64U) * 2U);
    check(hipMemset(half.p, 0, half.n), "half rows");
    check(hipDeviceSynchronize(), "init drain");
    const void* rows16 = nullptr;
    const auto time = [&](unsigned projection, unsigned output) {
        std::vector<float> samples;
        auto v = f.views();
        v.normalized16 = rows16;
        v.normalized16_pitch = kWidth + 64U;
        for (unsigned it = 0; it < 25U; ++it) {
            check(hipEventRecord(a, stream), "record");
            strix::testing::enqueue_wide_hyper_rows_test(c, v, projection, output, stream);
            check(hipEventRecord(b, stream), "record");
            check(hipEventSynchronize(b), "sync");
            float ms = 0; check(hipEventElapsedTime(&ms, a, b), "elapsed");
            if (it >= 5U) samples.push_back(ms * 1000.0F);
        }
        std::sort(samples.begin(), samples.end());
        return samples[samples.size() / 2U];
    };
    const double flop = 2.0 * n * 10240.0 * 324.0, flop_out = 2.0 * n * 10240.0 * 320.0;
    {
        // STRIX_QPF2K_HC output v3 (seam kernels 7..10) against the lane-chain output 1 it replaces: A B A B A B A
        // (A = output 1, B = form 7, the engine's form), then the attribution forms 8 (no low prefetch),
        // 9 (base epilogue), 10 (prefetch two blocks ahead), 11 (32-step loop), each followed by output 1. Medians of 20 launches.
        strix::testing::enqueue_wide_hyper_rows_test(c, f.views(), 2U, 255U, stream);
        for (unsigned i = 0; i < 7U; ++i) {
            const unsigned form = i % 2U ? 7U : 1U;
            const float us = time(255U, form);
            std::cout << "bench v3 rows " << n << " aba " << i << " output " << form << ": " << us << " us (" << flop_out / us / 1e6 << " TFLOPS)\n";
        }
        for (const unsigned form : {8U, 9U, 10U, 11U, 12U, 13U}) {
            const float us = time(255U, form), a2 = time(255U, 1U);
            std::cout << "bench v3 rows " << n << " output " << form << ": " << us << " us (" << flop_out / us / 1e6 << " TFLOPS) | output 1 after: " << a2 << " us\n";
        }
        if (std::getenv("STRIX_HC_WIDE_BENCH_V3_ONLY")) return;
    }
    for (unsigned projection = 0; projection <= 1U; ++projection) {
        const float us = time(projection, 255U);
        std::cout << "bench rows " << n << " projection " << projection << ": " << us << " us (" << flop / us / 1e6 << " TFLOPS)\n";
    }
    {
        const float us = time(2U, 1U);
        std::cout << "bench rows " << n << " projection 2 + output 1 (WMMA + lane chain, with pack): " << us << " us\n";
    }
    {
        const float us = time(2U, 255U), qus = time(3U, 255U), q4 = time(4U, 255U);
        std::cout << "bench rows " << n << " projection 2 (WMMA, with pack): " << us << " us (" << flop / us / 1e6 << " TFLOPS) | qpf2k projection 3: "
                  << qus << " us (" << flop / qus / 1e6 << " TFLOPS) | qpf2k projection 4 (v2): " << q4 << " us (" << flop / q4 / 1e6 << " TFLOPS)\n";
        rows16 = half.p;
        const float hus = time(2U, 255U);
        rows16 = nullptr;
        std::cout << "bench rows " << n << " projection 2, FP16 rows (with pack): " << hus << " us (" << flop / hus / 1e6 << " TFLOPS)\n";
    }
    {
        strix::testing::enqueue_wide_hyper_rows_test(c, f.views(), 2U, 255U, stream);
        const float us = time(255U, 1U), qus = time(255U, 3U);
        std::cout << "bench rows " << n << " output 1: " << us << " us (" << flop_out / us / 1e6 << " TFLOPS) | qpf2k output 3 (with pack): "
                  << qus << " us (" << flop_out / qus / 1e6 << " TFLOPS)\n";
        for (const unsigned form : {4U, 5U, 6U}) {
            const float fus = time(255U, form);
            std::cout << "bench rows " << n << " qpf2k output " << form << (form == 4U ? " (v2 blocks of 8)" : form == 5U ? " (blocks of 4)" : " (blocks of 10)")
                      << " (with pack): " << fus << " us (" << flop_out / fus / 1e6 << " TFLOPS)\n";
        }
        const float us2 = time(255U, 1U);
        std::cout << "bench rows " << n << " output 1 (again): " << us2 << " us\n";
    }
    if (std::getenv("STRIX_HC_WIDE_BENCH_INJECT")) {
        // Folded inject (+ combine) + norm kernel at n rows, in place (timing only).
        hipEvent_t a2, b2;
        check(hipEventCreate(&a2), "event");
        check(hipEventCreate(&b2), "event");
        Buffer h(static_cast<std::size_t>(n) * (kWidth + 64U) * 2U), d(static_cast<std::size_t>(n) * (kHidden + 32U) * 2U),
            r(static_cast<std::size_t>(n) * (kHidden + 32U) * 2U), g(static_cast<std::size_t>(n) * 32U), nn(static_cast<std::size_t>(n) * kNormalizedStride),
            w(kWidth * 4U);
        check(hipMemset(h.p, 0x3c, h.n), "init"); check(hipMemset(d.p, 0x3c, d.n), "init"); check(hipMemset(r.p, 0x3c, r.n), "init");
        check(hipMemset(g.p, 0, g.n), "init"); check(hipMemset(w.p, 0, w.n), "init");
        check(hipDeviceSynchronize(), "init drain");
        for (const bool combine : {false, true}) {
            strix::testing::Qpf2kInjectNormTestViews v;
            v.hyper = h.p; v.hyper_stride = (kWidth + 64U) * 2U; v.streams = h.p; v.streams_stride = v.hyper_stride;
            v.hidden = d.p; v.hidden_stride = (kHidden + 32U) * 2U;
            if (combine) { v.routed = r.p; v.routed_stride = v.hidden_stride; v.combined = d.p; v.combined_stride = v.hidden_stride; }
            v.gates = static_cast<const float*>(g.p); v.gates_stride = 32U;
            v.normalized = static_cast<float*>(nn.p); v.normalized_stride = kNormalizedStride;
            v.norm = static_cast<const float*>(w.p);
            v.rows = n;
            std::vector<float> samples;
            for (unsigned it = 0; it < 25U; ++it) {
                check(hipEventRecord(a2, stream), "record");
                strix::testing::enqueue_qpf2k_inject_norm_test(c, v, false, stream);
                check(hipEventRecord(b2, stream), "record");
                check(hipEventSynchronize(b2), "sync");
                float ms = 0;
                check(hipEventElapsedTime(&ms, a2, b2), "elapsed");
                if (it >= 5U) samples.push_back(ms * 1000.0F);
            }
            std::sort(samples.begin(), samples.end());
            const double bytes = static_cast<double>(n) * (kWidth * 2.0 * 2.0 + kWidth * 4.0 + kHidden * 2.0 * (combine ? 3.0 : 1.0));
            const double us = samples[samples.size() / 2U];
            // The same work as today's separate production launches: [combine], inject, HC norm.
            std::vector<float> base;
            for (unsigned it = 0; it < 25U; ++it) {
                check(hipEventRecord(a2, stream), "record");
                if (combine) strix::testing::enqueue_elementwise_prefill_rows_test(true, d.p, v.hidden_stride, r.p, v.hidden_stride, nullptr, 0U, d.p,
                    v.hidden_stride, n, kHidden, kStreams, stream);
                strix::testing::enqueue_elementwise_prefill_rows_test(false, h.p, v.hyper_stride, d.p, v.hidden_stride, static_cast<const float*>(g.p), 32U,
                    h.p, v.hyper_stride, n, kHidden, kStreams, stream);
                strix::testing::enqueue_qpf2k_inject_norm_test(c, v, true, stream);
                check(hipEventRecord(b2, stream), "record");
                check(hipEventSynchronize(b2), "sync");
                float ms = 0;
                check(hipEventElapsedTime(&ms, a2, b2), "elapsed");
                if (it >= 5U) base.push_back(ms * 1000.0F);
            }
            std::sort(base.begin(), base.end());
            std::cout << "bench rows " << n << " qpf2k inject+norm" << (combine ? "+combine" : "") << ": " << us << " us ("
                      << bytes / us / 1e3 << " GB/s of " << bytes / 1e6 << " MB) | separate production launches " << base[base.size() / 2U] << " us\n";
        }
    }
    for (unsigned output = 0; output <= 2U; ++output) {
        // The wide2 output kernel reads the transposed low values the wide2 projection writes, the WMMA output kernel
        // the FP16 low values of the WMMA projection.
        strix::testing::enqueue_wide_hyper_rows_test(c, f.views(), output == 2U ? 2U : 1U, 255U, stream);
        const float us = time(255U, output);
        std::cout << "bench rows " << n << " output " << output << ": " << us << " us (" << flop_out / us / 1e6 << " TFLOPS)\n";
    }
}
void fixtures_v3(hipStream_t stream) {
    // STRIX_QPF2K_HC output v3 (seam kernel 7 = the engine's form; 8..11 = the bench's attribution forms) == the
    // lane-chain output 1 it replaces (same inputs, every written byte), every flavor incl. non-finite scales / values,
    // partial 32-row groups; form 7 also == the reference forms.
    for (const bool bf16 : {true, false})
        for (const std::uint32_t rows : {8U, 40U, 264U, 2048U})
            for (unsigned flavor = 0; flavor < 4U; ++flavor) compare(rows, bf16, flavor, 1U, 7U, stream, 1U, 1U);
    for (const bool bf16 : {true, false})
        for (unsigned flavor = 0; flavor < 3U; ++flavor) compare(264U, bf16, flavor, 1U, 7U, stream);
    compare(8192U, true, 2U, 1U, 7U, stream, 1U, 1U);
    compare(8192U, true, 3U, 1U, 7U, stream, 1U, 1U);
    compare(8192U, false, 0U, 1U, 7U, stream, 1U, 1U);
    for (const unsigned form : {8U, 9U, 10U, 11U, 12U, 13U}) {
        for (const std::uint32_t rows : {40U, 264U})
            for (const unsigned flavor : {1U, 3U}) compare(rows, true, flavor, 1U, form, stream, 1U, 1U);
        compare(8192U, true, 0U, 1U, form, stream, 1U, 1U);
    }
    std::cout << "qpf2k HC output v3 == lane-chain output: ok\n";
}
}  // namespace

int main() try {
    hipStream_t stream = nullptr;
    check(hipStreamCreateWithFlags(&stream, hipStreamNonBlocking), "stream");
    // STRIX_HC_WIDE_V3_FIXTURES_ONLY=1: only the output v3 fixtures (then the bench if requested).
    if (std::getenv("STRIX_HC_WIDE_V3_FIXTURES_ONLY")) {
        fixtures_v3(stream);
        if (std::getenv("STRIX_HC_WIDE_BENCH")) bench(stream);
        check(hipStreamDestroy(stream), "stream");
        return 0;
    }
    if (!std::getenv("STRIX_HC_WIDE_BENCH_ONLY")) {
        for (const bool bf16 : {true, false})
            for (const std::uint32_t rows : {8U, 40U, 264U})
                for (unsigned flavor = 0; flavor < 3U; ++flavor) {
                    compare(rows, bf16, flavor, 1U, 0U, stream);
                    compare(rows, bf16, flavor, 1U, 1U, stream);
                }
        compare(2048U, true, 0U, 1U, 1U, stream);
        std::cout << "wide HC kernels == reference forms: ok\n";
        // STRIX_QPF2K_HC output (seam kernel 4 = the engine's v2 form; 3, 5, 6 = the v1 / blocks of 4 / blocks of 10 forms the bench
        // also times) == the reference forms; folded inject (+ combine) + norm == the window kernel.
        for (const bool bf16 : {true, false})
            for (const std::uint32_t rows : {8U, 40U, 264U})
                for (unsigned flavor = 0; flavor < 3U; ++flavor) compare(rows, bf16, flavor, 1U, 4U, stream);
        compare(2048U, true, 0U, 1U, 4U, stream);
        compare(8192U, true, 2U, 1U, 4U, stream);
        for (const unsigned form : {3U, 5U, 6U}) { compare(264U, true, 1U, 1U, form, stream); compare(8192U, true, 2U, 1U, form, stream); }
        // STRIX_QPF2K_HC projection (seam kernel 4 = the engine's v2 form; 3 = v1) == the WMMA projection it replaces
        // (kernel 2): low values, injection, and through the lane-chain output every output byte.
        for (const bool bf16 : {true, false})
            for (const std::uint32_t rows : {8U, 40U, 264U, 2048U})
                for (const unsigned flavor : {0U, 1U, 2U, 3U}) {
                    compare(rows, bf16, flavor, 4U, 1U, stream, 2U, 1U);
                    // The WMMA output reads the FP16 low values (low16): its bytes cover the projection's low16 stores.
                    if (flavor != 1U) compare(rows, bf16, flavor, 4U, 2U, stream, 2U, 2U);
                }
        compare(8192U, true, 0U, 4U, 1U, stream, 2U, 1U);
        compare(8192U, true, 0U, 4U, 4U, stream, 2U, 1U);
        compare(8192U, true, 3U, 3U, 1U, stream, 2U, 1U);
        for (const std::uint32_t rows : {8U, 40U, 264U, 2048U})
            for (const bool combine : {false, true})
                for (const bool in_place : {false, true})
                    for (unsigned flavor = 0; flavor < 5U; ++flavor) compare_inject_norm(rows, combine, in_place, flavor, stream);
        compare_inject_norm(8192U, true, true, 2U, stream);
        compare_inject_norm(8192U, false, true, 0U, stream);
        // Non-finite scales and inputs: the qpf2k output against the wide2 output it replaces (same chains, so NaN payloads too).
        for (const bool bf16 : {true, false}) for (const std::uint32_t rows : {40U, 264U}) compare(rows, bf16, 3U, 1U, 4U, stream, 1U, 1U);
        for (const unsigned form : {3U, 5U, 6U}) compare(264U, true, 3U, 1U, form, stream, 1U, 1U);
        // Packed coefficients vs the output kernel's own expression, every flavor of scales.
        for (unsigned flavor = 0; flavor < 4U; ++flavor) {
            std::mt19937 rng(91U + flavor);
            const auto up = q8_weights(kStreams * kHidden, kRank, flavor, rng);
            Buffer u(up.size()), pa(static_cast<std::size_t>(kRank) * kHidden * 16U), re(static_cast<std::size_t>(kRank) * kHidden * 16U);
            u.upload(up);
            check(hipMemset(pa.p, 0xa5, pa.n), "poison"); check(hipMemset(re.p, 0x5a, re.n), "poison");
            check(hipDeviceSynchronize(), "poison drain");
            strix::testing::enqueue_qpf2k_coef_test(u.p, kHidden, pa.p, re.p, stream);
            check(hipStreamSynchronize(stream), "coefficients");
            const auto a = pa.download(), b = re.download();
            std::size_t differ = 0;
            for (std::size_t i = 0; i < a.size(); ++i) differ += a[i] != b[i];
            std::cout << "qpf2k coefficients flavor " << flavor << ": " << a.size() << " bytes, differ " << differ << '\n';
            if (differ) throw std::runtime_error("qpf2k packed coefficients differ from the output kernel's expression");
        }
        fixtures_v3(stream);
        std::cout << "qpf2k HC kernels == reference forms: ok\n";
    }
    if (!std::getenv("STRIX_HC_WIDE_BENCH_ONLY")) {
        // Flavor 1 has FP16-overflowing effective weights (scale 65504); the fast kernels run on flavors 0 and 2.
        for (const bool bf16 : {true, false})
            for (const std::uint32_t rows : {8U, 40U, 264U, 2048U})
                for (const unsigned flavor : {0U, 2U})
                    for (const unsigned output : {1U, 2U}) compare_fast(rows, bf16, flavor, stream, output);
        std::cout << "fast HC kernels within the error bound: ok\n";
        for (const std::uint32_t rows : {40U, 2048U, 8192U}) compare_half(rows, true, 0U, stream);
        compare_half(264U, false, 2U, stream);
        compare_fast(8192U, true, 0U, stream, 1U);
        std::cout << "FP16-input HC projection == FP32-input form: ok\n";
    }
    if (std::getenv("STRIX_HC_WIDE_BENCH") || std::getenv("STRIX_HC_WIDE_BENCH_ONLY")) bench(stream);
    check(hipStreamDestroy(stream), "stream");
    return 0;
} catch (const std::exception& e) {
    std::cerr << "wide HC fixture failed: " << e.what() << '\n';
    return 1;
}
