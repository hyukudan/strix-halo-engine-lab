// GSQ-RCO dense A8 typed sites as the engine compiles and launches them (strix_hip_runtime, its own flags; the
// launchers' diagnostic probes build the real QsaInputRowsSite / QsaOutputRowsSite / GdnInputRowsSite /
// GdnOutputRowsSite / SharedGateUpRowsSite / SharedDownRowsSite (+ the R8/M640 narrow form) and launch them with a8):
// (1) the group decode (gsq_a8::dense_group) of every type == llama.cpp's dequantization, bit for bit;
// (2) every row of an 8-row set, alone (T = 1) and in every T-row window that contains it (T = 2..8 for GDN,
//     2/4/8 for QSA, every start position), gives the same bits, and those bits == the host contract: reference
//     A8 quantizer, ascending block chain acc = fma(x, y, acc) of the documented entries (one scale: (s0 * delta,
//     float(I)); halves: ((s0 * delta) * float(I0) + (s1 * delta) * float(I1), 1); min: ((s0 * delta) * float(I) -
//     (s1 * delta) * float(S), 1)), each operation rounded once, then the site's store rounding (BF16).
// (3) QSA input, 16 / 32 distinct rows of one typed kind through the cohort window dispatcher (8-row typed A8 sites)
//     == each row alone. The process sets STRIX_GSQ_Q20_A8=1 itself (the dispatcher admits typed kinds only then).
// Fixtures: random blocks (some FP16 scale headers zero or subnormal), standard normal activations with exact zeros, an
// all-zero and a tiny block, a block with +Inf (row 2) and one with NaN (row 5); the same weights and activations run
// every composition of a site (QSA: the R1 five-segment launch and the windows' launches grouped by type).
// usage: strix-hip-gsq-typed-sites-tests [--report]   (--report: run every case and print every count)
// exit: 0 PASS, 1 a word differs (alone vs batched or vs the host contract), 3 an error (exception)
#include "strix/hip_gdn_launcher.hpp"
#include "strix/hip_gsq_a8_groups.hpp"
#include "strix/hip_indexed_attention_launcher.hpp"
#include "strix/hip_shared_expert_launcher.hpp"
#include "strix/ggml_dequant.hpp"

#include <hip/hip_runtime.h>
#include <algorithm>
#include <array>
#include <bit>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <iostream>
#include <limits>
#include <memory>
#include <random>
#include <stdexcept>
#include <string>
#include <vector>

#pragma clang fp contract(off)

namespace {
using namespace strix;
void require(bool value, const std::string& message) { if (!value) throw std::runtime_error(message); }
void check(hipError_t error, const char* operation) {
    if (error != hipSuccess) throw std::runtime_error(std::string(operation) + ": " + hipGetErrorString(error));
}
constexpr std::uint32_t kNan = 0x7fc00000U;
const std::array<std::uint32_t, 10> kTypes{2U, 6U, 8U, 11U, 12U, 13U, 14U, 20U, 23U, 42U};

struct Exported { int v[32]; float s0, s1; };
template <std::uint32_t T>
__global__ void decode_groups(const unsigned char* weights, std::uint32_t width, std::uint32_t rows, Exported* out) {
    const std::uint32_t i = blockIdx.x * blockDim.x + threadIdx.x, groups = width / 32U;
    if (i >= rows * groups) return;
    const auto x = gsq_a8::dense_group<T>(weights + static_cast<std::uint64_t>(i / groups) * gsq_a8::dense_row_bytes(T, width), i % groups);
    Exported e{};
    for (unsigned k = 0; k < 8U; ++k)
        for (unsigned j = 0; j < 4U; ++j) e.v[4U * k + j] = static_cast<std::int8_t>(static_cast<std::uint32_t>(x.w[k]) >> (8U * j));
    e.s0 = x.s0; e.s1 = x.s1;
    out[i] = e;
}
template <class F> void dispatch(std::uint32_t type, F&& f) {
    switch (type) {
    case 2U: f.template operator()<2U>(); break;
    case 6U: f.template operator()<6U>(); break;
    case 8U: f.template operator()<8U>(); break;
    case 11U: f.template operator()<11U>(); break;
    case 12U: f.template operator()<12U>(); break;
    case 13U: f.template operator()<13U>(); break;
    case 14U: f.template operator()<14U>(); break;
    case 20U: f.template operator()<20U>(); break;
    case 23U: f.template operator()<23U>(); break;
    case 42U: f.template operator()<42U>(); break;
    default: throw std::invalid_argument("type");
    }
}
gsq_a8::DenseKind kind_of(std::uint32_t type) {
    gsq_a8::DenseKind k{};
    dispatch(type, [&]<std::uint32_t T>() { k = gsq_a8::dense_kind<T>(); });
    return k;
}
std::string type_name(std::uint32_t type) { return std::string(ggml_dequant::name(type)); }

template <class T> struct DeviceBuffer {
    T* p = nullptr; std::size_t n = 0;
    explicit DeviceBuffer(std::size_t count) : n(count) { check(hipMalloc(&p, std::max<std::size_t>(count, 1) * sizeof(T)), "malloc"); }
    ~DeviceBuffer() { if (p) (void)hipFree(p); }
    DeviceBuffer(const DeviceBuffer&) = delete; DeviceBuffer& operator=(const DeviceBuffer&) = delete;
    void upload(const T* src) { check(hipMemcpy(p, src, n * sizeof(T), hipMemcpyHostToDevice), "upload"); }
    std::vector<T> download() const { std::vector<T> out(n); check(hipMemcpy(out.data(), p, n * sizeof(T), hipMemcpyDeviceToHost), "download"); return out; }
    std::uintptr_t address(std::size_t at = 0) const { return reinterpret_cast<std::uintptr_t>(p + at); }
};

// Random rows of a type with finite, moderate FP16 header fields (d, dmin: exponent 1..max_exponent).
std::vector<std::uint8_t> random_rows(std::uint32_t type, std::uint32_t width, std::uint32_t rows, std::mt19937& rng, int max_exponent) {
    std::vector<std::uint8_t> w(gsq_a8::dense_row_bytes(type, width) * rows);
    for (auto& b : w) b = static_cast<std::uint8_t>(rng());
    const std::uint32_t bb = gsq_a8::dense_block_bytes(type);
    const std::vector<std::uint32_t> fields = type == 11U ? std::vector<std::uint32_t>{108} : type == 14U ? std::vector<std::uint32_t>{208} :
        type == 12U || type == 13U ? std::vector<std::uint32_t>{0, 2} : std::vector<std::uint32_t>{0};
    std::uniform_int_distribution<int> e(1, max_exponent), m(0, 1023), s(0, 1), special(0, 63);
    for (std::size_t b = 0; b < w.size() / bb; ++b)
        for (auto f : fields) {
            // about 1 in 64 headers zero, 1 in 64 subnormal
            const int pick = special(rng), exponent = pick == 0 || pick == 1 ? 0 : e(rng), mantissa = pick == 0 ? 0 : m(rng);
            const auto h = static_cast<std::uint16_t>((s(rng) << 15) | (exponent << 10) | mantissa);
            w[b * bb + f] = static_cast<std::uint8_t>(h & 255U); w[b * bb + f + 1U] = static_cast<std::uint8_t>(h >> 8U);
        }
    return w;
}

// Device decode of every group, checked bit for bit against llama.cpp's dequantization of the same rows.
std::vector<Exported> decode_checked(std::uint32_t type, const std::vector<std::uint8_t>& w, std::uint32_t width, std::uint32_t rows) {
    DeviceBuffer<std::uint8_t> dw(w.size()); dw.upload(w.data());
    const std::uint32_t groups = width / 32U;
    DeviceBuffer<Exported> out(static_cast<std::size_t>(rows) * groups);
    dispatch(type, [&]<std::uint32_t T>() {
        hipLaunchKernelGGL(decode_groups<T>, dim3((rows * groups + 255U) / 256U), dim3(256), 0, nullptr, dw.p, width, rows, out.p);
    });
    check(hipGetLastError(), "decode launch");
    const auto ex = out.download();
    const auto kind = kind_of(type);
    const auto rb = gsq_a8::dense_row_bytes(type, width);
    std::vector<float> deq(width);
    std::size_t bad = 0;
    for (std::uint32_t r = 0; r < rows; ++r) {
        ggml_dequant::dequantize_row(type, std::span<const std::byte>(reinterpret_cast<const std::byte*>(w.data()) + r * rb, rb), deq);
        for (std::uint32_t g = 0; g < groups; ++g) {
            const auto& e = ex[r * groups + g];
            for (unsigned j = 0; j < 32U; ++j) {
                const float x = kind == gsq_a8::DenseKind::min ? e.s0 * static_cast<float>(e.v[j]) - e.s1 :
                    static_cast<float>(e.v[j]) * (kind == gsq_a8::DenseKind::halves && j >= 16U ? e.s1 : e.s0);
                bad += std::memcmp(&x, &deq[g * 32U + j], 4) != 0;
            }
        }
    }
    require(bad == 0, "dense GSQ group decode of " + type_name(type) + " differs from llama.cpp dequantization (" + std::to_string(bad) + ")");
    return ex;
}

float bf16_value(std::uint16_t bits) { return std::bit_cast<float>(static_cast<std::uint32_t>(bits) << 16U); }
std::uint16_t to_bf16(float value) {  // the sites' float_to_bf16 (round half to even on the bit pattern)
    const auto bits = std::bit_cast<std::uint32_t>(value);
    return static_cast<std::uint16_t>((bits + 0x7fffU + ((bits >> 16U) & 1U)) >> 16U);
}

struct Quantized { std::vector<std::int8_t> codes; std::vector<float> deltas; };
// The dense A8 quantizer (dense_a8_quantize8) on the host: per 32-feature block delta = amax / 127 (one rounding),
// codes rint(x / delta) clamped to +-127; a non-finite block -> NaN delta, zero codes; zero or tiny -> 1, zeros.
Quantized quantize_reference(const std::vector<float>& x, std::uint32_t rows, std::uint32_t width) {
    const std::uint32_t blocks = width / 32U;
    Quantized q{std::vector<std::int8_t>(static_cast<std::size_t>(rows) * width), std::vector<float>(static_cast<std::size_t>(rows) * blocks)};
    for (std::uint32_t r = 0; r < rows; ++r)
        for (std::uint32_t b = 0; b < blocks; ++b) {
            const float* v = &x[static_cast<std::size_t>(r) * width + 32U * b];
            bool bad = false; float amax = 0.0F;
            for (unsigned k = 0; k < 32U; ++k) { bad |= !std::isfinite(v[k]); amax = std::max(amax, std::fabs(v[k])); }
            const bool tiny = amax > 0.0F && amax < 0x1p-119F;
            const float delta = bad ? std::bit_cast<float>(kNan) : amax == 0.0F || tiny ? 1.0F : amax / 127.0F;
            for (unsigned k = 0; k < 32U; ++k) {
                const float rounded = bad || tiny ? 0.0F : std::nearbyint(v[k] / delta);
                q.codes[static_cast<std::size_t>(r) * width + 32U * b + k] = static_cast<std::int8_t>(static_cast<int>(std::fmin(127.0F, std::fmax(-127.0F, rounded))));
            }
            q.deltas[static_cast<std::size_t>(r) * blocks + b] = delta;
        }
    return q;
}
// The host contract of one output (unrounded FP32 chain).
float reference_output(gsq_a8::DenseKind kind, const Exported* row_groups, const Quantized& q, std::uint32_t width, std::uint32_t r) {
    const std::uint32_t blocks = width / 32U;
    float acc = 0.0F;
    for (std::uint32_t b = 0; b < blocks; ++b) {
        const auto& e = row_groups[b];
        const std::int8_t* c = &q.codes[static_cast<std::size_t>(r) * width + 32U * b];
        const float delta = q.deltas[static_cast<std::size_t>(r) * blocks + b];
        long long i0 = 0, i1 = 0, sum = 0;
        for (unsigned j = 0; j < 32U; ++j) { (j < 16U ? i0 : i1) += static_cast<long long>(e.v[j]) * c[j]; sum += c[j]; }
        float x, y;
        if (kind == gsq_a8::DenseKind::one) { x = e.s0 * delta; y = static_cast<float>(i0 + i1); }
        else if (kind == gsq_a8::DenseKind::halves) {
            const float c0 = e.s0 * delta, c1 = e.s1 * delta;
            const float u0 = c0 * static_cast<float>(i0), u1 = c1 * static_cast<float>(i1);
            x = u0 + u1; y = 1.0F;
        } else {
            const float c0 = e.s0 * delta, c1 = e.s1 * delta;
            const float u0 = c0 * static_cast<float>(i0 + i1), u1 = c1 * static_cast<float>(sum);
            x = u0 - u1; y = 1.0F;
        }
        acc = std::fma(x, y, acc);
    }
    return std::isnan(acc) ? std::bit_cast<float>(kNan) : acc;
}

enum class SiteKind { qsa_input, qsa_output, gdn_input, gdn_output, shared_gate_up, shared_down };
const char* site_name(SiteKind k) {
    switch (k) {
    case SiteKind::qsa_input: return "QSA input";
    case SiteKind::qsa_output: return "QSA output";
    case SiteKind::gdn_input: return "GDN input";
    case SiteKind::gdn_output: return "GDN output";
    case SiteKind::shared_gate_up: return "shared gate/up";
    default: return "shared down";
    }
}
// BF16 activations in (QSA/GDN inputs: BF16-rounded FP32 out; shared gate/up: FP32 out); otherwise FP32 in, BF16 out.
bool input_side(SiteKind k) { return k == SiteKind::qsa_input || k == SiteKind::gdn_input || k == SiteKind::shared_gate_up; }
bool shared(SiteKind k) { return k == SiteKind::shared_gate_up || k == SiteKind::shared_down; }

struct Totals { std::size_t words = 0, row_vs_alone = 0, vs_reference = 0; };
struct Mismatch : std::runtime_error { using std::runtime_error::runtime_error; };

// One site: segments (types, counts) over 8 (cohort: 32) activation rows of `width`, one fixture. compositions: the ways the engine
// launches these segments (lists of segment index lists: all segments at once, or one launch per type); the T = 1
// reference ("alone") uses the first. Every composition runs at every launch shape (rows T, every first row) the site
// supports; every row's output words are compared with its alone words and the host contract. narrow: shared down also
// runs T = 8 as the R8/M640 narrow organization. cohort: QSA input over 32 distinct rows, also as 16- and 32-row cohorts.
Totals run_case(SiteKind kind, const std::vector<std::uint32_t>& types, const std::vector<std::uint32_t>& counts,
    const std::vector<std::vector<std::vector<unsigned>>>& compositions, std::uint32_t width, std::mt19937& rng, hipStream_t stream,
    bool report, const std::string& label, bool narrow = false, bool cohort = false) {
    const std::uint32_t total = cohort ? 32U : 8U;  // activation rows (a cohort case: 32 distinct rows)
    const auto n = static_cast<unsigned>(types.size());
    std::vector<std::vector<Exported>> ex(n);
    std::vector<std::unique_ptr<DeviceBuffer<std::uint8_t>>> weights;
    std::vector<std::uint32_t> first{0};
    for (unsigned m = 0; m < n; ++m) {
        const auto host = random_rows(types[m], width, counts[m], rng, 12);
        ex[m] = decode_checked(types[m], host, width, counts[m]);
        weights.push_back(std::make_unique<DeviceBuffer<std::uint8_t>>(host.size()));
        weights.back()->upload(host.data());
        first.push_back(first.back() + counts[m]);
    }
    const std::uint32_t outputs = first.back();
    // Activations: BF16 rows (input sites) or FP32 rows (output sites), standard normal with a few exact zeros, one
    // all-zero block in row 3, one block of a tiny value in row 6, +Inf in a block of row 2 and NaN in a block of row 5.
    std::normal_distribution<float> normal(0.0F, 1.0F);
    std::vector<float> x(static_cast<std::size_t>(total) * width);
    std::vector<std::uint16_t> xb(x.size());
    for (std::size_t i = 0; i < x.size(); ++i) {
        float v = normal(rng);
        if (i % 97U == 0U) v = 0.0F;
        if (i / width == 3U && i % width >= 64U && i % width < 96U) v = 0.0F;
        if (i / width == 6U && i % width >= 128U && i % width < 160U) v = (i % 2U ? 1.0F : -1.0F) * 0x1p-125F;
        if (i == 2U * width + 165U) v = std::numeric_limits<float>::infinity();
        if (i == 5U * width + 230U) v = std::numeric_limits<float>::quiet_NaN();
        xb[i] = to_bf16(v);
        x[i] = input_side(kind) ? bf16_value(xb[i]) : v;
    }
    DeviceBuffer<float> in_f(x.size()); in_f.upload(x.data());
    std::vector<float> gates(total);
    for (auto& g : gates) g = normal(rng);
    DeviceBuffer<float> gate_d(64U * total);  // one 16-byte aligned FP32 gate per row
    {
        std::vector<float> padded(64U * total, 0.0F);
        for (unsigned r = 0; r < total; ++r) padded[64U * r] = gates[r];
        gate_d.upload(padded.data());
    }
    DeviceBuffer<std::uint16_t> in_b(xb.size()); in_b.upload(xb.data());
    const auto q = quantize_reference(x, total, width);
    // expected[row * outputs + global output]: the stored value's bits (input sites: BF16-rounded FP32; output sites: BF16).
    std::vector<std::uint32_t> expected(static_cast<std::size_t>(total) * outputs);
    for (unsigned m = 0; m < n; ++m) {
        const auto k = kind_of(types[m]);
        for (std::uint32_t r = 0; r < total; ++r)
            for (std::uint32_t o = 0; o < counts[m]; ++o) {
                const float v = reference_output(k, &ex[m][static_cast<std::size_t>(o) * (width / 32U)], q, width, r);
                expected[static_cast<std::size_t>(r) * outputs + first[m] + o] = kind == SiteKind::shared_gate_up ? std::bit_cast<std::uint32_t>(v) :
                    input_side(kind) ? std::bit_cast<std::uint32_t>(bf16_value(to_bf16(v))) : shared(kind) ? to_bf16(gates[r] * v) : to_bf16(v);
            }
    }
    // One launch shape of one composition: got[row * outputs + global output] for rows [base, base + T).
    const auto launch = [&](const std::vector<std::vector<unsigned>>& groups, unsigned rows_t, unsigned base, std::vector<std::uint32_t>& got,
                            bool narrow_t) {
        for (const auto& g : groups) {
            std::vector<std::uint32_t> gt, gc;
            std::uint32_t g_out = 0;
            for (const auto m : g) { gt.push_back(types[m]); gc.push_back(counts[m]); g_out += counts[m]; }
            DeviceBuffer<float> out_f(input_side(kind) ? static_cast<std::size_t>(rows_t) * g_out : 1U);
            DeviceBuffer<std::uint16_t> out_b(input_side(kind) ? 1U : static_cast<std::size_t>(rows_t) * g_out);
            // sentinel: a word the launch does not write stays 0xffff... and mismatches. On the probes' stream: it is
            // non-blocking, so a legacy-stream hipMemset would not be ordered before the launch.
            check(hipMemsetAsync(out_f.p, 0xff, out_f.n * sizeof(float), stream), "sentinel");
            check(hipMemsetAsync(out_b.p, 0xff, out_b.n * sizeof(std::uint16_t), stream), "sentinel");
            const auto elem = input_side(kind) ? sizeof(float) : sizeof(std::uint16_t);
            const auto in_bytes = static_cast<std::uint64_t>(width) * (input_side(kind) ? 2U : 4U);
            if (shared(kind)) {
                std::vector<HipSharedQ8ProjectionView> w, in, gate, out;
                for (const auto m : g) w.push_back({weights[m]->address(), weights[m]->n});
                for (unsigned r = 0; r < rows_t; ++r) {
                    in.push_back({input_side(kind) ? in_b.address(static_cast<std::size_t>(base + r) * width) : in_f.address(static_cast<std::size_t>(base + r) * width), in_bytes});
                    if (!input_side(kind)) gate.push_back({gate_d.address(64U * (base + r)), 4U});
                    out.push_back({input_side(kind) ? out_f.address(static_cast<std::size_t>(r) * g_out) : out_b.address(static_cast<std::size_t>(r) * g_out), g_out * elem});
                }
                launch_hip_shared_gsq_rows_probe(input_side(kind) ? HipSharedQ8ProjectionRole::gate_up : HipSharedQ8ProjectionRole::down,
                    narrow_t, gt, w, gc, in, gate, out, width, stream);
            } else if (kind == SiteKind::qsa_input || kind == SiteKind::qsa_output) {
                std::vector<HipQsaInputProjectionView> w, in, out;
                for (const auto m : g) w.push_back({weights[m]->address(), weights[m]->n});
                for (unsigned r = 0; r < rows_t; ++r)
                    in.push_back({input_side(kind) ? in_b.address(static_cast<std::size_t>(base + r) * width) : in_f.address(static_cast<std::size_t>(base + r) * width), in_bytes});
                for (std::uint32_t i = 0, at = 0; i < g.size(); at += gc[i], ++i)
                    for (unsigned r = 0; r < rows_t; ++r) {
                        const auto offset = static_cast<std::size_t>(r) * g_out + at;
                        out.push_back({input_side(kind) ? out_f.address(offset) : out_b.address(offset), gc[i] * elem});
                    }
                launch_hip_qsa_gsq_rows_probe(input_side(kind) ? HipQsaRowsProjection::input : HipQsaRowsProjection::output,
                    gt, w, gc, in, out, width, stream);
            } else {
                std::vector<GdnKnownWindowProjectionView> w, in, out;
                for (const auto m : g) w.push_back({weights[m]->address(), weights[m]->n});
                for (unsigned r = 0; r < rows_t; ++r)
                    in.push_back({input_side(kind) ? in_b.address(static_cast<std::size_t>(base + r) * width) : in_f.address(static_cast<std::size_t>(base + r) * width), in_bytes});
                for (std::uint32_t i = 0, at = 0; i < g.size(); at += gc[i], ++i)
                    for (unsigned r = 0; r < rows_t; ++r) {
                        const auto offset = static_cast<std::size_t>(r) * g_out + at;
                        out.push_back({input_side(kind) ? out_f.address(offset) : out_b.address(offset), gc[i] * elem});
                    }
                gdn_gsq_rows_projection_probe(input_side(kind) ? GdnKnownWindowProjectionDirection::bf16_to_fp32 :
                    GdnKnownWindowProjectionDirection::fp32_to_bf16, gt, w, gc, in, out, width, stream);
            }
            check(hipStreamSynchronize(stream), "probe sync");
            std::vector<std::uint32_t> words(static_cast<std::size_t>(rows_t) * g_out);
            if (got.size() < static_cast<std::size_t>(base + rows_t) * outputs) throw std::logic_error("result buffer");
            if (input_side(kind)) { const auto v = out_f.download(); std::memcpy(words.data(), v.data(), words.size() * 4U); }
            else { const auto v = out_b.download(); for (std::size_t i = 0; i < v.size(); ++i) words[i] = v[i]; }
            for (unsigned r = 0; r < rows_t; ++r) {
                std::uint32_t at = 0;
                for (const auto m : g) {
                    std::memcpy(&got[static_cast<std::size_t>(base + r) * outputs + first[m]], &words[static_cast<std::size_t>(r) * g_out + at],
                        counts[m] * 4U);
                    at += counts[m];
                }
            }
        }
    };
    std::vector<std::uint32_t> alone(static_cast<std::size_t>(total) * outputs);
    for (unsigned r = 0; r < total; ++r) launch(compositions[0], 1U, r, alone, false);
    Totals t;
    std::string line = std::string(site_name(kind)) + " " + label + ":";
    // rows [base, base + T) of got against alone / expected
    const auto compare = [&](unsigned rows_t, unsigned base, const std::vector<std::uint32_t>& got) {
        std::size_t vs_alone = 0, vs_ref = 0;
        for (unsigned r = base; r < base + rows_t; ++r)
            for (std::uint32_t o = 0; o < outputs; ++o) {
                const auto i = static_cast<std::size_t>(r) * outputs + o, j = i;
                vs_alone += got[i] != alone[j];
                vs_ref += got[i] != expected[j];
                ++t.words;
            }
        t.row_vs_alone += vs_alone; t.vs_reference += vs_ref;
        return std::pair{vs_alone, vs_ref};
    };
    line += " T1 vs host " + std::to_string(compare(total, 0U, alone).second);
    const std::vector<unsigned> shapes = kind == SiteKind::gdn_input || kind == SiteKind::gdn_output ?
        std::vector<unsigned>{1U, 2U, 3U, 4U, 5U, 6U, 7U, 8U} : std::vector<unsigned>{1U, 2U, 4U, 8U};
    for (std::size_t c = 0; c < compositions.size(); ++c)
        for (const auto rows_t : shapes)
            for (const bool narrow_t : {false, true}) {
                if (narrow_t && !(narrow && rows_t == 8U)) continue;
                if (c == 0U && rows_t == 1U) continue;  // the alone reference itself
                std::size_t a_sum = 0, e_sum = 0;
                for (unsigned base = 0; base + rows_t <= total; ++base) {
                    std::vector<std::uint32_t> got(static_cast<std::size_t>(total) * outputs);
                    launch(compositions[c], rows_t, base, got, narrow_t);
                    const auto [a, e] = compare(rows_t, base, got);
                    a_sum += a; e_sum += e;
                }
                line += " | " + (compositions.size() > 1U ? "c" + std::to_string(c) + " " : std::string()) + "T" + std::to_string(rows_t) +
                    (narrow_t ? " narrow" : "") + " " + std::to_string(a_sum) + "/" + std::to_string(e_sum);
            }
    if (cohort)
        for (const unsigned rows_t : {16U, 32U}) {
            std::size_t a_sum = 0, e_sum = 0;
            for (unsigned base = 0; base + rows_t <= total; base += rows_t) {
                std::vector<std::uint32_t> got(static_cast<std::size_t>(total) * outputs);
                launch(compositions[0], rows_t, base, got, false);
                const auto [a, e] = compare(rows_t, base, got);
                a_sum += a; e_sum += e;
            }
            line += " | cohort T" + std::to_string(rows_t) + " " + std::to_string(a_sum) + "/" + std::to_string(e_sum);
        }
    line += "   (mismatching words vs alone / vs host)";
    std::cout << line << std::endl;
    if (!report && (t.row_vs_alone || t.vs_reference)) throw Mismatch(line);
    return t;
}
}  // namespace

int main(int argc, char** argv) try {
    try {
    // The engine's GSQ dense gate (q8_rows::dense_a8_type) reads STRIX_GSQ_Q20_A8 once; the cohort dispatcher needs it.
    if (setenv("STRIX_GSQ_Q20_A8", "1", 1) != 0) throw std::runtime_error("setenv STRIX_GSQ_Q20_A8");
    const bool report = argc > 1 && std::string(argv[1]) == "--report";
    hipStream_t stream = nullptr;
    check(hipStreamCreateWithFlags(&stream, hipStreamNonBlocking), "stream");
    std::mt19937 rng(20261025U);
    Totals all;
    const auto add = [&](const Totals& t) { all.words += t.words; all.row_vs_alone += t.row_vs_alone; all.vs_reference += t.vs_reference; };
    // Engine shapes of Qwen3.8-Flash-Next (hidden 2560, attention / value width 6144), output counts trimmed (odd
    // tails kept: the kernels' output groups and tails are covered; the arithmetic per output is shape-independent).
    for (const auto t : kTypes) {
        const auto name = type_name(t);
        const auto be = gsq_a8::dense_block_elements(t);
        // QSA input: the R1 site holds index_q, index_k, q, k, v in one launch; windows group the matrices by type.
        add(run_case(SiteKind::qsa_input, {8U, 8U, t, 8U, 8U}, {128U, 64U, 389U, 64U, 64U},
            {{{0U, 1U, 2U, 3U, 4U}}, {{0U, 1U, 3U, 4U}, {2U}}}, 2560U, rng, stream, report, "q " + name + " + Q8 (c0 R1 / c1 grouped)"));
        add(run_case(SiteKind::qsa_input, {t, t, t, t, t}, {128U, 64U, 389U, 64U, 64U}, {{{0U, 1U, 2U, 3U, 4U}}}, 2560U, rng, stream,
            report, "all " + name));
        if (t != 8U) add(run_case(SiteKind::qsa_input, {t}, {389U}, {{{0U}}}, 2560U, rng, stream, report, "q " + name + " (cohorts)", false, true));
        add(run_case(SiteKind::qsa_output, {t}, {389U}, {{{0U}}}, 6144U, rng, stream, report, name));
        // The engine's full output counts where the bisect points (typed QSA q 12288 + k/v 512, GDN qkv 10240 + z 6144):
        // BF16 stores hide most one-ulp FP32 differences, so these cases carry the most words.
        add(run_case(SiteKind::qsa_input, {8U, 8U, t, 8U, 8U}, {128U, 64U, 12288U, 512U, 512U},
            {{{0U, 1U, 2U, 3U, 4U}}, {{0U, 1U, 3U, 4U}, {2U}}}, 2560U, rng, stream, report, "q " + name + " + Q8, engine counts (c0 R1 / c1 grouped)"));
        add(run_case(SiteKind::gdn_input, {t, t}, {10240U, 6144U}, {{{0U, 1U}}}, 2560U, rng, stream, report, "qkv/z " + name + ", engine counts"));
        // GDN input: qkv + z (+ Q8 beta/alpha segments when the gates are Q8) in one launch.
        add(run_case(SiteKind::gdn_input, {t, t}, {389U, 197U}, {{{0U, 1U}}}, 2560U, rng, stream, report, "qkv/z " + name));
        add(run_case(SiteKind::gdn_input, {t, t, 8U, 8U}, {389U, 197U, 48U, 48U}, {{{0U, 1U, 2U, 3U}}}, 2560U, rng, stream, report,
            "qkv/z " + name + " + Q8 gates"));
        add(run_case(SiteKind::gdn_output, {t}, {389U}, {{{0U}}}, 6144U, rng, stream, report, name));
        // Shared expert: gate + up (intermediate 640) in one launch, down 640 -> hidden (+ the R8/M640 narrow organization).
        // 256-weight block types cannot have 640-wide rows: their down runs 1280 wide (no narrow form).
        add(run_case(SiteKind::shared_gate_up, {t, t}, {640U, 640U}, {{{0U, 1U}}}, 2560U, rng, stream, report, name));
        if (640U % be == 0U) add(run_case(SiteKind::shared_down, {t}, {389U}, {{{0U}}}, 640U, rng, stream, report, name + " M640", true));
        else add(run_case(SiteKind::shared_down, {t}, {389U}, {{{0U}}}, 1280U, rng, stream, report, name + " M1280"));
    }
    // Mixed types seen in the GSQ files (VAR-IQ3: Q6_K qkv + Q5_K z; QSA Q5_K q, Q6_K k, Q5_K v).
    add(run_case(SiteKind::gdn_input, {14U, 13U}, {389U, 197U}, {{{0U, 1U}}}, 2560U, rng, stream, report, "qkv Q6_K + z Q5_K"));
    add(run_case(SiteKind::qsa_input, {8U, 8U, 13U, 14U, 13U}, {128U, 64U, 389U, 64U, 64U},
        {{{0U, 1U, 2U, 3U, 4U}}, {{0U, 1U}, {2U, 4U}, {3U}}}, 2560U, rng, stream, report, "q Q5_K k Q6_K v Q5_K (c0 R1 / c1 grouped)"));
    check(hipStreamDestroy(stream), "stream destroy");
    std::cout << "GSQ typed sites: " << all.words << " words, row vs alone " << all.row_vs_alone << ", vs host " << all.vs_reference << std::endl;
    if (all.row_vs_alone || all.vs_reference) return 1;
    std::cout << "GSQ typed sites PASS" << std::endl;
    return 0;
    } catch (const Mismatch& e) {
        std::cerr << "MISMATCH " << e.what() << std::endl;
        return 1;
    }
} catch (const std::exception& e) {
    std::cerr << e.what() << std::endl;
    return 3;
}
