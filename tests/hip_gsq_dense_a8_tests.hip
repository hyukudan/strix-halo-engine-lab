// GSQ-RCO dense A8 (q8_rows typed segments, project_rows_dense_a8_gsq_kernel).
// (1) Device group decode (gsq_a8::dense_group) of every type == llama.cpp's dequantization (strix::ggml_dequant):
//     one scale: float(v) * s == x, halves: float(v) * s0|s1 == x, min: s0 * float(v) - s1 == x, bit for bit.
// (2) Engine launch (quantizer + GSQ projection) through a three-segment site (two GSQ types + Q8_0) at 1/2/4/8
//     rows == the serial contract (reference quantizer, ascending chain of the documented entries), bit for bit,
//     and every row gets the same bits from every row count.
#include "../src/hip_q8_rows_gemv.hpp"
#include "strix/ggml_dequant.hpp"

#include <hip/hip_runtime.h>
#include <algorithm>
#include <array>
#include <bit>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <iostream>
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

template <class T> struct DeviceBuffer {
    T* p = nullptr; std::size_t n = 0;
    explicit DeviceBuffer(std::size_t count) : n(count) { check(hipMalloc(&p, std::max<std::size_t>(count, 1) * sizeof(T)), "malloc"); }
    ~DeviceBuffer() { if (p) (void)hipFree(p); }
    DeviceBuffer(const DeviceBuffer&) = delete; DeviceBuffer& operator=(const DeviceBuffer&) = delete;
    void upload(const T* src) { check(hipMemcpy(p, src, n * sizeof(T), hipMemcpyHostToDevice), "upload"); }
    std::vector<T> download() const { std::vector<T> out(n); check(hipMemcpy(out.data(), p, n * sizeof(T), hipMemcpyDeviceToHost), "download"); return out; }
};

std::vector<std::uint8_t> random_rows(std::uint32_t type, std::uint32_t width, std::uint32_t rows, std::mt19937& rng, int max_exponent = 20) {
    const auto row_bytes = gsq_a8::dense_row_bytes(type, width);
    std::vector<std::uint8_t> w(row_bytes * rows);
    for (auto& b : w) b = static_cast<std::uint8_t>(rng());
    const std::uint32_t bb = gsq_a8::dense_block_bytes(type);
    // fp16 header fields (d, dmin) finite and moderate: exponent 1..20.
    std::vector<std::uint32_t> fields = type == 11U ? std::vector<std::uint32_t>{108} : type == 14U ? std::vector<std::uint32_t>{208} :
        type == 12U || type == 13U ? std::vector<std::uint32_t>{0, 2} : std::vector<std::uint32_t>{0};
    std::uniform_int_distribution<int> e(1, max_exponent), m(0, 1023), s(0, 1);
    for (std::size_t b = 0; b < w.size() / bb; ++b)
        for (auto f : fields) {
            const auto h = static_cast<std::uint16_t>((s(rng) << 15) | (e(rng) << 10) | m(rng));
            w[b * bb + f] = static_cast<std::uint8_t>(h & 255U); w[b * bb + f + 1U] = static_cast<std::uint8_t>(h >> 8U);
        }
    return w;
}

std::vector<Exported> decode(std::uint32_t type, const std::vector<std::uint8_t>& w, std::uint32_t width, std::uint32_t rows) {
    DeviceBuffer<std::uint8_t> dw(w.size()); dw.upload(w.data());
    const std::uint32_t groups = width / 32U;
    DeviceBuffer<Exported> out(static_cast<std::size_t>(rows) * groups);
    dispatch(type, [&]<std::uint32_t T>() {
        hipLaunchKernelGGL(decode_groups<T>, dim3((rows * groups + 255U) / 256U), dim3(256), 0, nullptr, dw.p, width, rows, out.p);
    });
    check(hipGetLastError(), "decode launch");
    return out.download();
}

// (1) decode == llama.cpp dequantization
void decode_test(std::uint32_t type, std::mt19937& rng) {
    const std::uint32_t width = 2560U, rows = 64U;
    const auto w = random_rows(type, width, rows, rng);
    const auto ex = decode(type, w, width, rows);
    const auto kind = kind_of(type);
    std::vector<float> deq(width);
    long long bad = 0;
    for (std::uint32_t r = 0; r < rows; ++r) {
        const auto rb = gsq_a8::dense_row_bytes(type, width);
        ggml_dequant::dequantize_row(type, std::span<const std::byte>(reinterpret_cast<const std::byte*>(w.data()) + r * rb, rb), deq);
        for (std::uint32_t g = 0; g < width / 32U; ++g) {
            const auto& e = ex[r * (width / 32U) + g];
            for (unsigned j = 0; j < 32U; ++j) {
                float x;
                if (kind == gsq_a8::DenseKind::min) { const float p = e.s0 * static_cast<float>(e.v[j]); x = p - e.s1; }
                else x = static_cast<float>(e.v[j]) * (kind == gsq_a8::DenseKind::halves && j >= 16U ? e.s1 : e.s0);
                bad += std::memcmp(&x, &deq[g * 32U + j], 4) != 0;
            }
        }
    }
    std::cout << "decode " << ggml_dequant::name(type) << " mismatches " << bad << std::endl;
    require(bad == 0, "dense GSQ group decode differs from llama.cpp dequantization");
}

template <unsigned Rows> struct Site {
    using Input = float;
    static constexpr unsigned rows = Rows, a8_family = 3U;
    const void* input[Rows];
    q8_rows::Segments segments;
    std::uint32_t width, outputs;
    float* output;
    __device__ static float coefficient(unsigned, std::int8_t) { return 0.0F; }
    __device__ static void fma(float&, float, float) {}
    __device__ static float add(float lower, float upper) { return lower + upper; }
    __device__ void store(std::uint32_t out, unsigned r, float value) const { output[static_cast<std::uint64_t>(r) * outputs + out] = value; }
};

struct Quantized { std::vector<std::int8_t> codes; std::vector<float> deltas; };
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
float reference_output(std::uint32_t type, const std::vector<Exported>& ex, const Quantized& q, std::uint32_t width, std::uint32_t r,
    std::uint32_t row) {
    const std::uint32_t blocks = width / 32U;
    const auto kind = kind_of(type);
    float acc = 0.0F;
    for (std::uint32_t b = 0; b < blocks; ++b) {
        const auto& e = ex[row * blocks + b];
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

// (2) three segments (types a, b, Q8_0), rows R = 1/2/4/8 of 8 activation rows.
void engine_test(std::uint32_t ta, std::uint32_t tb, std::uint32_t width, std::mt19937& rng) {
    const std::uint32_t na = 96U, nb = 64U, nc = 32U, outputs = na + nb + nc, total = 8U;
    const std::array<std::uint32_t, 3> types{ta, tb, 8U};
    const std::array<std::uint32_t, 3> counts{na, nb, nc};
    std::array<std::vector<std::uint8_t>, 3> host;
    std::array<std::vector<Exported>, 3> ex;
    std::vector<std::unique_ptr<DeviceBuffer<std::uint8_t>>> weights;
    for (unsigned m = 0; m < 3U; ++m) {
        host[m] = random_rows(types[m], width, counts[m], rng);
        ex[m] = decode(types[m], host[m], width, counts[m]);
        weights.push_back(std::make_unique<DeviceBuffer<std::uint8_t>>(host[m].size()));
        weights.back()->upload(host[m].data());
    }
    std::normal_distribution<float> normal(0.0F, 1.0F);
    std::vector<float> x(static_cast<std::size_t>(total) * width);
    for (auto& v : x) v = normal(rng);
    for (std::uint32_t k = 0; k < width; k += 97U) x[k] = 0.0F;  // a few exact zeros
    DeviceBuffer<float> input(x.size()); input.upload(x.data());
    const auto q = quantize_reference(x, total, width);
    std::vector<float> expected(static_cast<std::size_t>(total) * outputs);
    for (std::uint32_t r = 0; r < total; ++r)
        for (std::uint32_t n = 0, m = 0, first = 0; n < outputs; ++n) {
            while (n >= first + counts[m]) first += counts[m++];
            expected[static_cast<std::size_t>(r) * outputs + n] = reference_output(types[m], ex[m], q, width, r, n - first);
        }
    const auto run = [&]<unsigned R>() {
        for (std::uint32_t base = 0; base < total; base += R) {
            DeviceBuffer<float> out(static_cast<std::size_t>(R) * outputs);
            Site<R> site{};
            site.segments = q8_rows::segments({{weights[0]->p, na}, {weights[1]->p, nb}, {weights[2]->p, nc}});
            for (unsigned m = 0; m < 3U; ++m) site.segments.types[m] = types[m];
            for (unsigned r = 0; r < R; ++r) site.input[r] = input.p + static_cast<std::size_t>(base + r) * width;
            site.width = width; site.outputs = outputs; site.output = out.p;
            q8_rows::launch_dense_a8(site, nullptr, false);
            check(hipGetLastError(), "engine launch");
            const auto got = out.download();
            for (unsigned r = 0; r < R; ++r)
                for (std::uint32_t n = 0; n < outputs; ++n) {
                    const float e = expected[static_cast<std::size_t>(base + r) * outputs + n], a = got[static_cast<std::size_t>(r) * outputs + n];
                    if (std::memcmp(&e, &a, 4) != 0)
                        throw std::runtime_error("dense GSQ " + std::string(ggml_dequant::name(ta)) + "+" + std::string(ggml_dequant::name(tb)) +
                            " R" + std::to_string(R) + " row " + std::to_string(base + r) + " output " + std::to_string(n) + ": " +
                            std::to_string(a) + " vs reference " + std::to_string(e));
                }
        }
    };
    run.template operator()<1U>(); run.template operator()<2U>(); run.template operator()<4U>(); run.template operator()<8U>();
    std::cout << "engine " << ggml_dequant::name(ta) << "+" << ggml_dequant::name(tb) << "+Q8_0 width " << width << " R1/2/4/8 PASS" << std::endl;
}
// (3) GSQ FP16 WMMA prompt form (fast prefill semantics: FP16 operands, FP32 accumulation; not exact) against the
// host FP32 product of the dequantized weights: relative error of every output within 2e-2 of the row's scale.
template <unsigned R>
void f16_test(std::uint32_t ta, std::uint32_t tb, std::uint32_t width, std::mt19937& rng) {
    const std::uint32_t na = 128U, nb = 64U, outputs = na + nb;
    const std::array<std::uint32_t, 2> types{ta, tb};
    const std::array<std::uint32_t, 2> counts{na, nb};
    std::array<std::vector<std::uint8_t>, 2> host;
    std::vector<std::unique_ptr<DeviceBuffer<std::uint8_t>>> weights;
    std::vector<float> w(static_cast<std::size_t>(outputs) * width);
    for (unsigned m = 0, first = 0; m < 2U; first += counts[m], ++m) {
        host[m] = random_rows(types[m], width, counts[m], rng, 10);
        weights.push_back(std::make_unique<DeviceBuffer<std::uint8_t>>(host[m].size()));
        weights.back()->upload(host[m].data());
        const auto rb = gsq_a8::dense_row_bytes(types[m], width);
        for (std::uint32_t n = 0; n < counts[m]; ++n)
            ggml_dequant::dequantize_row(types[m], std::span<const std::byte>(reinterpret_cast<const std::byte*>(host[m].data()) + n * rb, rb),
                std::span<float>(w.data() + static_cast<std::size_t>(first + n) * width, width));
    }
    std::normal_distribution<float> normal(0.0F, 1.0F);
    std::vector<float> x(static_cast<std::size_t>(R) * width);
    for (auto& v : x) v = normal(rng);
    DeviceBuffer<float> input(x.size()); input.upload(x.data());
    DeviceBuffer<float> out(static_cast<std::size_t>(R) * outputs);
    Site<R> site{};
    site.segments = q8_rows::segments({{weights[0]->p, na}, {weights[1]->p, nb}});
    for (unsigned m = 0; m < 2U; ++m) site.segments.types[m] = types[m];
    for (unsigned r = 0; r < R; ++r) site.input[r] = input.p + static_cast<std::size_t>(r) * width;
    site.width = width; site.outputs = outputs; site.output = out.p;
    q8_rows::launch_prefill(site, true, true, nullptr);
    check(hipGetLastError(), "f16 launch");
    const auto got = out.download();
    double worst = 0.0;
    for (unsigned r = 0; r < R; ++r)
        for (std::uint32_t n = 0; n < outputs; ++n) {
            double ref = 0.0, scale = 0.0;
            for (std::uint32_t k = 0; k < width; ++k) {
                const double p = static_cast<double>(w[static_cast<std::size_t>(n) * width + k]) * x[static_cast<std::size_t>(r) * width + k];
                ref += p; scale += std::fabs(p);
            }
            worst = std::max(worst, std::fabs(got[static_cast<std::size_t>(r) * outputs + n] - ref) / std::max(scale, 1e-30));
        }
    std::cout << "f16 " << ggml_dequant::name(ta) << "+" << ggml_dequant::name(tb) << " R" << R << " worst rel " << worst << std::endl;
    require(worst < 2e-2, "GSQ FP16 prompt form deviates from the FP32 product");
}
// (4) Engine site compositions at engine shapes: the R1 site against the R2/4/8 window sites, bit for bit. The QSA R1
// input site holds every dense matrix in one launch (launch_qsa_dense_a8_inputs) while the windows group the matrices
// by type (enqueue_qsa_known_rows), so a typed q beside Q8_0 k/v runs the GSQ kernel at R1 and, for k/v, the Q8 dense
// kernel in windows. split = the window grouping by type; otherwise the window sites have the R1 composition (GDN).
// Returns the number of differing words (printed per segment).
std::size_t window_vs_r1(const std::vector<std::uint32_t>& types, const std::vector<std::uint32_t>& counts, bool split,
    std::uint32_t width, std::mt19937& rng, const std::string& name) {
    const auto n = static_cast<unsigned>(types.size());
    std::vector<std::unique_ptr<DeviceBuffer<std::uint8_t>>> weights;
    std::vector<std::uint32_t> first{0};
    for (unsigned m = 0; m < n; ++m) {
        const auto host = random_rows(types[m], width, counts[m], rng, 10);
        weights.push_back(std::make_unique<DeviceBuffer<std::uint8_t>>(host.size()));
        weights.back()->upload(host.data());
        first.push_back(first.back() + counts[m]);
    }
    const std::uint32_t outputs = first.back(), total = 8U;
    std::normal_distribution<float> normal(0.0F, 1.0F);
    std::vector<float> x(static_cast<std::size_t>(total) * width);
    for (auto& v : x) v = normal(rng);
    DeviceBuffer<float> input(x.size()); input.upload(x.data());
    // The sites of one launch: a list of segment indices (all of them, or one group per type).
    std::vector<std::vector<unsigned>> r1_sites{{}}, window_sites;
    for (unsigned m = 0; m < n; ++m) r1_sites[0].push_back(m);
    if (!split) window_sites = r1_sites;
    else for (unsigned m = 0; m < n; ++m) {
        bool placed = false;
        for (auto& s : window_sites) if (types[s[0]] == types[m]) { s.push_back(m); placed = true; break; }
        if (!placed) window_sites.push_back({m});
    }
    // out[row * outputs + global output] for one row count.
    const auto run = [&]<unsigned R>(const std::vector<std::vector<unsigned>>& sites) {
        std::vector<float> all(static_cast<std::size_t>(total) * outputs);
        for (std::uint32_t base = 0; base < total; base += R)
            for (const auto& s : sites) {
                std::uint32_t site_outputs = 0;
                for (const auto m : s) site_outputs += counts[m];
                DeviceBuffer<float> out(static_cast<std::size_t>(R) * site_outputs);
                Site<R> site{};
                std::uint32_t at = 0;
                for (unsigned i = 0; i < s.size(); ++i) {
                    site.segments.weights[i] = weights[s[i]]->p;
                    site.segments.types[i] = q8_rows::segment_type(types[s[i]]);
                    at += counts[s[i]]; site.segments.first[i + 1U] = at;
                }
                site.segments.count = static_cast<unsigned>(s.size());
                for (unsigned r = 0; r < R; ++r) site.input[r] = input.p + static_cast<std::size_t>(base + r) * width;
                site.width = width; site.outputs = site_outputs; site.output = out.p;
                q8_rows::launch_dense_a8(site, nullptr, false);
                check(hipGetLastError(), "window_vs_r1 launch");
                const auto got = out.download();
                for (unsigned r = 0; r < R; ++r) {
                    std::uint32_t local = 0;
                    for (const auto m : s) {
                        std::memcpy(&all[static_cast<std::size_t>(base + r) * outputs + first[m]],
                            &got[static_cast<std::size_t>(r) * site_outputs + local], counts[m] * sizeof(float));
                        local += counts[m];
                    }
                }
            }
        return all;
    };
    const auto ref = run.template operator()<1U>(r1_sites);
    std::size_t bad_total = 0;
    const auto compare = [&](const std::vector<float>& got, unsigned rows) {
        std::string line = name + (split ? " (windows grouped by type)" : "") + " R1 vs R" + std::to_string(rows) + ":";
        for (unsigned m = 0; m < n; ++m) {
            std::size_t bad = 0;
            for (std::uint32_t r = 0; r < total; ++r)
                for (std::uint32_t o = first[m]; o < first[m + 1U]; ++o)
                    bad += std::memcmp(&ref[static_cast<std::size_t>(r) * outputs + o], &got[static_cast<std::size_t>(r) * outputs + o], 4) != 0;
            line += " " + std::string(ggml_dequant::name(types[m])) + " " + std::to_string(bad) + "/" + std::to_string(total * counts[m]);
            bad_total += bad;
        }
        std::cout << line << std::endl;
    };
    compare(run.template operator()<1U>(window_sites), 1U);
    compare(run.template operator()<2U>(window_sites), 2U);
    compare(run.template operator()<4U>(window_sites), 4U);
    compare(run.template operator()<8U>(window_sites), 8U);
    return bad_total;
}
}  // namespace

int main(int argc, char** argv) try {
    if (argc > 1 && std::string(argv[1]) == "r1_vs_window") {
        // Engine shapes (Qwen3.8-Flash-Next, width 2560): QSA inputs q 12288 + k 512 + v 512 (attn_q / attn_k / attn_v),
        // GDN inputs qkv 10240 + z 6144 (attn_qkv / attn_gate). Type combinations of HYB3A, VAR-IQ3 and VAR-Q2 layers.
        std::mt19937 rng(20261025U);
        std::size_t qsa_mixed = 0, other = 0;
        for (const auto q : {14U, 13U, 23U, 12U}) qsa_mixed += window_vs_r1({q, 8U, 8U}, {12288U, 512U, 512U}, true, 2560U, rng, "QSA HYB3A");
        other += window_vs_r1({14U, 14U, 14U}, {12288U, 512U, 512U}, true, 2560U, rng, "QSA VAR-IQ3");
        other += window_vs_r1({13U, 14U, 13U}, {12288U, 512U, 512U}, true, 2560U, rng, "QSA VAR-IQ3");
        other += window_vs_r1({11U, 23U, 23U}, {12288U, 512U, 512U}, true, 2560U, rng, "QSA VAR-Q2");
        for (const auto [a, b] : {std::pair{14U, 13U}, std::pair{14U, 14U}, std::pair{13U, 12U}, std::pair{11U, 23U}, std::pair{23U, 42U}})
            other += window_vs_r1({a, b}, {10240U, 6144U}, false, 2560U, rng, "GDN");
        std::cout << "r1_vs_window: QSA typed-q + Q8 k/v differing words " << qsa_mixed << ", other compositions " << other << std::endl;
        return qsa_mixed || other ? 1 : 0;
    }
    std::mt19937 rng(20261024U);
    const std::array<std::uint32_t, 10> types{2U, 6U, 8U, 11U, 12U, 13U, 14U, 20U, 23U, 42U};
    for (auto t : types) decode_test(t, rng);
    for (std::size_t i = 0; i < types.size(); ++i) engine_test(types[i], types[(i + 3U) % types.size()], 2560U, rng);
    engine_test(13U, 42U, 6144U, rng);
    engine_test(20U, 6U, 640U, rng);
    engine_test(2U, 42U, 640U, rng);
    f16_test<32U>(11U, 13U, 2560U, rng);
    f16_test<16U>(42U, 23U, 2560U, rng);
    f16_test<32U>(14U, 12U, 6144U, rng);
    f16_test<16U>(20U, 2U, 640U, rng);
    std::cout << "GSQ dense A8 PASS" << std::endl;
    return 0;
} catch (const std::exception& e) {
    std::cerr << e.what() << std::endl;
    return 1;
}
