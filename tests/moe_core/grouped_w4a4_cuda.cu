// Device fixture + no-model bench of the grouped W4A4 expert tile (moe_core/grouped_w4a4.hpp, sm_120a).
// 1. Quantiser: FP32 rows by assignment index and BF16 rows, bitwise against a host reference of the family rule.
// 2. Projection: ragged tiles (1..64 rows, offsets, contiguous and interleaved-halves weights, K 2304 and 1152) against a double
//    reference of the same NVFP4 operands (FP32 tensor accumulation, bound 1e-4 of the absolute product sum), padding never
//    stored, two launches bitwise equal.
// 3. Bench (--bench): one Inkling-class chunk (128 experts x 2 tiles of 64 rows) per projection, BF16 shift-decode tile vs
//    quantiser + W4A4 tile, median of 20 launches each (cudaEvent).
#include "cuda/core/grouped_w4a4.hpp"
#include "cuda/core/mma_tile.cuh"

#include <cuda_runtime.h>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <random>
#include <vector>

namespace {

namespace gw = strix::moe_core::grouped_w4a4;
namespace packed = strix::weight::microscaled;
namespace view = strix::moe_core::grouped_microscaled;

#define CK(x)                                                                                    \
    do {                                                                                         \
        const cudaError_t e_ = (x);                                                              \
        if (e_ != cudaSuccess) {                                                                 \
            std::printf("CUDA %s at %s:%d\n", cudaGetErrorString(e_), __FILE__, __LINE__);     \
            std::exit(2);                                                                        \
        }                                                                                        \
    } while (0)

int failures = 0;
void check(bool ok, const char* what) {
    if (!ok) {
        ++failures;
        std::printf("FAIL %s\n", what);
    }
}

// ---- host reference of the family quantiser (exhaustive nearest, ties to the even code)
float e4m3_value(unsigned c) {
    const unsigned e = (c >> 3) & 15U, m = c & 7U;
    return e == 0 ? std::ldexp(static_cast<float>(m), -9) : std::ldexp(1.0F + static_cast<float>(m) / 8.0F, static_cast<int>(e) - 7);
}
std::uint8_t encode_e4m3_satfinite(float x) {  // x >= 0 finite
    unsigned best = 0;
    for (unsigned c = 1; c <= 0x7eU; ++c) {
        const float d = std::fabs(e4m3_value(c) - x), b = std::fabs(e4m3_value(best) - x);
        if (d < b || (d == b && (c & 1U) == 0U)) best = c;
    }
    return static_cast<std::uint8_t>(best);
}
float e2m1_value(unsigned n) {
    static constexpr float mag[8] = {0.F, .5F, 1.F, 1.5F, 2.F, 3.F, 4.F, 6.F};
    return (n & 8U) ? -mag[n & 7U] : mag[n & 7U];
}
std::uint8_t encode_e2m1(float x) {  // |x| <= 6
    const float a = std::fabs(x);
    unsigned best = 0;
    for (unsigned c = 1; c < 8U; ++c) {
        const float d = std::fabs(e2m1_value(c) - a), b = std::fabs(e2m1_value(best) - a);
        if (d < b || (d == b && (c & 1U) == 0U)) best = c;
    }
    return static_cast<std::uint8_t>(best | (std::signbit(x) ? 8U : 0U));
}
// Code bytes equal up to the sign of zero (E2M1 0x8 = -0 decodes to 0 like 0x0).
bool same_codes(const std::uint8_t* a, const std::uint8_t* b, std::size_t n) {
    const auto norm = [](std::uint8_t v) {
        const unsigned lo = (v & 15U) == 8U ? 0U : (v & 15U), hi = (v >> 4U) == 8U ? 0U : (v >> 4U);
        return lo | (hi << 4U);
    };
    for (std::size_t i = 0; i < n; ++i)
        if (norm(a[i]) != norm(b[i])) return false;
    return true;
}
void quantize_host(const float* x, unsigned width, std::uint8_t* codes, std::uint8_t* scales, float* rs) {
    float amax = 0.F;
    for (unsigned i = 0; i < width; ++i) amax = std::max(amax, std::fabs(x[i]));
    const float ts = amax / gw::kRowScaleDiv;
    *rs = ts;
    for (unsigned q = 0; q < width / 16U; ++q) {
        float a16 = 0.F;
        for (unsigned i = 0; i < 16U; ++i) a16 = std::max(a16, std::fabs(x[16U * q + i]));
        std::uint8_t sc = 0;
        float scale = 0.F;
        if (ts > 0.F) {
            sc = encode_e4m3_satfinite(std::min(448.F, (a16 / 6.F) / ts));
            scale = e4m3_value(sc) * ts;
        }
        scales[q] = sc;
        for (unsigned p = 0; p < 8U; ++p) {
            const float u = scale != 0.F ? std::min(6.F, std::max(-6.F, x[16U * q + 2U * p] / scale)) : 0.F;
            const float w = scale != 0.F ? std::min(6.F, std::max(-6.F, x[16U * q + 2U * p + 1U] / scale)) : 0.F;
            codes[8U * q + p] = static_cast<std::uint8_t>(encode_e2m1(u) | (encode_e2m1(w) << 4U));
        }
    }
}
float bf16_round(float x) {
    std::uint32_t u;
    std::memcpy(&u, &x, 4);
    u += 0x7fffU + ((u >> 16) & 1U);
    u &= 0xffff0000U;
    float y;
    std::memcpy(&y, &u, 4);
    return y;
}
std::uint16_t bf16_bits(float x) {
    std::uint32_t u;
    const float y = bf16_round(x);
    std::memcpy(&u, &y, 4);
    return static_cast<std::uint16_t>(u >> 16);
}

template <class T>
T* upload(const std::vector<T>& h) {
    T* d = nullptr;
    CK(cudaMalloc(&d, h.size() * sizeof(T) + 256));
    CK(cudaMemcpy(d, h.data(), h.size() * sizeof(T), cudaMemcpyHostToDevice));
    return d;
}

struct Expert {
    std::vector<std::uint8_t> codes, scales;
    packed::RowOrder order;
};

void correctness(std::mt19937& rng) {
    constexpr unsigned kOut = 256U, kTokens = 40U;
    const unsigned widths[2] = {2304U, 1152U};
    const struct { unsigned expert, rows; } tiles_spec[] = {{0, 64}, {0, 17}, {1, 1}, {2, 64}, {3, 33}, {1, 50}};
    for (unsigned wi = 0; wi < 2U; ++wi) {
        const unsigned K = widths[wi];
        std::uniform_int_distribution<int> byte(0, 255), sc(0x28, 0x48);
        std::normal_distribution<float> normal(0.F, 1.F);
        std::vector<Expert> experts(4);
        for (unsigned e = 0; e < 4U; ++e) {
            experts[e].codes.resize(std::size_t{kOut} * K / 2U);
            experts[e].scales.resize(std::size_t{kOut} * K / 16U);
            for (auto& b : experts[e].codes) b = static_cast<std::uint8_t>(byte(rng));
            for (auto& b : experts[e].scales) b = static_cast<std::uint8_t>(sc(rng));
            experts[e].order = (e & 1U) ? packed::RowOrder::interleaved_halves : packed::RowOrder::contiguous;
        }
        unsigned assignments = 0;
        for (const auto& t : tiles_spec) assignments += t.rows;
        std::vector<float> tokens(std::size_t{kTokens} * K);
        for (auto& v : tokens) v = normal(rng) * (1.F + static_cast<float>(byte(rng) % 7));
        std::vector<std::uint32_t> index(assignments);
        for (auto& i : index) i = static_cast<std::uint32_t>(byte(rng)) % kTokens;
        // device operands
        std::vector<std::uint8_t*> dcodes(4), dscales(4);
        for (unsigned e = 0; e < 4U; ++e) {
            dcodes[e] = upload(experts[e].codes);
            dscales[e] = upload(experts[e].scales);
        }
        float* dtokens = upload(tokens);
        std::uint32_t* dindex = upload(index);
        const auto layout = gw::image_layout(assignments, K);
        std::uint8_t* dimage = nullptr;
        CK(cudaMalloc(&dimage, layout.bytes));
        const auto image = gw::make_image(dimage, assignments, K);
        CK(strix::cuda::core::grouped_w4a4::quantize(
            gw::QuantizeArgs<float>{dtokens, K, dindex, assignments, K, image, nullptr}, nullptr));
        // quantiser bitwise
        std::vector<std::uint8_t> qc(std::size_t{assignments} * K / 2U), qs(std::size_t{assignments} * K / 16U);
        std::vector<float> qr(assignments);
        CK(cudaMemcpy(qc.data(), image.codes, qc.size(), cudaMemcpyDeviceToHost));
        CK(cudaMemcpy(qs.data(), image.scales, qs.size(), cudaMemcpyDeviceToHost));
        CK(cudaMemcpy(qr.data(), image.row_scale, qr.size() * 4, cudaMemcpyDeviceToHost));
        bool quant_ok = true;
        std::vector<std::uint8_t> hc(K / 2U), hs(K / 16U);
        for (unsigned a = 0; a < assignments; ++a) {
            float hr;
            quantize_host(tokens.data() + std::size_t{index[a]} * K, K, hc.data(), hs.data(), &hr);
            quant_ok = quant_ok && hr == qr[a] && same_codes(hc.data(), qc.data() + std::size_t{a} * K / 2U, hc.size()) &&
                       std::memcmp(hs.data(), qs.data() + std::size_t{a} * K / 16U, hs.size()) == 0;
        }
        check(quant_ok, "quantiser FP32 gather bitwise");
        // BF16 source rows (contiguous), bitwise as their FP32 widening
        std::vector<std::uint16_t> brows(std::size_t{assignments} * K);
        std::vector<float> bwide(brows.size());
        for (std::size_t i = 0; i < brows.size(); ++i) {
            brows[i] = bf16_bits(tokens[(i / K) % kTokens * K + i % K]);
            bwide[i] = bf16_round(tokens[(i / K) % kTokens * K + i % K]);
        }
        std::uint16_t* dbrows = upload(brows);
        std::uint8_t* dimage2 = nullptr;
        CK(cudaMalloc(&dimage2, layout.bytes));
        const auto image2 = gw::make_image(dimage2, assignments, K);
        CK(strix::cuda::core::grouped_w4a4::quantize(
            gw::QuantizeArgs<std::uint16_t>{dbrows, K, nullptr, assignments, K, image2, nullptr}, nullptr));
        CK(cudaMemcpy(qc.data(), image2.codes, qc.size(), cudaMemcpyDeviceToHost));
        CK(cudaMemcpy(qs.data(), image2.scales, qs.size(), cudaMemcpyDeviceToHost));
        CK(cudaMemcpy(qr.data(), image2.row_scale, qr.size() * 4, cudaMemcpyDeviceToHost));
        quant_ok = true;
        for (unsigned a = 0; a < assignments; ++a) {
            float hr;
            quantize_host(bwide.data() + std::size_t{a} * K, K, hc.data(), hs.data(), &hr);
            quant_ok = quant_ok && hr == qr[a] && same_codes(hc.data(), qc.data() + std::size_t{a} * K / 2U, hc.size()) &&
                       std::memcmp(hs.data(), qs.data() + std::size_t{a} * K / 16U, hs.size()) == 0;
        }
        check(quant_ok, "quantiser BF16 rows bitwise");
        // projection over the FP32-gather image
        CK(cudaMemcpy(qc.data(), image.codes, qc.size(), cudaMemcpyDeviceToHost));
        CK(cudaMemcpy(qs.data(), image.scales, qs.size(), cudaMemcpyDeviceToHost));
        CK(cudaMemcpy(qr.data(), image.row_scale, qr.size() * 4, cudaMemcpyDeviceToHost));
        std::vector<std::uint16_t> base(std::size_t{assignments} * K + 8U);  // index base only (never read)
        std::uint16_t* dbase = upload(base);
        float* dout = nullptr;
        const std::size_t out_bytes = std::size_t{assignments} * kOut * 4U;
        CK(cudaMalloc(&dout, out_bytes));
        CK(cudaMemset(dout, 0xff, out_bytes));  // NaN poison: padding rows must never be stored over live rows
        std::vector<view::TileRows> tiles;
        unsigned first = 0;
        const float outer = 0.0123F;
        for (const auto& t : tiles_spec) {
            const packed::PackedMatrix w{dcodes[t.expert], dscales[t.expert], kOut, K, outer, experts[t.expert].order};
            view::TileRows tile{w, dbase + std::size_t{first} * K, dout + std::size_t{first} * kOut, K, kOut, t.rows};
            check(gw::fits_tile(tile, K, kOut), "fixture tile admitted");
            tiles.push_back(tile);
            first += t.rows;
        }
        view::TileRows* dtiles = upload(tiles);
        const gw::Args args{{dtiles, static_cast<unsigned>(tiles.size()), K, kOut, nullptr, nullptr, 64U}, dbase, image};
        CK(strix::cuda::core::grouped_w4a4::launch(args, nullptr));
        CK(cudaDeviceSynchronize());
        std::vector<float> got(std::size_t{assignments} * kOut), again(got.size());
        CK(cudaMemcpy(got.data(), dout, out_bytes, cudaMemcpyDeviceToHost));
        CK(strix::cuda::core::grouped_w4a4::launch(args, nullptr));
        CK(cudaMemcpy(again.data(), dout, out_bytes, cudaMemcpyDeviceToHost));
        check(std::memcmp(got.data(), again.data(), out_bytes) == 0, "two launches bitwise equal");
        {  // 128-row plan: expert 0's two tiles (64 + 17 rows) as one 81-row tile, run as row slices: bitwise the 64-row plan.
            std::vector<view::TileRows> merged(tiles.begin() + 1, tiles.end());
            merged.front() = tiles[0];
            merged.front().rows = tiles[0].rows + tiles[1].rows;
            check(gw::fits_tile(merged.front(), K, kOut), "81-row tile of a 128-row plan admitted");
            view::TileRows* dmerged = upload(merged);
            const gw::Args margs{{dmerged, static_cast<unsigned>(merged.size()), K, kOut, nullptr, nullptr, 128U}, dbase,
                                 image};
            CK(cudaMemset(dout, 0xff, out_bytes));
            CK(strix::cuda::core::grouped_w4a4::launch(margs, nullptr));
            CK(cudaMemcpy(again.data(), dout, out_bytes, cudaMemcpyDeviceToHost));
            check(std::memcmp(got.data(), again.data(), out_bytes) == 0, "128-row plan (row slices) bitwise the 64-row plan");
            CK(cudaFree(dmerged));
        }
        double worst = 0.0;
        first = 0;
        for (const auto& t : tiles_spec) {
            const auto& ex = experts[t.expert];
            for (unsigned r = 0; r < t.rows; ++r) {
                const unsigned a = first + r;
                for (unsigned o = 0; o < kOut; ++o) {
                    const auto prow = packed::physical_row(ex.order, kOut, o);
                    double sum = 0.0, mag = 0.0;
                    for (unsigned k = 0; k < K; ++k) {
                        const double wv = e2m1_value((ex.codes[prow * (K / 2U) + k / 2U] >> (4U * (k & 1U))) & 15U) *
                                          e4m3_value(ex.scales[prow * (K / 16U) + k / 16U]);
                        const double xv = e2m1_value((qc[std::size_t{a} * K / 2U + k / 2U] >> (4U * (k & 1U))) & 15U) *
                                          e4m3_value(qs[std::size_t{a} * K / 16U + k / 16U]);
                        sum += wv * xv;
                        mag += std::fabs(wv * xv);
                    }
                    const double scale = static_cast<double>(outer) * qr[a];
                    const double want = sum * scale, bound = 1e-4 * mag * scale + 1e-30;
                    const double err = std::fabs(static_cast<double>(got[std::size_t{a} * kOut + o]) - want);
                    if (!(err <= bound)) {
                        if (worst == 0.0)
                            std::printf("  K %u assignment %u output %u got %.9g want %.9g\n", K, a, o,
                                        static_cast<double>(got[std::size_t{a} * kOut + o]), want);
                        worst = 1.0;
                    }
                }
            }
            first += t.rows;
        }
        check(worst == 0.0, "W4A4 projection within 1e-4 of the product magnitude (and every live row stored)");
        std::printf("fixture K %u: %u assignments, %zu tiles checked\n", K, assignments, tiles.size());
    }
}

__global__ void fill_bytes(std::uint8_t* p, std::size_t n, std::uint32_t seed, std::uint8_t lo, std::uint8_t span) {
    for (std::size_t i = blockIdx.x * std::size_t{blockDim.x} + threadIdx.x; i < n; i += std::size_t{gridDim.x} * blockDim.x) {
        std::uint32_t h = static_cast<std::uint32_t>(i) * 2654435761U ^ seed;
        h ^= h >> 15;
        h *= 2246822519U;
        h ^= h >> 13;
        p[i] = span ? static_cast<std::uint8_t>(lo + h % span) : static_cast<std::uint8_t>(h);
    }
}
__global__ void fill_floats(float* p, std::size_t n, std::uint32_t seed) {
    for (std::size_t i = blockIdx.x * std::size_t{blockDim.x} + threadIdx.x; i < n; i += std::size_t{gridDim.x} * blockDim.x) {
        std::uint32_t h = static_cast<std::uint32_t>(i) * 2246822519U ^ seed;
        h ^= h >> 16;
        h *= 2654435761U;
        p[i] = (static_cast<float>(h & 0xffffU) - 32768.F) / 16384.F;
    }
}
__global__ void to_bf16(const float* x, std::uint16_t* y, std::size_t n) {
    for (std::size_t i = blockIdx.x * std::size_t{blockDim.x} + threadIdx.x; i < n; i += std::size_t{gridDim.x} * blockDim.x)
        y[i] = static_cast<std::uint16_t>(__float_as_uint(x[i]) >> 16);
}

template <class F>
float median_ms(F&& f) {
    cudaEvent_t a, b;
    CK(cudaEventCreate(&a));
    CK(cudaEventCreate(&b));
    f();
    CK(cudaDeviceSynchronize());
    std::vector<float> t;
    for (int i = 0; i < 20; ++i) {
        CK(cudaEventRecord(a));
        f();
        CK(cudaEventRecord(b));
        CK(cudaEventSynchronize(b));
        float ms = 0.F;
        CK(cudaEventElapsedTime(&ms, a, b));
        t.push_back(ms);
    }
    std::sort(t.begin(), t.end());
    return t[t.size() / 2U];
}

void bench() {
    constexpr unsigned kExperts = 128U, kTilesPerExpert = 2U, kRows = 64U, kAssign = kExperts * kTilesPerExpert * kRows;
    constexpr unsigned kTokens = 2048U, kOut = 2304U;
    const struct { const char* name; unsigned K; packed::RowOrder order; } projs[] = {
        {"gate_up", 2304U, packed::RowOrder::interleaved_halves}, {"down", 1152U, packed::RowOrder::contiguous}};
    for (const auto& pj : projs) {
        const unsigned K = pj.K;
        const std::size_t wcodes = std::size_t{kOut} * K / 2U, wscales = std::size_t{kOut} * K / 16U;
        std::uint8_t *codes = nullptr, *scales = nullptr, *image_bytes = nullptr;
        CK(cudaMalloc(&codes, wcodes * kExperts));
        CK(cudaMalloc(&scales, wscales * kExperts));
        fill_bytes<<<1024, 256>>>(codes, wcodes * kExperts, 7U, 0, 0);
        fill_bytes<<<1024, 256>>>(scales, wscales * kExperts, 9U, 0x30, 0x10);
        float *tokens = nullptr, *out = nullptr;
        std::uint16_t* bf = nullptr;
        std::uint32_t* index = nullptr;
        CK(cudaMalloc(&tokens, std::size_t{kTokens} * K * 4U));
        CK(cudaMalloc(&bf, std::size_t{kAssign} * K * 2U));
        CK(cudaMalloc(&out, std::size_t{kAssign} * kOut * 4U));
        CK(cudaMalloc(&index, kAssign * 4U));
        fill_floats<<<1024, 256>>>(tokens, std::size_t{kTokens} * K, 3U);
        std::vector<std::uint32_t> hindex(kAssign);
        for (unsigned a = 0; a < kAssign; ++a) hindex[a] = (a * 2654435761U) % kTokens;
        CK(cudaMemcpy(index, hindex.data(), kAssign * 4U, cudaMemcpyHostToDevice));
        float* gathered = nullptr;
        CK(cudaMalloc(&gathered, std::size_t{kAssign} * K * 4U));
        fill_floats<<<1024, 256>>>(gathered, std::size_t{kAssign} * K, 5U);
        to_bf16<<<1024, 256>>>(gathered, bf, std::size_t{kAssign} * K);
        const auto layout = gw::image_layout(kAssign, K);
        CK(cudaMalloc(&image_bytes, layout.bytes));
        const auto image = gw::make_image(image_bytes, kAssign, K);
        std::vector<view::TileRows> tiles;
        for (unsigned t = 0; t < kExperts * kTilesPerExpert; ++t) {
            const unsigned e = t / kTilesPerExpert;
            const packed::PackedMatrix w{codes + wcodes * e, scales + wscales * e, kOut, K, 0.01F, pj.order};
            tiles.push_back({w, bf + std::size_t{t} * kRows * K, out + std::size_t{t} * kRows * kOut, K, kOut, kRows});
        }
        view::TileRows* dtiles = upload(tiles);
        view::Args bargs{dtiles, static_cast<unsigned>(tiles.size()), K, kOut, nullptr, nullptr, kRows};
        bargs.decode = view::Decode::shift;
        const gw::Args wargs{bargs, bf, image};
        const float t_bf16 = median_ms([&] { CK(strix::cuda::core::mma::microscaled::launch<packed::MatrixFormat::nvfp4>(bargs, nullptr)); });
        const float t_quant = median_ms([&] {
            CK(strix::cuda::core::grouped_w4a4::quantize(gw::QuantizeArgs<float>{tokens, K, index, kAssign, K, image, nullptr}, nullptr));
        });
        const float t_w4a4 = median_ms([&] { CK(strix::cuda::core::grouped_w4a4::launch(wargs, nullptr)); });
        const double macs = double(kAssign) * kOut * K;
        std::printf("BENCH %s K %u rows %u: bf16_shift %.3f ms (%.0f TFLOPS)  quant %.3f ms  w4a4 %.3f ms (%.0f TFLOPS)  ratio %.2fx\n",
                    pj.name, K, kAssign, t_bf16, 2.0 * macs / t_bf16 / 1e9, t_quant, t_w4a4, 2.0 * macs / t_w4a4 / 1e9,
                    t_bf16 / (t_w4a4 + t_quant));
        CK(cudaFree(codes));
        CK(cudaFree(scales));
        CK(cudaFree(tokens));
        CK(cudaFree(bf));
        CK(cudaFree(out));
        CK(cudaFree(index));
        CK(cudaFree(gathered));
        CK(cudaFree(image_bytes));
        CK(cudaFree(dtiles));
    }
}

}  // namespace

int main(int argc, char** argv) {
    std::mt19937 rng(20261122U);
    correctness(rng);
    if (failures != 0) {
        std::printf("grouped_w4a4_cuda: FAIL (%d)\n", failures);
        return 1;
    }
    std::printf("grouped_w4a4_cuda: PASS (quantiser bitwise FP32+BF16, projection ragged tiles K 2304/1152, deterministic)\n");
    if (argc > 1 && std::strcmp(argv[1], "--bench") == 0) bench();
    return 0;
}
