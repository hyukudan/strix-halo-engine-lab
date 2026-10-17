#include "../src/hip_q8_rows_gemv.hpp"

#include <hip/hip_runtime.h>
#include <algorithm>
#include <array>
#include <bit>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <functional>
#include <iostream>
#include <memory>
#include <random>
#include <stdexcept>
#include <string>
#include <vector>

#if defined(__FAST_MATH__) || (defined(__FINITE_MATH_ONLY__) && __FINITE_MATH_ONLY__)
#error "dense A8 requires strict floating-point semantics"
#endif
#pragma clang fp contract(off)
#if defined(__HIP_DEVICE_COMPILE__)
#if !defined(__gfx1151__)
#error "dense A8 prototypes target gfx1151"
#endif
#endif

// Dense A8 profile prototype (Q8_0 weights x A8 activations).
//
// Quantizer (one row r, one 32-feature block b; x = the BF16/FP32 activations as FP32):
//   any x non-finite      -> delta = NaN (0x7fc00000), codes 0
//   amax = max |x|; tiny = 0 < amax < 2^-119
//   delta = amax == 0 || tiny ? 1 : amax / 127          (IEEE division, round to nearest even)
//   code  = tiny ? 0 : clamp(rint(x / delta), -127, 127) (IEEE division, rint ties to even)
// Projection, per (row r, output n), blocks in ascending order:
//   acc = +0
//   acc = fma(float(d_w[n][b]) * delta[r][b], float(sum_k q_w[n][b][k] * code[r][b][k]), acc)
//   y   = isnan(acc) ? 0x7fc00000 : acc
// The int32 block dot is exact in any order (|sum| < 2^20, so its float is exact); the FP32
// product, FMA and the ascending block chain are the contract. Deltas are stored per block,
// transposed: delta[b * stride + r].
//
// (a) dense_a8_dot4_kernel: 1..8 rows, G lanes share one output's block (32/G code bytes
//     each, v_dot4_i32_iu8, integer DPP reduction), every lane of the group runs the chain.
// (b) dense_a8_wmma_kernel: 16x16 tiles, two v_wmma_i32_16x16x16_iu8 per block on a zero
//     accumulator, then the per-element chain step in the epilogue.
//
// STRIX_DENSE_A8_BENCH=1 adds timings: (b) at 64/128/256 rows and the FP32
// project_wide_rows_kernel at 64 rows, (a) at 1..8 rows and the exact q8_rows kernels.
namespace {
using namespace strix;
void require(bool value, const std::string& message) { if (!value) throw std::runtime_error(message); }
void check(hipError_t error, const char* operation) {
    if (error != hipSuccess) throw std::runtime_error(std::string(operation) + ": " + hipGetErrorString(error));
}

constexpr std::uint32_t kCanonicalNan = 0x7fc00000U;

// ---------------------------------------------------------------- device side
__device__ __forceinline__ float canonical(float v) { return v != v ? __uint_as_float(kCanonicalNan) : v; }
__device__ __forceinline__ float half_bits(unsigned bits) {
    return __half2float(__ushort_as_half(static_cast<unsigned short>(bits & 0xffffU)));
}
__device__ __forceinline__ float load_input(const std::uint16_t* p) { return __uint_as_float(static_cast<unsigned>(*p) << 16U); }
__device__ __forceinline__ float load_input(const float* p) { return *p; }
__device__ __forceinline__ float divide(float a, float b) { return a / b; }

// One wave per (row, block); lane = feature.
template <class Input>
__global__ __launch_bounds__(256) void dense_a8_quantize_kernel(const Input* input, std::uint32_t rows, std::uint32_t width,
    std::int8_t* codes, float* deltas, std::uint32_t stride) {
    const unsigned lane = threadIdx.x & 31U, blocks = width / 32U;
    const std::uint32_t item = blockIdx.x * 8U + threadIdx.x / 32U;
    if (item >= rows * blocks) return;
    const std::uint32_t r = item / blocks, b = item % blocks;
    const float x = load_input(input + static_cast<std::uint64_t>(r) * width + 32U * b + lane);
    const bool bad = __ballot(!isfinite(x)) != 0;
    float amax = fabsf(x);
    for (int d = 16; d != 0; d /= 2) amax = fmaxf(amax, __shfl_xor(amax, d, 32));
    const bool tiny = amax > 0.0F && amax < 0x1p-119F;
    const float delta = bad ? __uint_as_float(kCanonicalNan) : amax == 0.0F || tiny ? 1.0F : divide(amax, 127.0F);
    const float rounded = bad || tiny ? 0.0F : rintf(divide(x, delta));
    const int code = static_cast<int>(fminf(127.0F, fmaxf(-127.0F, rounded)));
    codes[static_cast<std::uint64_t>(r) * width + 32U * b + lane] = static_cast<std::int8_t>(code);
    if (lane == 0U) deltas[static_cast<std::uint64_t>(b) * stride + r] = delta;
}

__device__ __forceinline__ int dot4(int a, int b, int acc) { return __builtin_amdgcn_sudot4(true, a, true, b, acc, false); }
template <unsigned D> __device__ __forceinline__ int lane_xor_int(int v) {
    return __builtin_amdgcn_mov_dpp(v, 0x160 | static_cast<int>(D), 0xf, 0xf, false);
}
template <unsigned G> __device__ __forceinline__ int group_sum(int v) {
    if constexpr (G >= 2U) v += lane_xor_int<1U>(v);
    if constexpr (G >= 4U) v += lane_xor_int<2U>(v);
    if constexpr (G >= 8U) v += lane_xor_int<4U>(v);
    return v;
}
template <unsigned Words> struct Words4 { unsigned v[Words]; };

// (a) Rows rows starting at row0 of the quantized buffer; output n row r -> out[(row0 + r) * outputs + n].
template <unsigned Rows, unsigned G>
__global__ __launch_bounds__(256) void dense_a8_dot4_kernel(const std::int8_t* codes, const float* deltas, std::uint32_t stride,
    const unsigned char* weights, float* out, std::uint32_t row0, std::uint32_t width, std::uint32_t outputs) {
    constexpr unsigned Bytes = 32U / G, Words = Bytes / 4U, PerWave = 32U / G;
    const unsigned lane = threadIdx.x & 31U, j = lane % G;
    const std::uint32_t n = (blockIdx.x * 8U + threadIdx.x / 32U) * PerWave + lane / G, blocks = width / 32U;
    const std::uint32_t row = n < outputs ? n : outputs - 1U;
    const unsigned char* w = weights + static_cast<std::uint64_t>(row) * blocks * 34U;
    const std::int8_t* a = codes + static_cast<std::uint64_t>(row0) * width + j * Bytes;
    float acc[Rows];
#pragma unroll
    for (unsigned r = 0; r < Rows; ++r) acc[r] = 0.0F;
    for (std::uint32_t b = 0; b < blocks; ++b) {
        const unsigned char* block = w + b * 34U;
        const float dw = half_bits(*reinterpret_cast<const std::uint16_t*>(block));
        Words4<Words> q;
        __builtin_memcpy(&q, block + 2U + j * Bytes, Bytes);
#pragma unroll
        for (unsigned r = 0; r < Rows; ++r) {
            const auto x = *reinterpret_cast<const Words4<Words>*>(a + static_cast<std::uint64_t>(r) * width + 32U * b);
            int p = 0;
#pragma unroll
            for (unsigned k = 0; k < Words; ++k) p = dot4(static_cast<int>(x.v[k]), static_cast<int>(q.v[k]), p);
            p = group_sum<G>(p);
            const float s = dw * deltas[static_cast<std::uint64_t>(b) * stride + row0 + r];
            acc[r] = __builtin_fmaf(s, static_cast<float>(p), acc[r]);
        }
    }
    if (j == 0U && n < outputs)
#pragma unroll
        for (unsigned r = 0; r < Rows; ++r) out[static_cast<std::uint64_t>(row0 + r) * outputs + n] = canonical(acc[r]);
}

typedef int v4i __attribute__((ext_vector_type(4)));
typedef int v8i __attribute__((ext_vector_type(8)));
struct __attribute__((aligned(4))) Dword4 { unsigned x, y, z, w; };

// (b) A wave owns TM x TN 16x16 tiles (rows x outputs); a block of 4 waves shares the row tiles
// and consecutive blocks share the output tiles. Q8_0 block pairs (68 bytes) are dword aligned:
// the even block's codes are realigned by two bytes, the odd block's are aligned already.
// WMMA layouts (gfx11 wave32): A lane i%16 = row i (16 K bytes), B lane i%16 = output i,
// D element e of lane l = (row 2e + l/16, output l%16); the upper half wave repeats A/B.
template <unsigned TM, unsigned TN>
__global__ __launch_bounds__(128) void dense_a8_wmma_kernel(const std::int8_t* codes, const float* deltas, std::uint32_t stride,
    const unsigned char* weights, float* out, std::uint32_t rows, std::uint32_t width, std::uint32_t outputs) {
    const unsigned lane = threadIdx.x & 31U, wave = threadIdx.x / 32U, i = lane & 15U, h = lane >> 4U;
    const std::uint32_t row_tiles = (rows + 16U * TM - 1U) / (16U * TM), pairs = width / 64U, blocks = width / 32U;
    const std::uint32_t m0 = blockIdx.x % row_tiles * 16U * TM, n0 = (blockIdx.x / row_tiles * 4U + wave) * 16U * TN;
    if (n0 >= outputs) return;
    const unsigned char* w[TN];
#pragma unroll
    for (unsigned t = 0; t < TN; ++t)
        w[t] = weights + static_cast<std::uint64_t>(min(n0 + 16U * t + i, outputs - 1U)) * blocks * 34U;
    const std::int8_t* a[TM];
#pragma unroll
    for (unsigned t = 0; t < TM; ++t) a[t] = codes + static_cast<std::uint64_t>(min(m0 + 16U * t + i, rows - 1U)) * width;
    // Epilogue rows of this lane: m0 + 16 t + 2 e + h. The delta stride covers whole row tiles
    // (padding rows are read but never stored), so they are immediate offsets of one index.
    const std::uint32_t erow = m0 + h;
    float acc[TM][TN][8];
#pragma unroll
    for (unsigned t = 0; t < TM; ++t)
#pragma unroll
        for (unsigned u = 0; u < TN; ++u)
#pragma unroll
            for (unsigned e = 0; e < 8U; ++e) acc[t][u][e] = 0.0F;
    const v8i zero{};
    for (std::uint32_t p = 0; p < pairs; ++p) {
        v4i wb[2][TN][2];
        float dw[2][TN];
#pragma unroll
        for (unsigned u = 0; u < TN; ++u) {
            const auto* q4 = reinterpret_cast<const Dword4*>(w[u] + 68U * p);
            const Dword4 c0 = q4[0], c1 = q4[1], c2 = q4[2], c3 = q4[3];
            const unsigned last = reinterpret_cast<const unsigned*>(w[u] + 68U * p)[16];
            const unsigned q[17]{c0.x, c0.y, c0.z, c0.w, c1.x, c1.y, c1.z, c1.w, c2.x, c2.y, c2.z, c2.w, c3.x, c3.y, c3.z, c3.w, last};
            dw[0][u] = half_bits(q[0]);
            dw[1][u] = half_bits(q[8] >> 16U);
#pragma unroll
            for (unsigned k = 0; k < 4U; ++k) {
                wb[0][u][0][k] = static_cast<int>(__builtin_amdgcn_alignbyte(q[k + 1U], q[k], 2U));
                wb[0][u][1][k] = static_cast<int>(__builtin_amdgcn_alignbyte(q[k + 5U], q[k + 4U], 2U));
                wb[1][u][0][k] = static_cast<int>(q[9U + k]);
                wb[1][u][1][k] = static_cast<int>(q[13U + k]);
            }
        }
        v4i ab[2][TM][2];
#pragma unroll
        for (unsigned t = 0; t < TM; ++t) {
            const auto* x = reinterpret_cast<const v4i*>(a[t] + 64U * p);
            ab[0][t][0] = x[0]; ab[0][t][1] = x[1]; ab[1][t][0] = x[2]; ab[1][t][1] = x[3];
        }
#pragma unroll
        for (unsigned s = 0; s < 2U; ++s) {
            const float* d = deltas + ((2U * p + s) * stride + erow);
            float da[TM][8];
#pragma unroll
            for (unsigned t = 0; t < TM; ++t)
#pragma unroll
                for (unsigned e = 0; e < 8U; ++e) da[t][e] = d[16U * t + 2U * e];
#pragma unroll
            for (unsigned t = 0; t < TM; ++t)
#pragma unroll
                for (unsigned u = 0; u < TN; ++u) {
                    v8i c = __builtin_amdgcn_wmma_i32_16x16x16_iu8_w32(true, ab[s][t][0], true, wb[s][u][0], zero, false);
                    c = __builtin_amdgcn_wmma_i32_16x16x16_iu8_w32(true, ab[s][t][1], true, wb[s][u][1], c, false);
#pragma unroll
                    for (unsigned e = 0; e < 8U; ++e) {
                        const float scale = dw[s][u] * da[t][e];
                        acc[t][u][e] = __builtin_fmaf(scale, static_cast<float>(c[e]), acc[t][u][e]);
                    }
                }
        }
    }
#pragma unroll
    for (unsigned t = 0; t < TM; ++t)
#pragma unroll
        for (unsigned u = 0; u < TN; ++u) {
            const std::uint32_t n = n0 + 16U * u + i;
#pragma unroll
            for (unsigned e = 0; e < 8U; ++e) {
                const std::uint32_t m = m0 + 16U * t + 2U * e + h;
                if (m < rows && n < outputs) out[static_cast<std::uint64_t>(m) * outputs + n] = canonical(acc[t][u][e]);
            }
        }
}

// Current FP32 path (the GDN sites' coefficient/FMA/tree) for the baseline timings.
__device__ __forceinline__ float baseline_coefficient(unsigned scale, std::int8_t code) {
    float result;
    asm("v_fma_mix_f32 %0, %1, %2, neg(0) op_sel_hi:[1,0,0]" : "=v"(result) : "v"(scale), "v"(static_cast<float>(code)));
    return result;
}
template <class In, unsigned Rows> struct BaselineSite {
    using Input = In;
    static constexpr unsigned rows = Rows;
    const void* input[Rows];
    q8_rows::Segments segments;
    std::uint32_t width, outputs;
    float* output;
    __device__ static float coefficient(unsigned scale, std::int8_t code) { return baseline_coefficient(scale, code); }
    __device__ static void fma(float& acc, float c, float x) { acc = __builtin_fmaf(c, x, acc); }
    __device__ static float add(float lower, float upper) {
        float result;
        asm("v_add_f32 %0, %1, %2" : "=v"(result) : "v"(lower), "v"(upper));
        return result;
    }
    __device__ void store(std::uint32_t out, unsigned r, float value) const { output[static_cast<std::uint64_t>(r) * outputs + out] = value; }
};

// ---------------------------------------------------------------- host side
struct Buffer {
    void* p = nullptr; std::size_t bytes = 0;
    explicit Buffer(std::size_t size) : bytes(size) { check(hipMalloc(&p, size), "allocate"); }
    ~Buffer() { if (hipFree(p) != hipSuccess) std::_Exit(3); }
    Buffer(const Buffer&) = delete; Buffer& operator=(const Buffer&) = delete;
    template <class T> T* as() const { return static_cast<T*>(p); }
    template <class T> void upload(const std::vector<T>& v) const {
        require(v.size() * sizeof(T) == bytes, "upload size"); check(hipMemcpy(p, v.data(), bytes, hipMemcpyHostToDevice), "upload");
    }
    template <class T> std::vector<T> download() const {
        std::vector<T> v(bytes / sizeof(T)); check(hipMemcpy(v.data(), p, bytes, hipMemcpyDeviceToHost), "download"); return v;
    }
    void poison() const { check(hipMemset(p, 0xa5, bytes), "poison"); }
};

float half_to_float(std::uint16_t h) {
    const std::uint32_t sign = static_cast<std::uint32_t>(h & 0x8000U) << 16U, e = (h >> 10U) & 31U, m = h & 1023U;
    if (e == 31U) return std::bit_cast<float>(sign | 0x7f800000U | (m << 13U));
    if (e == 0U) {
        const float v = std::ldexp(static_cast<float>(m), -24);
        return sign ? -v : v;
    }
    return std::bit_cast<float>(sign | ((e + 112U) << 23U) | (m << 13U));
}
std::uint16_t float_to_half(float v) {  // finite, in range; round to nearest even via the hardware type
    return std::bit_cast<std::uint16_t>(static_cast<_Float16>(v));
}
std::uint16_t bf16(float value) {
    auto bits = std::bit_cast<std::uint32_t>(value);
    bits += 0x7fffU + ((bits >> 16U) & 1U);
    return static_cast<std::uint16_t>(bits >> 16U);
}

// Reference quantizer and projection (the contract, serially).
struct Quantized { std::vector<std::int8_t> codes; std::vector<float> deltas; };
Quantized quantize_reference(const std::vector<float>& x, std::uint32_t rows, std::uint32_t width, std::uint32_t stride) {
    const std::uint32_t blocks = width / 32U;
    Quantized q{std::vector<std::int8_t>(static_cast<std::size_t>(rows) * width), std::vector<float>(static_cast<std::size_t>(blocks) * stride)};
    for (std::uint32_t r = 0; r < rows; ++r)
        for (std::uint32_t b = 0; b < blocks; ++b) {
            const float* v = &x[static_cast<std::size_t>(r) * width + 32U * b];
            bool bad = false; float amax = 0.0F;
            for (unsigned k = 0; k < 32U; ++k) { bad |= !std::isfinite(v[k]); amax = std::max(amax, std::fabs(v[k])); }
            const bool tiny = amax > 0.0F && amax < 0x1p-119F;
            const float delta = bad ? std::bit_cast<float>(kCanonicalNan) : amax == 0.0F || tiny ? 1.0F : amax / 127.0F;
            for (unsigned k = 0; k < 32U; ++k) {
                const float rounded = bad || tiny ? 0.0F : std::nearbyint(v[k] / delta);
                q.codes[static_cast<std::size_t>(r) * width + 32U * b + k] =
                    static_cast<std::int8_t>(static_cast<int>(std::fmin(127.0F, std::fmax(-127.0F, rounded))));
            }
            q.deltas[static_cast<std::size_t>(b) * stride + r] = delta;
        }
    return q;
}
float reference_output(const Quantized& q, std::uint32_t stride, const std::vector<std::uint8_t>& weights, std::uint32_t width,
    std::uint32_t r, std::uint32_t n) {
    const std::uint32_t blocks = width / 32U;
    float acc = 0.0F;
    for (std::uint32_t b = 0; b < blocks; ++b) {
        const std::uint8_t* block = &weights[(static_cast<std::size_t>(n) * blocks + b) * 34U];
        const float dw = half_to_float(static_cast<std::uint16_t>(block[0] | (block[1] << 8U)));
        int dot = 0;
        for (unsigned k = 0; k < 32U; ++k)
            dot += static_cast<int>(static_cast<std::int8_t>(block[2U + k])) * q.codes[static_cast<std::size_t>(r) * width + 32U * b + k];
        const float s = dw * q.deltas[static_cast<std::size_t>(b) * stride + r];
        acc = std::fma(s, static_cast<float>(dot), acc);
    }
    return std::isnan(acc) ? std::bit_cast<float>(kCanonicalNan) : acc;
}

// Fixture flavors. 0: random (Gaussian activations with a few outlier channels, realistic
// scales). 1: edges (zero / tiny / subnormal / extreme blocks, RNE ties, -128 codes, signed
// zero, subnormal/zero/negative fp16 scales). 2: sparse NaN/Inf activations and Inf/NaN scales.
std::vector<std::uint8_t> make_weights(std::uint32_t width, std::uint32_t outputs, unsigned flavor, std::mt19937& rng) {
    const std::size_t blocks = static_cast<std::size_t>(width / 32U) * outputs;
    std::vector<std::uint8_t> w(blocks * 34U);
    std::normal_distribution<float> code(0.0F, 40.0F);
    std::uniform_real_distribution<float> scale(0.0004F, 0.004F);
    constexpr std::array<std::uint16_t, 8> edge_scales{0x0000, 0x8000, 0x0001, 0x83ff, 0x0400, 0xa000, 0x7bff, 0x1c00};
    for (std::size_t b = 0; b < blocks; ++b) {
        std::uint16_t d = float_to_half(scale(rng) * (b % 7U == 3U ? -1.0F : 1.0F));
        if (flavor == 1U && b % 5U == 0U) d = edge_scales[(b / 5U) % edge_scales.size()];
        if (flavor == 2U && b % 997U == 13U) d = b % 2U ? 0x7c00 : 0x7e01;
        w[b * 34U] = static_cast<std::uint8_t>(d & 0xffU); w[b * 34U + 1U] = static_cast<std::uint8_t>(d >> 8U);
        for (unsigned k = 0; k < 32U; ++k) {
            int c = static_cast<int>(std::lround(code(rng)));
            c = std::clamp(c, -128, 127);
            if (flavor == 1U && b % 3U == 1U) c = k % 2U ? -128 : 127;
            w[b * 34U + 2U + k] = static_cast<std::uint8_t>(static_cast<std::int8_t>(c));
        }
    }
    return w;
}
// Activations as FP32 values exactly representable in the input type (BF16 rows are rounded).
std::vector<float> make_inputs(std::uint32_t rows, std::uint32_t width, bool bf, unsigned flavor, std::mt19937& rng) {
    std::vector<float> x(static_cast<std::size_t>(rows) * width);
    std::normal_distribution<float> g(0.0F, 1.0F);
    for (std::uint32_t r = 0; r < rows; ++r)
        for (std::uint32_t i = 0; i < width; ++i) {
            float v = g(rng) * (i % 97U == 5U ? 30.0F : 1.0F);
            const std::uint32_t b = i / 32U, k = i % 32U;
            if (flavor == 1U) {
                switch ((b + r) % 9U) {
                case 0: v = 0.0F; break;                                     // zero block
                case 1: v = k == 7U ? 0x1p-120F : 0.0F; break;               // tiny block
                case 2: v = std::ldexp(static_cast<float>(k) - 16.0F, -133); break;  // subnormal values
                case 3: v = (k % 2U ? -1.0F : 1.0F) * 3.0e38F; break;        // near FP32 max
                case 4: v = (static_cast<float>(k) - 15.5F) / 127.0F * 4.0F; break;  // half-step ties
                case 5: v = k == 0U ? -0.0F : (k == 1U ? 254.0F : 1.0F); break;  // exact x/delta = 0.5
                case 6: v = k % 2U ? -0.0F : 0.0F; break;                    // signed zeros
                default: break;
                }
            }
            if (flavor == 2U && r % 5U == 2U && i == (r * 37U) % width)
                v = std::bit_cast<float>(i % 3U == 0U ? 0x7f800000U : i % 3U == 1U ? 0xff800000U : 0x7fc01234U + r);
            if (bf) v = std::bit_cast<float>(static_cast<std::uint32_t>(bf16(v)) << 16U);
            if (bf && std::isnan(v)) v = std::bit_cast<float>(0x7fc10000U);
            x[static_cast<std::size_t>(r) * width + i] = v;
        }
    return x;
}

struct Shape { const char* name; std::uint32_t width, outputs; bool bf16; };
constexpr std::array<Shape, 5> shapes{{
    {"GDN qkv+z", 2560U, 18432U, true},
    {"GDN/QSA out", 6144U, 2560U, false},
    {"QSA q+k+v", 2560U, 13312U, true},
    {"shared gate+up", 2560U, 1280U, true},
    {"shared down", 640U, 2560U, false},
}};

struct Activations {
    std::uint32_t rows, width, stride;
    Buffer input, codes, deltas;
    bool bf;
    Activations(std::uint32_t r, std::uint32_t w, bool bf16_input)
        : rows(r), width(w), stride((r + 63U) / 64U * 64U), input(static_cast<std::size_t>(r) * w * (bf16_input ? 2U : 4U)),
          codes(static_cast<std::size_t>(r) * w), deltas(static_cast<std::size_t>(w / 32U) * stride * 4U), bf(bf16_input) {}
    void upload(const std::vector<float>& x) const {
        if (bf) {
            std::vector<std::uint16_t> h(x.size());
            for (std::size_t i = 0; i < x.size(); ++i) h[i] = static_cast<std::uint16_t>(std::bit_cast<std::uint32_t>(x[i]) >> 16U);
            input.upload(h);
        } else input.upload(x);
    }
    void quantize(hipStream_t stream) const {
        const std::uint32_t items = rows * (width / 32U), grid = (items + 7U) / 8U;
        if (bf) hipLaunchKernelGGL(dense_a8_quantize_kernel<std::uint16_t>, dim3(grid), dim3(256), 0, stream,
            input.as<std::uint16_t>(), rows, width, codes.as<std::int8_t>(), deltas.as<float>(), stride);
        else hipLaunchKernelGGL(dense_a8_quantize_kernel<float>, dim3(grid), dim3(256), 0, stream,
            input.as<float>(), rows, width, codes.as<std::int8_t>(), deltas.as<float>(), stride);
        check(hipGetLastError(), "quantize launch");
    }
};

template <unsigned Rows, unsigned G>
void launch_dot4(const Activations& a, const Buffer& w, Buffer& out, std::uint32_t row0, std::uint32_t outputs, hipStream_t stream) {
    constexpr unsigned PerBlock = 8U * (32U / G);
    hipLaunchKernelGGL((dense_a8_dot4_kernel<Rows, G>), dim3((outputs + PerBlock - 1U) / PerBlock), dim3(256), 0, stream,
        a.codes.as<std::int8_t>(), a.deltas.as<float>(), a.stride, w.as<unsigned char>(), out.as<float>(), row0, a.width, outputs);
}
template <unsigned G>
void launch_dot4_rows(unsigned rows, const Activations& a, const Buffer& w, Buffer& out, std::uint32_t row0, std::uint32_t outputs,
    hipStream_t stream) {
    switch (rows) {
    case 1: launch_dot4<1U, G>(a, w, out, row0, outputs, stream); break;
    case 2: launch_dot4<2U, G>(a, w, out, row0, outputs, stream); break;
    case 3: launch_dot4<3U, G>(a, w, out, row0, outputs, stream); break;
    case 4: launch_dot4<4U, G>(a, w, out, row0, outputs, stream); break;
    case 5: launch_dot4<5U, G>(a, w, out, row0, outputs, stream); break;
    case 6: launch_dot4<6U, G>(a, w, out, row0, outputs, stream); break;
    case 7: launch_dot4<7U, G>(a, w, out, row0, outputs, stream); break;
    default: launch_dot4<8U, G>(a, w, out, row0, outputs, stream); break;
    }
    check(hipGetLastError(), "dot4 launch");
}
template <unsigned TM, unsigned TN>
void launch_wmma(const Activations& a, const Buffer& w, Buffer& out, std::uint32_t outputs, hipStream_t stream) {
    const std::uint32_t row_tiles = (a.rows + 16U * TM - 1U) / (16U * TM), groups = (outputs + 16U * TN - 1U) / (16U * TN);
    hipLaunchKernelGGL((dense_a8_wmma_kernel<TM, TN>), dim3(row_tiles * ((groups + 3U) / 4U)), dim3(128), 0, stream,
        a.codes.as<std::int8_t>(), a.deltas.as<float>(), a.stride, w.as<unsigned char>(), out.as<float>(), a.rows, a.width, outputs);
    check(hipGetLastError(), "wmma launch");
}
enum class Tile { t11, t21, t12, t22, t41, t42 };
const char* tile_name(Tile t) {
    switch (t) { case Tile::t11: return "1x1"; case Tile::t21: return "2x1"; case Tile::t12: return "1x2";
    case Tile::t22: return "2x2"; case Tile::t41: return "4x1"; default: return "4x2"; }
}
void launch_wmma(Tile t, const Activations& a, const Buffer& w, Buffer& out, std::uint32_t outputs, hipStream_t stream) {
    switch (t) {
    case Tile::t11: launch_wmma<1U, 1U>(a, w, out, outputs, stream); break;
    case Tile::t21: launch_wmma<2U, 1U>(a, w, out, outputs, stream); break;
    case Tile::t12: launch_wmma<1U, 2U>(a, w, out, outputs, stream); break;
    case Tile::t22: launch_wmma<2U, 2U>(a, w, out, outputs, stream); break;
    case Tile::t41: launch_wmma<4U, 1U>(a, w, out, outputs, stream); break;
    default: launch_wmma<4U, 2U>(a, w, out, outputs, stream); break;
    }
}
constexpr std::array<Tile, 6> tiles{Tile::t11, Tile::t21, Tile::t12, Tile::t22, Tile::t41, Tile::t42};

std::string describe(std::uint32_t bits) {
    char text[64]; std::snprintf(text, sizeof text, "0x%08x (%g)", bits, std::bit_cast<float>(bits)); return text;
}
void compare(const std::vector<float>& expected, const std::vector<float>& actual, std::uint32_t outputs, const std::string& what) {
    require(expected.size() == actual.size(), what + ": size");
    if (!std::memcmp(expected.data(), actual.data(), expected.size() * 4U)) return;
    for (std::size_t i = 0; i < expected.size(); ++i)
        if (std::bit_cast<std::uint32_t>(expected[i]) != std::bit_cast<std::uint32_t>(actual[i]))
            throw std::runtime_error(what + ": row " + std::to_string(i / outputs) + " output " + std::to_string(i % outputs) +
                " expected " + describe(std::bit_cast<std::uint32_t>(expected[i])) + " got " + describe(std::bit_cast<std::uint32_t>(actual[i])));
}

// One shape, rows, flavor: GPU quantizer == reference quantizer; WMMA (every tile) == dot4 (every
// group width, rows 1..8 per launch) bitwise; sampled (or all) outputs == the serial reference.
void exact_case(hipStream_t stream, const Shape& shape, std::uint32_t rows, unsigned flavor, std::uint32_t outputs,
    bool full_reference, std::uint32_t& cases) {
    std::mt19937 rng(1234U + flavor * 77U + rows + shape.width * 3U + outputs);
    const auto weights_host = make_weights(shape.width, outputs, flavor, rng);
    const auto x = make_inputs(rows, shape.width, shape.bf16, flavor, rng);
    Buffer weights(weights_host.size()); weights.upload(weights_host);
    Activations a(rows, shape.width, shape.bf16); a.upload(x);
    check(hipMemset(a.deltas.p, 0xff, a.deltas.bytes), "poison padding deltas (NaN)");
    a.quantize(stream);
    check(hipStreamSynchronize(stream), "quantize");
    const auto q = quantize_reference(x, rows, shape.width, a.stride);
    const auto codes = a.codes.download<std::int8_t>();
    const auto deltas = a.deltas.download<float>();
    const std::string tag = std::string(shape.name) + " " + std::to_string(shape.width) + "->" + std::to_string(outputs) +
        " rows=" + std::to_string(rows) + " flavor=" + std::to_string(flavor);
    require(codes == q.codes, tag + ": quantizer codes differ from the reference");
    for (std::uint32_t b = 0; b < shape.width / 32U; ++b)
        require(!std::memcmp(&deltas[static_cast<std::size_t>(b) * a.stride], &q.deltas[static_cast<std::size_t>(b) * a.stride], rows * 4U),
            tag + ": quantizer deltas differ from the reference");

    Buffer out(static_cast<std::size_t>(rows) * outputs * 4U);
    // Decode form: 8 rows per launch, group width 8 (the reference result for the WMMA kernels).
    out.poison();
    for (std::uint32_t r = 0; r < rows; r += 8U) launch_dot4_rows<8U>(std::min(8U, rows - r), a, weights, out, r, outputs, stream);
    check(hipStreamSynchronize(stream), "dot4");
    const auto decode = out.download<float>();
    // Serial reference: every output of every row, or a sample.
    const std::uint32_t step_n = full_reference ? 1U : 37U;
    for (std::uint32_t r = 0; r < rows; ++r)
        for (std::uint32_t n = (r * 13U) % step_n; n < outputs; n += step_n) {
            const float e = reference_output(q, a.stride, weights_host, shape.width, r, n);
            const float g = decode[static_cast<std::size_t>(r) * outputs + n];
            if (std::bit_cast<std::uint32_t>(e) != std::bit_cast<std::uint32_t>(g))
                throw std::runtime_error(tag + ": dot4 row " + std::to_string(r) + " output " + std::to_string(n) + " reference " +
                    describe(std::bit_cast<std::uint32_t>(e)) + " dot4 " + describe(std::bit_cast<std::uint32_t>(g)));
        }
    // Other group widths and row counts (1..7 rows per launch at varying offsets).
    for (const unsigned g : {1U, 2U, 4U}) {
        out.poison();
        for (std::uint32_t r = 0, k = 0; r < rows; ++k) {
            const unsigned count = std::min(1U + (k * 3U + g) % 8U, rows - r);
            if (g == 1U) launch_dot4_rows<1U>(count, a, weights, out, r, outputs, stream);
            else if (g == 2U) launch_dot4_rows<2U>(count, a, weights, out, r, outputs, stream);
            else launch_dot4_rows<4U>(count, a, weights, out, r, outputs, stream);
            r += count;
        }
        check(hipStreamSynchronize(stream), "dot4 groups");
        compare(decode, out.download<float>(), outputs, tag + " dot4 G=" + std::to_string(g));
    }
    // Prefill form.
    for (const auto t : tiles) {
        out.poison();
        launch_wmma(t, a, weights, out, outputs, stream);
        check(hipStreamSynchronize(stream), "wmma");
        compare(decode, out.download<float>(), outputs, tag + " wmma " + tile_name(t));
    }
    if (flavor == 0U) {
        std::size_t finite = 0;
        for (const float v : decode) finite += std::isfinite(v) && v != 0.0F;
        require(finite * 10U > decode.size() * 9U, tag + ": degenerate fixture");
    }
    ++cases;
}

float median_us(const std::function<void()>& enqueue, hipStream_t stream, unsigned iterations = 30U) {
    hipEvent_t start, stop;
    check(hipEventCreate(&start), "event"); check(hipEventCreate(&stop), "event");
    std::vector<float> samples;
    for (unsigned it = 0; it < iterations + 5U; ++it) {
        check(hipEventRecord(start, stream), "record"); enqueue(); check(hipEventRecord(stop, stream), "record");
        check(hipEventSynchronize(stop), "sync");
        float ms = 0; check(hipEventElapsedTime(&ms, start, stop), "elapsed");
        if (it >= 5U) samples.push_back(ms * 1000.0F);
    }
    check(hipEventDestroy(start), "event"); check(hipEventDestroy(stop), "event");
    std::sort(samples.begin(), samples.end());
    return samples[samples.size() / 2U];
}

template <class In, unsigned Rows>
float baseline_us(const Shape& shape, const Buffer& weights, const std::vector<std::unique_ptr<Buffer>>& rows_in, Buffer& out,
    hipStream_t stream) {
    BaselineSite<In, Rows> site{};
    for (unsigned r = 0; r < Rows; ++r) site.input[r] = rows_in[r]->p;
    site.segments = q8_rows::segments({{weights.p, shape.outputs}});
    site.width = shape.width; site.outputs = shape.outputs; site.output = out.as<float>();
    return median_us([&] { q8_rows::launch(site, false, stream); }, stream);
}
template <unsigned Rows>
float baseline_shape_us(const Shape& shape, const Buffer& weights, hipStream_t stream) {
    std::vector<std::unique_ptr<Buffer>> rows_in;
    std::mt19937 rng(7U);
    const auto x = make_inputs(Rows, shape.width, shape.bf16, 0U, rng);
    for (unsigned r = 0; r < Rows; ++r) {
        rows_in.push_back(std::make_unique<Buffer>(static_cast<std::size_t>(shape.width) * (shape.bf16 ? 2U : 4U)));
        if (shape.bf16) {
            std::vector<std::uint16_t> h(shape.width);
            for (std::uint32_t i = 0; i < shape.width; ++i) h[i] = static_cast<std::uint16_t>(std::bit_cast<std::uint32_t>(x[r * shape.width + i]) >> 16U);
            rows_in.back()->upload(h);
        } else rows_in.back()->upload(std::vector<float>(x.begin() + r * shape.width, x.begin() + (r + 1U) * shape.width));
    }
    Buffer out(static_cast<std::size_t>(Rows) * shape.outputs * 4U);
    return shape.bf16 ? baseline_us<std::uint16_t, Rows>(shape, weights, rows_in, out, stream)
                      : baseline_us<float, Rows>(shape, weights, rows_in, out, stream);
}

void bench(hipStream_t stream) {
    std::cout.setf(std::ios::fixed); std::cout.precision(1);
    for (const auto& shape : shapes) {
        std::mt19937 rng(99U);
        const auto weights_host = make_weights(shape.width, shape.outputs, 0U, rng);
        Buffer weights(weights_host.size()); weights.upload(weights_host);
        const double mb = static_cast<double>(weights_host.size()) / 1e6;
        std::cout << shape.name << " " << shape.width << "->" << shape.outputs << " (" << mb << " MB Q8_0, warm L2/MALL)\n";
        const float base64 = baseline_shape_us<64U>(shape, weights, stream);
        const double ops64 = 2.0 * 64.0 * shape.width * shape.outputs;
        std::cout << "  FP32 project_wide_rows_kernel T=64: " << base64 << " us  " << ops64 / (base64 * 1e6) << " TOPS\n";
        for (const std::uint32_t rows : {64U, 128U, 256U}) {
            Activations a(rows, shape.width, shape.bf16);
            a.upload(make_inputs(rows, shape.width, shape.bf16, 0U, rng));
            a.quantize(stream);
            Buffer out(static_cast<std::size_t>(rows) * shape.outputs * 4U);
            const double ops = 2.0 * rows * shape.width * shape.outputs;
            const float quant = median_us([&] { a.quantize(stream); }, stream);
            float best = 1e30F; Tile best_tile = Tile::t11;
            std::cout << "  A8 WMMA T=" << rows << " (quantize " << quant << " us):";
            for (const auto t : tiles) {
                const float us = median_us([&] { launch_wmma(t, a, weights, out, shape.outputs, stream); }, stream);
                std::cout << "  " << tile_name(t) << " " << us;
                if (us < best) { best = us; best_tile = t; }
            }
            std::cout << "\n    best " << tile_name(best_tile) << ": " << best << " us  " << ops / (best * 1e6) << " TOPS  "
                      << mb * 1e3 / best << " GB/s weights";
            if (rows == 64U) std::cout << "  speedup vs FP32 x" << base64 / best << " (x" << base64 / (best + quant) << " with quantize)";
            std::cout << '\n';
        }
        // Decode form vs the exact q8_rows kernels.
        Activations a(8U, shape.width, shape.bf16);
        a.upload(make_inputs(8U, shape.width, shape.bf16, 0U, rng));
        a.quantize(stream);
        Buffer out(static_cast<std::size_t>(8U) * shape.outputs * 4U);
        std::cout << "  decode (us)   exact-q8_rows  A8 dot4 G=1 / 2 / 4 / 8\n";
        const auto exact = [&](unsigned rows) {
            switch (rows) {
            case 1: return baseline_shape_us<1U>(shape, weights, stream);
            case 2: return baseline_shape_us<2U>(shape, weights, stream);
            case 4: return baseline_shape_us<4U>(shape, weights, stream);
            default: return baseline_shape_us<8U>(shape, weights, stream);
            }
        };
        for (const unsigned rows : {1U, 2U, 4U, 8U}) {
            std::cout << "    T=" << rows << "  " << exact(rows) << "  ";
            std::cout << median_us([&] { launch_dot4_rows<1U>(rows, a, weights, out, 0U, shape.outputs, stream); }, stream) << " / "
                      << median_us([&] { launch_dot4_rows<2U>(rows, a, weights, out, 0U, shape.outputs, stream); }, stream) << " / "
                      << median_us([&] { launch_dot4_rows<4U>(rows, a, weights, out, 0U, shape.outputs, stream); }, stream) << " / "
                      << median_us([&] { launch_dot4_rows<8U>(rows, a, weights, out, 0U, shape.outputs, stream); }, stream) << "  ("
                      << mb * 1e3 / median_us([&] { launch_dot4_rows<4U>(rows, a, weights, out, 0U, shape.outputs, stream); }, stream)
                      << " GB/s at G=4)\n";
        }
    }
}
} // namespace

int main() try {
    hipStream_t stream = nullptr;
    check(hipStreamCreateWithFlags(&stream, hipStreamNonBlocking), "create stream");
    if (!std::getenv("STRIX_DENSE_A8_BENCH_ONLY")) {
        std::uint32_t cases = 0;
        for (const auto& shape : shapes) {
            for (const std::uint32_t rows : {64U, 128U, 256U})
                for (unsigned flavor = 0; flavor < 3U; ++flavor)
                    exact_case(stream, shape, rows, flavor, shape.outputs, rows == 64U && flavor == 0U && shape.outputs <= 2560U, cases);
            // Row/output tails (not multiples of the tiles) and odd row counts.
            for (const std::uint32_t rows : {1U, 17U, 100U})
                for (const std::uint32_t outputs : {1U, 33U, 1000U})
                    exact_case(stream, shape, rows, rows % 3U, outputs, true, cases);
            std::cout << shape.name << ": ok\n";
        }
        std::cout << "dense A8: quantizer == reference, WMMA == dot4 == serial reference bitwise: " << cases << " cases\n";
    }
    if (std::getenv("STRIX_DENSE_A8_BENCH")) bench(stream);
    check(hipStreamDestroy(stream), "destroy stream");
    return 0;
} catch (const std::exception& error) {
    std::cerr << "dense A8 fixture failed: " << error.what() << '\n';
    return 1;
}
