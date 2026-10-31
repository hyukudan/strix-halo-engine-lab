#include "device.hpp"
#include "gemma4/exl3/device.hpp"
#include <hip/hip_runtime.h>
#include <cstdlib>
#include <stdexcept>
#include <string_view>

namespace strix::gemma4::prefill {
namespace {
using Half16 = _Float16 __attribute__((ext_vector_type(16)));
using Float8 = float __attribute__((ext_vector_type(8)));
void checked() {
    if (const auto e = hipGetLastError(); e != hipSuccess) throw std::runtime_error(hipGetErrorString(e));
}
__device__ _Float16 rounded(float x, unsigned* failure) {
    // Native conversion is RNE under the target's fixed mode; fixture-tested
    // against half_bits, including half subnormals and the overflow midpoint.
    const auto h = static_cast<_Float16>(x);
    if ((__builtin_bit_cast(unsigned short, h) & 0x7c00U) == 0x7c00U) atomicOr(failure, 4U);
    return h;
}
template <bool Panels>
__global__ void convert(const float* x, _Float16* out, unsigned count, unsigned width, unsigned stride,
                        unsigned* failure, unsigned* maximum) {
    const auto i = std::size_t(blockIdx.x) * blockDim.x + threadIdx.x;
    unsigned bits = 0;
    if (i < std::size_t(count) * stride) {
        const unsigned r = unsigned(i / stride), c = unsigned(i % stride);
        const float a = c < width ? x[std::size_t(r) * width + c] : 0.0F;
        bits = __builtin_bit_cast(unsigned, a) & 0x7fffffffU;
        // Panels [K16][row][K] let adjacent matrix lanes load adjacent rows.
        const auto index = Panels ? std::size_t(c / 16) * count * 16 + r * 16 + c % 16 : i;
        out[index] = rounded(a, failure);
    }
    for (unsigned d = 16; d; d >>= 1U) bits = max(bits, __shfl_down(bits, d, 32));
    __shared__ unsigned maxima[8];
    if (threadIdx.x % 32U == 0) maxima[threadIdx.x / 32U] = bits;
    __syncthreads();
    if (!threadIdx.x) {
        unsigned result = 0;
        for (unsigned j = 0; j < 8; ++j) result = max(result, maxima[j]);
        atomicMax(maximum, result);
    }
}
// Panels are [N16][K16][N][K]. Packing is outside prompt timing and consumes
// exactly the same decoded FP32 coefficients and FP16 conversion as gemm.
template <WeightFormat Format>
__global__ void prepare_kernel(LinearView w, _Float16* out, unsigned stride, unsigned* failure) {
    const auto i = std::size_t(blockIdx.x) * blockDim.x + threadIdx.x;
    const auto count = std::size_t((w.outputs + 15U) / 16U) * 16 * stride;
    if (i >= count) return;
    const auto n = unsigned(i / (16ULL * stride)) * 16 + unsigned(i / 16) % 16;
    const auto k = unsigned(i / 256) % (stride / 16) * 16 + unsigned(i % 16);
    w.format = Format;
    out[i] = rounded(n < w.outputs && k < w.inputs ? gpu::coefficient(w, n, k) : 0.0F, failure);
}
// Stage four K16 panels per barrier. Four waves share a 64x64 output tile;
// each wave owns four accumulators throughout the increasing K16 sequence.
// A transposed LDS layout makes lanes visit adjacent rows/columns, avoiding
// the previous stride-16 bank conflicts. Arithmetic is still issued K16 by K16.
__global__ __launch_bounds__(128) void prepared_gemm(const _Float16* weights, const _Float16* x,
                                                    unsigned rows, unsigned outputs, unsigned stride,
                                                    float* out, unsigned* failure) {
    __shared__ _Float16 ax[2][64 * 64], bw[2][64 * 64];
    const unsigned tid = threadIdx.x, lane = tid % 32, wave = tid / 32;
    const unsigned m0 = blockIdx.y * 64, n0 = blockIdx.x * 64;
    const unsigned wm = (wave / 2) * 32, wn = (wave % 2) * 32;
    Float8 c00{}, c01{}, c10{}, c11{};
    for (unsigned base = 0; base < stride; base += 64) {
        const unsigned slot = (base / 64) % 2;
        for (unsigned i = tid; i < 64 * 64; i += 128) {
            const unsigned r = i / 64, k = i % 64;
            ax[slot][k * 64 + r] = m0 + r < rows && base + k < stride
                ? x[std::size_t(m0 + r) * stride + base + k] : _Float16(0);
            bw[slot][k * 64 + r] = n0 + r < outputs && base + k < stride
                ? weights[(std::size_t((n0 + r) / 16) * (stride / 16) + (base + k) / 16) * 256 + (r % 16) * 16 + k % 16]
                : _Float16(0);
        }
        __syncthreads();
#pragma unroll
        for (unsigned t = 0; t < 4; ++t) {
            if (base + t * 16 >= stride) break;
            Half16 a0, a1, b0, b1;
#pragma unroll
            for (unsigned k = 0; k < 16; ++k) {
                const auto index = (t * 16 + k) * 64 + lane % 16;
                a0[k] = ax[slot][index + wm]; a1[k] = ax[slot][index + wm + 16];
                b0[k] = bw[slot][index + wn]; b1[k] = bw[slot][index + wn + 16];
            }
            c00 = __builtin_amdgcn_wmma_f32_16x16x16_f16_w32(a0, b0, c00);
            c01 = __builtin_amdgcn_wmma_f32_16x16x16_f16_w32(a0, b1, c01);
            c10 = __builtin_amdgcn_wmma_f32_16x16x16_f16_w32(a1, b0, c10);
            c11 = __builtin_amdgcn_wmma_f32_16x16x16_f16_w32(a1, b1, c11);
        }
        __syncthreads();
    }
#pragma unroll
    for (unsigned e = 0; e < 8; ++e) {
        const auto r = m0 + wm + 2 * e + lane / 16, n = n0 + wn + lane % 16;
        const float values[4] = {c00[e], c01[e], c10[e], c11[e]};
#pragma unroll
        for (unsigned i = 0; i < 2; ++i)
#pragma unroll
            for (unsigned j = 0; j < 2; ++j)
                if (r + i * 16 < rows && n + j * 16 < outputs) {
                    const float a = values[i * 2 + j];
                    if ((__builtin_bit_cast(unsigned, a) & 0x7f800000U) == 0x7f800000U) atomicOr(failure, 1U);
                    out[std::size_t(r + i * 16) * outputs + n + j * 16] = a;
                }
    }
}
// A register-only alternative: four waves own four independent 64x64 tiles in
// a 128x128 workgroup. Contiguous panel vectors feed sixteen WMMA accumulators
// per wave; the device caches provide immutable operand reuse across waves.
template <unsigned RowWaves, bool Panels = false>
__global__ __launch_bounds__(RowWaves * 64) void direct_gemm(const _Float16* weights, const _Float16* x,
                                                  unsigned rows, unsigned outputs, unsigned stride,
                                                  float* out, unsigned* failure) {
    const unsigned lane = threadIdx.x % 32, wave = threadIdx.x / 32;
    const unsigned m0 = blockIdx.y * (RowWaves * 64) + (wave / 2) * 64, n0 = blockIdx.x * 128 + (wave % 2) * 64;
    Float8 c[4][4]{};
    for (unsigned k0 = 0; k0 < stride; k0 += 16) {
        Half16 a[4], b[4];
#pragma unroll
        for (unsigned i = 0; i < 4; ++i) {
            const auto r = m0 + i * 16 + lane % 16, n = n0 + i * 16 + lane % 16;
            const auto index = Panels ? std::size_t(k0) * rows + r * 16 : std::size_t(r) * stride + k0;
            a[i] = r < rows ? *reinterpret_cast<const Half16*>(x + index) : Half16{};
            b[i] = n < outputs ? *reinterpret_cast<const Half16*>(weights +
                (std::size_t(n / 16) * (stride / 16) + k0 / 16) * 256 + (n % 16) * 16) : Half16{};
        }
#pragma unroll
        for (unsigned i = 0; i < 4; ++i)
#pragma unroll
            for (unsigned j = 0; j < 4; ++j)
                c[i][j] = __builtin_amdgcn_wmma_f32_16x16x16_f16_w32(a[i], b[j], c[i][j]);
    }
#pragma unroll
    for (unsigned i = 0; i < 4; ++i)
#pragma unroll
        for (unsigned j = 0; j < 4; ++j)
#pragma unroll
            for (unsigned e = 0; e < 8; ++e) {
                const auto r = m0 + i * 16 + 2 * e + lane / 16, n = n0 + j * 16 + lane % 16;
                if (r < rows && n < outputs) {
                    const float a = c[i][j][e];
                    if ((bit_cast<unsigned>(a) & 0x7f800000U) == 0x7f800000U) atomicOr(failure, 1U);
                    out[std::size_t(r) * outputs + n] = a;
                }
            }
}
// Narrow projections need more workgroups than a 128-column output tile gives.
// Four waves own a 64x64 tile without changing any output's K16 recurrence.
__global__ __launch_bounds__(128) void narrow_gemm(const _Float16* weights, const _Float16* x,
                                                  unsigned rows, unsigned outputs, unsigned stride,
                                                  float* out, unsigned* failure) {
    const unsigned lane = threadIdx.x % 32, wave = threadIdx.x / 32;
    const unsigned m0 = blockIdx.y * 64 + (wave / 2) * 32, n0 = blockIdx.x * 64 + (wave % 2) * 32;
    Float8 c[2][2]{};
    for (unsigned k0 = 0; k0 < stride; k0 += 16) {
        Half16 a[2], b[2];
#pragma unroll
        for (unsigned i = 0; i < 2; ++i) {
            const auto r = m0 + i * 16 + lane % 16, n = n0 + i * 16 + lane % 16;
            a[i] = r < rows ? *reinterpret_cast<const Half16*>(x + std::size_t(k0) * rows + r * 16) : Half16{};
            b[i] = n < outputs ? *reinterpret_cast<const Half16*>(weights +
                (std::size_t(n / 16) * (stride / 16) + k0 / 16) * 256 + (n % 16) * 16) : Half16{};
        }
#pragma unroll
        for (unsigned i = 0; i < 2; ++i)
#pragma unroll
            for (unsigned j = 0; j < 2; ++j)
                c[i][j] = __builtin_amdgcn_wmma_f32_16x16x16_f16_w32(a[i], b[j], c[i][j]);
    }
#pragma unroll
    for (unsigned i = 0; i < 2; ++i)
#pragma unroll
        for (unsigned j = 0; j < 2; ++j)
#pragma unroll
            for (unsigned e = 0; e < 8; ++e) {
                const auto r = m0 + i * 16 + 2 * e + lane / 16, n = n0 + j * 16 + lane % 16;
                if (r < rows && n < outputs) {
                    const float a = c[i][j][e];
                    if ((bit_cast<unsigned>(a) & 0x7f800000U) == 0x7f800000U) atomicOr(failure, 1U);
                    out[std::size_t(r) * outputs + n] = a;
                }
            }
}
// Small last-row head calls also reuse the prepared panels. One wave owns N16,
// avoiding the padded row work of the large workgroup while retaining K order.
__global__ __launch_bounds__(32) void prepared_rows(const _Float16* weights, const _Float16* x,
                                                   unsigned rows, unsigned outputs, unsigned stride,
                                                   float* out, unsigned* failure) {
    const unsigned lane = threadIdx.x, r = blockIdx.y * 16 + lane % 16, n = blockIdx.x * 16 + lane % 16;
    Float8 c{};
    for (unsigned k0 = 0; k0 < stride; k0 += 16) {
        const Half16 a = r < rows ? *reinterpret_cast<const Half16*>(x + std::size_t(k0) * rows + r * 16) : Half16{};
        const Half16 b = n < outputs ? *reinterpret_cast<const Half16*>(weights +
            (std::size_t(n / 16) * (stride / 16) + k0 / 16) * 256 + (n % 16) * 16) : Half16{};
        c = __builtin_amdgcn_wmma_f32_16x16x16_f16_w32(a, b, c);
    }
#pragma unroll
    for (unsigned e = 0; e < 8; ++e) {
        const auto row = blockIdx.y * 16 + 2 * e + lane / 16;
        if (row < rows && n < outputs) {
            if ((bit_cast<unsigned>(c[e]) & 0x7f800000U) == 0x7f800000U) atomicOr(failure, 1U);
            out[std::size_t(row) * outputs + n] = c[e];
        }
    }
}
// Four waves, each with 2x2 independent 16x16 accumulators. LDS shares each
// decoded weight tile across 64 prompt rows; only increasing K tiles are issued.
template <WeightFormat Format>
__global__ __launch_bounds__(128) void gemm(LinearView w, const _Float16* x, unsigned rows, unsigned stride,
                                           float* out, unsigned* failure) {
    __shared__ _Float16 ax[64 * 16], bw[64 * 16];
    const unsigned tid = threadIdx.x, lane = tid % 32, wave = tid / 32;
    const unsigned m0 = blockIdx.y * 64, n0 = blockIdx.x * 64;
    const unsigned wm = (wave / 2) * 32, wn = (wave % 2) * 32;
    Float8 c00{}, c01{}, c10{}, c11{};
    w.format = Format;
    for (unsigned k0 = 0; k0 < stride; k0 += 16) {
        for (unsigned i = tid; i < 64 * 16; i += 128) {
            const auto r = i / 16, k = i % 16;
            ax[i] = m0 + r < rows ? x[std::size_t(m0 + r) * stride + k0 + k] : _Float16(0);
            const float b = n0 + r < w.outputs && k0 + k < w.inputs ? gpu::coefficient(w, n0 + r, k0 + k) : 0.0F;
            bw[i] = rounded(b, failure);
        }
        __syncthreads();
        Half16 a0, a1, b0, b1;
#pragma unroll
        for (unsigned k = 0; k < 16; ++k) {
            a0[k] = ax[(wm + lane % 16) * 16 + k];
            a1[k] = ax[(wm + 16 + lane % 16) * 16 + k];
            b0[k] = bw[(wn + lane % 16) * 16 + k];
            b1[k] = bw[(wn + 16 + lane % 16) * 16 + k];
        }
        c00 = __builtin_amdgcn_wmma_f32_16x16x16_f16_w32(a0, b0, c00);
        c01 = __builtin_amdgcn_wmma_f32_16x16x16_f16_w32(a0, b1, c01);
        c10 = __builtin_amdgcn_wmma_f32_16x16x16_f16_w32(a1, b0, c10);
        c11 = __builtin_amdgcn_wmma_f32_16x16x16_f16_w32(a1, b1, c11);
        __syncthreads();
    }
#pragma unroll
    for (unsigned e = 0; e < 8; ++e) {
        const auto r = m0 + wm + 2 * e + lane / 16, n = n0 + wn + lane % 16;
        const float values[4] = {c00[e], c01[e], c10[e], c11[e]};
#pragma unroll
        for (unsigned i = 0; i < 2; ++i)
#pragma unroll
            for (unsigned j = 0; j < 2; ++j)
                if (r + i * 16 < rows && n + j * 16 < w.outputs) {
                    const auto a = values[i * 2 + j];
                    if ((__builtin_bit_cast(unsigned, a) & 0x7f800000U) == 0x7f800000U) atomicOr(failure, 1U);
                    out[std::size_t(r + i * 16) * w.outputs + n + j * 16] = a;
                }
    }
}
template <WeightFormat Format>
__global__ __launch_bounds__(32) void reference_gemm(LinearView w, const _Float16* x, unsigned rows, unsigned stride,
                                                    float* out, unsigned* failure) {
    const auto lane = threadIdx.x, r = blockIdx.y * 16 + lane % 16, n = blockIdx.x * 16 + lane % 16;
    Float8 c{};
    w.format = Format;
    for (unsigned k0 = 0; k0 < stride; k0 += 16) {
        Half16 a, b;
#pragma unroll
        for (unsigned k = 0; k < 16; ++k) {
            a[k] = r < rows ? x[std::size_t(r) * stride + k0 + k] : _Float16(0);
            b[k] = rounded(n < w.outputs && k0 + k < w.inputs ? gpu::coefficient(w, n, k0 + k) : 0.0F, failure);
        }
        c = __builtin_amdgcn_wmma_f32_16x16x16_f16_w32(a, b, c);
    }
#pragma unroll
    for (unsigned e = 0; e < 8; ++e) {
        const auto row = blockIdx.y * 16 + 2 * e + lane / 16;
        if (row < rows && n < w.outputs) {
            if ((__builtin_bit_cast(unsigned, c[e]) & 0x7f800000U) == 0x7f800000U) atomicOr(failure, 1U);
            out[std::size_t(row) * w.outputs + n] = c[e];
        }
    }
}
// LDS-staged variant of direct_gemm<.., true>: a workgroup of 2 x WavesN waves
// stages 128 rows x (64*WavesN) outputs x K32 of the panel operands into
// double-buffered LDS (next stage prefetched into registers while the current
// one is multiplied), so each operand is fetched once per workgroup instead of
// once per wave. Every accumulator still issues the same increasing K16
// instructions on the same FP16 operands. Requires stride % 32 == 0.
template <unsigned WavesN>
__global__ __launch_bounds__(WavesN * 64) void staged_gemm(const _Float16* weights, const _Float16* x,
                                                          unsigned rows, unsigned outputs, unsigned stride,
                                                          float* out, unsigned* failure) {
    constexpr unsigned threads = WavesN * 64, tile_m = 128, tile_n = 64 * WavesN;
    constexpr unsigned a_vectors = tile_m * 32 / 8, b_vectors = tile_n * 32 / 8; // 16-byte vectors per stage
    constexpr unsigned a_loads = a_vectors / threads, b_loads = b_vectors / threads;
    using Vec = unsigned __attribute__((ext_vector_type(4)));
    // [stage][k16 of the stage][row or output][16]
    __shared__ _Float16 la[2][2][tile_m * 16], lb[2][2][tile_n * 16];
    const unsigned tid = threadIdx.x, lane = tid % 32, wave = tid / 32;
    const unsigned m_tile = blockIdx.y * tile_m, n_tile = blockIdx.x * tile_n;
    const unsigned wm = (wave / WavesN) * 64, wn = (wave % WavesN) * 64;
    const unsigned k16s = stride / 16, n16s = (outputs + 15U) / 16U;
    Vec ra[a_loads], rb[b_loads];
    // Vector v of a stage: k16 half h = v / (rows*2), then row and 16-byte half.
    auto fetch = [&](unsigned k0) {
#pragma unroll
        for (unsigned i = 0; i < a_loads; ++i) {
            const unsigned v = tid + i * threads, h = v / (tile_m * 2), row = (v / 2) % tile_m, part = v % 2;
            const unsigned r = m_tile + row;
            ra[i] = r < rows ? *reinterpret_cast<const Vec*>(x + (std::size_t(k0 / 16 + h) * rows + r) * 16 + part * 8) : Vec{};
        }
#pragma unroll
        for (unsigned i = 0; i < b_loads; ++i) {
            const unsigned v = tid + i * threads, h = v / (tile_n * 2), n = (v / 2) % tile_n, part = v % 2;
            const unsigned j = (n_tile + n) / 16;
            rb[i] = j < n16s ? *reinterpret_cast<const Vec*>(weights + (std::size_t(j) * k16s + k0 / 16 + h) * 256 + (n % 16) * 16 + part * 8)
                             : Vec{};
        }
    };
    auto store = [&](unsigned slot) {
#pragma unroll
        for (unsigned i = 0; i < a_loads; ++i) {
            const unsigned v = tid + i * threads, h = v / (tile_m * 2), row = (v / 2) % tile_m, part = v % 2;
            *reinterpret_cast<Vec*>(&la[slot][h][row * 16 + part * 8]) = ra[i];
        }
#pragma unroll
        for (unsigned i = 0; i < b_loads; ++i) {
            const unsigned v = tid + i * threads, h = v / (tile_n * 2), n = (v / 2) % tile_n, part = v % 2;
            *reinterpret_cast<Vec*>(&lb[slot][h][n * 16 + part * 8]) = rb[i];
        }
    };
    Float8 c[4][4]{};
    fetch(0); store(0);
    __syncthreads();
    for (unsigned k0 = 0; k0 < stride; k0 += 32) {
        const unsigned slot = (k0 / 32) % 2;
        if (k0 + 32 < stride) fetch(k0 + 32);
#pragma unroll
        for (unsigned h = 0; h < 2; ++h) {
            Half16 a[4], b[4];
#pragma unroll
            for (unsigned i = 0; i < 4; ++i) {
                a[i] = *reinterpret_cast<const Half16*>(&la[slot][h][(wm + i * 16 + lane % 16) * 16]);
                b[i] = *reinterpret_cast<const Half16*>(&lb[slot][h][(wn + i * 16 + lane % 16) * 16]);
            }
#pragma unroll
            for (unsigned i = 0; i < 4; ++i)
#pragma unroll
                for (unsigned j = 0; j < 4; ++j)
                    c[i][j] = __builtin_amdgcn_wmma_f32_16x16x16_f16_w32(a[i], b[j], c[i][j]);
        }
        if (k0 + 32 < stride) store(slot ^ 1U);
        __syncthreads();
    }
    unsigned bad = 0;
#pragma unroll
    for (unsigned i = 0; i < 4; ++i)
#pragma unroll
        for (unsigned j = 0; j < 4; ++j)
#pragma unroll
            for (unsigned e = 0; e < 8; ++e) {
                const auto r = m_tile + wm + i * 16 + 2 * e + lane / 16, n = n_tile + wn + j * 16 + lane % 16;
                if (r < rows && n < outputs) {
                    const float a = c[i][j][e];
                    bad |= (bit_cast<unsigned>(a) & 0x7f800000U) == 0x7f800000U;
                    out[std::size_t(r) * outputs + n] = a;
                }
            }
    if (bad) atomicOr(failure, 1U);
}
// Gate and up projections of the same panel input in one workgroup, in the
// register-only style of direct_gemm: eight waves, each owning 64 rows x 32
// outputs of both matrices, with the same increasing K16 chains as two
// separate GEMMs. The epilogue forms gelu_tanh(gate) * up in FP32 exactly as
// the product kernel, checks it as check_finite would, and stores it straight
// into the next linear's FP16 input panels with convert's RNE/overflow rule
// and abs-bits maximum. 256 rows x 64 outputs per workgroup.
__global__ __launch_bounds__(256) void gelu_pair_gemm(const _Float16* gate, const _Float16* up, const _Float16* x,
                                                     unsigned rows, unsigned outputs, unsigned stride,
                                                     _Float16* product, unsigned* failure, unsigned* maximum) {
    const unsigned lane = threadIdx.x % 32, wave = threadIdx.x / 32;
    const unsigned m0 = blockIdx.y * 256 + (wave / 2) * 64, n0 = blockIdx.x * 64 + (wave % 2) * 32;
    Float8 cg[4][2]{}, cu[4][2]{};
    for (unsigned k0 = 0; k0 < stride; k0 += 16) {
        Half16 a[4], bg[2], bu[2];
#pragma unroll
        for (unsigned i = 0; i < 4; ++i) {
            const auto r = m0 + i * 16 + lane % 16;
            a[i] = r < rows ? *reinterpret_cast<const Half16*>(x + std::size_t(k0) * rows + r * 16) : Half16{};
        }
#pragma unroll
        for (unsigned j = 0; j < 2; ++j) {
            const auto n = n0 + j * 16 + lane % 16;
            const auto offset = (std::size_t(n / 16) * (stride / 16) + k0 / 16) * 256 + (n % 16) * 16;
            bg[j] = n < outputs ? *reinterpret_cast<const Half16*>(gate + offset) : Half16{};
            bu[j] = n < outputs ? *reinterpret_cast<const Half16*>(up + offset) : Half16{};
        }
#pragma unroll
        for (unsigned i = 0; i < 4; ++i)
#pragma unroll
            for (unsigned j = 0; j < 2; ++j) {
                cg[i][j] = __builtin_amdgcn_wmma_f32_16x16x16_f16_w32(a[i], bg[j], cg[i][j]);
                cu[i][j] = __builtin_amdgcn_wmma_f32_16x16x16_f16_w32(a[i], bu[j], cu[i][j]);
            }
    }
    unsigned bad = 0, high = 0;
#pragma unroll
    for (unsigned i = 0; i < 4; ++i)
#pragma unroll
        for (unsigned j = 0; j < 2; ++j)
#pragma unroll
            for (unsigned e = 0; e < 8; ++e) {
                const auto r = m0 + i * 16 + 2 * e + lane / 16, n = n0 + j * 16 + lane % 16;
                if (r < rows && n < outputs) {
                    const float g = cg[i][j][e], u = cu[i][j][e];
                    bad |= (bit_cast<unsigned>(g) & 0x7f800000U) == 0x7f800000U ? 1U : 0U;
                    bad |= (bit_cast<unsigned>(u) & 0x7f800000U) == 0x7f800000U ? 1U : 0U;
                    float p = gelu_tanh(g) * u;
                    // The product is an FP32 value first (it was stored), then
                    // converted: never fold the multiply into v_fma_mix.
                    asm volatile("" : "+v"(p));
                    bad |= (bit_cast<unsigned>(p) & 0x7f800000U) == 0x7f800000U ? 1U : 0U;
                    high = max(high, bit_cast<unsigned>(p) & 0x7fffffffU);
                    const auto hp = static_cast<_Float16>(p);
                    bad |= (bit_cast<unsigned short>(hp) & 0x7c00U) == 0x7c00U ? 4U : 0U;
                    product[std::size_t(n / 16) * rows * 16 + std::size_t(r) * 16 + n % 16] = hp;
                }
            }
    if (bad) atomicOr(failure, bad);
    for (unsigned d = 16; d; d >>= 1U) high = max(high, __shfl_down(high, d, 32));
    if (lane == 0) atomicMax(maximum, high);
}
} // namespace
bool staged_enabled() {
    static const bool enabled = [] {
        const auto* p = std::getenv("STRIX_G4_PREFILL_STAGED");
        return !p || std::string_view(p) != "0";
    }();
    return enabled;
}
bool tiled_enabled() {
    static const bool enabled = [] {
        const auto* p = std::getenv("STRIX_G4_PREFILL_TILING");
        return !p || std::string_view(p) != "0";
    }();
    return enabled;
}
bool prepared_enabled() {
    static const bool enabled = [] {
        const auto* p = std::getenv("STRIX_G4_PREFILL_PREPARE");
        return !p || std::string_view(p) != "0";
    }();
    return enabled;
}
unsigned direct_mode() {
    static const unsigned mode = [] {
        const auto* p = std::getenv("STRIX_G4_PREFILL_DIRECT");
        if (!p || std::string_view(p) == "3") return 3U;
        if (std::string_view(p) == "4") return 4U;
        if (std::string_view(p) == "2") return 2U;
        if (std::string_view(p) == "0") return 0U;
        if (std::string_view(p) == "1") return 1U;
        throw std::invalid_argument("unknown prefill direct tile");
    }();
    return mode;
}
PreparedLinear::PreparedLinear(LinearView w, gpu::Queue queue) : weights_(w) {
    if (!w.data || !w.inputs || w.inputs > 10752 || !w.outputs || w.outputs > vocab ||
        (w.format == WeightFormat::exl3_mul1 ? !exl3::geometry(w) : w.row_bytes < packed_row_bytes(w.format, w.inputs))) throw std::invalid_argument("prepared linear geometry");
    const auto stride = (w.inputs + 15U) / 16U * 16U;
    panels_.resize(std::size_t((w.outputs + 15U) / 16U) * 16 * stride * sizeof(std::uint16_t));
    gpu::Buffer failure(sizeof(unsigned)); failure.zero();
    const auto q = static_cast<hipStream_t>(queue);
    switch (w.format) {
#define G4_PREPARE(F) case WeightFormat::F: \
        hipLaunchKernelGGL((prepare_kernel<WeightFormat::F>), dim3(unsigned((panels_.size() / 2 + 255) / 256)), dim3(256), 0, q, \
                           w, panels_.as<_Float16>(), stride, failure.as<unsigned>()); break
        G4_PREPARE(bf16); G4_PREPARE(f32); G4_PREPARE(q8_0); G4_PREPARE(q4_k); G4_PREPARE(q6_k);
        G4_PREPARE(exl3_mul1);
#undef G4_PREPARE
        default: throw std::invalid_argument("prepared linear format");
    }
    checked(); gpu::synchronize(queue);
    unsigned flag = 0; failure.download(&flag, sizeof(flag));
    if (flag) throw std::runtime_error("prepared linear FP16 overflow/nonfinite coefficient");
}
void linear(LinearView w, const float* x, unsigned rows, float* out, void* scratch, unsigned* failure,
            unsigned* maximum, gpu::Queue queue, bool tiled, const PreparedLinear* prepared, void* converted_event, bool converted) {
    if (w.format == WeightFormat::exl3_mul1) {
        if (prepared) {
            const auto p = prepared->weights();
            if (p.data != w.data || p.inputs != w.inputs || p.outputs != w.outputs || p.row_bytes != w.row_bytes || p.format != w.format)
                throw std::invalid_argument("EXL3 prepared linear mismatch");
        }
        exl3::device_prefill(w, x, rows, out, scratch, failure, maximum,
            prepared ? prepared->panels().data() : nullptr, queue, converted_event);
        return;
    }
    if (!rows || rows > chunk_rows || !w.data || !w.inputs || !w.outputs || !x || !out || !scratch || !failure || !maximum)
        throw std::invalid_argument("WMMA linear geometry or scratch");
    if (w.inputs > 10752 || w.outputs > vocab || w.row_bytes < packed_row_bytes(w.format, w.inputs))
        throw std::invalid_argument("WMMA linear tensor bounds");
    if (w.format != WeightFormat::bf16 && w.format != WeightFormat::f32 && w.format != WeightFormat::q8_0 &&
        w.format != WeightFormat::q4_k && w.format != WeightFormat::q6_k)
        throw std::invalid_argument("WMMA linear weight format");
    const auto stride = ((w.inputs + 15U) / 16U) * 16U;
    const auto q = static_cast<hipStream_t>(queue);
    tiled &= tiled_enabled();
    const auto mode = prepared && tiled ? direct_mode() : 0;
    // The staged kernel measured faster only for long-K projections (down:
    // 33 vs 29 TF at M512); wide outputs keep the register-only tiles.
    const bool staged = mode >= 3 && staged_enabled() && stride % 32 == 0 && stride >= 8192 && w.outputs > 1024 && w.outputs <= 4096 && rows > 8;
    const auto direct = mode == 4 ? 3U : mode;
    if (converted && direct != 3) throw std::invalid_argument("pre-converted input requires panel GEMMs");
    if (converted) {}
    else if (direct == 3)
        hipLaunchKernelGGL((convert<true>), dim3(unsigned((std::size_t(rows) * stride + 255) / 256)), dim3(256), 0, q,
                           x, static_cast<_Float16*>(scratch), rows, w.inputs, stride, failure, maximum);
    else
        hipLaunchKernelGGL((convert<false>), dim3(unsigned((std::size_t(rows) * stride + 255) / 256)), dim3(256), 0, q,
                           x, static_cast<_Float16*>(scratch), rows, w.inputs, stride, failure, maximum);
    checked();
    if (converted_event)
        if (const auto e = hipEventRecord(static_cast<hipEvent_t>(converted_event), q); e != hipSuccess)
            throw std::runtime_error(hipGetErrorString(e));
    if (prepared && tiled) {
        const auto p = prepared->weights();
        if (p.data != w.data || p.inputs != w.inputs || p.outputs != w.outputs || p.row_bytes != w.row_bytes || p.format != w.format)
            throw std::invalid_argument("prepared linear tensor mismatch");
        if (staged && w.outputs >= 4096)
            hipLaunchKernelGGL(staged_gemm<4>, dim3((w.outputs + 255U) / 256U, (rows + 127U) / 128U), dim3(256), 0, q,
                               prepared->panels().as<_Float16>(), static_cast<const _Float16*>(scratch), rows, w.outputs, stride, out, failure);
        else if (staged)
            hipLaunchKernelGGL(staged_gemm<2>, dim3((w.outputs + 127U) / 128U, (rows + 127U) / 128U), dim3(128), 0, q,
                               prepared->panels().as<_Float16>(), static_cast<const _Float16*>(scratch), rows, w.outputs, stride, out, failure);
        else if (direct == 3 && rows <= 8)
            hipLaunchKernelGGL(prepared_rows, dim3((w.outputs + 15U) / 16U, (rows + 15U) / 16U), dim3(32), 0, q,
                               prepared->panels().as<_Float16>(), static_cast<const _Float16*>(scratch), rows, w.outputs, stride, out, failure);
        else if (direct == 3 && w.outputs <= 1024)
            hipLaunchKernelGGL(narrow_gemm, dim3((w.outputs + 63U) / 64U, (rows + 63U) / 64U), dim3(128), 0, q,
                               prepared->panels().as<_Float16>(), static_cast<const _Float16*>(scratch), rows, w.outputs, stride, out, failure);
        else if (direct == 3 && w.outputs >= 8192)
            hipLaunchKernelGGL((direct_gemm<4, true>), dim3((w.outputs + 127U) / 128U, (rows + 255U) / 256U), dim3(256), 0, q,
                               prepared->panels().as<_Float16>(), static_cast<const _Float16*>(scratch), rows, w.outputs, stride, out, failure);
        else if (direct == 3)
            hipLaunchKernelGGL((direct_gemm<2, true>), dim3((w.outputs + 127U) / 128U, (rows + 127U) / 128U), dim3(128), 0, q,
                               prepared->panels().as<_Float16>(), static_cast<const _Float16*>(scratch), rows, w.outputs, stride, out, failure);
        else if (direct == 2 && w.outputs > 512)
            hipLaunchKernelGGL((direct_gemm<4>), dim3((w.outputs + 127U) / 128U, (rows + 255U) / 256U), dim3(256), 0, q,
                               prepared->panels().as<_Float16>(), static_cast<const _Float16*>(scratch), rows, w.outputs, stride, out, failure);
        else if (direct)
            hipLaunchKernelGGL((direct_gemm<2>), dim3((w.outputs + 127U) / 128U, (rows + 127U) / 128U), dim3(128), 0, q,
                               prepared->panels().as<_Float16>(), static_cast<const _Float16*>(scratch), rows, w.outputs, stride, out, failure);
        else
            hipLaunchKernelGGL(prepared_gemm, dim3((w.outputs + 63U) / 64U, (rows + 63U) / 64U), dim3(128), 0, q,
                               prepared->panels().as<_Float16>(), static_cast<const _Float16*>(scratch), rows, w.outputs, stride, out, failure);
        checked(); return;
    }
    switch (w.format) {
#define G4_PF_FORMAT(F) case WeightFormat::F: \
    if (tiled) \
        hipLaunchKernelGGL((gemm<WeightFormat::F>), dim3((w.outputs + 63U) / 64U, (rows + 63U) / 64U), dim3(128), 0, q, \
                           w, static_cast<const _Float16*>(scratch), rows, stride, out, failure); \
    else \
        hipLaunchKernelGGL((reference_gemm<WeightFormat::F>), dim3((w.outputs + 15U) / 16U, (rows + 15U) / 16U), dim3(32), 0, q, \
                           w, static_cast<const _Float16*>(scratch), rows, stride, out, failure); break
        G4_PF_FORMAT(bf16);
        G4_PF_FORMAT(f32);
        G4_PF_FORMAT(q8_0);
        G4_PF_FORMAT(q4_k);
        G4_PF_FORMAT(q6_k);
#undef G4_PF_FORMAT
    }
    checked();
}
bool fused_ffn_enabled() {
    static const bool enabled = [] {
        const auto* p = std::getenv("STRIX_G4_PREFILL_FUSE_FFN");
        return !p || std::string_view(p) != "0";
    }();
    return enabled;
}
bool gelu_pair_supported(const PreparedLinear& gate, const PreparedLinear& up) {
    const auto g = gate.weights(), u = up.weights();
    return g.inputs == u.inputs && g.outputs == u.outputs && g.format != WeightFormat::exl3_mul1 &&
           u.format != WeightFormat::exl3_mul1 && g.outputs % 16U == 0;
}
void gelu_pair(const PreparedLinear& gate, const PreparedLinear& up, const float* x, unsigned rows, void* scratch,
               void* product, unsigned* failure, unsigned* maximum, gpu::Queue queue) {
    const auto g = gate.weights();
    if (!rows || rows > chunk_rows || !x || !scratch || !product || !failure || !maximum || !gelu_pair_supported(gate, up))
        throw std::invalid_argument("fused gate/up geometry");
    const auto stride = (g.inputs + 15U) / 16U * 16U;
    const auto q = static_cast<hipStream_t>(queue);
    hipLaunchKernelGGL((convert<true>), dim3(unsigned((std::size_t(rows) * stride + 255) / 256)), dim3(256), 0, q,
                       x, static_cast<_Float16*>(scratch), rows, g.inputs, stride, failure, maximum);
    checked();
    // Both projections read the same converted input: same recorded maximum.
    if (const auto e = hipMemcpyAsync(maximum + 1, maximum, sizeof(unsigned), hipMemcpyDeviceToDevice, q); e != hipSuccess)
        throw std::runtime_error(hipGetErrorString(e));
    hipLaunchKernelGGL(gelu_pair_gemm, dim3((g.outputs + 63U) / 64U, (rows + 255U) / 256U), dim3(256), 0, q,
                       gate.panels().as<_Float16>(), up.panels().as<_Float16>(), static_cast<const _Float16*>(scratch),
                       rows, g.outputs, stride, static_cast<_Float16*>(product), failure, maximum + 2);
    checked();
}
} // namespace strix::gemma4::prefill
